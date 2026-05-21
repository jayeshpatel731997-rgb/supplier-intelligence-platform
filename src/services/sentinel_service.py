"""Production Sentinel service with safe demo and failure handling."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from news_intelligence import SupplierNewsImpact, run_sentinel_scan
from src.config import Settings
from src.models import NewsEvent, SupplierEventMatch
from src.repositories.alerts import AlertRepository
from src.repositories.audit import AuditRepository
from src.repositories.suppliers import SupplierRepository
from src.tenancy import DEMO_TENANT_ID


@dataclass(slots=True)
class SentinelScanResult:
    events: list[dict]
    mode_used: str
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _supplier_frame(session: Session, tenant_id: str) -> pd.DataFrame:
    rows = []
    for supplier in SupplierRepository(session, tenant_id).list(limit=10_000):
        rows.append(
            {
                "supplier_name": supplier.name,
                "country": supplier.country,
                "category": supplier.category,
                "annual_spend": supplier.annual_spend,
            }
        )
    return pd.DataFrame(rows)


def _impact_to_dict(impact: SupplierNewsImpact) -> dict:
    return {
        "title": impact.article.title,
        "source": impact.article.source,
        "url": impact.article.url,
        "published_at": impact.article.published_at.isoformat() if impact.article.published_at else None,
        "disruption_type": impact.disruption_type,
        "severity": impact.severity,
        "severity_score": impact.severity_score,
        "confidence": impact.confidence,
        "affected_suppliers": impact.affected_suppliers,
        "affected_countries": impact.affected_countries,
        "affected_categories": impact.affected_categories,
        "estimated_exposure_usd": impact.estimated_exposure_usd,
        "summary": impact.summary,
        "recommended_actions": impact.recommended_actions,
        "analysis_method": impact.analysis_method,
    }


class SentinelService:
    def __init__(self, session: Session, settings: Settings, tenant_id: str = DEMO_TENANT_ID):
        self.session = session
        self.settings = settings
        self.tenant_id = tenant_id

    def scan(self, supplier_df: pd.DataFrame | None = None, mode: str = "demo", provider=None) -> SentinelScanResult:
        df = supplier_df if supplier_df is not None else _supplier_frame(self.session, self.tenant_id)
        if df.empty:
            df = pd.DataFrame([{"supplier_name": "Demo Supplier", "country": "USA", "category": "Machining", "annual_spend": 0.0}])

        try:
            if provider is not None:
                impacts, mode_used, error = provider(df, mode)
            else:
                chosen_mode = mode
                if mode == "live_ai" and not (self.settings.newsapi_key and (self.settings.openai_api_key or self.settings.anthropic_api_key)):
                    message = "Live News + AI requires NewsAPI plus OpenAI or Anthropic key."
                    AlertRepository(self.session, self.tenant_id).create_alert("sentinel_api_failure", "medium", message)
                    AuditRepository(self.session, self.tenant_id).log("sentinel.scan_skipped", details={"mode": mode, "error": message})
                    return SentinelScanResult(events=[], mode_used="Live News + AI", error=message)
                impacts, mode_used, error = run_sentinel_scan(
                    news_api_key=self.settings.newsapi_key,
                    supplier_df=df,
                    anthropic_api_key=self.settings.anthropic_api_key,
                    openai_api_key=self.settings.openai_api_key,
                    llm_provider="openai" if self.settings.openai_api_key else "anthropic",
                    mode=chosen_mode,
                )
        except Exception as exc:
            message = str(exc)
            AlertRepository(self.session, self.tenant_id).create_alert("sentinel_api_failure", "medium", f"Sentinel scan failed: {message}")
            AuditRepository(self.session, self.tenant_id).log("sentinel.scan_failed", details={"mode": mode, "error": message})
            return SentinelScanResult(events=[], mode_used=mode, error=message)

        events = []
        for impact in impacts:
            event_dict = _impact_to_dict(impact)
            events.append(event_dict)
            event_id = impact.article.article_id
            existing = self.session.scalar(
                select(NewsEvent).where(NewsEvent.tenant_id == self.tenant_id, NewsEvent.event_id == event_id)
            )
            if existing is None:
                existing = NewsEvent(tenant_id=self.tenant_id, event_id=event_id, title=impact.article.title)
                self.session.add(existing)
            existing.title = impact.article.title
            existing.source = impact.article.source
            existing.url = impact.article.url
            existing.published_at = impact.article.published_at
            existing.disruption_type = impact.disruption_type
            existing.severity = impact.severity
            existing.confidence = impact.confidence
            existing.summary = impact.summary

            if impact.severity in {"critical", "high"} or impact.estimated_exposure_usd >= self.settings.alert_exposure_threshold:
                AlertRepository(self.session, self.tenant_id).create_alert(
                    "new_disruption_event",
                    impact.severity,
                    impact.summary or impact.article.title,
                    exposure=impact.estimated_exposure_usd,
                )
            for supplier_name in impact.affected_suppliers:
                supplier = next((s for s in SupplierRepository(self.session, self.tenant_id).list() if s.name == supplier_name), None)
                if supplier:
                    self.session.add(
                        SupplierEventMatch(
                            tenant_id=self.tenant_id,
                            event_id=event_id,
                            supplier_id=supplier.supplier_id,
                            exposure=impact.estimated_exposure_usd,
                            match_reason="country/category/name match",
                        )
                    )

        AuditRepository(self.session, self.tenant_id).log(
            "sentinel.scan_completed",
            details={"mode": mode_used, "event_count": len(events), "error": error},
        )
        return SentinelScanResult(events=events, mode_used=mode_used, error=error or "")

    def list_events(self, limit: int = 100) -> list[dict]:
        rows = self.session.scalars(
            select(NewsEvent).where(NewsEvent.tenant_id == self.tenant_id).order_by(NewsEvent.created_at.desc()).limit(limit)
        )
        return [
            {
                "event_id": row.event_id,
                "title": row.title,
                "source": row.source,
                "severity": row.severity,
                "confidence": row.confidence,
                "disruption_type": row.disruption_type,
                "summary": row.summary,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
