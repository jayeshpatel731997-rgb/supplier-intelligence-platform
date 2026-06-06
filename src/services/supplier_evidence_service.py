"""Tenant-scoped supplier weak-signal and evidence-chain persistence."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from src.config import Settings
from src.models import (
    SupplierConnectorSync,
    SupplierEvidenceAction,
    SupplierEvidenceRun,
    SupplierEvidenceRunSupplier,
    SupplierEvidenceScoringVersion,
    SupplierWeakSignal,
)
from src.observability.logging import redact_secret_text
from src.repositories.audit import AuditRepository
from src.services.connectors import run_connector
from src.services.narrative_provider import build_evidence_narrative
from src.services.supplier_risk_evidence_chain import (
    SIGNAL_TYPE_WEIGHTS,
    build_supplier_risk_evidence_chains,
    load_demo_supplier_signals,
)
from src.tenancy import DEMO_TENANT_ID


VALID_ACTION_STATUSES = {"open", "in_progress", "blocked", "completed", "dismissed"}
SUPPORTED_CONNECTORS = [
    {
        "source_system": "erp",
        "connector_type": "api",
        "signal_types": ["operational", "erp"],
        "description": "ERP receiving history, open POs, vendor master, and performance data.",
    },
    {
        "source_system": "supplier_portal",
        "connector_type": "api",
        "signal_types": ["operational", "audit", "email"],
        "description": "Supplier portal status, acknowledgements, recovery plans, and audit artifacts.",
    },
    {
        "source_system": "financial_filings",
        "connector_type": "filing_feed",
        "signal_types": ["financial"],
        "description": "Public filings, liquidity indicators, bankruptcy notices, and credit signals.",
    },
    {
        "source_system": "email",
        "connector_type": "mailbox_digest",
        "signal_types": ["email"],
        "description": "Supplier email digests, expedite requests, force-majeure notices, and escalation threads.",
    },
    {
        "source_system": "hiring",
        "connector_type": "labor_market_feed",
        "signal_types": ["hiring"],
        "description": "Hiring trend feeds and staffing contraction signals.",
    },
    {
        "source_system": "logistics",
        "connector_type": "logistics_feed",
        "signal_types": ["news", "operational"],
        "description": "Port disruption, lane delays, carrier events, and geographic logistics signals.",
    },
]


def _loads(raw: str, fallback: Any) -> Any:
    try:
        return json.loads(raw or "")
    except json.JSONDecodeError:
        return fallback


def _parse_observed_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    except ValueError:
        return None


def _signal_to_dict(row: SupplierWeakSignal) -> dict[str, Any]:
    return {
        "id": row.id,
        "tenant_id": row.tenant_id,
        "signal_id": row.signal_id,
        "supplier_id": row.supplier_id,
        "supplier_name": row.supplier_name,
        "signal_type": row.signal_type,
        "driver": row.driver,
        "source": row.source,
        "source_url": row.source_url,
        "source_system": row.source_system,
        "observed_at": row.observed_at.isoformat() if row.observed_at else "",
        "severity": row.severity,
        "confidence": row.confidence,
        "summary": row.summary,
    }


def _run_supplier_to_report(row: SupplierEvidenceRunSupplier) -> dict[str, Any]:
    return {
        "supplier_id": row.supplier_id,
        "supplier_name": row.supplier_name,
        "risk_score": row.risk_score,
        "risk_level": row.risk_level,
        "top_risk_drivers": _loads(row.top_risk_drivers_json, []),
        "evidence_chain": _loads(row.evidence_chain_json, []),
        "recommended_actions": _loads(row.recommended_actions_json, []),
        "confidence": row.confidence,
    }


def _action_to_dict(row: SupplierEvidenceAction) -> dict[str, Any]:
    return {
        "id": row.id,
        "tenant_id": row.tenant_id,
        "run_id": row.run_id,
        "supplier_id": row.supplier_id,
        "supplier_name": row.supplier_name,
        "action": row.action,
        "source_driver": row.source_driver,
        "status": row.status,
        "owner": row.owner,
        "updated_by": row.updated_by,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _sync_to_dict(row: SupplierConnectorSync) -> dict[str, Any]:
    return {
        "id": row.id,
        "tenant_id": row.tenant_id,
        "source_system": row.source_system,
        "connector": row.source_system,
        "connector_type": row.connector_type,
        "status": row.status,
        "records_received": row.records_received,
        "records_accepted": row.records_accepted,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "error": redact_secret_text(row.error),
        "metadata": _safe_metadata(_loads(row.metadata_json, {})),
    }


def _safe_source_label(value: object) -> str:
    text = redact_secret_text(value)
    try:
        parsed = urlsplit(text)
    except ValueError:
        return text
    if not parsed.scheme or not parsed.netloc:
        return text
    host = parsed.hostname or parsed.netloc.rsplit("@", 1)[-1]
    try:
        port = parsed.port
    except ValueError:
        return f"{parsed.scheme}://***"
    if port:
        host = f"{host}:{port}"
    return urlunsplit((parsed.scheme, host, "", "", ""))


def _safe_path(path: str) -> str:
    credential_markers = ("token", "secret", "api-key", "apikey", "auth", "signature", "credential")
    segments = path.split("/")
    redact_next = False
    safe_segments: list[str] = []
    for segment in segments:
        lowered = segment.lower()
        opaque_secret = len(segment) >= 32 and segment.replace("-", "").replace("_", "").isalnum()
        sensitive = redact_next or any(marker in lowered for marker in credential_markers) or opaque_secret
        safe_segments.append("***" if sensitive and segment else segment)
        redact_next = lowered in {"token", "key", "secret", "auth", "signature", "credential"}
    return "/".join(safe_segments)


def _safe_signal_url(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = urlsplit(text)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    host = parsed.hostname
    try:
        port = parsed.port
    except ValueError:
        return f"{parsed.scheme}://***"
    if port:
        host = f"{host}:{port}"
    return urlunsplit((parsed.scheme, host, _safe_path(parsed.path), "", ""))


def _safe_metadata(metadata: Any) -> Any:
    if isinstance(metadata, dict):
        safe: dict[str, Any] = {}
        for key, value in metadata.items():
            if key in {"source_names", "source_urls", "urls"} and isinstance(value, list):
                safe[key] = [_safe_source_label(item) for item in value]
            else:
                safe[key] = _safe_metadata(value)
        return safe
    if isinstance(metadata, list):
        return [_safe_metadata(item) for item in metadata]
    if isinstance(metadata, str):
        return redact_secret_text(metadata)
    return metadata


class SupplierEvidenceService:
    def __init__(self, session: Session, tenant_id: str = DEMO_TENANT_ID):
        self.session = session
        self.tenant_id = tenant_id

    @staticmethod
    def connector_catalog(settings: Settings | None = None) -> dict[str, Any]:
        mode = settings.connector_mode if settings is not None else "demo"
        convex_configured = bool(settings and settings.convex_url and settings.convex_deploy_key)
        return {
            "connectors": SUPPORTED_CONNECTORS,
            "ingestion_endpoint": "/evidence/signals/import",
            "sync_endpoint": "/evidence/connectors/{connector_name}/sync",
            "storage_provider": "sqlalchemy",
            "mode": mode,
            "stub_available": True,
            "public_available": True,
            "convex_status": "configured" if convex_configured else "not_configured",
        }

    def save_scoring_config(
        self,
        version: str,
        description: str = "",
        signal_type_weights: dict[str, float] | None = None,
        supplier_criticality: dict[str, float] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        version = (version or "default-v1").strip()
        self.session.execute(
            update(SupplierEvidenceScoringVersion)
            .where(SupplierEvidenceScoringVersion.tenant_id == self.tenant_id)
            .values(is_active=False)
        )
        row = self.session.scalar(
            select(SupplierEvidenceScoringVersion).where(
                SupplierEvidenceScoringVersion.tenant_id == self.tenant_id,
                SupplierEvidenceScoringVersion.version == version,
            )
        )
        if row is None:
            row = SupplierEvidenceScoringVersion(tenant_id=self.tenant_id, version=version)
            self.session.add(row)
        row.description = description or ""
        row.signal_type_weights_json = json.dumps(signal_type_weights or SIGNAL_TYPE_WEIGHTS)
        row.supplier_criticality_json = json.dumps(supplier_criticality or {})
        row.is_active = True
        row.created_by = actor
        AuditRepository(self.session, self.tenant_id).log(
            "evidence.scoring_config_saved",
            username=actor,
            details={"version": version},
        )
        self.session.flush()
        return self.scoring_config_to_dict(row)

    def current_scoring_config(self, requested_version: str = "") -> dict[str, Any]:
        stmt = select(SupplierEvidenceScoringVersion).where(SupplierEvidenceScoringVersion.tenant_id == self.tenant_id)
        if requested_version:
            stmt = stmt.where(SupplierEvidenceScoringVersion.version == requested_version)
        else:
            stmt = stmt.where(SupplierEvidenceScoringVersion.is_active.is_(True))
        row = self.session.scalar(stmt.order_by(SupplierEvidenceScoringVersion.created_at.desc()))
        if row is None:
            return {
                "id": None,
                "tenant_id": self.tenant_id,
                "version": "default-v1",
                "description": "Built-in deterministic scoring defaults.",
                "signal_type_weights": dict(SIGNAL_TYPE_WEIGHTS),
                "supplier_criticality": {},
                "is_active": True,
                "created_by": "built-in",
                "persisted": False,
            }
        return self.scoring_config_to_dict(row)

    @staticmethod
    def scoring_config_to_dict(row: SupplierEvidenceScoringVersion) -> dict[str, Any]:
        return {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "version": row.version,
            "description": row.description,
            "signal_type_weights": _loads(row.signal_type_weights_json, {}),
            "supplier_criticality": _loads(row.supplier_criticality_json, {}),
            "is_active": row.is_active,
            "created_by": row.created_by,
            "persisted": True,
        }

    def import_signals(
        self,
        source_system: str,
        signals: list[dict[str, Any]],
        connector_type: str = "api",
        actor: str = "system",
    ) -> dict[str, Any]:
        accepted = self.upsert_signals(source_system, signals)
        sync = SupplierConnectorSync(
            tenant_id=self.tenant_id,
            source_system=source_system,
            connector_type=connector_type,
            records_received=len(signals),
            records_accepted=accepted,
            status="completed",
        )
        self.session.add(sync)
        AuditRepository(self.session, self.tenant_id).log(
            "evidence.signals_imported",
            username=actor,
            details={"source_system": source_system, "accepted": accepted},
        )
        self.session.flush()
        return {
            "tenant_id": self.tenant_id,
            "source_system": source_system,
            "connector_type": connector_type,
            "received": len(signals),
            "accepted": accepted,
            "sync_id": sync.id,
        }

    def upsert_signals(self, source_system: str, signals: list[dict[str, Any]]) -> int:
        accepted = 0
        for signal in signals:
            safe_signal = dict(signal)
            safe_signal["source_url"] = _safe_signal_url(signal.get("source_url"))
            signal_id = str(signal.get("signal_id") or uuid.uuid4().hex)
            row = self.session.scalar(
                select(SupplierWeakSignal).where(
                    SupplierWeakSignal.tenant_id == self.tenant_id,
                    SupplierWeakSignal.signal_id == signal_id,
                )
            )
            if row is None:
                row = SupplierWeakSignal(tenant_id=self.tenant_id, signal_id=signal_id)
                self.session.add(row)
            row.supplier_id = str(signal.get("supplier_id") or "unknown-supplier")
            row.supplier_name = str(signal.get("supplier_name") or row.supplier_id)
            row.signal_type = str(signal.get("signal_type") or "operational").lower()
            row.driver = str(signal.get("driver") or "Unclassified signal")
            row.source = str(signal.get("source") or source_system)
            row.source_url = safe_signal["source_url"]
            row.source_system = source_system
            row.observed_at = _parse_observed_at(signal.get("observed_at"))
            row.severity = float(signal.get("severity") or 0.0)
            row.confidence = float(signal.get("confidence") or 0.5)
            row.summary = str(signal.get("summary") or "")
            row.raw_payload_json = json.dumps(safe_signal, default=str)
            accepted += 1
        return accepted

    def run_connector_sync(self, connector_name: str, settings: Settings, actor: str = "system") -> dict[str, Any]:
        result = run_connector(connector_name, settings, self.tenant_id)
        accepted = 0
        if result.status == "completed" and result.signals:
            accepted = self.upsert_signals(result.connector, result.signals)
        sync = SupplierConnectorSync(
            tenant_id=self.tenant_id,
            source_system=result.connector,
            connector_type=result.connector_type,
            status=result.status,
            records_received=result.records_received,
            records_accepted=accepted,
            started_at=result.started_at,
            finished_at=result.finished_at,
            error=redact_secret_text(result.error),
            metadata_json=json.dumps(_safe_metadata(result.metadata), default=str),
        )
        self.session.add(sync)
        AuditRepository(self.session, self.tenant_id).log(
            "evidence.connector_synced",
            username=actor,
            details={"connector": result.connector, "status": result.status, "accepted": accepted},
        )
        self.session.flush()
        payload = _sync_to_dict(sync)
        payload["connector"] = result.connector
        return payload

    def list_connector_syncs(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.session.scalars(
            select(SupplierConnectorSync)
            .where(SupplierConnectorSync.tenant_id == self.tenant_id)
            .order_by(SupplierConnectorSync.created_at.desc())
            .limit(limit)
        )
        return [_sync_to_dict(row) for row in rows]

    def seed_demo_signals_if_empty(self) -> None:
        existing = self.session.scalar(
            select(SupplierWeakSignal.id).where(SupplierWeakSignal.tenant_id == self.tenant_id).limit(1)
        )
        if existing is None:
            self.import_signals("demo_seed", load_demo_supplier_signals(), connector_type="local_demo")

    def list_signals(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = self.session.scalars(
            select(SupplierWeakSignal)
            .where(SupplierWeakSignal.tenant_id == self.tenant_id)
            .order_by(SupplierWeakSignal.created_at.desc())
            .limit(limit)
        )
        return [_signal_to_dict(row) for row in rows]

    def run_evidence_chain(
        self,
        scoring_version: str = "",
        include_demo_signals: bool = False,
        actor: str = "system",
        settings: Settings | None = None,
    ) -> dict[str, Any]:
        if include_demo_signals:
            if settings is None or not settings.demo_mode or settings.is_staging_or_production:
                raise ValueError("Demo signals require local demo mode.")
            if self.tenant_id != DEMO_TENANT_ID:
                raise ValueError("Demo signals may only be injected for demo-tenant.")
            self.seed_demo_signals_if_empty()
        config = self.current_scoring_config(scoring_version)
        signals = self.list_signals(limit=10_000)
        reports = build_supplier_risk_evidence_chains(
            signals,
            signal_type_weights=config["signal_type_weights"],
            supplier_criticality=config["supplier_criticality"],
        )
        run_id = f"evr_{uuid.uuid4().hex[:16]}"
        narrative = build_evidence_narrative(reports, config["version"], settings=settings)
        narrative["summary"] = narrative["risk_summary"]
        narrative["evidence_supplier_count"] = len(reports)
        run = SupplierEvidenceRun(
            tenant_id=self.tenant_id,
            run_id=run_id,
            scoring_version=config["version"],
            status="completed",
            supplier_count=len(reports),
            narrative_json=json.dumps(narrative),
            result_json=json.dumps(reports, default=str),
            llm_provider=narrative["provider"] if narrative["provider"] != "none" else "deterministic",
            llm_model=narrative["model"],
            prompt_policy=narrative["policy"],
        )
        self.session.add(run)
        action_rows: list[SupplierEvidenceAction] = []
        for report in reports:
            self.session.add(
                SupplierEvidenceRunSupplier(
                    tenant_id=self.tenant_id,
                    run_id=run_id,
                    supplier_id=report["supplier_id"],
                    supplier_name=report["supplier_name"],
                    risk_score=float(report["risk_score"]),
                    risk_level=report["risk_level"],
                    confidence=float(report["confidence"]),
                    top_risk_drivers_json=json.dumps(report["top_risk_drivers"], default=str),
                    evidence_chain_json=json.dumps(report["evidence_chain"], default=str),
                    recommended_actions_json=json.dumps(report["recommended_actions"], default=str),
                )
            )
            source_driver = report["top_risk_drivers"][0]["driver"] if report["top_risk_drivers"] else ""
            for action in report["recommended_actions"]:
                action_row = SupplierEvidenceAction(
                    tenant_id=self.tenant_id,
                    run_id=run_id,
                    supplier_id=report["supplier_id"],
                    supplier_name=report["supplier_name"],
                    action=action,
                    source_driver=source_driver,
                )
                self.session.add(action_row)
                action_rows.append(action_row)
        AuditRepository(self.session, self.tenant_id).log(
            "evidence.run_completed",
            username=actor,
            details={"run_id": run_id, "supplier_count": len(reports), "scoring_version": config["version"]},
        )
        self.session.flush()
        return self.run_to_dict(run, reports=reports, actions=[_action_to_dict(row) for row in action_rows])

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.session.scalars(
            select(SupplierEvidenceRun)
            .where(SupplierEvidenceRun.tenant_id == self.tenant_id)
            .order_by(SupplierEvidenceRun.created_at.desc())
            .limit(limit)
        )
        return [self.run_to_dict(row, include_details=False) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.session.scalar(
            select(SupplierEvidenceRun).where(
                SupplierEvidenceRun.tenant_id == self.tenant_id,
                SupplierEvidenceRun.run_id == run_id,
            )
        )
        if row is None:
            return None
        suppliers = [
            _run_supplier_to_report(item)
            for item in self.session.scalars(
                select(SupplierEvidenceRunSupplier).where(
                    SupplierEvidenceRunSupplier.tenant_id == self.tenant_id,
                    SupplierEvidenceRunSupplier.run_id == run_id,
                )
            )
        ]
        actions = [
            _action_to_dict(item)
            for item in self.session.scalars(
                select(SupplierEvidenceAction)
                .where(
                    SupplierEvidenceAction.tenant_id == self.tenant_id,
                    SupplierEvidenceAction.run_id == run_id,
                )
                .order_by(SupplierEvidenceAction.id)
            )
        ]
        return self.run_to_dict(row, reports=suppliers, actions=actions)

    def update_action(self, action_id: int, status: str, owner: str = "", actor: str = "system") -> dict[str, Any]:
        status = (status or "").lower()
        if status not in VALID_ACTION_STATUSES:
            raise ValueError(f"Unsupported evidence action status: {status}")
        row = self.session.scalar(
            select(SupplierEvidenceAction).where(
                SupplierEvidenceAction.tenant_id == self.tenant_id,
                SupplierEvidenceAction.id == action_id,
            )
        )
        if row is None:
            raise ValueError(f"Evidence action {action_id} not found")
        row.status = status
        row.owner = owner or row.owner
        row.updated_by = actor
        AuditRepository(self.session, self.tenant_id).log(
            "evidence.action_updated",
            username=actor,
            details={"action_id": action_id, "status": status, "owner": row.owner},
        )
        self.session.flush()
        return _action_to_dict(row)

    def run_to_dict(
        self,
        run: SupplierEvidenceRun,
        reports: list[dict[str, Any]] | None = None,
        actions: list[dict[str, Any]] | None = None,
        include_details: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "id": run.id,
            "tenant_id": run.tenant_id,
            "run_id": run.run_id,
            "scoring_version": run.scoring_version,
            "status": run.status,
            "supplier_count": run.supplier_count,
            "narrative": _loads(run.narrative_json, {}),
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }
        if include_details:
            payload["suppliers"] = reports if reports is not None else _loads(run.result_json, [])
            payload["actions"] = actions or []
        return payload
