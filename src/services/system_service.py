"""System health aggregation for API and Streamlit command center."""

from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import select

from src.config import Settings
from src.database import database_health
from src.observability.logging import redact_secret_text
from src.repositories.alerts import AlertRepository
from src.repositories.jobs import JobRepository
from src.repositories.suppliers import SupplierRepository
from src.models import SupplierEvidenceScoringVersion
from src.services.scheduler import LocalJobScheduler
from src.services.supplier_evidence_service import SupplierEvidenceService
from src.services.worker_queue import get_worker_mode
from src.tenancy import DEMO_TENANT_ID


def _base_status(settings: Settings, database: dict) -> dict:
    return {
        "database": database,
        "backend": {
            "requested": settings.data_backend,
            "active": settings.active_data_backend,
            "fallback": settings.data_backend == "convex",
        },
        "api": {"ok": bool(database.get("ok")), "status": "ready" if database.get("ok") else "degraded"},
        "worker": {"status": "unknown", "enabled": settings.scheduler_enabled},
        "worker_mode": get_worker_mode(settings),
        "sentinel": {
            "configured": bool(settings.newsapi_key and (settings.openai_api_key or settings.anthropic_api_key)),
            "demo_mode": settings.demo_mode,
        },
        "auth": {
            "provider": settings.auth_provider,
            "mfa_required": settings.mfa_required,
            "scim_enabled": settings.scim_enabled,
        },
        "connectors": {
            "mode": settings.connector_mode,
            "stub_available": True,
            "public_available": True,
            "last_sync_status": "unknown",
        },
        "scoring_config": {"available": False, "version": ""},
        "convex": {
            "configured": settings.convex_configured,
            "status": "configured" if settings.convex_configured else "not_configured",
        },
        "llm_narrative": {
            "provider": settings.llm_narrative_provider,
            "configured": settings.llm_narrative_provider == "none",
            "available": settings.llm_narrative_provider == "none",
            "status": "deterministic" if settings.llm_narrative_provider == "none" else "interface_only",
            "fallback": "deterministic",
        },
        "rate_limit": {
            "enabled": settings.rate_limit_enabled,
            "requests": settings.rate_limit_requests,
            "window_seconds": settings.rate_limit_window_seconds,
        },
        "secrets": {"provider": settings.secrets_provider, "kms_provider": settings.kms_provider},
        "retention": {"enabled": settings.retention_enabled, "days": settings.retention_days},
        "siem": {"sink": settings.siem_sink, "configured": bool(settings.siem_webhook_url or settings.siem_sink == "file")},
        "monitored_suppliers": 0,
        "open_alerts": 0,
        "security_mode": settings.security_mode,
        "deployment_mode": settings.deployment_mode,
        "production_issues": settings.validate_runtime(),
        "last_failed_job": None,
        "status_error": "",
    }


def system_status(
    settings: Settings,
    session_factory: sessionmaker[Session] | None,
    tenant_id: str = DEMO_TENANT_ID,
    startup_error: str = "",
) -> dict:
    status = _base_status(settings, database_health(settings))
    status["tenant_id"] = tenant_id
    if not status["database"]["ok"]:
        status["status_error"] = redact_secret_text(status["database"].get("error", "") or startup_error)
        return status
    if session_factory is None:
        status["api"] = {"ok": False, "status": "degraded"}
        status["status_error"] = redact_secret_text(startup_error or "Database session factory is unavailable.")
        return status
    if startup_error:
        status["api"] = {"ok": False, "status": "degraded"}
        status["status_error"] = redact_secret_text(startup_error)
        return status

    try:
        with session_factory() as session:
            suppliers = SupplierRepository(session, tenant_id).list(limit=10_000)
            alerts = AlertRepository(session, tenant_id)
            jobs = JobRepository(session, tenant_id)
            last_failure = jobs.last_failure()
            scheduler = LocalJobScheduler(settings, session_factory)
            evidence = SupplierEvidenceService(session, tenant_id)
            scoring_config = session.scalar(
                select(SupplierEvidenceScoringVersion)
                .where(
                    SupplierEvidenceScoringVersion.tenant_id == tenant_id,
                    SupplierEvidenceScoringVersion.is_active.is_(True),
                )
                .order_by(SupplierEvidenceScoringVersion.created_at.desc())
            )
            syncs = evidence.list_connector_syncs(limit=1)
            status["worker"] = scheduler.status(tenant_id)
            status["monitored_suppliers"] = len(suppliers)
            status["open_alerts"] = alerts.count_open()
            status["last_failed_job"] = JobRepository.to_dict(last_failure) if last_failure else None
            status["scoring_config"] = {
                "available": True,
                "version": scoring_config.version if scoring_config else "default-v1",
            }
            if syncs:
                status["connectors"]["last_sync_status"] = syncs[0]["status"]
            return status
    except Exception as exc:
        status["api"] = {"ok": False, "status": "degraded"}
        status["worker"] = {"status": "unknown", "enabled": settings.scheduler_enabled}
        status["status_error"] = redact_secret_text(exc)
        return status
