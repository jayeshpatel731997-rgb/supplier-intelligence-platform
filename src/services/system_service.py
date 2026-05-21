"""System health aggregation for API and Streamlit command center."""

from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from src.config import Settings
from src.database import database_health
from src.repositories.alerts import AlertRepository
from src.repositories.jobs import JobRepository
from src.repositories.suppliers import SupplierRepository
from src.services.scheduler import LocalJobScheduler
from src.services.worker_queue import get_worker_mode
from src.tenancy import DEMO_TENANT_ID


def system_status(settings: Settings, session_factory: sessionmaker, tenant_id: str = DEMO_TENANT_ID) -> dict:
    with session_factory() as session:
        suppliers = SupplierRepository(session, tenant_id).list(limit=10_000)
        alerts = AlertRepository(session, tenant_id)
        jobs = JobRepository(session, tenant_id)
        last_failure = jobs.last_failure()
        scheduler = LocalJobScheduler(settings, session_factory)
        return {
            "database": database_health(settings),
            "api": {"ok": True, "status": "ready"},
            "worker": scheduler.status(tenant_id),
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
            "rate_limit": {
                "enabled": settings.rate_limit_enabled,
                "requests": settings.rate_limit_requests,
                "window_seconds": settings.rate_limit_window_seconds,
            },
            "secrets": {"provider": settings.secrets_provider, "kms_provider": settings.kms_provider},
            "retention": {"enabled": settings.retention_enabled, "days": settings.retention_days},
            "siem": {"sink": settings.siem_sink, "configured": bool(settings.siem_webhook_url or settings.siem_sink == "file")},
            "monitored_suppliers": len(suppliers),
            "open_alerts": alerts.count_open(),
            "security_mode": settings.security_mode,
            "deployment_mode": settings.deployment_mode,
            "production_issues": settings.validate_runtime(),
            "last_failed_job": JobRepository.to_dict(last_failure) if last_failure else None,
        }
