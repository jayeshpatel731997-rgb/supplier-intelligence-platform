"""Celery/Redis-ready worker architecture with local fallback."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass

from sqlalchemy.orm import Session, sessionmaker

from src.config import Settings, get_settings
from src.repositories.alerts import AlertRepository
from src.repositories.audit import AuditRepository
from src.repositories.jobs import JobRepository
from src.services.audit_export import AuditExportService
from src.services.backup_service import BackupService
from src.services.retention_service import RetentionService
from src.services.risk_service import RiskService
from src.services.sentinel_service import SentinelService
from src.tenancy import DEMO_TENANT_ID

try:
    from celery import Celery
except Exception:  # pragma: no cover - optional dependency
    Celery = None


@dataclass(slots=True)
class TaskResult:
    run_id: str
    task_name: str
    tenant_id: str
    status: str
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


TASK_ALIASES = {
    "sentinel_scan_task": "sentinel_scan",
    "risk_recalculation_task": "risk_recalculate",
    "exposure_recalculation_task": "exposure_recalculate",
    "retention_cleanup_task": "retention_cleanup",
    "audit_export_task": "audit_export",
    "backup_metadata_task": "backup_metadata",
}


def get_worker_mode(settings: Settings | None = None) -> dict:
    active = settings or get_settings()
    celery_available = Celery is not None and bool(active.redis_url)
    return {
        "requested": active.worker_mode,
        "active": "celery" if active.worker_mode == "celery" and celery_available else "local",
        "available": celery_available,
        "redis_configured": bool(active.redis_url),
        "fallback": "local" if not celery_available else "",
    }


def build_celery_app(settings: Settings | None = None):
    active = settings or get_settings()
    if Celery is None:
        return None
    broker = active.redis_url or "redis://redis:6379/0"
    app = Celery("supplier_intelligence", broker=broker, backend=broker)
    app.conf.task_serializer = "json"
    app.conf.result_serializer = "json"
    app.conf.accept_content = ["json"]
    return app


class EnterpriseTaskRunner:
    def __init__(self, settings: Settings, session_factory: sessionmaker[Session]):
        self.settings = settings
        self.session_factory = session_factory

    def run_task(
        self,
        task_name: str,
        tenant_id: str = DEMO_TENANT_ID,
        correlation_id: str = "",
        retry_count: int = 0,
    ) -> TaskResult:
        run_id = str(uuid.uuid4())
        job_name = TASK_ALIASES.get(task_name, task_name)
        with self.session_factory() as session:
            jobs = JobRepository(session, tenant_id)
            row = jobs.start(
                run_id,
                job_name,
                task_name=task_name,
                request_id=correlation_id,
                correlation_id=correlation_id,
                retry_count=retry_count,
            )
            audit = AuditRepository(session, tenant_id)
            try:
                self._execute(task_name, session, tenant_id)
                jobs.finish(row, "completed")
                audit.log("worker.task_completed", details={"task_name": task_name, "run_id": run_id})
                session.commit()
                return TaskResult(run_id, task_name, tenant_id, "completed")
            except Exception as exc:
                message = str(exc)
                jobs.finish(row, "failed", message)
                audit.log("worker.task_failed", details={"task_name": task_name, "run_id": run_id, "error_summary": message[:500]})
                AlertRepository(session, tenant_id).create_alert(
                    "background_job_failure",
                    "medium",
                    f"{task_name} failed: {message}",
                )
                session.commit()
                return TaskResult(run_id, task_name, tenant_id, "failed", message)

    def run_task_for_all_tenants(self, task_name: str) -> list[TaskResult]:
        from src.repositories.tenants import TenantRepository

        with self.session_factory() as session:
            tenant_ids = [tenant.tenant_id for tenant in TenantRepository(session).list_tenants()]
        return [self.run_task(task_name, tenant_id) for tenant_id in tenant_ids]

    def _execute(self, task_name: str, session: Session, tenant_id: str) -> None:
        if task_name == "sentinel_scan_task":
            SentinelService(session, self.settings, tenant_id).scan(mode="demo" if self.settings.demo_mode else "live_ai")
        elif task_name == "risk_recalculation_task":
            RiskService(session, tenant_id).recalculate()
        elif task_name == "exposure_recalculation_task":
            RiskService(session, tenant_id).financial_exposure()
        elif task_name == "retention_cleanup_task":
            RetentionService(session, tenant_id, self.settings).cleanup(dry_run=True)
        elif task_name == "audit_export_task":
            AuditExportService(session, tenant_id).export_jsonl()
        elif task_name == "backup_metadata_task":
            BackupService(session, tenant_id).record_metadata("metadata_recorded", "local://metadata-only")
        else:
            raise ValueError(f"Unknown task: {task_name}")


def register_celery_tasks(celery_app, settings: Settings, session_factory):
    if celery_app is None:
        return None

    @celery_app.task(name="sentinel_scan_task")
    def sentinel_scan_task(tenant_id: str = DEMO_TENANT_ID, correlation_id: str = ""):
        return EnterpriseTaskRunner(settings, session_factory).run_task("sentinel_scan_task", tenant_id, correlation_id).to_dict()

    @celery_app.task(name="risk_recalculation_task")
    def risk_recalculation_task(tenant_id: str = DEMO_TENANT_ID, correlation_id: str = ""):
        return EnterpriseTaskRunner(settings, session_factory).run_task("risk_recalculation_task", tenant_id, correlation_id).to_dict()

    @celery_app.task(name="exposure_recalculation_task")
    def exposure_recalculation_task(tenant_id: str = DEMO_TENANT_ID, correlation_id: str = ""):
        return EnterpriseTaskRunner(settings, session_factory).run_task("exposure_recalculation_task", tenant_id, correlation_id).to_dict()

    @celery_app.task(name="retention_cleanup_task")
    def retention_cleanup_task(tenant_id: str = DEMO_TENANT_ID, correlation_id: str = ""):
        return EnterpriseTaskRunner(settings, session_factory).run_task("retention_cleanup_task", tenant_id, correlation_id).to_dict()

    @celery_app.task(name="audit_export_task")
    def audit_export_task(tenant_id: str = DEMO_TENANT_ID, correlation_id: str = ""):
        return EnterpriseTaskRunner(settings, session_factory).run_task("audit_export_task", tenant_id, correlation_id).to_dict()

    @celery_app.task(name="backup_metadata_task")
    def backup_metadata_task(tenant_id: str = DEMO_TENANT_ID, correlation_id: str = ""):
        return EnterpriseTaskRunner(settings, session_factory).run_task("backup_metadata_task", tenant_id, correlation_id).to_dict()

    @celery_app.task(name="sentinel_scan_all_tenants_task")
    def sentinel_scan_all_tenants_task():
        return [result.to_dict() for result in EnterpriseTaskRunner(settings, session_factory).run_task_for_all_tenants("sentinel_scan_task")]

    @celery_app.task(name="risk_recalculation_all_tenants_task")
    def risk_recalculation_all_tenants_task():
        return [result.to_dict() for result in EnterpriseTaskRunner(settings, session_factory).run_task_for_all_tenants("risk_recalculation_task")]

    @celery_app.task(name="exposure_recalculation_all_tenants_task")
    def exposure_recalculation_all_tenants_task():
        return [result.to_dict() for result in EnterpriseTaskRunner(settings, session_factory).run_task_for_all_tenants("exposure_recalculation_task")]

    return celery_app
