"""Local background job infrastructure with APScheduler-compatible shape."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass

from sqlalchemy.orm import Session, sessionmaker

from src.config import Settings
from src.repositories.alerts import AlertRepository
from src.repositories.jobs import JobRepository
from src.repositories.tenants import TenantRepository
from src.services.risk_service import RiskService
from src.services.sentinel_service import SentinelService
from src.tenancy import DEMO_TENANT_ID
from src.services.worker_queue import get_worker_mode


@dataclass(slots=True)
class JobResult:
    run_id: str
    job_name: str
    status: str
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class LocalJobScheduler:
    """Synchronous local scheduler fallback for demos, tests, and single-container pilots."""

    def __init__(self, settings: Settings, session_factory: sessionmaker[Session]):
        self.settings = settings
        self.session_factory = session_factory

    def run_job_now(self, job_name: str, tenant_id: str = DEMO_TENANT_ID) -> JobResult:
        run_id = str(uuid.uuid4())
        with self.session_factory() as session:
            repo = JobRepository(session, tenant_id)
            row = repo.start(run_id, job_name)
            try:
                if job_name == "risk_recalculate":
                    RiskService(session, tenant_id).recalculate()
                elif job_name == "sentinel_scan":
                    SentinelService(session, self.settings, tenant_id).scan(mode="demo" if self.settings.demo_mode else "live_ai")
                elif job_name == "exposure_recalculate":
                    RiskService(session, tenant_id).financial_exposure()
                elif job_name == "retention_cleanup":
                    from src.services.retention_service import RetentionService

                    RetentionService(session, tenant_id, self.settings).cleanup(dry_run=True)
                elif job_name == "audit_export":
                    from src.services.audit_export import AuditExportService

                    AuditExportService(session, tenant_id).export_jsonl()
                elif job_name == "backup_metadata":
                    from src.services.backup_service import BackupService

                    BackupService(session, tenant_id).record_metadata("metadata_recorded", "local://metadata-only")
                else:
                    raise ValueError(f"Unknown job: {job_name}")
                repo.finish(row, "completed")
                session.commit()
                return JobResult(run_id=run_id, job_name=job_name, status="completed")
            except Exception as exc:
                message = str(exc)
                repo.finish(row, "failed", message)
                AlertRepository(session, tenant_id).create_alert("background_job_failure", "medium", f"{job_name} failed: {message}")
                from src.repositories.audit import AuditRepository

                AuditRepository(session, tenant_id).log("worker.job_failed", details={"job_name": job_name, "error_summary": message[:500]})
                session.commit()
                return JobResult(run_id=run_id, job_name=job_name, status="failed", error=message)

    def run_all_tenants(self, job_name: str) -> list[JobResult]:
        with self.session_factory() as session:
            tenant_ids = [tenant.tenant_id for tenant in TenantRepository(session).list_tenants()]
        return [self.run_job_now(job_name, tenant_id) for tenant_id in tenant_ids]

    def status(self, tenant_id: str = DEMO_TENANT_ID) -> dict:
        with self.session_factory() as session:
            jobs = JobRepository(session, tenant_id)
            last_failure = jobs.last_failure()
            last_sentinel = jobs.last_success("sentinel_scan")
            return {
                "enabled": self.settings.scheduler_enabled,
                "mode": get_worker_mode(self.settings)["active"],
                "requested_mode": self.settings.worker_mode,
                "redis_configured": bool(self.settings.redis_url),
                "last_successful_sentinel_scan": last_sentinel.finished_at.isoformat() if last_sentinel and last_sentinel.finished_at else None,
                "last_failed_job": JobRepository.to_dict(last_failure) if last_failure else None,
            }
