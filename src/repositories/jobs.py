"""Background job repository."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models import BackgroundJobRun
from src.tenancy import DEMO_TENANT_ID


class JobRepository:
    def __init__(self, session: Session, tenant_id: str = DEMO_TENANT_ID):
        self.session = session
        self.tenant_id = tenant_id

    def start(
        self,
        run_id: str,
        job_name: str,
        task_name: str = "",
        request_id: str = "",
        correlation_id: str = "",
        retry_count: int = 0,
    ) -> BackgroundJobRun:
        row = BackgroundJobRun(
            tenant_id=self.tenant_id,
            run_id=run_id,
            job_name=job_name,
            task_name=task_name or job_name,
            status="running",
            request_id=request_id,
            correlation_id=correlation_id or request_id,
            retry_count=retry_count,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def finish(self, row: BackgroundJobRun, status: str, error: str = "") -> BackgroundJobRun:
        row.status = status
        row.error = error or ""
        row.error_summary = (error or "")[:500]
        row.finished_at = datetime.now(UTC)
        row.duration_ms = max(0, int((row.finished_at - row.started_at).total_seconds() * 1000))
        return row

    def list(self, limit: int = 100) -> list[BackgroundJobRun]:
        return list(
            self.session.scalars(
                select(BackgroundJobRun)
                .where(BackgroundJobRun.tenant_id == self.tenant_id)
                .order_by(BackgroundJobRun.started_at.desc())
                .limit(limit)
            )
        )

    def last_success(self, job_name: str) -> BackgroundJobRun | None:
        return self.session.scalar(
            select(BackgroundJobRun)
            .where(
                BackgroundJobRun.tenant_id == self.tenant_id,
                BackgroundJobRun.job_name == job_name,
                BackgroundJobRun.status == "completed",
            )
            .order_by(BackgroundJobRun.finished_at.desc())
        )

    def last_failure(self) -> BackgroundJobRun | None:
        return self.session.scalar(
            select(BackgroundJobRun)
            .where(BackgroundJobRun.tenant_id == self.tenant_id, BackgroundJobRun.status == "failed")
            .order_by(BackgroundJobRun.finished_at.desc())
        )

    @staticmethod
    def to_dict(row: BackgroundJobRun) -> dict:
        return {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "run_id": row.run_id,
            "job_name": row.job_name,
            "task_name": row.task_name,
            "status": row.status,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            "duration_ms": row.duration_ms,
            "retry_count": row.retry_count,
            "error": row.error,
            "error_summary": row.error_summary,
            "request_id": row.request_id,
            "correlation_id": row.correlation_id,
        }
