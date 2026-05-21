"""Backup metadata service and local backup helpers."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.models import BackupRun
from src.tenancy import DEMO_TENANT_ID


class BackupService:
    def __init__(self, session: Session, tenant_id: str = DEMO_TENANT_ID):
        self.session = session
        self.tenant_id = tenant_id

    def record_metadata(self, status: str, location: str = "", error: str = "") -> BackupRun:
        row = BackupRun(tenant_id=self.tenant_id, status=status, location=location, error=error)
        self.session.add(row)
        self.session.flush()
        return row

    @staticmethod
    def to_dict(row: BackupRun) -> dict:
        return {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "status": row.status,
            "location": row.location,
            "error": row.error,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
