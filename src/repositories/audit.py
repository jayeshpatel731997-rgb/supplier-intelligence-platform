"""Audit log repository."""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models import AuditLog
from src.tenancy import DEMO_TENANT_ID


class AuditRepository:
    def __init__(self, session: Session, tenant_id: str = DEMO_TENANT_ID):
        self.session = session
        self.tenant_id = tenant_id

    def log(self, action: str, username: str = "system", role: str = "system", details: dict | None = None) -> AuditLog:
        row = AuditLog(
            tenant_id=self.tenant_id,
            username=username or "system",
            role=role or "system",
            action=action,
            details_json=json.dumps(details or {}, default=str),
        )
        self.session.add(row)
        self.session.flush()
        return row

    def list(self, limit: int = 200) -> list[AuditLog]:
        return list(
            self.session.scalars(
                select(AuditLog).where(AuditLog.tenant_id == self.tenant_id).order_by(AuditLog.timestamp.desc()).limit(limit)
            )
        )

    @staticmethod
    def to_dict(row: AuditLog) -> dict:
        return {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            "username": row.username,
            "role": row.role,
            "action": row.action,
            "details": json.loads(row.details_json or "{}"),
        }
