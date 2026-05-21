"""Compliance evidence collection stubs for readiness workflows."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.repositories.audit import AuditRepository
from src.repositories.tenants import TenantRepository
from src.services.backup_service import BackupService
from src.tenancy import DEMO_TENANT_ID


class EvidenceService:
    def __init__(self, session: Session, tenant_id: str = DEMO_TENANT_ID):
        self.session = session
        self.tenant_id = tenant_id

    def collect_access_control_evidence(self) -> dict:
        repo = TenantRepository(self.session)
        return {
            "tenant_id": self.tenant_id,
            "control": "access_control",
            "memberships": [TenantRepository.membership_to_dict(row) for row in repo.list_memberships(self.tenant_id)],
            "access_reviews": [
                {
                    "id": row.id,
                    "status": row.status,
                    "reviewer": row.reviewer,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in repo.list_access_reviews(self.tenant_id)
            ],
        }

    def collect_operational_evidence(self) -> dict:
        audits = AuditRepository(self.session, self.tenant_id).list(limit=50)
        return {
            "tenant_id": self.tenant_id,
            "control": "operations",
            "recent_audit_events": [AuditRepository.to_dict(row) for row in audits],
            "backup_metadata_supported": hasattr(BackupService, "record_metadata"),
        }
