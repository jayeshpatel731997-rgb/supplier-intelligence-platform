"""Retention cleanup service with disabled-by-default and dry-run-safe behavior."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.config import Settings
from src.tenancy import DEMO_TENANT_ID


class RetentionService:
    def __init__(self, session: Session, tenant_id: str = DEMO_TENANT_ID, settings: Settings | None = None):
        self.session = session
        self.tenant_id = tenant_id
        self.settings = settings or Settings()

    def cleanup(self, dry_run: bool = True) -> dict:
        if not self.settings.retention_enabled:
            return {
                "tenant_id": self.tenant_id,
                "enabled": False,
                "dry_run": True,
                "deleted_rows": 0,
                "message": "Retention cleanup is disabled by default.",
            }
        # Production deletion should be implemented per table with legal review.
        return {
            "tenant_id": self.tenant_id,
            "enabled": True,
            "dry_run": dry_run,
            "deleted_rows": 0,
            "message": "Retention policy evaluated; destructive deletes are not enabled in this local implementation.",
        }
