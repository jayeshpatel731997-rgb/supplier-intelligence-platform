"""Tenant-scoped audit export service for SIEM and WORM storage handoff."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models import AuditLog
from src.repositories.audit import AuditRepository
from src.tenancy import DEMO_TENANT_ID


class AuditExportService:
    def __init__(self, session: Session, tenant_id: str = DEMO_TENANT_ID):
        self.session = session
        self.tenant_id = tenant_id

    def rows(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        action: str | None = None,
        limit: int = 10_000,
    ) -> list[AuditLog]:
        stmt = select(AuditLog).where(AuditLog.tenant_id == self.tenant_id)
        if start is not None:
            stmt = stmt.where(AuditLog.timestamp >= start)
        if end is not None:
            stmt = stmt.where(AuditLog.timestamp <= end)
        if action:
            stmt = stmt.where(AuditLog.action == action)
        return list(self.session.scalars(stmt.order_by(AuditLog.timestamp.asc()).limit(limit)))

    def export_jsonl(self, **filters) -> str:
        return "\n".join(json.dumps(AuditRepository.to_dict(row), default=str) for row in self.rows(**filters))

    def export_csv(self, **filters) -> str:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["id", "tenant_id", "timestamp", "username", "role", "action", "details"])
        writer.writeheader()
        for row in self.rows(**filters):
            writer.writerow(AuditRepository.to_dict(row))
        return output.getvalue()


class SIEMSink:
    def send(self, payload: str) -> dict:
        raise NotImplementedError


class FileSIEMSink(SIEMSink):
    def __init__(self, path: str):
        self.path = path

    def send(self, payload: str) -> dict:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(payload)
            if payload and not payload.endswith("\n"):
                handle.write("\n")
        return {"sink": "file", "path": self.path, "bytes": len(payload.encode("utf-8"))}


class WebhookSIEMSink(SIEMSink):
    """Placeholder for Splunk/Datadog/Elastic/Sentinel HTTP Event Collector integrations."""

    def __init__(self, url: str):
        self.url = url

    def send(self, payload: str) -> dict:
        if not self.url:
            return {"sink": "webhook", "sent": False, "reason": "No webhook URL configured."}
        return {"sink": "webhook", "sent": False, "reason": "Webhook delivery is intentionally stubbed for local mode."}
