"""Alert repository and state transitions."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.models import Alert
from src.tenancy import DEMO_TENANT_ID


class AlertRepository:
    def __init__(self, session: Session, tenant_id: str = DEMO_TENANT_ID):
        self.session = session
        self.tenant_id = tenant_id

    def create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        supplier_id: str | None = None,
        exposure: float = 0.0,
    ) -> Alert:
        alert = Alert(
            tenant_id=self.tenant_id,
            supplier_id=supplier_id,
            alert_type=alert_type,
            severity=str(severity or "medium").lower(),
            message=message,
            exposure=float(exposure or 0.0),
            status="open",
        )
        self.session.add(alert)
        self.session.flush()
        return alert

    def list(self, status: str | None = None, limit: int = 200) -> list[Alert]:
        stmt = select(Alert).where(Alert.tenant_id == self.tenant_id).order_by(Alert.created_at.desc()).limit(limit)
        if status:
            stmt = stmt.where(Alert.status == status)
        return list(self.session.scalars(stmt))

    def get(self, alert_id: int) -> Alert | None:
        return self.session.scalar(select(Alert).where(Alert.tenant_id == self.tenant_id, Alert.id == alert_id))

    def count_open(self) -> int:
        return int(
            self.session.scalar(
                select(func.count()).select_from(Alert).where(Alert.tenant_id == self.tenant_id, Alert.status == "open")
            )
            or 0
        )

    def acknowledge(self, alert_id: int, actor: str = "system") -> Alert:
        alert = self.get(alert_id)
        if alert is None:
            raise ValueError(f"Alert {alert_id} not found")
        alert.status = "acknowledged"
        alert.acknowledged_by = actor
        return alert

    def resolve(self, alert_id: int, actor: str = "system") -> Alert:
        alert = self.get(alert_id)
        if alert is None:
            raise ValueError(f"Alert {alert_id} not found")
        alert.status = "resolved"
        alert.resolved_by = actor
        return alert

    @staticmethod
    def to_dict(alert: Alert) -> dict:
        return {
            "id": alert.id,
            "tenant_id": alert.tenant_id,
            "supplier_id": alert.supplier_id,
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "message": alert.message,
            "exposure": alert.exposure,
            "status": alert.status,
            "created_at": alert.created_at.isoformat() if alert.created_at else None,
            "updated_at": alert.updated_at.isoformat() if alert.updated_at else None,
        }
