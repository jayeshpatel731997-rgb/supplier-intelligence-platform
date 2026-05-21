"""SQLAlchemy models for production-ready persistence."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class Role(Base, TimestampMixin):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String(255), default="", nullable=False)


class Tenant(Base, TimestampMixin):
    __tablename__ = "tenants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="active", nullable=False)


class Organization(Base, TimestampMixin):
    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[str] = mapped_column(String(255), default="", nullable=False)


class Membership(Base, TimestampMixin):
    __tablename__ = "memberships"
    __table_args__ = (
        UniqueConstraint("tenant_id", "username", name="uq_membership_tenant_username"),
        Index("ix_memberships_tenant_role", "tenant_id", "role"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class TenantApiKey(Base, TimestampMixin):
    __tablename__ = "tenant_api_keys"
    __table_args__ = (Index("ix_tenant_api_keys_tenant_active", "tenant_id", "is_active"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(150), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    label: Mapped[str] = mapped_column(String(150), default="", nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    prefix: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class AccessReview(Base, TimestampMixin):
    __tablename__ = "access_reviews"
    __table_args__ = (Index("ix_access_reviews_tenant_status", "tenant_id", "status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), nullable=False, index=True)
    reviewer: Mapped[str] = mapped_column(String(150), default="", nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="open", nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)


class RetentionPolicy(Base, TimestampMixin):
    __tablename__ = "retention_policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), nullable=False, index=True)
    data_type: Mapped[str] = mapped_column(String(120), nullable=False)
    retention_days: Mapped[int] = mapped_column(Integer, default=365, nullable=False)


class BackupRun(Base, TimestampMixin):
    __tablename__ = "backup_runs"
    __table_args__ = (Index("ix_backup_runs_tenant_status", "tenant_id", "status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    location: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    error: Mapped[str] = mapped_column(Text, default="", nullable=False)


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(150), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="viewer", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    failed_login_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Supplier(Base, TimestampMixin):
    __tablename__ = "suppliers"
    __table_args__ = (
        UniqueConstraint("tenant_id", "supplier_id", name="uq_suppliers_tenant_supplier"),
        Index("ix_suppliers_tenant_name", "tenant_id", "name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), default="demo-tenant", nullable=False, index=True)
    supplier_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    country: Mapped[str] = mapped_column(String(120), default="Unknown", nullable=False)
    category: Mapped[str] = mapped_column(String(120), default="Uncategorized", nullable=False)
    tier: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    annual_spend: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    unit_cost: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    on_time_delivery_pct: Mapped[float] = mapped_column(Float, default=85.0, nullable=False)
    defect_rate_pct: Mapped[float] = mapped_column(Float, default=2.0, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, default=50.0, nullable=False)
    source: Mapped[str] = mapped_column(String(100), default="manual", nullable=False)


class SupplierKPI(Base, TimestampMixin):
    __tablename__ = "supplier_kpis"
    __table_args__ = (Index("ix_supplier_kpis_tenant_supplier", "tenant_id", "supplier_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), default="demo-tenant", nullable=False, index=True)
    supplier_id: Mapped[str] = mapped_column(String(120), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(120), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    period: Mapped[str] = mapped_column(String(50), default="current", nullable=False)


class SupplierRiskScore(Base, TimestampMixin):
    __tablename__ = "supplier_risk_scores"
    __table_args__ = (Index("ix_supplier_risk_scores_tenant_supplier", "tenant_id", "supplier_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), default="demo-tenant", nullable=False, index=True)
    supplier_id: Mapped[str] = mapped_column(String(120), nullable=False)
    risk_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(50), nullable=False)
    dominant_factor: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    confidence: Mapped[str] = mapped_column(String(50), default="medium", nullable=False)
    drivers_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)


class NewsEvent(Base, TimestampMixin):
    __tablename__ = "news_events"
    __table_args__ = (
        UniqueConstraint("tenant_id", "event_id", name="uq_news_events_tenant_event"),
        Index("ix_news_events_tenant_severity", "tenant_id", "severity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), default="demo-tenant", nullable=False, index=True)
    event_id: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source: Mapped[str] = mapped_column(String(150), default="", nullable=False)
    url: Mapped[str] = mapped_column(String(1000), default="", nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    disruption_type: Mapped[str] = mapped_column(String(120), default="General Supply Chain", nullable=False)
    severity: Mapped[str] = mapped_column(String(50), default="medium", nullable=False)
    confidence: Mapped[str] = mapped_column(String(50), default="medium", nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)


class SupplierEventMatch(Base, TimestampMixin):
    __tablename__ = "supplier_event_matches"
    __table_args__ = (Index("ix_supplier_event_matches_tenant_event", "tenant_id", "event_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), default="demo-tenant", nullable=False, index=True)
    event_id: Mapped[str] = mapped_column(String(150), nullable=False)
    supplier_id: Mapped[str] = mapped_column(String(120), nullable=False)
    exposure: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    match_reason: Mapped[str] = mapped_column(String(255), default="", nullable=False)


class ScenarioRun(Base, TimestampMixin):
    __tablename__ = "scenario_runs"
    __table_args__ = (Index("ix_scenario_runs_tenant_status", "tenant_id", "status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), default="demo-tenant", nullable=False, index=True)
    scenario_name: Mapped[str] = mapped_column(String(150), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="completed", nullable=False)
    result_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)


class FinancialExposureRun(Base, TimestampMixin):
    __tablename__ = "financial_exposure_runs"
    __table_args__ = (Index("ix_financial_exposure_runs_tenant_supplier", "tenant_id", "supplier_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), default="demo-tenant", nullable=False, index=True)
    supplier_id: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    expected_loss: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    var95: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    cvar95: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    result_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)


class Alert(Base, TimestampMixin):
    __tablename__ = "alerts"
    __table_args__ = (Index("ix_alerts_tenant_status", "tenant_id", "status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), default="demo-tenant", nullable=False, index=True)
    supplier_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    alert_type: Mapped[str] = mapped_column(String(120), nullable=False)
    severity: Mapped[str] = mapped_column(String(50), default="medium", nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    exposure: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="open", nullable=False)
    acknowledged_by: Mapped[str | None] = mapped_column(String(150), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(150), nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_logs_tenant_timestamp", "tenant_id", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), default="demo-tenant", nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    username: Mapped[str] = mapped_column(String(150), default="system", nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="system", nullable=False)
    action: Mapped[str] = mapped_column(String(150), nullable=False)
    details_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)


class IngestionJob(Base, TimestampMixin):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (Index("ix_ingestion_jobs_tenant_status", "tenant_id", "status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), default="demo-tenant", nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    report_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    error: Mapped[str] = mapped_column(Text, default="", nullable=False)


class SystemHealthEvent(Base, TimestampMixin):
    __tablename__ = "system_health_events"
    __table_args__ = (Index("ix_system_health_events_tenant_component", "tenant_id", "component"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), default="demo-tenant", nullable=False, index=True)
    component: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str] = mapped_column(Text, default="", nullable=False)


class BackgroundJobRun(Base, TimestampMixin):
    __tablename__ = "background_job_runs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "job_name", "run_id", name="uq_background_job_run"),
        Index("ix_background_job_runs_tenant_job", "tenant_id", "job_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(120), ForeignKey("tenants.tenant_id"), default="demo-tenant", nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(120), nullable=False)
    job_name: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    task_name: Mapped[str] = mapped_column(String(150), default="", nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="running", nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str] = mapped_column(Text, default="", nullable=False)
    error_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    request_id: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    correlation_id: Mapped[str] = mapped_column(String(120), default="", nullable=False)
