"""Migration safety helpers for tenant backfill and schema validation."""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from src.models import Base
from src.repositories.tenants import TenantRepository
from src.tenancy import DEMO_TENANT_ID


BUSINESS_TABLES = {
    "suppliers",
    "supplier_kpis",
    "supplier_risk_scores",
    "news_events",
    "supplier_event_matches",
    "scenario_runs",
    "financial_exposure_runs",
    "alerts",
    "audit_logs",
    "ingestion_jobs",
    "system_health_events",
    "background_job_runs",
}


def tenant_scoped_tables() -> list[str]:
    return sorted(BUSINESS_TABLES)


def validate_tenant_schema(session: Session) -> dict:
    inspector = inspect(session.bind)
    tables = set(inspector.get_table_names())
    missing_tables = sorted(BUSINESS_TABLES - tables)
    missing_tenant_id: list[str] = []
    nullable_tenant_id: list[str] = []
    for table in sorted(BUSINESS_TABLES & tables):
        columns = inspector.get_columns(table)
        by_name = {column["name"]: column for column in columns}
        if "tenant_id" not in by_name:
            missing_tenant_id.append(table)
        elif by_name["tenant_id"].get("nullable"):
            nullable_tenant_id.append(table)
    return {
        "ok": not missing_tenant_id and not nullable_tenant_id,
        "known_model_tables": sorted(Base.metadata.tables.keys()),
        "business_tables": sorted(BUSINESS_TABLES),
        "missing_tables": missing_tables,
        "tables_missing_tenant_id": missing_tenant_id,
        "tables_with_nullable_tenant_id": nullable_tenant_id,
    }


def backfill_demo_tenant(session: Session, production_mode: bool = False, tenant_id: str = DEMO_TENANT_ID) -> dict:
    if production_mode:
        raise RuntimeError("Demo tenant backfill is disabled in production mode.")
    TenantRepository(session).ensure_demo_seed()
    inspector = inspect(session.bind)
    tables = set(inspector.get_table_names())
    backfilled = 0
    for table in sorted(BUSINESS_TABLES & tables):
        columns = {column["name"] for column in inspector.get_columns(table)}
        if "tenant_id" not in columns:
            continue
        result = session.execute(
            text(f"UPDATE {table} SET tenant_id = :tenant_id WHERE tenant_id IS NULL OR tenant_id = ''"),
            {"tenant_id": tenant_id},
        )
        backfilled += int(result.rowcount or 0)
    return {"tenant_id": tenant_id, "backfilled_rows": backfilled}
