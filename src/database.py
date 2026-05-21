"""Database setup with Postgres-ready SQLAlchemy and SQLite fallback."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import Settings, get_settings
from src.models import Base


def _sqlite_path(database_url: str) -> Path | None:
    if not database_url.startswith("sqlite:///"):
        return None
    raw = database_url.replace("sqlite:///", "", 1)
    if raw == ":memory:":
        return None
    return Path(raw)


def create_database_engine(settings: Settings | None = None) -> Engine:
    active = settings or get_settings()
    sqlite_path = _sqlite_path(active.database_url)
    if sqlite_path is not None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False} if active.database_url.startswith("sqlite") else {}
    return create_engine(active.database_url, connect_args=connect_args, pool_pre_ping=True, future=True)


def create_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    engine = create_database_engine(settings)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_database(session_factory: sessionmaker[Session] | None = None) -> None:
    factory = session_factory or create_session_factory()
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    _ensure_sqlite_compat_columns(engine)


def seed_demo_tenant(session_factory: sessionmaker[Session]) -> None:
    from src.repositories.tenants import TenantRepository

    with session_factory() as session:
        TenantRepository(session).ensure_demo_seed()
        session.commit()


def _ensure_sqlite_compat_columns(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return
    tenant_tables = {
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
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table in tenant_tables & existing_tables:
            columns = {column["name"] for column in inspector.get_columns(table)}
            if "tenant_id" not in columns:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN tenant_id VARCHAR(120) NOT NULL DEFAULT 'demo-tenant'"))
        if "background_job_runs" in existing_tables:
            columns = {column["name"] for column in inspector.get_columns("background_job_runs")}
            extra_columns = {
                "task_name": "VARCHAR(150) NOT NULL DEFAULT ''",
                "duration_ms": "INTEGER NOT NULL DEFAULT 0",
                "retry_count": "INTEGER NOT NULL DEFAULT 0",
                "error_summary": "TEXT NOT NULL DEFAULT ''",
                "request_id": "VARCHAR(120) NOT NULL DEFAULT ''",
                "correlation_id": "VARCHAR(120) NOT NULL DEFAULT ''",
            }
            for column_name, ddl in extra_columns.items():
                if column_name not in columns:
                    conn.execute(text(f"ALTER TABLE background_job_runs ADD COLUMN {column_name} {ddl}"))


@contextmanager
def session_scope(session_factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    factory = session_factory or create_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def database_health(settings: Settings | None = None) -> dict:
    active = settings or get_settings()
    try:
        engine = create_database_engine(active)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "driver": active.database_driver, "url": _safe_url(active.database_url)}
    except Exception as exc:
        return {"ok": False, "driver": active.database_driver, "url": _safe_url(active.database_url), "error": str(exc)}


def _safe_url(database_url: str) -> str:
    if "@" not in database_url:
        return database_url
    scheme, rest = database_url.split("://", 1)
    _, host = rest.rsplit("@", 1)
    return f"{scheme}://***:***@{host}"
