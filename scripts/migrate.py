"""Run Alembic migrations with a SQLite create_all fallback for local demos."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy.engine import make_url

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.database import create_session_factory, init_database, seed_demo_tenant
from src.observability.logging import redact_secret_text


def _seed_demo_if_safe(settings) -> None:
    if settings.demo_mode and not settings.is_production:
        factory = create_session_factory(settings)
        init_database(factory)
        seed_demo_tenant(factory)


def _requires_managed_database(settings) -> bool:
    return settings.is_staging_or_production


def _configured_database_url() -> str:
    return (os.getenv("SUPPLIER_DATABASE_URL") or os.getenv("DATABASE_URL") or "").strip()


def _validate_database_url(settings) -> str:
    raw_url = _configured_database_url()
    if _requires_managed_database(settings) and not raw_url:
        return "SUPPLIER_DATABASE_URL or DATABASE_URL is required for staging/production migrations."
    try:
        parsed = make_url(settings.database_url)
    except Exception as exc:
        return f"Database URL is invalid: {redact_secret_text(exc)}"
    if _requires_managed_database(settings) and parsed.drivername.startswith("sqlite"):
        return "Staging/production migrations require a Postgres SUPPLIER_DATABASE_URL or DATABASE_URL; SQLite is not allowed."
    if _requires_managed_database(settings) and not parsed.drivername.startswith("postgresql"):
        return "Staging/production migrations require a Postgres database URL."
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run database migrations safely.")
    parser.add_argument("--create-all-fallback", action="store_true", help="Use SQLAlchemy create_all fallback after Alembic.")
    args = parser.parse_args(argv)
    settings = get_settings()
    database_issue = _validate_database_url(settings)
    if database_issue:
        print(f"Migration configuration error: {database_issue}", file=sys.stderr)
        return 2
    if args.create_all_fallback and _requires_managed_database(settings):
        print("Migration configuration error: --create-all-fallback is disabled for staging/production.", file=sys.stderr)
        return 2
    command = [sys.executable, "-m", "alembic", "upgrade", "head"]
    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        _seed_demo_if_safe(settings)
        return 0
    if not args.create_all_fallback and _requires_managed_database(settings):
        print("Alembic migration failed; fix the migration/database issue and rerun scripts/migrate.py.", file=sys.stderr)
        return result.returncode
    factory = create_session_factory(settings)
    init_database(factory)
    _seed_demo_if_safe(settings)
    print("Alembic failed; SQLAlchemy create_all fallback applied for local/demo mode.")
    return 0 if not settings.is_production else result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
