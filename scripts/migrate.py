"""Run Alembic migrations with a SQLite create_all fallback for local demos."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.database import create_session_factory, init_database, seed_demo_tenant


def _seed_demo_if_safe(settings) -> None:
    if settings.demo_mode and not settings.is_production:
        factory = create_session_factory(settings)
        init_database(factory)
        seed_demo_tenant(factory)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run database migrations safely.")
    parser.add_argument("--create-all-fallback", action="store_true", help="Use SQLAlchemy create_all fallback after Alembic.")
    args = parser.parse_args()
    settings = get_settings()
    command = [sys.executable, "-m", "alembic", "upgrade", "head"]
    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        _seed_demo_if_safe(settings)
        return 0
    if not args.create_all_fallback and settings.is_production:
        return result.returncode
    factory = create_session_factory(settings)
    init_database(factory)
    _seed_demo_if_safe(settings)
    print("Alembic failed; SQLAlchemy create_all fallback applied for local/demo mode.")
    return 0 if not settings.is_production else result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
