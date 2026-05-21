"""Backfill legacy single-tenant rows into the demo tenant in non-production mode."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.database import create_session_factory, init_database
from src.services.migration_service import backfill_demo_tenant


def main() -> int:
    settings = get_settings()
    factory = create_session_factory(settings)
    init_database(factory)
    with factory() as session:
        result = backfill_demo_tenant(session, production_mode=settings.is_production)
        session.commit()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
