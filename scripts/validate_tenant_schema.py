"""Validate that business tables are tenant-scoped."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.database import create_session_factory, init_database
from src.services.migration_service import validate_tenant_schema


def main() -> int:
    settings = get_settings()
    factory = create_session_factory(settings)
    if not settings.is_production:
        init_database(factory)
    with factory() as session:
        result = validate_tenant_schema(session)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
