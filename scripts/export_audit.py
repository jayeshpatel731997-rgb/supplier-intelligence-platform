"""Export tenant-scoped audit logs as JSONL or CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.database import create_session_factory, init_database
from src.services.audit_export import AuditExportService
from src.tenancy import DEMO_TENANT_ID


def main() -> int:
    parser = argparse.ArgumentParser(description="Export audit logs for a tenant.")
    parser.add_argument("--tenant-id", default=DEMO_TENANT_ID)
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    args = parser.parse_args()
    factory = create_session_factory(get_settings())
    init_database(factory)
    with factory() as session:
        service = AuditExportService(session, args.tenant_id)
        payload = service.export_csv() if args.format == "csv" else service.export_jsonl()
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
