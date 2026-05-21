"""Collect tenant-scoped compliance evidence snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.database import create_session_factory, init_database
from src.services.compliance_evidence import EvidenceService
from src.tenancy import DEMO_TENANT_ID


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect compliance evidence for a tenant.")
    parser.add_argument("--tenant-id", default=DEMO_TENANT_ID)
    args = parser.parse_args()
    factory = create_session_factory(get_settings())
    init_database(factory)
    with factory() as session:
        service = EvidenceService(session, args.tenant_id)
        payload = {
            "access_control": service.collect_access_control_evidence(),
            "operations": service.collect_operational_evidence(),
        }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
