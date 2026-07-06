"""Local, non-network API smoke checks for demo/staging evidence collection."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="supplier-smoke-", ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "smoke.db"
        os.environ["SUPPLIER_SECURITY_MODE"] = "local"
        os.environ["SUPPLIER_DEPLOYMENT_MODE"] = "local-smoke"
        os.environ["SUPPLIER_DEMO_MODE"] = "true"
        os.environ["SUPPLIER_DATABASE_URL"] = f"sqlite:///{db_path}"
        os.environ["RATE_LIMIT_ENABLED"] = "false"

        import src.config as config
        import backend.main as backend_main

        config.get_settings.cache_clear()
        reloaded = importlib.reload(backend_main)
        headers = {"X-Tenant-ID": "demo-tenant", "X-API-Key": "demo-api-key"}

        with TestClient(reloaded.app) as client:
            checks = {
                "/live": client.get("/live"),
                "/health": client.get("/health"),
                "/ready": client.get("/ready"),
                "/system/status": client.get("/system/status", headers=headers),
                "/suppliers": client.get("/suppliers", headers=headers),
            }

        payload = {}
        failures = []
        for path, response in checks.items():
            payload[path] = {"status_code": response.status_code}
            if response.headers.get("X-Request-ID"):
                payload[path]["request_id_present"] = True
            if response.status_code >= 400:
                failures.append(path)

        print(json.dumps(payload, indent=2))
        try:
            bind = reloaded.runtime.session_factory.kw.get("bind")
            if bind is not None:
                bind.dispose()
        except Exception:
            pass
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
