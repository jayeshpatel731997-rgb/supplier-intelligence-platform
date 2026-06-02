from __future__ import annotations

import importlib

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLIER_SECURITY_MODE", "local")
    monkeypatch.setenv("SUPPLIER_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("SUPPLIER_DEMO_MODE", "true")

    import src.config as config
    import backend.main as backend_main

    config.get_settings.cache_clear()
    importlib.reload(backend_main)
    return TestClient(backend_main.app)


def _headers(tenant_id: str = "demo-tenant", api_key: str = "demo-api-key"):
    return {"X-Tenant-ID": tenant_id, "X-API-Key": api_key, "X-Request-ID": "test-request-id"}


def test_fastapi_health_and_system_status(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    health = client.get("/health")
    live = client.get("/live")
    ready = client.get("/ready")
    status = client.get("/system/status", headers=_headers())

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert live.status_code == 200
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["database"]["ok"] is True
    assert status.status_code == 200
    assert status.json()["database"]["ok"] is True
    assert status.headers["X-Request-ID"]


def test_ready_reports_503_when_database_is_unavailable(monkeypatch):
    monkeypatch.setenv("SUPPLIER_SECURITY_MODE", "production")
    monkeypatch.setenv("SUPPLIER_DATABASE_URL", "postgresql+missingdriver://user:pass@db:5432/app")
    monkeypatch.setenv("SUPPLIER_DEMO_MODE", "false")

    import src.config as config
    import backend.main as backend_main

    config.get_settings.cache_clear()
    reloaded = importlib.reload(backend_main)

    assert reloaded.app_import_smoke() == "ok"
    with TestClient(reloaded.app) as client:
        live = client.get("/live")
        ready = client.get("/ready")

    assert live.status_code == 200
    assert ready.status_code == 503
    assert ready.json()["status"] == "degraded"
    assert ready.json()["database"]["ok"] is False


def test_ready_reports_503_for_degraded_production_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLIER_SECURITY_MODE", "production")
    monkeypatch.setenv("SUPPLIER_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("SUPPLIER_DEMO_MODE", "true")
    monkeypatch.setenv("AUTH_PROVIDER", "local")
    monkeypatch.delenv("AUTH_ALLOW_LOCAL_IN_PRODUCTION", raising=False)

    import src.config as config
    import backend.main as backend_main

    config.get_settings.cache_clear()
    reloaded = importlib.reload(backend_main)

    with TestClient(reloaded.app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"]["ok"] is True
    assert body["production_issues"]


def test_production_startup_does_not_create_schema_automatically(monkeypatch, tmp_path):
    database_path = tmp_path / "api.db"
    monkeypatch.setenv("SUPPLIER_SECURITY_MODE", "production")
    monkeypatch.setenv("SUPPLIER_DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("SUPPLIER_DEMO_MODE", "false")
    monkeypatch.setenv("AUTH_PROVIDER", "local")
    monkeypatch.setenv("AUTH_ALLOW_LOCAL_IN_PRODUCTION", "true")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://staging.example.com")

    import src.config as config
    import backend.main as backend_main

    config.get_settings.cache_clear()
    reloaded = importlib.reload(backend_main)

    with TestClient(reloaded.app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    engine = create_engine(f"sqlite:///{database_path}")
    assert "suppliers" not in inspect(engine).get_table_names()


def test_protected_routes_require_tenant_and_api_key(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    assert client.get("/suppliers").status_code == 401
    assert client.get("/suppliers", headers={"X-Tenant-ID": "demo-tenant"}).status_code == 401
    assert client.get("/suppliers", headers=_headers(api_key="bad-key")).status_code == 403
    assert client.get("/suppliers", headers=_headers(tenant_id="missing-tenant")).status_code == 403


def test_staging_production_suppliers_rejects_missing_auth(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLIER_SECURITY_MODE", "production")
    monkeypatch.setenv("SUPPLIER_DEPLOYMENT_MODE", "render-staging")
    monkeypatch.setenv("SUPPLIER_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("SUPPLIER_DEMO_MODE", "false")
    monkeypatch.setenv("AUTH_PROVIDER", "oidc")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://staging.example.com")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")

    import src.config as config
    import backend.main as backend_main

    config.get_settings.cache_clear()
    reloaded = importlib.reload(backend_main)

    with TestClient(reloaded.app) as client:
        response = client.get("/suppliers")

    assert response.status_code == 401
    assert "Bearer token" in response.json()["detail"]


def test_supplier_risk_sentinel_alert_and_acknowledge_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLIER_UPLOAD_STORAGE_PATH", str(tmp_path / "uploads"))
    client = _client(monkeypatch, tmp_path)

    upload = client.post(
        "/ingestion/upload",
        files={
            "file": (
                "suppliers.csv",
                b"Supplier,Country,Category,Annual Spend,On Time Delivery\nApex,Mexico,Machining,100000,88%",
                "text/csv",
            )
        },
        headers=_headers(),
    )
    assert upload.status_code == 200
    upload_body = upload.json()
    assert upload_body["success"] is True
    assert upload_body["upload_storage_provider"] == "local"
    assert upload_body["upload_key"].startswith("demo-tenant/")
    assert (tmp_path / "uploads" / upload_body["upload_key"]).exists()

    suppliers = client.get("/suppliers", headers=_headers())
    assert suppliers.status_code == 200
    assert suppliers.json()[0]["name"] == "Apex"

    risk = client.get("/risk/scores", headers=_headers())
    assert risk.status_code == 200
    assert len(risk.json()) == 1

    scan = client.post("/sentinel/scan", json={"mode": "demo"}, headers=_headers())
    assert scan.status_code == 200
    assert scan.json()["mode_used"] == "Demo Mode"

    alerts = client.get("/alerts", headers=_headers())
    assert alerts.status_code == 200
    assert len(alerts.json()) >= 1

    alert_id = alerts.json()[0]["id"]
    ack = client.post(f"/alerts/{alert_id}/acknowledge", headers=_headers())
    assert ack.status_code == 200
    assert ack.json()["status"] == "acknowledged"


def test_api_safe_without_external_keys(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    response = client.post("/sentinel/scan", json={"mode": "live_ai"}, headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["events"] == []
    assert body["error"]


def test_viewer_api_key_cannot_mutate_but_can_read(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    admin_headers = _headers()
    created = client.post(
        "/tenants/demo-tenant/api-keys",
        json={"username": "viewer@example.com", "role": "viewer", "label": "viewer key"},
        headers=admin_headers,
    )
    assert created.status_code == 200
    viewer_headers = _headers(api_key=created.json()["api_key"])

    assert client.get("/suppliers", headers=viewer_headers).status_code == 200
    denied = client.post(
        "/ingestion/upload",
        files={"file": ("suppliers.csv", b"Supplier\nApex", "text/csv")},
        headers=viewer_headers,
    )
    assert denied.status_code == 403


def test_org_admin_cannot_grant_platform_admin_role(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    admin_headers = _headers()
    created = client.post(
        "/tenants/demo-tenant/api-keys",
        json={"username": "org-admin@example.com", "role": "org_admin", "label": "org admin key"},
        headers=admin_headers,
    )
    assert created.status_code == 200
    org_admin_headers = _headers(api_key=created.json()["api_key"])

    key_escalation = client.post(
        "/tenants/demo-tenant/api-keys",
        json={"username": "new-platform@example.com", "role": "platform_admin", "label": "bad escalation"},
        headers=org_admin_headers,
    )
    membership_escalation = client.post(
        "/tenants/demo-tenant/memberships",
        json={"username": "new-platform@example.com", "role": "platform_admin"},
        headers=org_admin_headers,
    )

    assert key_escalation.status_code == 403
    assert membership_escalation.status_code == 403


def test_upload_rejects_unsupported_file_type(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    response = client.post(
        "/ingestion/upload",
        files={"file": ("suppliers.exe", b"not a supplier file", "application/octet-stream")},
        headers=_headers(),
    )

    assert response.status_code == 400
    assert "Unsupported upload file type" in response.json()["detail"]


def test_upload_rejects_unsafe_filename(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    response = client.post(
        "/ingestion/upload",
        files={"file": ("../suppliers.csv", b"Supplier\nApex", "text/csv")},
        headers=_headers(),
    )

    assert response.status_code == 400
    assert "Unsafe upload filename" in response.json()["detail"]


def test_upload_rejects_files_over_configured_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLIER_MAX_UPLOAD_BYTES", "8")
    client = _client(monkeypatch, tmp_path)

    response = client.post(
        "/ingestion/upload",
        files={"file": ("suppliers.csv", b"Supplier\nApex\nTooLarge", "text/csv")},
        headers=_headers(),
    )

    assert response.status_code == 413
    assert "Upload exceeds" in response.json()["detail"]


def test_ready_reports_503_when_production_upload_storage_config_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLIER_SECURITY_MODE", "production")
    monkeypatch.setenv("SUPPLIER_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("SUPPLIER_DEMO_MODE", "false")
    monkeypatch.setenv("AUTH_PROVIDER", "local")
    monkeypatch.setenv("AUTH_ALLOW_LOCAL_IN_PRODUCTION", "true")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://staging.example.com")
    monkeypatch.setenv("SUPPLIER_UPLOAD_STORAGE_PROVIDER", "s3")
    monkeypatch.delenv("SUPPLIER_UPLOAD_STORAGE_BUCKET", raising=False)
    monkeypatch.delenv("SUPPLIER_UPLOAD_STORAGE_ENDPOINT_URL", raising=False)

    import src.config as config
    import backend.main as backend_main

    config.get_settings.cache_clear()
    reloaded = importlib.reload(backend_main)

    with TestClient(reloaded.app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert any("SUPPLIER_UPLOAD_STORAGE_BUCKET" in issue for issue in response.json()["production_issues"])
