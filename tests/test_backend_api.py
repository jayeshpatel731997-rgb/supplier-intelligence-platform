from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


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
    assert status.status_code == 200
    assert status.json()["database"]["ok"] is True
    assert status.headers["X-Request-ID"]


def test_protected_routes_require_tenant_and_api_key(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    assert client.get("/suppliers").status_code == 401
    assert client.get("/suppliers", headers={"X-Tenant-ID": "demo-tenant"}).status_code == 401
    assert client.get("/suppliers", headers=_headers(api_key="bad-key")).status_code == 403
    assert client.get("/suppliers", headers=_headers(tenant_id="missing-tenant")).status_code == 403


def test_supplier_risk_sentinel_alert_and_acknowledge_flow(monkeypatch, tmp_path):
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
    assert upload.json()["success"] is True

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
