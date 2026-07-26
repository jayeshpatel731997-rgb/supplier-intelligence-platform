from __future__ import annotations

from tests.test_backend_api import _client, _headers


def test_evidence_chain_run_persists_signals_run_actions_and_narrative(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    imported = client.post(
        "/evidence/signals/import",
        json={
            "source_system": "erp",
            "signals": [
                {
                    "supplier_id": "sup-api-1",
                    "supplier_name": "API Components Mexico",
                    "signal_id": "api-sig-1",
                    "signal_type": "operational",
                    "driver": "Delivery reliability decline",
                    "source": "ERP receiving history",
                    "observed_at": "2026-06-01",
                    "severity": 72,
                    "confidence": 0.86,
                    "summary": "On-time delivery fell below the customer threshold for three weekly releases.",
                },
                {
                    "supplier_id": "sup-api-1",
                    "supplier_name": "API Components Mexico",
                    "signal_id": "api-sig-2",
                    "signal_type": "email",
                    "driver": "Expedite request pattern",
                    "source": "Supplier email digest",
                    "observed_at": "2026-06-02",
                    "severity": 64,
                    "confidence": 0.76,
                    "summary": "Supplier requested expedited handling for two open POs.",
                },
            ],
        },
        headers=_headers(),
    )

    assert imported.status_code == 200
    assert imported.json()["accepted"] == 2

    scoring = client.put(
        "/evidence/scoring-config",
        json={
            "version": "customer-criticality-v1",
            "description": "Customer criticality test config.",
            "signal_type_weights": {},
            "supplier_criticality": {},
        },
        headers=_headers(),
    )
    assert scoring.status_code == 200
    assert scoring.json()["persisted"] is True

    run = client.post(
        "/evidence/runs",
        json={"scoring_version": "customer-criticality-v1"},
        headers=_headers(),
    )

    assert run.status_code == 200
    body = run.json()
    assert body["tenant_id"] == "demo-tenant"
    assert body["status"] == "completed"
    assert body["scoring_version"] == "customer-criticality-v1"
    assert body["suppliers"][0]["supplier_id"] == "sup-api-1"
    assert body["suppliers"][0]["evidence_chain"][0]["signal_id"] in {"api-sig-1", "api-sig-2"}
    assert "STRUCTURED EVIDENCE ONLY" in body["narrative"]["policy"]
    assert body["actions"]

    action_id = body["actions"][0]["id"]
    updated = client.patch(
        f"/evidence/actions/{action_id}",
        json={"status": "in_progress", "owner": "buyer@example.com"},
        headers=_headers(),
    )

    assert updated.status_code == 200
    assert updated.json()["status"] == "in_progress"
    assert updated.json()["owner"] == "buyer@example.com"

    fetched = client.get(f"/evidence/runs/{body['run_id']}", headers=_headers())
    assert fetched.status_code == 200
    assert fetched.json()["run_id"] == body["run_id"]
    assert fetched.json()["actions"][0]["status"] == "in_progress"


def test_evidence_scoring_config_versions_are_tenant_scoped(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    saved = client.put(
        "/evidence/scoring-config",
        json={
            "version": "automotive-critical-v2",
            "description": "Automotive line-stop criticality weights",
            "signal_type_weights": {"financial": 1.4, "operational": 1.25, "email": 1.0},
            "supplier_criticality": {"sup-critical": 1.3},
        },
        headers=_headers(),
    )

    assert saved.status_code == 200
    assert saved.json()["version"] == "automotive-critical-v2"
    assert saved.json()["is_active"] is True

    current = client.get("/evidence/scoring-config/current", headers=_headers())

    assert current.status_code == 200
    assert current.json()["version"] == "automotive-critical-v2"
    assert current.json()["signal_type_weights"]["financial"] == 1.4


def test_evidence_connector_catalog_lists_real_signal_sources(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    response = client.get("/evidence/connectors", headers=_headers())

    assert response.status_code == 200
    source_keys = {item["source_system"] for item in response.json()["connectors"]}
    assert {"erp", "supplier_portal", "financial_filings", "email", "hiring", "logistics"} <= source_keys


def test_viewer_can_read_evidence_runs_but_cannot_mutate(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    created = client.post(
        "/tenants/demo-tenant/api-keys",
        json={"username": "viewer@example.com", "role": "viewer", "label": "viewer key"},
        headers=_headers(),
    )
    viewer_headers = _headers(api_key=created.json()["api_key"])

    assert client.get("/evidence/runs", headers=viewer_headers).status_code == 200
    assert client.post("/evidence/runs", json={}, headers=viewer_headers).status_code == 403


def test_demo_signal_injection_is_restricted_to_local_demo_tenant(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    client.post("/tenants", json={"tenant_id": "tenant-b", "name": "Tenant B"}, headers=_headers())
    key = client.post(
        "/tenants/tenant-b/api-keys",
        json={"username": "risk-b@example.com", "role": "risk_manager", "label": "tenant-b key"},
        headers=_headers(),
    ).json()["api_key"]

    response = client.post(
        "/evidence/runs",
        json={"include_demo_signals": True},
        headers=_headers(tenant_id="tenant-b", api_key=key),
    )

    assert response.status_code == 400
    assert "demo-tenant" in response.json()["detail"]
