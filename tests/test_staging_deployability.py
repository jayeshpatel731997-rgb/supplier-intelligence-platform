from __future__ import annotations

import importlib
import json
from pathlib import Path

from sqlalchemy import select

from tests.test_backend_api import _client, _headers


def test_ready_reports_connector_scoring_convex_and_llm_status(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLIER_CONNECTOR_MODE", "demo")
    monkeypatch.setenv("SUPPLIER_LLM_NARRATIVE_PROVIDER", "none")
    monkeypatch.setenv("CONVEX_URL", "")
    client = _client(monkeypatch, tmp_path)

    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["backend"]["requested"] == "sqlalchemy"
    assert body["backend"]["active"] == "sqlalchemy"
    assert body["backend"]["fallback"] is False
    assert body["connectors"]["mode"] == "demo"
    assert body["connectors"]["stub_available"] is True
    assert body["scoring_config"]["available"] is True
    assert body["convex"]["configured"] is False
    assert body["convex"]["status"] == "not_configured"
    assert body["llm_narrative"]["provider"] == "none"
    assert body["llm_narrative"]["available"] is True
    assert "secret" not in json.dumps(body).lower()


def test_ready_reports_unimplemented_llm_provider_as_interface_only(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLIER_LLM_NARRATIVE_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-exposed")
    client = _client(monkeypatch, tmp_path)

    response = client.get("/ready")

    assert response.status_code == 200
    narrative = response.json()["llm_narrative"]
    assert narrative["provider"] == "openai"
    assert narrative["configured"] is False
    assert narrative["available"] is False
    assert narrative["status"] == "interface_only"
    assert "must-not-be-exposed" not in response.text


def test_seed_demo_data_is_idempotent(monkeypatch, tmp_path):
    database_path = tmp_path / "seed.db"
    monkeypatch.setenv("SUPPLIER_DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("SUPPLIER_DEMO_MODE", "true")

    import src.config as config
    from scripts.seed_demo_data import seed_demo_data
    from src.database import create_session_factory, init_database
    from src.models import (
        SupplierConnectorSync,
        SupplierEvidenceAction,
        SupplierEvidenceRun,
        SupplierWeakSignal,
    )

    config.get_settings.cache_clear()
    settings = config.get_settings()
    session_factory = create_session_factory(settings)
    init_database(session_factory)

    first = seed_demo_data(session_factory=session_factory, tenant_id="demo-tenant")
    second = seed_demo_data(session_factory=session_factory, tenant_id="demo-tenant")

    assert first["tenant_id"] == "demo-tenant"
    assert second["weak_signals"] == first["weak_signals"]
    with session_factory() as session:
        signals = session.scalars(select(SupplierWeakSignal)).all()
        runs = session.scalars(select(SupplierEvidenceRun)).all()
        actions = session.scalars(select(SupplierEvidenceAction)).all()
        seed_syncs = session.scalars(
            select(SupplierConnectorSync).where(SupplierConnectorSync.source_system == "demo_seed")
        ).all()

    assert len(signals) == first["weak_signals"]
    assert len(runs) == 1
    assert len(actions) == first["actions"]
    assert len(seed_syncs) == 1


def test_seed_demo_data_creates_its_own_run_when_other_run_exists(monkeypatch, tmp_path):
    database_path = tmp_path / "seed-existing.db"
    monkeypatch.setenv("SUPPLIER_DATABASE_URL", f"sqlite:///{database_path}")

    import src.config as config
    from scripts.seed_demo_data import seed_demo_data
    from src.database import create_session_factory, init_database
    from src.models import SupplierEvidenceRun
    from src.repositories.tenants import TenantRepository

    config.get_settings.cache_clear()
    factory = create_session_factory(config.get_settings())
    init_database(factory)
    with factory() as session:
        TenantRepository(session).ensure_demo_seed()
        session.add(
            SupplierEvidenceRun(
                tenant_id="demo-tenant",
                run_id="unrelated-run",
                scoring_version="unrelated-v1",
            )
        )
        session.commit()

    result = seed_demo_data(session_factory=factory, tenant_id="demo-tenant")

    assert result["evidence_run"] != "unrelated-run"
    with factory() as session:
        versions = set(session.scalars(select(SupplierEvidenceRun.scoring_version)).all())
    assert versions == {"unrelated-v1", "demo-staging-v1"}


def test_staging_seed_rejects_predictable_demo_api_key(monkeypatch, tmp_path):
    import pytest

    from scripts.seed_demo_data import seed_demo_data
    from src.config import Settings
    from src.database import create_session_factory, init_database

    monkeypatch.delenv("SUPPLIER_DEMO_API_KEY", raising=False)
    settings = Settings(
        deployment_mode="render-staging",
        demo_mode=False,
        database_url=f"sqlite:///{tmp_path / 'staging-seed.db'}",
    )
    factory = create_session_factory(settings)
    init_database(factory)

    with pytest.raises(RuntimeError, match="SUPPLIER_DEMO_API_KEY"):
        seed_demo_data(session_factory=factory, tenant_id="demo-tenant", settings=settings)


def test_staging_seed_creates_explicit_key_for_existing_demo_tenant(monkeypatch, tmp_path):
    from scripts.seed_demo_data import seed_demo_data
    from src.config import Settings
    from src.database import create_session_factory, init_database
    from src.repositories.tenants import TenantRepository

    database_url = f"sqlite:///{tmp_path / 'staging-existing.db'}"
    local_settings = Settings(database_url=database_url, demo_mode=True)
    factory = create_session_factory(local_settings)
    init_database(factory)
    with factory() as session:
        TenantRepository(session).ensure_demo_seed()
        session.commit()

    monkeypatch.setenv("SUPPLIER_DEMO_API_KEY", "staging-rotated-secret")
    staging_settings = Settings(
        deployment_mode="render-staging",
        demo_mode=False,
        database_url=database_url,
    )
    seed_demo_data(session_factory=factory, tenant_id="demo-tenant", settings=staging_settings)

    with factory() as session:
        repo = TenantRepository(session)
        context = repo.validate_api_key(
            "demo-tenant",
            "staging-rotated-secret",
        )
        legacy_context = repo.validate_api_key("demo-tenant", "demo-api-key")
    assert context is not None
    assert context.role == "risk_manager"
    assert legacy_context is None


def test_staging_oidc_seed_creates_membership_without_api_key(monkeypatch, tmp_path):
    from scripts.seed_demo_data import seed_demo_data
    from src.config import Settings
    from src.database import create_session_factory, init_database
    from src.repositories.tenants import TenantRepository

    database_url = f"sqlite:///{tmp_path / 'staging-oidc.db'}"
    settings = Settings(
        deployment_mode="render-staging",
        demo_mode=False,
        auth_provider="oidc",
        database_url=database_url,
    )
    factory = create_session_factory(settings)
    init_database(factory)
    with factory() as session:
        TenantRepository(session).ensure_demo_seed()
        session.commit()
    monkeypatch.setenv("SUPPLIER_STAGING_SEED_USERNAME", "risk.manager@example.test")
    monkeypatch.delenv("SUPPLIER_DEMO_API_KEY", raising=False)

    first = seed_demo_data(session_factory=factory, tenant_id="demo-tenant", settings=settings)
    second = seed_demo_data(session_factory=factory, tenant_id="demo-tenant", settings=settings)

    assert second["weak_signals"] == first["weak_signals"]
    with factory() as session:
        repo = TenantRepository(session)
        membership = repo.get_membership("demo-tenant", "risk.manager@example.test")
        assert membership is not None
        assert membership.role == "risk_manager"
        assert repo.validate_api_key("demo-tenant", "demo-api-key") is None


def test_staging_oidc_seed_requires_explicit_membership_identity(monkeypatch, tmp_path):
    import pytest

    from scripts.seed_demo_data import seed_demo_data
    from src.config import Settings
    from src.database import create_session_factory, init_database

    settings = Settings(
        deployment_mode="render-staging",
        demo_mode=False,
        auth_provider="oidc",
        database_url=f"sqlite:///{tmp_path / 'staging-oidc-missing-user.db'}",
    )
    factory = create_session_factory(settings)
    init_database(factory)
    monkeypatch.delenv("SUPPLIER_STAGING_SEED_USERNAME", raising=False)

    with pytest.raises(RuntimeError, match="SUPPLIER_STAGING_SEED_USERNAME"):
        seed_demo_data(session_factory=factory, tenant_id="demo-tenant", settings=settings)


def test_demo_seed_rejects_custom_tenant(monkeypatch, tmp_path):
    import pytest

    from scripts.seed_demo_data import seed_demo_data
    from src.config import Settings
    from src.database import create_session_factory, init_database

    settings = Settings(database_url=f"sqlite:///{tmp_path / 'custom-seed.db'}")
    factory = create_session_factory(settings)
    init_database(factory)

    with pytest.raises(ValueError, match="demo-tenant"):
        seed_demo_data(
            session_factory=factory,
            tenant_id="customer-tenant",
            settings=settings,
        )


def test_smoke_script_exercises_evidence_workflow(monkeypatch):
    import scripts.smoke_staging as smoke

    calls: list[tuple[str, str]] = []
    payloads: dict[str, object] = {}

    def fake_request_json(base_url, path, headers=None, timeout=10, method="GET", payload=None):
        del base_url, timeout
        calls.append((method, path))
        payloads[path] = payload
        if path == "/suppliers" and method == "GET" and not headers:
            return smoke.SmokeResponse(401, json.dumps({"detail": "auth required"}), "application/json")
        if path == "/system/status":
            if headers and headers.get("X-Tenant-ID") == "cross-tenant-smoke-probe":
                return smoke.SmokeResponse(403, json.dumps({"detail": "invalid tenant"}), "application/json")
            return smoke.SmokeResponse(200, json.dumps({"tenant_id": "demo-tenant"}), "application/json")
        if path == "/evidence/runs" and method == "POST":
            return smoke.SmokeResponse(
                200,
                json.dumps({"run_id": "evr_smoke", "actions": [{"id": 7}], "status": "completed"}),
                "application/json",
            )
        if path == "/evidence/connectors/news/sync":
            return smoke.SmokeResponse(
                200,
                json.dumps({"status": "completed", "records_accepted": 1}),
                "application/json",
            )
        if path == "/evidence/scoring-config/current":
            return smoke.SmokeResponse(
                200,
                json.dumps({"version": "default-v1", "is_active": True}),
                "application/json",
            )
        if path == "/evidence/actions/7":
            return smoke.SmokeResponse(
                200,
                json.dumps({"id": 7, "status": "in_progress"}),
                "application/json",
            )
        return smoke.SmokeResponse(200, json.dumps({"status": "ok", "id": 7}), "application/json")

    monkeypatch.setattr(smoke, "request_json", fake_request_json)

    result = smoke.run_smoke(
        "https://api.example.test/",
        {"X-Tenant-ID": "demo-tenant", "X-API-Key": "demo-api-key"},
    )

    assert result == 0
    assert ("GET", "/health") in calls
    assert ("GET", "/ready") in calls
    assert ("POST", "/evidence/connectors/news/sync") in calls
    assert ("POST", "/evidence/runs") in calls
    assert ("PATCH", "/evidence/actions/7") in calls
    assert ("GET", "/evidence/scoring-config/current") in calls
    assert payloads["/evidence/runs"] == {"include_demo_signals": False}


def test_smoke_script_fails_when_connector_workflow_reports_failure(monkeypatch):
    import scripts.smoke_staging as smoke

    def fake_request_json(base_url, path, headers=None, timeout=10, method="GET", payload=None):
        del base_url, timeout, method, payload
        if path == "/suppliers" and not headers:
            return smoke.SmokeResponse(401, json.dumps({"detail": "auth required"}), "application/json")
        if path == "/suppliers":
            return smoke.SmokeResponse(200, json.dumps([]), "application/json")
        if path == "/system/status":
            if headers and headers.get("X-Tenant-ID") == "cross-tenant-smoke-probe":
                return smoke.SmokeResponse(403, json.dumps({"detail": "invalid tenant"}), "application/json")
            return smoke.SmokeResponse(200, json.dumps({"tenant_id": "demo-tenant"}), "application/json")
        if path == "/evidence/connectors/news/sync":
            return smoke.SmokeResponse(
                200,
                json.dumps({"status": "failed", "records_accepted": 0}),
                "application/json",
            )
        if path == "/evidence/runs":
            return smoke.SmokeResponse(
                200,
                json.dumps({"run_id": "evr_smoke", "actions": [{"id": 7}], "status": "completed"}),
                "application/json",
            )
        if path == "/evidence/actions/7":
            return smoke.SmokeResponse(200, json.dumps({"status": "in_progress"}), "application/json")
        if path == "/evidence/scoring-config/current":
            return smoke.SmokeResponse(200, json.dumps({"version": "default-v1"}), "application/json")
        return smoke.SmokeResponse(200, json.dumps({"status": "ok"}), "application/json")

    monkeypatch.setattr(smoke, "request_json", fake_request_json)

    result = smoke.run_smoke(
        "https://api.example.test/",
        {"X-Tenant-ID": "demo-tenant", "X-API-Key": "demo-api-key"},
    )

    assert result == 1


def test_smoke_script_requires_auth_unless_health_only(monkeypatch):
    import scripts.smoke_staging as smoke

    def fake_request_json(base_url, path, headers=None, timeout=10, method="GET", payload=None):
        del base_url, headers, timeout, method, payload
        if path == "/suppliers":
            return smoke.SmokeResponse(401, json.dumps({"detail": "auth required"}), "application/json")
        return smoke.SmokeResponse(200, json.dumps({"status": "ok"}), "application/json")

    monkeypatch.setattr(smoke, "request_json", fake_request_json)

    assert smoke.run_smoke("https://api.example.test/", {}) == 1
    assert smoke.run_smoke("https://api.example.test/", {}, health_only=True) == 0


def test_smoke_health_only_does_not_run_workflow_with_credentials(monkeypatch):
    import scripts.smoke_staging as smoke

    calls: list[str] = []

    def fake_request_json(base_url, path, headers=None, timeout=10, method="GET", payload=None):
        del base_url, headers, timeout, method, payload
        calls.append(path)
        if path == "/suppliers":
            return smoke.SmokeResponse(401, json.dumps({"detail": "auth required"}), "application/json")
        return smoke.SmokeResponse(200, json.dumps({"status": "ok"}), "application/json")

    monkeypatch.setattr(smoke, "request_json", fake_request_json)

    result = smoke.run_smoke(
        "https://api.example.test/",
        {"X-Tenant-ID": "demo-tenant", "X-API-Key": "demo-api-key"},
        health_only=True,
    )

    assert result == 0
    assert "/evidence/runs" not in calls
    assert "/evidence/connectors/news/sync" not in calls


def test_smoke_allows_visible_public_connector_degradation(monkeypatch):
    import scripts.smoke_staging as smoke

    def fake_request_json(base_url, path, headers=None, timeout=10, method="GET", payload=None):
        del base_url, timeout, method, payload
        if path == "/suppliers" and not headers:
            return smoke.SmokeResponse(401, json.dumps({"detail": "auth required"}), "application/json")
        if path == "/ready":
            return smoke.SmokeResponse(
                200,
                json.dumps({"status": "ready", "connectors": {"mode": "public"}}),
                "application/json",
            )
        if path == "/system/status":
            if headers and headers.get("X-Tenant-ID") == "cross-tenant-smoke-probe":
                return smoke.SmokeResponse(403, json.dumps({"detail": "invalid tenant"}), "application/json")
            return smoke.SmokeResponse(200, json.dumps({"tenant_id": "demo-tenant"}), "application/json")
        if path == "/evidence/connectors/news/sync":
            return smoke.SmokeResponse(
                200,
                json.dumps({"connector": "news", "status": "skipped", "records_accepted": 0}),
                "application/json",
            )
        if path == "/evidence/scoring-config/current":
            return smoke.SmokeResponse(200, json.dumps({"version": "default-v1"}), "application/json")
        if path == "/evidence/runs":
            return smoke.SmokeResponse(
                200,
                json.dumps({"run_id": "evr_smoke", "actions": [{"id": 7}], "status": "completed"}),
                "application/json",
            )
        if path == "/evidence/actions/7":
            return smoke.SmokeResponse(200, json.dumps({"status": "in_progress"}), "application/json")
        return smoke.SmokeResponse(200, json.dumps({"status": "ok"}), "application/json")

    monkeypatch.setattr(smoke, "request_json", fake_request_json)

    result = smoke.run_smoke(
        "https://api.example.test/",
        {"X-Tenant-ID": "demo-tenant", "X-API-Key": "demo-api-key"},
    )

    assert result == 0


def test_scoring_config_read_returns_unpersisted_builtin_default(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    response = client.get("/evidence/scoring-config/current", headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "default-v1"
    assert body["persisted"] is False

    from backend import main
    from src.models import SupplierEvidenceScoringVersion

    with main.runtime.session_factory() as session:
        assert session.scalars(select(SupplierEvidenceScoringVersion)).all() == []


def test_unknown_scoring_version_uses_named_builtin_default(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    response = client.post(
        "/evidence/runs",
        json={"scoring_version": "missing-v9", "include_demo_signals": True},
        headers=_headers(),
    )

    assert response.status_code == 200
    assert response.json()["scoring_version"] == "default-v1"


def test_evidence_api_rejects_non_finite_and_out_of_range_numbers(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    signal_response = client.post(
        "/evidence/signals/import",
        json={
            "source_system": "manual",
            "signals": [
                {
                    "supplier_id": "sup-1",
                    "supplier_name": "Supplier One",
                    "signal_type": "financial",
                    "severity": 101,
                    "confidence": 2,
                }
            ],
        },
        headers=_headers(),
    )
    scoring_response = client.put(
        "/evidence/scoring-config",
        content=(
            '{"version":"invalid-v1","signal_type_weights":{"financial":Infinity},'
            '"supplier_criticality":{}}'
        ),
        headers={"Content-Type": "application/json", **_headers()},
    )

    assert signal_response.status_code == 422
    assert scoring_response.status_code == 422


def test_evidence_signal_url_redacts_credential_path_segments(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    response = client.post(
        "/evidence/signals/import",
        json={
            "source_system": "manual",
            "signals": [
                {
                    "supplier_id": "sup-1",
                    "supplier_name": "Supplier One",
                    "signal_id": "secret-url",
                    "source_url": "https://feed.example.test/token/hidden-secret/article?id=secret",
                }
            ],
        },
        headers=_headers(),
    )
    signals = client.get("/evidence/signals", headers=_headers()).json()

    assert response.status_code == 200
    assert "hidden-secret" not in str(signals)
    assert signals[0]["source_url"] == "https://feed.example.test/***/***/article"


def test_secret_redaction_handles_malformed_url_port():
    from src.observability.logging import redact_secret_text

    safe = redact_secret_text("GET https://feed.example.test:invalid/token-secret failed")

    assert "token-secret" not in safe
    assert "invalid" not in safe


def test_streamlit_api_client_uses_configured_base_url(monkeypatch):
    monkeypatch.setenv("SUPPLIER_API_BASE_URL", "https://staging-api.example.test")

    import src.services.streamlit_api_client as client_module

    importlib.reload(client_module)
    client = client_module.StreamlitApiClient.from_env()

    assert client.base_url == "https://staging-api.example.test/"
    assert client.build_url("/ready") == "https://staging-api.example.test/ready"
    assert "unreachable" in client_module.friendly_api_error(RuntimeError("timed out")).lower()


def test_streamlit_api_client_uses_active_tenant_override(monkeypatch):
    monkeypatch.setenv("STAGING_TENANT_ID", "configured-tenant")
    monkeypatch.setenv("STAGING_API_KEY", "tenant-key")

    from src.services.streamlit_api_client import StreamlitApiClient

    client = StreamlitApiClient.from_env(
        "https://staging-api.example.test",
        tenant_id="active-tenant",
    )

    assert client.headers()["X-Tenant-ID"] == "active-tenant"
    assert client.headers()["X-API-Key"] == "tenant-key"


def test_streamlit_api_client_uses_demo_key_only_for_local_api(monkeypatch):
    monkeypatch.delenv("STAGING_API_KEY", raising=False)
    monkeypatch.delenv("SUPPLIER_DEMO_API_KEY", raising=False)

    from src.services.streamlit_api_client import StreamlitApiClient

    local_client = StreamlitApiClient.from_env("http://localhost:8000")
    staging_client = StreamlitApiClient.from_env("https://staging-api.example.test")

    assert local_client.api_key == "demo-api-key"
    assert staging_client.api_key == ""
