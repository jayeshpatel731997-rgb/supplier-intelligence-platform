from __future__ import annotations

import importlib
import os
from pathlib import Path

from src.config import Settings


ROOT = Path(__file__).resolve().parents[1]


def test_staging_runtime_requires_managed_database_and_safe_runtime_config():
    settings = Settings(
        security_mode="local",
        deployment_mode="render-staging",
        database_url="sqlite:///data/local.db",
        demo_mode=True,
        auth_provider="local",
        cors_allow_origins="*",
    )

    issues = settings.validate_runtime()

    assert any("Postgres" in issue for issue in issues)
    assert any("disable demo mode" in issue for issue in issues)
    assert any("CORS_ALLOW_ORIGINS" in issue for issue in issues)


def test_migrate_script_rejects_missing_database_url_for_staging(monkeypatch):
    monkeypatch.delenv("SUPPLIER_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SUPPLIER_DEPLOYMENT_MODE", "render-staging")
    monkeypatch.setenv("SUPPLIER_SECURITY_MODE", "production")
    monkeypatch.setenv("SUPPLIER_DEMO_MODE", "false")

    import src.config as config
    import scripts.migrate as migrate

    config.get_settings.cache_clear()
    reloaded = importlib.reload(migrate)

    assert reloaded.main([]) == 2


def test_migrate_script_rejects_sqlite_database_url_for_staging(monkeypatch):
    monkeypatch.setenv("SUPPLIER_DATABASE_URL", "sqlite:///data/local.db")
    monkeypatch.setenv("SUPPLIER_DEPLOYMENT_MODE", "render-staging")
    monkeypatch.setenv("SUPPLIER_SECURITY_MODE", "production")
    monkeypatch.setenv("SUPPLIER_DEMO_MODE", "false")

    import src.config as config
    import scripts.migrate as migrate

    config.get_settings.cache_clear()
    reloaded = importlib.reload(migrate)

    assert reloaded.main([]) == 2


def test_render_blueprints_use_alembic_without_create_all_fallback():
    for blueprint in ("render.yaml", "render.full.yaml"):
        text = (ROOT / blueprint).read_text(encoding="utf-8")

        assert "dockerCommand: sh scripts/start_api_render.sh" in text
        assert "--create-all-fallback" not in text
        assert "healthCheckPath: /live" in text


def test_render_blueprints_keep_api_and_ui_services_separate():
    for blueprint in ("render.yaml", "render.full.yaml"):
        text = (ROOT / blueprint).read_text(encoding="utf-8")

        assert "name: supplier-intelligence-api" in text
        assert "dockerfilePath: ./backend/Dockerfile" in text
        assert "dockerCommand: sh scripts/start_api_render.sh" in text
        assert "name: supplier-intelligence-ui" in text
        assert "dockerfilePath: ./Dockerfile" in text
        assert "dockerCommand: sh scripts/start_ui_render.sh" in text
        assert "healthCheckPath: /_stcore/health" in text


def test_render_startup_scripts_use_exec_and_render_port():
    api_script = (ROOT / "scripts" / "start_api_render.sh").read_text(encoding="utf-8")
    ui_script = (ROOT / "scripts" / "start_ui_render.sh").read_text(encoding="utf-8")

    assert api_script.startswith("#!/usr/bin/env sh\nset -e\n")
    assert "python scripts/migrate.py" in api_script
    assert 'exec uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-10000}"' in api_script
    assert ui_script.startswith("#!/usr/bin/env sh\nset -e\n")
    assert 'exec streamlit run app.py --server.port="${PORT:-10000}" --server.address=0.0.0.0' in ui_script


def test_render_web_services_do_not_embed_quoted_shell_commands():
    for blueprint in ("render.yaml", "render.full.yaml"):
        text = (ROOT / blueprint).read_text(encoding="utf-8")

        assert 'dockerCommand: /bin/sh -c "python scripts/migrate.py && uvicorn' not in text
        assert 'dockerCommand: /bin/sh -c "streamlit run app.py' not in text


def test_render_runbook_matches_blueprint_auth_and_resource_truth():
    blueprint = (ROOT / "render.yaml").read_text(encoding="utf-8")
    runbook = (ROOT / "RENDER_STAGING_RUNBOOK.md").read_text(encoding="utf-8")

    assert "name: supplier-intelligence-api" in blueprint
    assert "name: supplier-intelligence-ui" in blueprint
    assert "name: supplier-intelligence-postgres" in blueprint
    assert "API + Streamlit + Postgres" in runbook
    assert "does not generate" in runbook
    assert "`SUPPLIER_DEMO_API_KEY`" in runbook
    assert "STAGING_BEARER_TOKEN" in runbook
    assert "STAGING_UI_BASE_URL" in runbook


def test_smoke_script_redacts_secret_like_values():
    import scripts.smoke_staging as smoke

    text = smoke.redact("Authorization: Bearer secret-token X-API-Key=secret-key DATABASE_URL=postgres://user:pass@host/db")

    assert "secret-token" not in text
    assert "secret-key" not in text
    assert "user:pass" not in text
    assert "***" in text


def test_smoke_script_redacts_cookie_and_client_secret_values():
    import scripts.smoke_staging as smoke

    text = smoke.redact("Cookie: session=private-value client_secret=private-client-value")

    assert "private-value" not in text
    assert "private-client-value" not in text
    assert "***" in text


def test_smoke_script_summaries_omit_database_urls_and_evidence_payloads():
    import scripts.smoke_staging as smoke

    response = smoke.SmokeResponse(
        200,
        (
            '{"status":"completed","run_id":"run-1","database":'
            '{"ok":true,"driver":"postgresql+psycopg","url":"postgresql://hidden"},'
            '"suppliers":[{"supplier_name":"Confidential Supplier"}]}'
        ),
        "application/json",
    )

    summary = smoke._json_summary(response)

    assert "run-1" in summary
    assert "postgresql://hidden" not in summary
    assert "Confidential Supplier" not in summary


def test_smoke_script_builds_auth_headers_without_printing_values(monkeypatch):
    monkeypatch.setenv("STAGING_TENANT_ID", "tenant-a")
    monkeypatch.setenv("STAGING_API_KEY", "api-key-value")
    monkeypatch.delenv("STAGING_BEARER_TOKEN", raising=False)

    import scripts.smoke_staging as smoke

    headers = smoke.auth_headers(os.environ)

    assert headers == {"X-Tenant-ID": "tenant-a", "X-API-Key": "api-key-value"}


def test_smoke_script_prefers_staging_api_base_url_alias():
    import scripts.smoke_staging as smoke

    assert (
        smoke.staging_base_url(
            {
                "STAGING_API_BASE_URL": "https://api.example.test",
                "STAGING_BASE_URL": "https://legacy.example.test",
            }
        )
        == "https://api.example.test"
    )
    assert smoke.staging_base_url({"STAGING_BASE_URL": "https://legacy.example.test"}) == "https://legacy.example.test"


def test_smoke_script_preflight_requires_oidc_tenant_and_ui_urls():
    import scripts.smoke_staging as smoke

    env = {
        "STAGING_API_BASE_URL": "https://api.example.test",
        "STAGING_BEARER_TOKEN": "not-printed",
    }
    headers = smoke.auth_headers(env)

    errors = smoke.configuration_errors(env, headers, health_only=False, skip_ui=False)

    assert any("STAGING_EXPECTED_TENANT_ID" in error for error in errors)
    assert any("STAGING_UI_BASE_URL" in error for error in errors)
    assert all("not-printed" not in error for error in errors)


def test_smoke_script_checks_oidc_tenant_header_override(monkeypatch):
    import scripts.smoke_staging as smoke

    calls: list[tuple[str, dict[str, str]]] = []

    def fake_request(base_url, path, headers=None, timeout=10, method="GET", payload=None):
        del base_url, timeout, method, payload
        active_headers = dict(headers or {})
        calls.append((path, active_headers))
        if path == "/system/status":
            return smoke.SmokeResponse(
                200,
                '{"tenant_id":"tenant-a","status":"ok"}',
                "application/json",
            )
        return smoke.SmokeResponse(200, "{}", "application/json")

    monkeypatch.setattr(smoke, "request_json", fake_request)

    checks = smoke._tenant_isolation_checks(
        "https://api.example.test/",
        {"Authorization": "Bearer hidden"},
        "tenant-a",
    )

    assert all(ok for _name, ok, _detail in checks)
    assert any(headers.get("X-Tenant-ID") == "cross-tenant-smoke-probe" for _path, headers in calls)


def test_smoke_script_rejects_streamlit_html_fallback(monkeypatch):
    import scripts.smoke_staging as smoke

    def fake_request(_base_url, path, headers=None, timeout=10):
        if path == "/suppliers":
            return smoke.SmokeResponse(
                status=200,
                body="<html><title>Supplier Intelligence Platform</title></html>",
                content_type="text/html; charset=utf-8",
            )
        return smoke.SmokeResponse(
            status=200,
            body='{"status":"ok","database":{"ok":true},"api":{"ok":true},"production_issues":[]}',
            content_type="application/json",
        )

    monkeypatch.setattr(smoke, "request_json", fake_request)

    assert smoke.run_smoke("https://staging.example.com/", {}) == 1


def test_smoke_script_rejects_non_json_health_response(monkeypatch):
    import scripts.smoke_staging as smoke

    def fake_request(_base_url, path, headers=None, timeout=10):
        if path == "/health":
            return smoke.SmokeResponse(
                status=200,
                body="<html><title>Streamlit</title></html>",
                content_type="text/html",
            )
        return smoke.SmokeResponse(status=200, body='{"status":"alive"}', content_type="application/json")

    monkeypatch.setattr(smoke, "request_json", fake_request)

    assert smoke.run_smoke("https://staging.example.com/", {}) == 1
