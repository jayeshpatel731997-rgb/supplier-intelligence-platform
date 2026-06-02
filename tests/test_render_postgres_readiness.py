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

        assert "python scripts/migrate.py" in text
        assert "--create-all-fallback" not in text
        assert "healthCheckPath: /live" in text


def test_render_blueprints_keep_api_and_ui_services_separate():
    for blueprint in ("render.yaml", "render.full.yaml"):
        text = (ROOT / blueprint).read_text(encoding="utf-8")

        assert "name: supplier-intelligence-api" in text
        assert "dockerfilePath: ./backend/Dockerfile" in text
        assert "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}" in text
        assert "name: supplier-intelligence-ui" in text
        assert "dockerfilePath: ./Dockerfile" in text
        assert "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0" in text
        assert "healthCheckPath: /_stcore/health" in text


def test_smoke_script_redacts_secret_like_values():
    import scripts.smoke_staging as smoke

    text = smoke.redact("Authorization: Bearer secret-token X-API-Key=secret-key DATABASE_URL=postgres://user:pass@host/db")

    assert "secret-token" not in text
    assert "secret-key" not in text
    assert "user:pass" not in text
    assert "***" in text


def test_smoke_script_builds_auth_headers_without_printing_values(monkeypatch):
    monkeypatch.setenv("STAGING_TENANT_ID", "tenant-a")
    monkeypatch.setenv("STAGING_API_KEY", "api-key-value")
    monkeypatch.delenv("STAGING_BEARER_TOKEN", raising=False)

    import scripts.smoke_staging as smoke

    headers = smoke.auth_headers(os.environ)

    assert headers == {"X-Tenant-ID": "tenant-a", "X-API-Key": "api-key-value"}


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
