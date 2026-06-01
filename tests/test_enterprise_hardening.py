from __future__ import annotations

import os

from fastapi.testclient import TestClient

from src.config import Settings
from src.config import _normalize_database_url
from src.database import create_session_factory, init_database
from src.models import AuditLog, Supplier
from src.repositories.audit import AuditRepository
from src.repositories.suppliers import SupplierRepository
from src.repositories.tenants import TenantRepository
from src.security.auth_providers import LocalAuthProvider, OIDCAuthProvider
from src.security.secrets import EnvSecretProvider, LocalDevKMSProvider, require_production_secret
from src.services.audit_export import AuditExportService
from src.services.backup_service import BackupService
from src.services.compliance_evidence import EvidenceService
from src.services.migration_service import backfill_demo_tenant, validate_tenant_schema
from src.services.retention_service import RetentionService
from src.services.worker_queue import EnterpriseTaskRunner, get_worker_mode


def _settings(tmp_path, **overrides) -> Settings:
    values = {
        "security_mode": "local",
        "deployment_mode": "test",
        "database_url": f"sqlite:///{tmp_path / 'enterprise.db'}",
        "demo_mode": True,
        "worker_mode": "local",
        "redis_url": "",
        "retention_enabled": False,
    }
    values.update(overrides)
    return Settings(**values)


def _session_factory(tmp_path, **overrides):
    settings = _settings(tmp_path, **overrides)
    factory = create_session_factory(settings)
    init_database(factory)
    return settings, factory


def test_tenant_schema_validation_and_backfill_are_sqlite_safe(tmp_path):
    _, factory = _session_factory(tmp_path)

    with factory() as session:
        session.add(Supplier(supplier_id="LEGACY-1", tenant_id="", name="Legacy Supplier"))
        session.add(AuditLog(tenant_id="", username="legacy", action="legacy.action"))
        session.commit()

        result = backfill_demo_tenant(session, production_mode=False)
        session.commit()
        validation = validate_tenant_schema(session)

        assert result["backfilled_rows"] >= 2
        assert validation["ok"] is True
        assert validation["tables_missing_tenant_id"] == []
        assert SupplierRepository(session, "demo-tenant").get("LEGACY-1").name == "Legacy Supplier"


def test_enterprise_task_runner_records_success_and_failure_with_alert_and_audit(tmp_path):
    settings, factory = _session_factory(tmp_path)
    runner = EnterpriseTaskRunner(settings, factory)

    success = runner.run_task("risk_recalculation_task", tenant_id="demo-tenant", correlation_id="corr-1")
    failure = runner.run_task("missing_task", tenant_id="demo-tenant", correlation_id="corr-2")

    with factory() as session:
        audits = [row.action for row in AuditRepository(session, "demo-tenant").list()]

    assert success.status == "completed"
    assert failure.status == "failed"
    assert "Unknown task" in failure.error
    assert "worker.task_failed" in audits


def test_auth_provider_scaffold_maps_oidc_claims_and_blocks_unsafe_local_production():
    local = LocalAuthProvider(Settings(security_mode="production", demo_mode=False, auth_allow_local_in_production=False))
    assert local.validate_runtime()["ok"] is False

    oidc = OIDCAuthProvider(
        Settings(
            auth_provider="oidc",
            oidc_issuer_url="https://issuer.example.com",
            oidc_client_id="client-id",
            oidc_audience="supplier-api",
            oidc_jwks_url="https://issuer.example.com/.well-known/jwks.json",
        )
    )
    mapped = oidc.map_claims(
        {
            "sub": "user-123",
            "email": "buyer@example.com",
            "name": "Buyer User",
            "tenant_id": "tenant-a",
            "roles": ["risk_manager"],
        }
    )

    assert oidc.validate_runtime()["ok"] is True
    assert mapped.username == "buyer@example.com"
    assert mapped.tenant_id == "tenant-a"
    assert mapped.role == "risk_manager"


def test_rate_limit_request_id_and_safe_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLIER_SECURITY_MODE", "local")
    monkeypatch.setenv("SUPPLIER_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("SUPPLIER_DEMO_MODE", "true")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")

    import importlib
    import backend.main as backend_main
    import src.config as config

    config.get_settings.cache_clear()
    importlib.reload(backend_main)
    client = TestClient(backend_main.app)
    headers = {"X-Tenant-ID": "demo-tenant", "X-API-Key": "demo-api-key", "X-Request-ID": "rid-1"}

    assert client.get("/suppliers", headers=headers).headers["X-Request-ID"] == "rid-1"
    assert client.get("/suppliers", headers=headers).status_code == 200
    limited = client.get("/suppliers", headers=headers)

    assert limited.status_code == 429
    assert "traceback" not in limited.text.lower()


def test_secrets_and_kms_abstractions_do_not_leak_values(monkeypatch):
    monkeypatch.setenv("PRIVATE_API_TOKEN", "super-secret-value")

    provider = EnvSecretProvider()
    kms = LocalDevKMSProvider(master_key="dev-key")
    encrypted = kms.encrypt("payload")

    assert provider.get_secret("PRIVATE_API_TOKEN").value == "super-secret-value"
    assert str(provider.get_secret("PRIVATE_API_TOKEN")) == "***"
    assert kms.decrypt(encrypted) == "payload"
    assert require_production_secret("MISSING_SECRET", provider, production=True).ok is False
    assert "super-secret-value" not in repr(provider.get_secret("PRIVATE_API_TOKEN"))


def test_backup_retention_audit_and_evidence_exports_are_tenant_scoped(tmp_path):
    settings, factory = _session_factory(tmp_path)

    with factory() as session:
        tenants = TenantRepository(session)
        tenants.create_tenant("tenant-a", "Tenant A")
        tenants.create_tenant("tenant-b", "Tenant B")
        AuditRepository(session, "tenant-a").log("supplier.updated", username="alice")
        AuditRepository(session, "tenant-b").log("supplier.deleted", username="bob")
        backup = BackupService(session, "tenant-a").record_metadata("dry_run", "local://backup.sqlite")
        retention = RetentionService(session, "tenant-a", settings).cleanup(dry_run=True)
        session.commit()

        jsonl = AuditExportService(session, "tenant-a").export_jsonl()
        evidence = EvidenceService(session, "tenant-a").collect_access_control_evidence()

        assert backup.status == "dry_run"
        assert retention["enabled"] is False
        assert "supplier.updated" in jsonl
        assert "supplier.deleted" not in jsonl
        assert evidence["tenant_id"] == "tenant-a"
        assert "memberships" in evidence


def test_worker_mode_reports_celery_ready_without_requiring_redis(monkeypatch):
    monkeypatch.setenv("WORKER_MODE", "celery")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    mode = get_worker_mode(Settings(worker_mode="celery", redis_url=os.getenv("REDIS_URL", "")))

    assert mode["requested"] == "celery"
    assert "available" in mode


def test_render_postgres_url_uses_installed_psycopg_driver():
    assert _normalize_database_url("postgresql://user:pass@host:5432/db") == "postgresql+psycopg://user:pass@host:5432/db"
    assert _normalize_database_url("postgresql+psycopg://user:pass@host:5432/db") == "postgresql+psycopg://user:pass@host:5432/db"
