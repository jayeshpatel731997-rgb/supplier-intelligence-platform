from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import Settings, get_settings
from src.database import create_session_factory, database_health, init_database
from src.repositories.alerts import AlertRepository
from src.repositories.audit import AuditRepository
from src.repositories.jobs import JobRepository
from src.repositories.suppliers import SupplierRepository
from src.repositories.tenants import TenantRepository
from src.security.auth import AuthService, PasswordPolicyError, validate_password_strength
from src.services.decision_service import build_decision_brief
from src.services.ingestion_service import IngestionService
from src.services.scheduler import LocalJobScheduler
from src.services.sentinel_service import SentinelService
from src.tenancy import DEMO_TENANT_ID, TenantContext, require_permission


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        security_mode="local",
        database_url=f"sqlite:///{tmp_path / 'production.db'}",
        demo_mode=True,
        newsapi_key="",
        anthropic_api_key="",
        openai_api_key="",
    )


def _session_factory(tmp_path: Path):
    settings = _settings(tmp_path)
    factory = create_session_factory(settings)
    init_database(factory)
    return settings, factory


def test_sqlite_database_fallback_stores_suppliers_and_health(tmp_path):
    settings, session_factory = _session_factory(tmp_path)

    with session_factory() as session:
        repo = SupplierRepository(session)
        supplier = repo.upsert_supplier(
            supplier_id="SUP-1",
            name="Apex Components",
            country="Mexico",
            category="Machining",
            annual_spend=125000.0,
        )
        session.commit()

        assert supplier.name == "Apex Components"
        assert repo.get("SUP-1").country == "Mexico"
        assert len(repo.list()) == 1

    health = database_health(settings)
    assert health["ok"] is True
    assert health["driver"] == "sqlite"


def test_ingestion_handles_messy_procurement_columns_and_creates_job(tmp_path):
    _, session_factory = _session_factory(tmp_path)
    csv_bytes = (
        b"Vendor Name,Country,Product Category,Annual Spend ($),OTD %,Defects,Unit Price\n"
        b" Apex Components ,Mexico,Machining,\"$125,000\",97%,1.2%,12.50\n"
    )

    with session_factory() as session:
        result = IngestionService(session).process_upload(csv_bytes, "supplier_kpis.csv")
        session.commit()

        suppliers = SupplierRepository(session).list()
        assert result.success is True
        assert result.job_status == "completed"
        assert result.row_count == 1
        assert suppliers[0].name == "Apex Components"
        assert suppliers[0].annual_spend == 125000.0


def test_sentinel_missing_api_key_and_provider_failure_do_not_crash(tmp_path):
    settings, session_factory = _session_factory(tmp_path)
    suppliers = pd.DataFrame(
        [{"supplier_name": "Apex Components", "country": "Mexico", "category": "Machining", "annual_spend": 125000.0}]
    )

    with session_factory() as session:
        service = SentinelService(session, settings)
        missing_key_result = service.scan(suppliers, mode="live_ai")
        failure_result = service.scan(suppliers, mode="live_ai", provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
        demo_result = service.scan(suppliers, mode="demo")
        session.commit()

        assert missing_key_result.error
        assert missing_key_result.events == []
        assert "boom" in failure_result.error
        assert len(demo_result.events) >= 1
        assert AlertRepository(session).count_open() >= 1


def test_decision_brief_falls_back_without_llm_key():
    brief = build_decision_brief(
        supplier_name="Apex Components",
        risk_score=0.82,
        risk_drivers=["geopolitical", "concentration"],
        financial_exposure=250000.0,
        confidence="high",
        llm_api_key="",
    )

    assert brief.final_decision in {"monitor", "renegotiate", "dual-source", "escalate", "replace supplier"}
    assert brief.recommended_action
    assert brief.alternatives


def test_password_strength_lockout_rbac_and_alert_acknowledgement(tmp_path):
    _, session_factory = _session_factory(tmp_path)

    with pytest.raises(PasswordPolicyError):
        validate_password_strength("password")

    with session_factory() as session:
        auth = AuthService(session, max_failed_attempts=3, lockout_minutes=15)
        auth.create_user("admin", "StrongerPass123!", "admin")
        assert auth.authenticate("admin", "wrong") is None
        assert auth.authenticate("admin", "wrong") is None
        assert auth.authenticate("admin", "wrong") is None
        assert auth.authenticate("admin", "StrongerPass123!") is None
        assert auth.user_has_role("admin", "admin") is True

        alerts = AlertRepository(session)
        alert = alerts.create_alert(
            alert_type="supplier_high_risk",
            severity="high",
            message="Supplier risk exceeded threshold.",
            supplier_id="SUP-1",
            exposure=100000.0,
        )
        alerts.acknowledge(alert.id, actor="admin")
        session.commit()

        assert alerts.get(alert.id).status == "acknowledged"


def test_background_jobs_capture_success_and_failure(tmp_path):
    settings, session_factory = _session_factory(tmp_path)
    scheduler = LocalJobScheduler(settings, session_factory)

    success = scheduler.run_job_now("risk_recalculate")
    failure = scheduler.run_job_now("missing_job")

    assert success.status == "completed"
    assert failure.status == "failed"
    assert "Unknown job" in (failure.error or "")


def test_tenant_scoped_repositories_prevent_cross_tenant_leakage(tmp_path):
    _, session_factory = _session_factory(tmp_path)

    with session_factory() as session:
        tenants = TenantRepository(session)
        tenants.create_tenant("tenant-a", "Tenant A")
        tenants.create_tenant("tenant-b", "Tenant B")

        SupplierRepository(session, tenant_id="tenant-a").upsert_supplier(
            supplier_id="SUP-SHARED",
            name="Apex A",
            annual_spend=1000,
        )
        SupplierRepository(session, tenant_id="tenant-b").upsert_supplier(
            supplier_id="SUP-SHARED",
            name="Apex B",
            annual_spend=2000,
        )

        alerts_a = AlertRepository(session, tenant_id="tenant-a")
        alerts_b = AlertRepository(session, tenant_id="tenant-b")
        alert_a = alerts_a.create_alert("supplier_high_risk", "high", "A alert", supplier_id="SUP-SHARED")
        alert_b = alerts_b.create_alert("supplier_high_risk", "high", "B alert", supplier_id="SUP-SHARED")

        AuditRepository(session, tenant_id="tenant-a").log("tenant_a.action", username="alice")
        AuditRepository(session, tenant_id="tenant-b").log("tenant_b.action", username="bob")
        JobRepository(session, tenant_id="tenant-a").start("run-a", "risk_recalculate")
        JobRepository(session, tenant_id="tenant-b").start("run-b", "risk_recalculate")
        session.commit()

        assert SupplierRepository(session, tenant_id="tenant-a").get("SUP-SHARED").name == "Apex A"
        assert SupplierRepository(session, tenant_id="tenant-b").get("SUP-SHARED").name == "Apex B"
        assert [alert.message for alert in alerts_a.list()] == ["A alert"]
        assert [alert.message for alert in alerts_b.list()] == ["B alert"]
        assert alerts_a.get(alert_b.id) is None
        assert alerts_b.get(alert_a.id) is None
        assert [row.action for row in AuditRepository(session, tenant_id="tenant-a").list()] == ["tenant_a.action"]
        assert [row.run_id for row in JobRepository(session, tenant_id="tenant-b").list()] == ["run-b"]


def test_tenant_rbac_permissions_are_role_scoped(tmp_path):
    _, session_factory = _session_factory(tmp_path)

    with session_factory() as session:
        tenants = TenantRepository(session)
        tenants.create_tenant("tenant-a", "Tenant A")
        tenants.create_membership("tenant-a", username="owner@example.com", role="org_admin")
        tenants.create_membership("tenant-a", username="analyst@example.com", role="analyst")
        tenants.create_membership("tenant-a", username="viewer@example.com", role="viewer")
        tenants.create_membership("tenant-a", username="auditor@example.com", role="auditor")
        session.commit()

        assert require_permission(TenantContext("tenant-a", "owner@example.com", "org_admin"), "tenant.manage_users")
        assert require_permission(TenantContext("tenant-a", "analyst@example.com", "analyst"), "ingestion.upload")
        assert not require_permission(TenantContext("tenant-a", "viewer@example.com", "viewer"), "ingestion.upload")
        assert require_permission(TenantContext("tenant-a", "auditor@example.com", "auditor"), "audit.read")
        assert not require_permission(TenantContext("tenant-a", "auditor@example.com", "auditor"), "alerts.acknowledge")


def test_demo_tenant_seed_creates_membership_and_api_key(tmp_path):
    _, session_factory = _session_factory(tmp_path)

    with session_factory() as session:
        tenants = TenantRepository(session)
        seed = tenants.ensure_demo_seed()
        session.commit()

        assert seed.tenant_id == DEMO_TENANT_ID
        assert seed.api_key == "demo-api-key"
        assert tenants.validate_api_key(DEMO_TENANT_ID, "demo-api-key").role == "platform_admin"
        assert tenants.validate_api_key(DEMO_TENANT_ID, "wrong") is None


def test_invalid_numeric_environment_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "not-an-int")
    monkeypatch.setenv("SUPPLIER_MAX_UPLOAD_BYTES", "also-bad")
    get_settings.cache_clear()

    try:
        settings = get_settings()
    finally:
        get_settings.cache_clear()

    assert settings.rate_limit_requests == 120
    assert settings.max_upload_bytes == 5_000_000


def test_production_runtime_rejects_wildcard_cors_and_incomplete_oidc():
    settings = Settings(
        security_mode="production",
        database_url="postgresql+psycopg://user:pass@db:5432/app",
        demo_mode=False,
        auth_provider="oidc",
        oidc_issuer_url="https://issuer.example.com",
        oidc_client_id="supplier-platform",
        oidc_audience="supplier-api",
        cors_allow_origins="*",
    )

    issues = settings.validate_runtime()

    assert any("CORS" in issue for issue in issues)
    assert any("OIDC_CLIENT_SECRET" in issue for issue in issues)


def test_upload_storage_defaults_to_local_demo_path(tmp_path):
    settings = Settings(upload_storage_path=str(tmp_path / "uploads"))

    assert settings.upload_storage_provider == "local"
    assert settings.upload_storage_path.endswith("uploads")
    assert settings.allowed_upload_mime_types


def test_production_runtime_requires_complete_object_storage_config():
    settings = Settings(
        security_mode="production",
        database_url="postgresql+psycopg://user:pass@db:5432/app",
        demo_mode=False,
        auth_provider="local",
        auth_allow_local_in_production=True,
        cors_allow_origins="https://staging.example.com",
        upload_storage_provider="s3",
        upload_storage_bucket="",
        upload_storage_endpoint_url="",
    )

    issues = settings.validate_runtime()

    assert any("SUPPLIER_UPLOAD_STORAGE_BUCKET" in issue for issue in issues)
    assert any("SUPPLIER_UPLOAD_STORAGE_ENDPOINT_URL" in issue for issue in issues)


def test_production_runtime_fails_when_upload_scanner_is_required_but_missing():
    settings = Settings(
        security_mode="production",
        database_url="postgresql+psycopg://user:pass@db:5432/app",
        demo_mode=False,
        auth_provider="local",
        auth_allow_local_in_production=True,
        cors_allow_origins="https://staging.example.com",
        upload_storage_provider="s3",
        upload_storage_bucket="supplier-uploads",
        upload_storage_endpoint_url="https://object-storage.example.com",
        upload_storage_access_key_id="access-key",
        upload_storage_secret_access_key="secret-key",
        upload_scanner_required=True,
        upload_scanner_provider="none",
    )

    issues = settings.validate_runtime()

    assert any("SUPPLIER_UPLOAD_SCANNER_PROVIDER" in issue for issue in issues)


def test_system_status_reports_degraded_when_database_queries_fail(monkeypatch):
    from src.services import system_service

    class BrokenSessionFactory:
        def __call__(self):
            raise RuntimeError("database unavailable")

    settings = Settings(security_mode="production", database_url="postgresql+psycopg://user:pass@db:5432/app")
    monkeypatch.setattr(
        system_service,
        "database_health",
        lambda _settings: {
            "ok": False,
            "driver": "postgresql+psycopg",
            "url": "postgresql+psycopg://***:***@db:5432/app",
            "error": "database unavailable",
        },
    )

    status = system_service.system_status(settings, BrokenSessionFactory())

    assert status["database"]["ok"] is False
    assert status["api"]["status"] == "degraded"
    assert status["monitored_suppliers"] == 0
    assert status["open_alerts"] == 0
    assert status["worker"]["status"] == "unknown"
    assert "database unavailable" in status["status_error"]


def test_system_status_redacts_credentials_from_startup_errors(monkeypatch):
    from src.services import system_service

    settings = Settings(
        security_mode="production",
        database_url="postgresql+psycopg://user:super-secret-password@db:5432/app",
        demo_mode=False,
        auth_provider="local",
        auth_allow_local_in_production=True,
        cors_allow_origins="https://staging.example.com",
    )
    monkeypatch.setattr(
        system_service,
        "database_health",
        lambda _settings: {
            "ok": True,
            "driver": "postgresql+psycopg",
            "url": "postgresql+psycopg://***:***@db:5432/app",
        },
    )

    status = system_service.system_status(
        settings,
        None,
        startup_error="could not connect to postgresql+psycopg://user:super-secret-password@db:5432/app",
    )

    assert "super-secret-password" not in status["status_error"]
    assert "***:***@db:5432/app" in status["status_error"]


def test_tenant_schema_validation_fails_when_business_tables_are_missing(tmp_path):
    from src.services.migration_service import validate_tenant_schema

    settings = Settings(database_url=f"sqlite:///{tmp_path / 'empty.db'}")
    factory = create_session_factory(settings)

    with factory() as session:
        result = validate_tenant_schema(session)

    assert result["ok"] is False
    assert "suppliers" in result["missing_tables"]
