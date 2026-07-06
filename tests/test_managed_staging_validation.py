from __future__ import annotations

from scripts.validate_managed_staging import (
    readiness_label,
    redact,
    run_validation,
    staging_api_token,
    staging_api_url,
)
from scripts.smoke_staging import auth_headers, staging_base_url


def test_staging_env_aliases_prefer_new_names():
    env = {
        "STAGING_API_URL": "https://api.example.com",
        "STAGING_API_BASE_URL": "https://old.example.com",
        "STAGING_API_TOKEN": "new-token",
        "STAGING_BEARER_TOKEN": "old-token",
    }

    assert staging_api_url(env) == "https://api.example.com"
    assert staging_api_token(env) == "new-token"
    assert staging_base_url(env) == "https://api.example.com"
    assert auth_headers(env) == {"Authorization": "Bearer new-token"}


def test_redaction_removes_secret_values():
    text = (
        "Authorization: Bearer abc.def DATABASE_URL=postgresql://user:pass@host/db "
        "RENDER_API_KEY=render-secret"
    )

    redacted = redact(text)

    assert "abc.def" not in redacted
    assert "pass@host" not in redacted
    assert "render-secret" not in redacted
    assert "***" in redacted


def test_validation_skips_missing_external_configuration():
    results = run_validation({})

    assert results
    assert all(result.status == "SKIP" for result in results)
    assert readiness_label(results) == "Conditional go for managed staging"


def test_incomplete_object_storage_configuration_fails():
    results = run_validation({"SUPPLIER_UPLOAD_STORAGE_PROVIDER": "s3", "SUPPLIER_UPLOAD_STORAGE_BUCKET": "bucket"})

    assert any(result.name == "object_storage_config" and result.status == "FAIL" for result in results)
    assert readiness_label(results) == "Conditional go for managed staging"


def test_complete_supabase_object_storage_configuration_passes():
    results = run_validation(
        {
            "SUPPLIER_UPLOAD_STORAGE_PROVIDER": "supabase",
            "SUPABASE_EVIDENCE_BUCKET": "evidence",
            "SUPABASE_UPLOAD_QUARANTINE_BUCKET": "quarantine",
            "SUPABASE_UPLOAD_CLEAN_BUCKET": "clean",
        }
    )

    assert any(result.name == "object_storage_config" and result.status == "PASS" for result in results)


def test_incomplete_supabase_object_storage_configuration_fails():
    results = run_validation(
        {
            "SUPPLIER_UPLOAD_STORAGE_PROVIDER": "supabase",
            "SUPABASE_EVIDENCE_BUCKET": "evidence",
        }
    )

    assert any(result.name == "object_storage_config" and result.status == "FAIL" for result in results)
