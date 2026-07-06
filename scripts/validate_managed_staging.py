"""Managed staging validation without leaking secrets or mutating production.

This script is intentionally conservative. It can collect real external
evidence when credentials are present, but it skips missing or unapproved
external controls instead of pretending they passed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


SECRET_PATTERNS = [
    re.compile(r"(Authorization:\s*Bearer\s+)[^\s]+", re.IGNORECASE),
    re.compile(r"((?:api[_-]?key|token|secret|password|client[_-]?secret)[=:]\s*)[^,\s]+", re.IGNORECASE),
    re.compile(r"((?:Cookie|Set-Cookie):\s*)[^\r\n]+", re.IGNORECASE),
    re.compile(r"(postgres(?:ql)?(?:\+psycopg)?://[^:/@\s]+:)[^@\s]+(@)", re.IGNORECASE),
    re.compile(r"((?:DATABASE|POSTGRES)_URL\s*=\s*)[^\s]+", re.IGNORECASE),
    re.compile(r"((?:AWS|S3|RENDER|OIDC|STAGING)[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)\s*=\s*)[^\s]+", re.IGNORECASE),
]


@dataclass(slots=True)
class ValidationResult:
    name: str
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status in {"PASS", "SKIP"}

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": redact(self.detail)}


def redact(value: object) -> str:
    text = str(value)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda match: f"{match.group(1)}***{match.group(2) if len(match.groups()) > 1 else ''}", text)
    return text


def _env_first(env: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = env.get(name, "").strip()
        if value:
            return value
    return ""


def _bool_env(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def staging_api_url(env: Mapping[str, str]) -> str:
    return _env_first(env, "STAGING_API_URL", "STAGING_API_BASE_URL", "STAGING_BASE_URL")


def staging_api_token(env: Mapping[str, str]) -> str:
    return _env_first(env, "STAGING_API_TOKEN", "STAGING_BEARER_TOKEN")


def postgres_url(env: Mapping[str, str]) -> str:
    return _env_first(env, "POSTGRES_URL", "DATABASE_URL", "SUPPLIER_DATABASE_URL")


def normalize_url(value: str) -> str:
    if not value:
        raise ValueError("URL is required.")
    if not value.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://.")
    return value.rstrip("/") + "/"


def _request_json(base_url: str, path: str, headers: Mapping[str, str] | None = None, timeout: int = 10) -> tuple[int, object, dict[str, str]]:
    request = Request(urljoin(base_url, path.lstrip("/")), headers=dict(headers or {}), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(body), dict(response.headers)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload: object = json.loads(body)
        except json.JSONDecodeError:
            payload = body[:200]
        return exc.code, payload, dict(exc.headers)


def validate_api(env: Mapping[str, str]) -> list[ValidationResult]:
    raw_url = staging_api_url(env)
    if not raw_url:
        return [ValidationResult("staging_api", "SKIP", "STAGING_API_URL/STAGING_API_BASE_URL not configured.")]

    results: list[ValidationResult] = []
    try:
        base_url = normalize_url(raw_url)
        live_status, live_payload, live_headers = _request_json(base_url, "/live")
        results.append(
            ValidationResult(
                "staging_api_live",
                "PASS" if live_status == 200 else "FAIL",
                f"HTTP {live_status}; request_id_present={bool(live_headers.get('X-Request-ID'))}; payload={live_payload}",
            )
        )
        health_status, health_payload, _health_headers = _request_json(base_url, "/health")
        results.append(
            ValidationResult(
                "staging_api_health",
                "PASS" if health_status == 200 else "FAIL",
                f"HTTP {health_status}; payload={health_payload}",
            )
        )
        ready_status, ready_payload, _ready_headers = _request_json(base_url, "/ready")
        results.append(
            ValidationResult(
                "staging_api_ready",
                "PASS" if ready_status == 200 else "FAIL",
                f"HTTP {ready_status}; payload={ready_payload}",
            )
        )
        unauth_status, unauth_payload, _unauth_headers = _request_json(base_url, "/suppliers")
        results.append(
            ValidationResult(
                "staging_api_auth_rejection",
                "PASS" if unauth_status in {401, 403} else "FAIL",
                f"HTTP {unauth_status}; payload={unauth_payload}",
            )
        )

        token = staging_api_token(env)
        if token:
            expected_tenant = _env_first(env, "STAGING_EXPECTED_TENANT_ID")
            headers = {"Authorization": f"Bearer {token}"}
            status_code, payload, response_headers = _request_json(base_url, "/system/status", headers=headers)
            tenant_ok = True
            if expected_tenant and isinstance(payload, dict):
                tenant_ok = payload.get("tenant_id") == expected_tenant
            results.append(
                ValidationResult(
                    "staging_api_authenticated_status",
                    "PASS" if status_code == 200 and tenant_ok else "FAIL",
                    f"HTTP {status_code}; expected_tenant={expected_tenant or 'not-set'}; request_id_present={bool(response_headers.get('X-Request-ID'))}; payload={payload}",
                )
            )
        else:
            results.append(ValidationResult("staging_api_authenticated_status", "SKIP", "STAGING_API_TOKEN/STAGING_BEARER_TOKEN not configured."))
    except (ValueError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        results.append(ValidationResult("staging_api", "FAIL", f"API validation failed: {redact(exc)}"))
    return results


def validate_database(env: Mapping[str, str]) -> list[ValidationResult]:
    url = postgres_url(env)
    if not url:
        return [ValidationResult("postgres_readonly", "SKIP", "POSTGRES_URL/DATABASE_URL/SUPPLIER_DATABASE_URL not configured.")]
    if not _bool_env(env, "STAGING_DB_READONLY_APPROVED"):
        return [ValidationResult("postgres_readonly", "SKIP", "Database URL present but STAGING_DB_READONLY_APPROVED=true is not set.")]
    try:
        import psycopg

        safe_url = url.replace("postgresql+psycopg://", "postgresql://", 1)
        rows: dict[str, object] = {}
        with psycopg.connect(safe_url, connect_timeout=10) as conn:
            conn.read_only = True
            with conn.cursor() as cur:
                cur.execute("select current_database(), current_user")
                db_name, user_name = cur.fetchone()
                rows["database"] = db_name
                rows["user_present"] = bool(user_name)
                cur.execute("select to_regclass('public.alembic_version')")
                has_alembic = cur.fetchone()[0] is not None
                rows["alembic_version_table"] = has_alembic
                if has_alembic:
                    cur.execute("select version_num from alembic_version limit 1")
                    row = cur.fetchone()
                    rows["alembic_version_present"] = bool(row and row[0])
        return [ValidationResult("postgres_readonly", "PASS", json.dumps(rows, sort_keys=True))]
    except Exception as exc:
        return [ValidationResult("postgres_readonly", "FAIL", f"Read-only Postgres validation failed: {redact(exc)}")]


def validate_backup_restore(env: Mapping[str, str]) -> list[ValidationResult]:
    if not _bool_env(env, "STAGING_BACKUP_RESTORE_APPROVED"):
        return [ValidationResult("postgres_backup_restore_drill", "SKIP", "STAGING_BACKUP_RESTORE_APPROVED=true not set.")]
    if not shutil.which("pg_dump") or not shutil.which("pg_restore"):
        return [ValidationResult("postgres_backup_restore_drill", "SKIP", "pg_dump/pg_restore not available.")]
    source = postgres_url(env)
    target = _env_first(env, "POSTGRES_RESTORE_TARGET_URL", "STAGING_RESTORE_DATABASE_URL")
    if not source or not target:
        return [ValidationResult("postgres_backup_restore_drill", "SKIP", "Source and isolated restore target URLs are required.")]
    if not _bool_env(env, "STAGING_RESTORE_TARGET_CONFIRMED_DISPOSABLE"):
        return [ValidationResult("postgres_backup_restore_drill", "SKIP", "STAGING_RESTORE_TARGET_CONFIRMED_DISPOSABLE=true not set.")]
    return [
        ValidationResult(
            "postgres_backup_restore_drill",
            "SKIP",
            "Backup/restore command execution is intentionally manual in this script; run scripts/backup_postgres.ps1 and scripts/restore_postgres.ps1 against the confirmed disposable target and attach redacted logs.",
        )
    ]


def validate_object_storage(env: Mapping[str, str]) -> list[ValidationResult]:
    provider = _env_first(env, "SUPPLIER_UPLOAD_STORAGE_PROVIDER", "STAGING_UPLOAD_STORAGE_PROVIDER")
    supabase_evidence_bucket = _env_first(env, "SUPABASE_EVIDENCE_BUCKET")
    supabase_quarantine_bucket = _env_first(env, "SUPABASE_UPLOAD_QUARANTINE_BUCKET")
    supabase_clean_bucket = _env_first(env, "SUPABASE_UPLOAD_CLEAN_BUCKET")
    supabase_configured = any([supabase_evidence_bucket, supabase_quarantine_bucket, supabase_clean_bucket])
    if provider == "supabase" or supabase_configured:
        missing = [
            name
            for name, value in {
                "SUPABASE_EVIDENCE_BUCKET": supabase_evidence_bucket,
                "SUPABASE_UPLOAD_QUARANTINE_BUCKET": supabase_quarantine_bucket,
                "SUPABASE_UPLOAD_CLEAN_BUCKET": supabase_clean_bucket,
            }.items()
            if not value
        ]
        if missing:
            return [ValidationResult("object_storage_config", "FAIL", f"Missing Supabase storage settings: {', '.join(missing)}.")]
        return [
            ValidationResult(
                "object_storage_config",
                "PASS",
                "Supabase storage bucket configuration is complete; live bucket/object check must be captured separately.",
            )
        ]

    bucket = _env_first(env, "SUPPLIER_UPLOAD_STORAGE_BUCKET", "STAGING_S3_BUCKET")
    endpoint = _env_first(env, "SUPPLIER_UPLOAD_STORAGE_ENDPOINT_URL", "STAGING_S3_ENDPOINT_URL")
    access_key = _env_first(env, "SUPPLIER_UPLOAD_STORAGE_ACCESS_KEY_ID", "STAGING_S3_ACCESS_KEY_ID")
    secret_key = _env_first(env, "SUPPLIER_UPLOAD_STORAGE_SECRET_ACCESS_KEY", "STAGING_S3_SECRET_ACCESS_KEY")
    region = _env_first(env, "SUPPLIER_UPLOAD_STORAGE_REGION", "STAGING_S3_REGION")
    configured = any([provider, bucket, endpoint, access_key, secret_key, region])
    if not configured:
        return [ValidationResult("object_storage_config", "SKIP", "Object storage environment variables are not configured.")]
    missing = [
        name
        for name, value in {
            "provider": provider,
            "bucket": bucket,
            "endpoint": endpoint,
            "access_key": access_key,
            "secret_key": secret_key,
        }.items()
        if not value
    ]
    if provider and provider != "s3":
        return [ValidationResult("object_storage_config", "FAIL", f"Expected s3 or supabase provider for managed staging, got {provider}.")]
    if missing:
        return [ValidationResult("object_storage_config", "FAIL", f"Missing object storage settings: {', '.join(missing)}.")]
    if not _bool_env(env, "STAGING_OBJECT_STORAGE_VALIDATE"):
        return [ValidationResult("object_storage_config", "PASS", "S3-compatible object storage configuration is complete; live bucket check skipped.")]
    try:
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=endpoint or None,
            region_name=region or None,
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
        )
        client.head_bucket(Bucket=bucket)
        return [ValidationResult("object_storage_head_bucket", "PASS", "Bucket is reachable with provided credentials.")]
    except Exception as exc:
        return [ValidationResult("object_storage_head_bucket", "FAIL", f"Bucket validation failed: {redact(exc)}")]


def validate_render(env: Mapping[str, str]) -> list[ValidationResult]:
    token = _env_first(env, "RENDER_API_KEY", "RENDER_TOKEN")
    service_id = _env_first(env, "RENDER_SERVICE_ID", "RENDER_API_SERVICE_ID", "STAGING_RENDER_API_SERVICE_ID")
    if not token or not service_id:
        return [ValidationResult("render_service_evidence", "SKIP", "RENDER_API_KEY/RENDER_SERVICE_ID not configured.")]
    try:
        request = Request(
            f"https://api.render.com/v1/services/{service_id}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            method="GET",
        )
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        service = payload.get("service") if isinstance(payload, dict) else {}
        detail = {
            "status_code": 200,
            "name": service.get("name") if isinstance(service, dict) else None,
            "type": service.get("type") if isinstance(service, dict) else None,
            "service_id_present": True,
        }
        return [ValidationResult("render_service_evidence", "PASS", json.dumps(detail, sort_keys=True))]
    except Exception as exc:
        return [ValidationResult("render_service_evidence", "FAIL", f"Render service evidence failed: {redact(exc)}")]


def validate_malware_scanner(env: Mapping[str, str]) -> list[ValidationResult]:
    required = _env_first(env, "SUPPLIER_UPLOAD_SCANNER_REQUIRED", "STAGING_UPLOAD_SCANNER_REQUIRED")
    provider = _env_first(env, "SUPPLIER_UPLOAD_SCANNER_PROVIDER", "STAGING_UPLOAD_SCANNER_PROVIDER")
    endpoint = _env_first(env, "SUPPLIER_UPLOAD_SCANNER_ENDPOINT_URL", "STAGING_UPLOAD_SCANNER_ENDPOINT_URL")
    if not any([required, provider, endpoint]):
        return [ValidationResult("upload_scanner_config", "SKIP", "Upload scanner environment variables are not configured.")]
    if required.lower() in {"1", "true", "yes", "on"} and (not provider or provider == "none" or not endpoint):
        return [ValidationResult("upload_scanner_config", "FAIL", "Scanner is required but provider/endpoint is incomplete.")]
    if provider in {"eicar-test", "staging-safe", "none"}:
        return [ValidationResult("upload_scanner_config", "SKIP", "Only mock/staging-safe scanner provider is configured; real malware scanning not validated.")]
    if provider and endpoint:
        return [ValidationResult("upload_scanner_config", "PASS", f"Scanner provider and endpoint configured for provider={provider}; live malware test must be captured separately.")]
    return [ValidationResult("upload_scanner_config", "SKIP", "Scanner is not required and no real provider is configured.")]


def run_validation(env: Mapping[str, str]) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    results.extend(validate_api(env))
    results.extend(validate_database(env))
    results.extend(validate_backup_restore(env))
    results.extend(validate_object_storage(env))
    results.extend(validate_malware_scanner(env))
    results.extend(validate_render(env))
    return results


def readiness_label(results: list[ValidationResult]) -> str:
    if any(result.status == "FAIL" for result in results):
        return "Conditional go for managed staging"
    validated_external = {
        result.name
        for result in results
        if result.status == "PASS"
        and result.name
        in {
            "staging_api_ready",
            "staging_api_authenticated_status",
            "postgres_readonly",
            "object_storage_head_bucket",
            "render_service_evidence",
        }
    }
    required = {"staging_api_ready", "staging_api_authenticated_status", "postgres_readonly"}
    if required.issubset(validated_external):
        return "Staging-ready with external controls validated"
    if any(result.status == "PASS" for result in results):
        return "Staging-ready with mocks for external controls"
    return "Conditional go for managed staging"


def main() -> int:
    results = run_validation(os.environ)
    payload = {
        "results": [result.to_dict() for result in results],
        "final_readiness_label": readiness_label(results),
    }
    print(json.dumps(payload, indent=2))
    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
