"""Centralized configuration and safe secret loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _secret_or_env(name: str, default: str = "") -> str:
    try:
        import streamlit as st

        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def _bool_env(name: str, default: bool) -> bool:
    raw = _secret_or_env(name, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


@dataclass(slots=True)
class Settings:
    app_name: str = "Supplier Intelligence Platform"
    security_mode: str = "local"
    deployment_mode: str = "demo"
    database_url: str = "sqlite:///data/production_app.db"
    demo_mode: bool = True
    newsapi_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    anthropic_model: str = "claude-3-5-haiku-latest"
    openai_model: str = "gpt-4o-mini"
    auth_provider: str = "local"
    auth_allow_local_in_production: bool = False
    oidc_issuer_url: str = ""
    oidc_client_id: str = ""
    oidc_client_secret: str = ""
    oidc_audience: str = ""
    oidc_redirect_uri: str = ""
    saml_metadata_url: str = ""
    saml_metadata_file: str = ""
    scim_enabled: bool = False
    mfa_required: bool = False
    cors_allow_origins: str = "*"
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    redis_url: str = ""
    worker_mode: str = "local"
    secrets_provider: str = "env"
    kms_provider: str = "local"
    retention_enabled: bool = False
    retention_days: int = 365
    audit_retention_days: int = 2555
    backup_retention_days: int = 35
    siem_sink: str = "file"
    siem_webhook_url: str = ""
    session_timeout_minutes: int = 60
    alert_exposure_threshold: float = 100_000.0
    scheduler_enabled: bool = True
    sentinel_interval_minutes: int = 60
    risk_interval_minutes: int = 120

    @property
    def is_production(self) -> bool:
        return self.security_mode.lower() == "production"

    @property
    def database_driver(self) -> str:
        return self.database_url.split(":", 1)[0]

    def validate_runtime(self) -> list[str]:
        issues: list[str] = []
        if self.is_production:
            if self.database_driver == "sqlite":
                issues.append("Production mode should use Postgres or another managed database.")
            if self.demo_mode:
                issues.append("Production mode should disable demo mode.")
            if self.auth_provider == "local" and not self.auth_allow_local_in_production:
                issues.append("Production mode should use OIDC/SAML or explicitly allow local auth.")
        return issues


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        security_mode=_secret_or_env("SUPPLIER_SECURITY_MODE", _secret_or_env("SECURITY_MODE", "local")).lower(),
        deployment_mode=_secret_or_env("SUPPLIER_DEPLOYMENT_MODE", "demo").lower(),
        database_url=_normalize_database_url(
            _secret_or_env("SUPPLIER_DATABASE_URL", _secret_or_env("DATABASE_URL", "sqlite:///data/production_app.db"))
        ),
        demo_mode=_bool_env("SUPPLIER_DEMO_MODE", True),
        newsapi_key=_secret_or_env("NEWSAPI_KEY", ""),
        anthropic_api_key=_secret_or_env("ANTHROPIC_API_KEY", ""),
        openai_api_key=_secret_or_env("OPENAI_API_KEY", ""),
        anthropic_model=_secret_or_env("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
        openai_model=_secret_or_env("OPENAI_MODEL", "gpt-4o-mini"),
        auth_provider=_secret_or_env("AUTH_PROVIDER", "local").lower(),
        auth_allow_local_in_production=_bool_env("AUTH_ALLOW_LOCAL_IN_PRODUCTION", False),
        oidc_issuer_url=_secret_or_env("OIDC_ISSUER_URL", ""),
        oidc_client_id=_secret_or_env("OIDC_CLIENT_ID", ""),
        oidc_client_secret=_secret_or_env("OIDC_CLIENT_SECRET", ""),
        oidc_audience=_secret_or_env("OIDC_AUDIENCE", ""),
        oidc_redirect_uri=_secret_or_env("OIDC_REDIRECT_URI", ""),
        saml_metadata_url=_secret_or_env("SAML_METADATA_URL", ""),
        saml_metadata_file=_secret_or_env("SAML_METADATA_FILE", ""),
        scim_enabled=_bool_env("SCIM_ENABLED", False),
        mfa_required=_bool_env("MFA_REQUIRED", False),
        cors_allow_origins=_secret_or_env("CORS_ALLOW_ORIGINS", "*"),
        rate_limit_enabled=_bool_env("RATE_LIMIT_ENABLED", True),
        rate_limit_requests=int(_secret_or_env("RATE_LIMIT_REQUESTS", "120")),
        rate_limit_window_seconds=int(_secret_or_env("RATE_LIMIT_WINDOW_SECONDS", "60")),
        redis_url=_secret_or_env("REDIS_URL", ""),
        worker_mode=_secret_or_env("WORKER_MODE", "local").lower(),
        secrets_provider=_secret_or_env("SECRETS_PROVIDER", "env").lower(),
        kms_provider=_secret_or_env("KMS_PROVIDER", "local").lower(),
        retention_enabled=_bool_env("RETENTION_ENABLED", False),
        retention_days=int(_secret_or_env("RETENTION_DAYS", "365")),
        audit_retention_days=int(_secret_or_env("AUDIT_RETENTION_DAYS", "2555")),
        backup_retention_days=int(_secret_or_env("BACKUP_RETENTION_DAYS", "35")),
        siem_sink=_secret_or_env("SIEM_SINK", "file").lower(),
        siem_webhook_url=_secret_or_env("SIEM_WEBHOOK_URL", ""),
        session_timeout_minutes=int(_secret_or_env("SUPPLIER_APP_SESSION_TIMEOUT_MINUTES", "60")),
        alert_exposure_threshold=float(_secret_or_env("SUPPLIER_ALERT_EXPOSURE_THRESHOLD", "100000")),
        scheduler_enabled=_bool_env("SUPPLIER_SCHEDULER_ENABLED", True),
        sentinel_interval_minutes=int(_secret_or_env("SUPPLIER_SENTINEL_INTERVAL_MINUTES", "60")),
        risk_interval_minutes=int(_secret_or_env("SUPPLIER_RISK_INTERVAL_MINUTES", "120")),
    )
