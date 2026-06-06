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


def _int_env(name: str, default: int) -> int:
    raw = _secret_or_env(name, str(default)).strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    raw = _secret_or_env(name, str(default)).strip()
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


@dataclass(slots=True)
class Settings:
    app_name: str = "Supplier Intelligence Platform"
    security_mode: str = "local"
    deployment_mode: str = "demo"
    data_backend: str = "sqlalchemy"
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
    oidc_jwks_url: str = ""
    oidc_jwks_json: str = ""
    oidc_algorithms: str = "RS256"
    oidc_clock_skew_seconds: int = 60
    oidc_redirect_uri: str = ""
    saml_metadata_url: str = ""
    saml_metadata_file: str = ""
    scim_enabled: bool = False
    mfa_required: bool = False
    cors_allow_origins: str = "*"
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    max_upload_bytes: int = 5_000_000
    allowed_upload_extensions: str = ".csv,.xlsx,.xls,.json"
    allowed_upload_mime_types: str = (
        "text/csv,application/csv,text/plain,application/json,"
        "application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    upload_storage_provider: str = "local"
    upload_storage_path: str = "data/uploads"
    upload_storage_bucket: str = ""
    upload_storage_region: str = ""
    upload_storage_endpoint_url: str = ""
    upload_storage_access_key_id: str = ""
    upload_storage_secret_access_key: str = ""
    upload_storage_key_prefix: str = "uploads"
    upload_scanner_required: bool = False
    upload_scanner_provider: str = "none"
    upload_scanner_endpoint_url: str = ""
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
    connector_mode: str = "demo"
    connector_timeout_seconds: int = 10
    connector_retry_count: int = 1
    news_rss_urls: str = ""
    news_require_supplier_match: bool = True
    filings_company_identifier: str = ""
    filings_source_urls: str = ""
    filings_user_agent: str = "Supplier Intelligence Platform staging@example.invalid"
    hiring_source_urls: str = ""
    public_connector_supplier_id: str = "public-supplier"
    public_connector_supplier_name: str = "Public Source Supplier"
    convex_url: str = ""
    convex_deploy_key: str = ""
    llm_narrative_provider: str = "none"
    streamlit_api_base_url: str = "http://localhost:8000"
    demo_scenario_path: str = "data/demo_supplier_scenarios.json"
    calibration_outcomes_path: str = "data/demo_historical_outcomes.json"

    @property
    def is_production(self) -> bool:
        return self.security_mode.lower() == "production"

    @property
    def is_staging_or_production(self) -> bool:
        mode = self.deployment_mode.lower()
        return self.is_production or "staging" in mode or "production" in mode

    @property
    def database_driver(self) -> str:
        return self.database_url.split(":", 1)[0]

    @property
    def convex_configured(self) -> bool:
        return bool(self.convex_url and self.convex_deploy_key)

    @property
    def active_data_backend(self) -> str:
        return "sqlalchemy"

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_allow_origins.split(",") if item.strip()]

    @property
    def effective_cors_origins(self) -> list[str]:
        origins = self.cors_origins
        if self.is_production and "*" in origins:
            return [origin for origin in origins if origin != "*"]
        return origins or ["*"]

    @property
    def oidc_allowed_algorithms(self) -> list[str]:
        return [item.strip() for item in self.oidc_algorithms.split(",") if item.strip()]

    @property
    def oidc_effective_audience(self) -> str:
        return self.oidc_audience or self.oidc_client_id

    def validate_runtime(self) -> list[str]:
        issues: list[str] = []
        if self.data_backend not in {"sqlalchemy", "auto", "convex"}:
            issues.append("SUPPLIER_DATA_BACKEND must be sqlalchemy, auto, or convex.")
        if self.connector_mode not in {"stub", "demo", "public"}:
            issues.append("SUPPLIER_CONNECTOR_MODE must be stub, demo, or public.")
        if self.connector_timeout_seconds <= 0:
            issues.append("SUPPLIER_CONNECTOR_TIMEOUT_SECONDS must be greater than zero.")
        if self.connector_retry_count < 0:
            issues.append("SUPPLIER_CONNECTOR_RETRY_COUNT must be zero or greater.")
        if self.llm_narrative_provider not in {"none", "openai", "anthropic"}:
            issues.append("SUPPLIER_LLM_NARRATIVE_PROVIDER must be none, openai, or anthropic.")
        if bool(self.convex_url) != bool(self.convex_deploy_key):
            issues.append("CONVEX_URL and CONVEX_DEPLOY_KEY must be configured together.")
        if self.data_backend == "convex":
            issues.append(
                "Convex data backend is not active in this build; use SUPPLIER_DATA_BACKEND=sqlalchemy."
            )
        if self.is_staging_or_production:
            if self.database_driver == "sqlite":
                issues.append("Staging/production mode should use Postgres or another managed database.")
            if self.demo_mode:
                issues.append("Staging/production mode should disable demo mode.")
            if not self.cors_origins or "*" in self.cors_origins:
                issues.append("Staging/production mode should set CORS_ALLOW_ORIGINS to explicit trusted origins.")
            if self.auth_provider == "local" and not self.auth_allow_local_in_production:
                issues.append("Staging/production mode should use OIDC/SAML or explicitly allow local auth.")
            if self.auth_provider == "oidc":
                if not self.oidc_issuer_url:
                    issues.append("OIDC_ISSUER_URL is required in staging/production OIDC mode.")
                if not self.oidc_client_id:
                    issues.append("OIDC_CLIENT_ID is required in staging/production OIDC mode.")
                if not self.oidc_client_secret:
                    issues.append("OIDC_CLIENT_SECRET is required in staging/production OIDC mode.")
                if not self.oidc_effective_audience:
                    issues.append("OIDC_AUDIENCE or OIDC_CLIENT_ID is required in staging/production OIDC mode.")
                if not self.oidc_jwks_url:
                    issues.append("OIDC_JWKS_URL is required in staging/production OIDC mode.")
                if not self.oidc_allowed_algorithms:
                    issues.append("OIDC_ALGORITHMS must include at least one JWT signing algorithm.")
                if self.oidc_clock_skew_seconds < 0:
                    issues.append("OIDC_CLOCK_SKEW_SECONDS must be zero or greater.")
            elif self.auth_provider == "saml" and not (self.saml_metadata_url or self.saml_metadata_file):
                issues.append("SAML metadata URL or file is required in staging/production SAML mode.")
            elif self.auth_provider not in {"local", "oidc", "saml"}:
                issues.append(f"Unsupported AUTH_PROVIDER for staging/production: {self.auth_provider}.")
            if self.upload_storage_provider != "s3":
                issues.append("SUPPLIER_UPLOAD_STORAGE_PROVIDER must be s3 in staging/production mode.")
            if self.upload_storage_provider == "s3":
                if not self.upload_storage_bucket:
                    issues.append("SUPPLIER_UPLOAD_STORAGE_BUCKET is required for staging/production upload storage.")
                if not self.upload_storage_endpoint_url:
                    issues.append("SUPPLIER_UPLOAD_STORAGE_ENDPOINT_URL is required for staging/production upload storage.")
                if not self.upload_storage_access_key_id:
                    issues.append("SUPPLIER_UPLOAD_STORAGE_ACCESS_KEY_ID is required for staging/production upload storage.")
                if not self.upload_storage_secret_access_key:
                    issues.append("SUPPLIER_UPLOAD_STORAGE_SECRET_ACCESS_KEY is required for staging/production upload storage.")
            if self.upload_scanner_required and self.upload_scanner_provider == "none":
                issues.append("SUPPLIER_UPLOAD_SCANNER_PROVIDER is required when upload scanning is required in staging/production.")
            if self.upload_scanner_required and self.upload_scanner_provider != "none" and not self.upload_scanner_endpoint_url:
                issues.append("SUPPLIER_UPLOAD_SCANNER_ENDPOINT_URL is required when upload scanning is required in staging/production.")
        return issues


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        security_mode=_secret_or_env("SUPPLIER_SECURITY_MODE", _secret_or_env("SECURITY_MODE", "local")).lower(),
        deployment_mode=_secret_or_env("SUPPLIER_DEPLOYMENT_MODE", "demo").lower(),
        data_backend=_secret_or_env("SUPPLIER_DATA_BACKEND", "sqlalchemy").lower(),
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
        oidc_jwks_url=_secret_or_env("OIDC_JWKS_URL", ""),
        oidc_jwks_json=_secret_or_env("OIDC_JWKS_JSON", ""),
        oidc_algorithms=_secret_or_env("OIDC_ALGORITHMS", "RS256"),
        oidc_clock_skew_seconds=_int_env("OIDC_CLOCK_SKEW_SECONDS", 60),
        oidc_redirect_uri=_secret_or_env("OIDC_REDIRECT_URI", ""),
        saml_metadata_url=_secret_or_env("SAML_METADATA_URL", ""),
        saml_metadata_file=_secret_or_env("SAML_METADATA_FILE", ""),
        scim_enabled=_bool_env("SCIM_ENABLED", False),
        mfa_required=_bool_env("MFA_REQUIRED", False),
        cors_allow_origins=_secret_or_env("CORS_ALLOW_ORIGINS", "*"),
        rate_limit_enabled=_bool_env("RATE_LIMIT_ENABLED", True),
        rate_limit_requests=_int_env("RATE_LIMIT_REQUESTS", 120),
        rate_limit_window_seconds=_int_env("RATE_LIMIT_WINDOW_SECONDS", 60),
        max_upload_bytes=_int_env("SUPPLIER_MAX_UPLOAD_BYTES", 5_000_000),
        allowed_upload_extensions=_secret_or_env("SUPPLIER_ALLOWED_UPLOAD_EXTENSIONS", ".csv,.xlsx,.xls,.json"),
        allowed_upload_mime_types=_secret_or_env(
            "SUPPLIER_ALLOWED_UPLOAD_MIME_TYPES",
            "text/csv,application/csv,text/plain,application/json,"
            "application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        upload_storage_provider=_secret_or_env("SUPPLIER_UPLOAD_STORAGE_PROVIDER", "local").lower(),
        upload_storage_path=_secret_or_env("SUPPLIER_UPLOAD_STORAGE_PATH", "data/uploads"),
        upload_storage_bucket=_secret_or_env("SUPPLIER_UPLOAD_STORAGE_BUCKET", ""),
        upload_storage_region=_secret_or_env("SUPPLIER_UPLOAD_STORAGE_REGION", ""),
        upload_storage_endpoint_url=_secret_or_env("SUPPLIER_UPLOAD_STORAGE_ENDPOINT_URL", ""),
        upload_storage_access_key_id=_secret_or_env("SUPPLIER_UPLOAD_STORAGE_ACCESS_KEY_ID", ""),
        upload_storage_secret_access_key=_secret_or_env("SUPPLIER_UPLOAD_STORAGE_SECRET_ACCESS_KEY", ""),
        upload_storage_key_prefix=_secret_or_env("SUPPLIER_UPLOAD_STORAGE_KEY_PREFIX", "uploads"),
        upload_scanner_required=_bool_env("SUPPLIER_UPLOAD_SCANNER_REQUIRED", False),
        upload_scanner_provider=_secret_or_env("SUPPLIER_UPLOAD_SCANNER_PROVIDER", "none").lower(),
        upload_scanner_endpoint_url=_secret_or_env("SUPPLIER_UPLOAD_SCANNER_ENDPOINT_URL", ""),
        redis_url=_secret_or_env("REDIS_URL", ""),
        worker_mode=_secret_or_env("WORKER_MODE", "local").lower(),
        secrets_provider=_secret_or_env("SECRETS_PROVIDER", "env").lower(),
        kms_provider=_secret_or_env("KMS_PROVIDER", "local").lower(),
        retention_enabled=_bool_env("RETENTION_ENABLED", False),
        retention_days=_int_env("RETENTION_DAYS", 365),
        audit_retention_days=_int_env("AUDIT_RETENTION_DAYS", 2555),
        backup_retention_days=_int_env("BACKUP_RETENTION_DAYS", 35),
        siem_sink=_secret_or_env("SIEM_SINK", "file").lower(),
        siem_webhook_url=_secret_or_env("SIEM_WEBHOOK_URL", ""),
        session_timeout_minutes=_int_env("SUPPLIER_APP_SESSION_TIMEOUT_MINUTES", 60),
        alert_exposure_threshold=_float_env("SUPPLIER_ALERT_EXPOSURE_THRESHOLD", 100000.0),
        scheduler_enabled=_bool_env("SUPPLIER_SCHEDULER_ENABLED", True),
        sentinel_interval_minutes=_int_env("SUPPLIER_SENTINEL_INTERVAL_MINUTES", 60),
        risk_interval_minutes=_int_env("SUPPLIER_RISK_INTERVAL_MINUTES", 120),
        connector_mode=_secret_or_env("SUPPLIER_CONNECTOR_MODE", "demo").lower(),
        connector_timeout_seconds=_int_env("SUPPLIER_CONNECTOR_TIMEOUT_SECONDS", 10),
        connector_retry_count=_int_env("SUPPLIER_CONNECTOR_RETRY_COUNT", 1),
        news_rss_urls=_secret_or_env("SUPPLIER_NEWS_RSS_URLS", ""),
        news_require_supplier_match=_bool_env("SUPPLIER_NEWS_REQUIRE_SUPPLIER_MATCH", True),
        filings_company_identifier=_secret_or_env("SUPPLIER_FILINGS_COMPANY_IDENTIFIER", ""),
        filings_source_urls=_secret_or_env("SUPPLIER_FILINGS_SOURCE_URLS", ""),
        filings_user_agent=_secret_or_env(
            "SUPPLIER_FILINGS_USER_AGENT",
            "Supplier Intelligence Platform staging@example.invalid",
        ),
        hiring_source_urls=_secret_or_env("SUPPLIER_HIRING_SOURCE_URLS", ""),
        public_connector_supplier_id=_secret_or_env("SUPPLIER_PUBLIC_CONNECTOR_SUPPLIER_ID", "public-supplier"),
        public_connector_supplier_name=_secret_or_env("SUPPLIER_PUBLIC_CONNECTOR_SUPPLIER_NAME", "Public Source Supplier"),
        convex_url=_secret_or_env("CONVEX_URL", _secret_or_env("SUPPLIER_CONVEX_URL", "")),
        convex_deploy_key=_secret_or_env("CONVEX_DEPLOY_KEY", _secret_or_env("SUPPLIER_CONVEX_DEPLOY_KEY", "")),
        llm_narrative_provider=_secret_or_env("SUPPLIER_LLM_NARRATIVE_PROVIDER", "none").lower(),
        streamlit_api_base_url=_secret_or_env("SUPPLIER_API_BASE_URL", "http://localhost:8000"),
        demo_scenario_path=_secret_or_env("SUPPLIER_DEMO_SCENARIO_PATH", "data/demo_supplier_scenarios.json"),
        calibration_outcomes_path=_secret_or_env("SUPPLIER_CALIBRATION_OUTCOMES_PATH", "data/demo_historical_outcomes.json"),
    )
