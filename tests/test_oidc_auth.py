from __future__ import annotations

import base64
import importlib
import json
from datetime import UTC, datetime, timedelta

import jwt
from fastapi.testclient import TestClient


OIDC_ISSUER = "https://issuer.example.com"
OIDC_AUDIENCE = "supplier-api"
OIDC_CLIENT_ID = "supplier-platform"
OIDC_KID = "test-key-1"
OIDC_SECRET = b"test-oidc-signing-secret-32-bytes"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _jwks_json() -> str:
    return json.dumps(
        {
            "keys": [
                {
                    "kty": "oct",
                    "kid": OIDC_KID,
                    "alg": "HS256",
                    "use": "sig",
                    "k": _b64url(OIDC_SECRET),
                }
            ]
        }
    )


def _token(**overrides) -> str:
    now = datetime.now(UTC)
    claims = {
        "iss": OIDC_ISSUER,
        "aud": OIDC_AUDIENCE,
        "sub": "demo-platform-admin",
        "email": "demo-platform-admin",
        "tenant_id": "demo-tenant",
        "roles": ["platform_admin"],
        "iat": now,
        "nbf": now - timedelta(seconds=5),
        "exp": now + timedelta(minutes=5),
    }
    claims.update(overrides)
    return jwt.encode(claims, OIDC_SECRET, algorithm="HS256", headers={"kid": OIDC_KID})


def _oidc_client(monkeypatch, tmp_path, **env_overrides):
    monkeypatch.setenv("SUPPLIER_SECURITY_MODE", "local")
    monkeypatch.setenv("SUPPLIER_DATABASE_URL", f"sqlite:///{tmp_path / 'oidc.db'}")
    monkeypatch.setenv("SUPPLIER_DEMO_MODE", "true")
    monkeypatch.setenv("AUTH_PROVIDER", "oidc")
    monkeypatch.setenv("OIDC_ISSUER_URL", OIDC_ISSUER)
    monkeypatch.setenv("OIDC_CLIENT_ID", OIDC_CLIENT_ID)
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "not-used-for-jwt-tests")
    monkeypatch.setenv("OIDC_AUDIENCE", OIDC_AUDIENCE)
    monkeypatch.setenv("OIDC_JWKS_URL", "https://issuer.example.com/.well-known/jwks.json")
    monkeypatch.setenv("OIDC_JWKS_JSON", _jwks_json())
    monkeypatch.setenv("OIDC_ALGORITHMS", "HS256")
    monkeypatch.setenv("OIDC_CLOCK_SKEW_SECONDS", "30")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    for key, value in env_overrides.items():
        monkeypatch.setenv(key, value)

    import backend.main as backend_main
    import src.config as config

    config.get_settings.cache_clear()
    reloaded = importlib.reload(backend_main)
    return TestClient(reloaded.app)


def test_valid_oidc_token_is_accepted_for_protected_route(monkeypatch, tmp_path):
    client = _oidc_client(monkeypatch, tmp_path)

    response = client.get("/suppliers", headers={"Authorization": f"Bearer {_token()}"})

    assert response.status_code == 200


def test_oidc_missing_or_unknown_tenant_claim_is_denied(monkeypatch, tmp_path):
    client = _oidc_client(monkeypatch, tmp_path)

    missing_tenant = client.get(
        "/suppliers",
        headers={"Authorization": f"Bearer {_token(tenant_id=None)}"},
    )
    unknown_tenant = client.get(
        "/suppliers",
        headers={"Authorization": f"Bearer {_token(tenant_id='tenant-not-found')}"},
    )

    assert missing_tenant.status_code == 403
    assert unknown_tenant.status_code == 403
    assert missing_tenant.json()["detail"] == "Invalid OIDC bearer token."


def test_oidc_ignores_cross_tenant_header_override(monkeypatch, tmp_path):
    client = _oidc_client(monkeypatch, tmp_path)

    response = client.get(
        "/system/status",
        headers={
            "Authorization": f"Bearer {_token()}",
            "X-Tenant-ID": "tenant-not-authorized",
        },
    )

    assert response.status_code == 200
    assert response.json()["tenant_id"] == "demo-tenant"


def test_oidc_database_membership_role_overrides_token_role(monkeypatch, tmp_path):
    client = _oidc_client(monkeypatch, tmp_path)
    assert client.get("/suppliers", headers={"Authorization": f"Bearer {_token()}"}).status_code == 200

    from backend import main
    from src.repositories.tenants import TenantRepository

    with main.runtime.session_factory() as session:
        TenantRepository(session).create_membership(
            "demo-tenant",
            "demo-platform-admin",
            "viewer",
        )
        session.commit()

    response = client.post(
        "/evidence/runs",
        json={},
        headers={"Authorization": f"Bearer {_token(role='platform_admin')}"},
    )

    assert response.status_code == 403
    assert "evidence.run" in response.json()["detail"]


def test_oidc_protected_route_rejects_missing_token(monkeypatch, tmp_path):
    client = _oidc_client(monkeypatch, tmp_path)

    response = client.get("/suppliers")

    assert response.status_code == 401
    assert "Bearer token" in response.json()["detail"]


def test_oidc_protected_route_rejects_invalid_signature(monkeypatch, tmp_path):
    client = _oidc_client(monkeypatch, tmp_path)
    bad_token = jwt.encode(
        {
            "iss": OIDC_ISSUER,
            "aud": OIDC_AUDIENCE,
            "sub": "demo-platform-admin",
            "email": "demo-platform-admin",
            "tenant_id": "demo-tenant",
            "roles": ["platform_admin"],
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        b"wrong-oidc-signing-secret-32byte",
        algorithm="HS256",
        headers={"kid": OIDC_KID},
    )

    response = client.get("/suppliers", headers={"Authorization": f"Bearer {bad_token}"})

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid OIDC bearer token."


def test_oidc_protected_route_rejects_wrong_issuer_and_audience(monkeypatch, tmp_path):
    client = _oidc_client(monkeypatch, tmp_path)

    wrong_issuer = client.get(
        "/suppliers",
        headers={"Authorization": f"Bearer {_token(iss='https://wrong-issuer.example.com')}"},
    )
    wrong_audience = client.get(
        "/suppliers",
        headers={"Authorization": f"Bearer {_token(aud='wrong-audience')}"},
    )

    assert wrong_issuer.status_code == 403
    assert wrong_audience.status_code == 403


def test_incomplete_production_oidc_config_fails_readiness(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLIER_SECURITY_MODE", "production")
    monkeypatch.setenv("SUPPLIER_DATABASE_URL", f"sqlite:///{tmp_path / 'prod.db'}")
    monkeypatch.setenv("SUPPLIER_DEMO_MODE", "false")
    monkeypatch.setenv("AUTH_PROVIDER", "oidc")
    monkeypatch.setenv("OIDC_ISSUER_URL", OIDC_ISSUER)
    monkeypatch.setenv("OIDC_CLIENT_ID", OIDC_CLIENT_ID)
    monkeypatch.setenv("OIDC_AUDIENCE", OIDC_AUDIENCE)
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://staging.example.com")

    import backend.main as backend_main
    import src.config as config

    config.get_settings.cache_clear()
    reloaded = importlib.reload(backend_main)

    with TestClient(reloaded.app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert any("OIDC_JWKS_URL" in issue for issue in response.json()["production_issues"])


def test_production_oidc_runtime_accepts_client_id_as_audience_when_audience_is_empty():
    from src.config import Settings

    settings = Settings(
        security_mode="production",
        database_url="postgresql+psycopg://user:pass@db:5432/app",
        demo_mode=False,
        auth_provider="oidc",
        auth_allow_local_in_production=False,
        oidc_issuer_url=OIDC_ISSUER,
        oidc_client_id=OIDC_CLIENT_ID,
        oidc_client_secret="client-secret",
        oidc_audience="",
        oidc_jwks_url="https://issuer.example.com/.well-known/jwks.json",
        oidc_algorithms="RS256",
        cors_allow_origins="https://staging.example.com",
    )

    assert not [issue for issue in settings.validate_runtime() if "OIDC_AUDIENCE" in issue]
    assert settings.oidc_effective_audience == OIDC_CLIENT_ID


def test_local_demo_api_key_behavior_still_works_when_local_auth_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLIER_SECURITY_MODE", "local")
    monkeypatch.setenv("SUPPLIER_DATABASE_URL", f"sqlite:///{tmp_path / 'local.db'}")
    monkeypatch.setenv("SUPPLIER_DEMO_MODE", "true")
    monkeypatch.setenv("AUTH_PROVIDER", "local")

    import backend.main as backend_main
    import src.config as config

    config.get_settings.cache_clear()
    reloaded = importlib.reload(backend_main)

    with TestClient(reloaded.app) as client:
        response = client.get(
            "/suppliers",
            headers={"X-Tenant-ID": "demo-tenant", "X-API-Key": "demo-api-key"},
        )

    assert response.status_code == 200
