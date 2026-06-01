"""Provider-agnostic enterprise authentication scaffolding.

The local provider keeps demo mode usable. OIDC/SAML/SCIM providers expose
runtime validation and claim mapping without requiring a real identity provider
for local tests.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from urllib.request import urlopen
from typing import Any

import jwt
from jwt import InvalidTokenError

from src.config import Settings
from src.tenancy import DEMO_TENANT_ID, TenantContext


@dataclass(slots=True)
class AuthenticatedPrincipal:
    subject: str
    username: str
    display_name: str = ""
    tenant_id: str = DEMO_TENANT_ID
    role: str = "viewer"
    groups: list[str] = field(default_factory=list)

    def to_context(self, request_id: str = "") -> TenantContext:
        return TenantContext(
            tenant_id=self.tenant_id,
            username=self.username,
            role=self.role,
            request_id=request_id,
        )


class AuthProvider:
    name = "base"

    def __init__(self, settings: Settings):
        self.settings = settings

    def validate_runtime(self) -> dict:
        return {"ok": True, "provider": self.name, "issues": []}

    def map_claims(self, claims: dict[str, Any]) -> AuthenticatedPrincipal:
        subject = str(claims.get("sub") or claims.get("subject") or "")
        username = str(claims.get("email") or claims.get("preferred_username") or subject)
        role = _role_from_claims(claims)
        tenant_id = str(claims.get("tenant_id") or claims.get("org_id") or DEMO_TENANT_ID)
        groups = claims.get("groups") or claims.get("roles") or []
        if isinstance(groups, str):
            groups = [groups]
        return AuthenticatedPrincipal(
            subject=subject,
            username=username,
            display_name=str(claims.get("name") or username),
            tenant_id=tenant_id,
            role=role,
            groups=[str(group) for group in groups],
        )


class LocalAuthProvider(AuthProvider):
    name = "local"

    def validate_runtime(self) -> dict:
        issues: list[str] = []
        if self.settings.is_production and not self.settings.auth_allow_local_in_production:
            issues.append("Local auth is disabled in production unless AUTH_ALLOW_LOCAL_IN_PRODUCTION=true.")
        return {"ok": not issues, "provider": self.name, "issues": issues}


class OIDCAuthProvider(AuthProvider):
    name = "oidc"

    def validate_runtime(self) -> dict:
        issues: list[str] = []
        if not self.settings.oidc_issuer_url:
            issues.append("OIDC_ISSUER_URL is required.")
        if not self.settings.oidc_client_id:
            issues.append("OIDC_CLIENT_ID is required.")
        if not self.settings.oidc_effective_audience:
            issues.append("OIDC_AUDIENCE or OIDC_CLIENT_ID is required for API token validation.")
        if not self.settings.oidc_jwks_url:
            issues.append("OIDC_JWKS_URL is required.")
        if not self.settings.oidc_allowed_algorithms:
            issues.append("OIDC_ALGORITHMS must include at least one JWT signing algorithm.")
        return {
            "ok": not issues,
            "provider": self.name,
            "issuer": self.settings.oidc_issuer_url,
            "audience": self.settings.oidc_effective_audience,
            "issues": issues,
            "jwks_url": self.settings.oidc_jwks_url,
            "algorithms": self.settings.oidc_allowed_algorithms,
            "clock_skew_seconds": self.settings.oidc_clock_skew_seconds,
        }

    def verify_token(self, token: str) -> AuthenticatedPrincipal:
        if not token:
            raise ValueError("OIDC bearer token is required.")
        runtime = self.validate_runtime()
        if not runtime["ok"]:
            raise ValueError("OIDC runtime is not configured.")

        try:
            header = jwt.get_unverified_header(token)
            jwk = self._select_jwk(str(header.get("kid") or ""))
            key = _jwk_to_key(jwk)
            claims = jwt.decode(
                token,
                key=key,
                algorithms=self.settings.oidc_allowed_algorithms,
                issuer=self.settings.oidc_issuer_url,
                audience=self.settings.oidc_effective_audience,
                leeway=self.settings.oidc_clock_skew_seconds,
            )
        except (InvalidTokenError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid OIDC bearer token.") from exc
        return self.map_claims(claims)

    def _load_jwks(self) -> dict[str, Any]:
        if self.settings.oidc_jwks_json:
            return json.loads(self.settings.oidc_jwks_json)
        with urlopen(self.settings.oidc_jwks_url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def _select_jwk(self, kid: str) -> dict[str, Any]:
        jwks = self._load_jwks()
        keys = jwks.get("keys") or []
        if not isinstance(keys, list) or not keys:
            raise ValueError("OIDC JWKS does not contain signing keys.")
        if kid:
            for key in keys:
                if str(key.get("kid") or "") == kid:
                    return key
            raise ValueError("OIDC signing key not found.")
        if len(keys) == 1:
            return keys[0]
        raise ValueError("OIDC token missing key id.")


class SAMLAuthProvider(AuthProvider):
    name = "saml"

    def validate_runtime(self) -> dict:
        configured = bool(self.settings.saml_metadata_url or self.settings.saml_metadata_file)
        return {
            "ok": configured,
            "provider": self.name,
            "issues": [] if configured else ["SAML metadata URL or file is required."],
        }


class SCIMProvisioningProvider(AuthProvider):
    name = "scim"

    def validate_runtime(self) -> dict:
        return {
            "ok": self.settings.scim_enabled,
            "provider": self.name,
            "enabled": self.settings.scim_enabled,
            "issues": [] if self.settings.scim_enabled else ["SCIM is disabled."],
        }


def build_auth_provider(settings: Settings) -> AuthProvider:
    provider = settings.auth_provider.lower()
    if provider == "oidc":
        return OIDCAuthProvider(settings)
    if provider == "saml":
        return SAMLAuthProvider(settings)
    return LocalAuthProvider(settings)


def _role_from_claims(claims: dict[str, Any]) -> str:
    allowed = {"platform_admin", "org_admin", "risk_manager", "analyst", "viewer", "auditor"}
    role = claims.get("role")
    if isinstance(role, str) and role in allowed:
        return role
    roles = claims.get("roles") or claims.get("groups") or []
    if isinstance(roles, str):
        roles = [roles]
    for candidate in roles:
        clean = str(candidate).split("/")[-1]
        if clean in allowed:
            return clean
    return "viewer"


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}".encode("ascii"))


def _jwk_to_key(jwk: dict[str, Any]):
    if jwk.get("kty") == "oct":
        return _base64url_decode(str(jwk["k"]))
    return jwt.PyJWK.from_dict(jwk).key
