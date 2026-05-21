"""Provider-agnostic enterprise authentication scaffolding.

The local provider keeps demo mode usable. OIDC/SAML/SCIM providers expose
runtime validation and claim mapping without requiring a real identity provider
for local tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
        if not self.settings.oidc_audience:
            issues.append("OIDC_AUDIENCE is recommended for API token validation.")
        return {
            "ok": not issues,
            "provider": self.name,
            "issuer": self.settings.oidc_issuer_url,
            "audience": self.settings.oidc_audience,
            "issues": issues,
            "jwks_url": f"{self.settings.oidc_issuer_url.rstrip('/')}/.well-known/jwks.json"
            if self.settings.oidc_issuer_url
            else "",
        }

    def verify_token(self, token: str) -> AuthenticatedPrincipal:
        if not token:
            raise ValueError("OIDC bearer token is required.")
        raise NotImplementedError("Configure PyJWT/JWKS verification for the selected identity provider.")


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
