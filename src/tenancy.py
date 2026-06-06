"""Tenant context and tenant-scoped RBAC helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEMO_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "demo-tenant")
DEMO_API_KEY = os.getenv("SUPPLIER_DEMO_API_KEY", os.getenv("DEFAULT_TENANT_API_KEY", "demo-api-key"))
DEMO_PLATFORM_ADMIN = os.getenv("DEFAULT_PLATFORM_ADMIN", "demo-platform-admin")


ROLE_PERMISSIONS: dict[str, set[str]] = {
    "platform_admin": {"*"},
    "org_admin": {
        "tenant.read",
        "tenant.manage_users",
        "tenant.manage_api_keys",
        "supplier.read",
        "supplier.write",
        "risk.run",
        "evidence.read",
        "evidence.run",
        "evidence.write",
        "scenario.run",
        "sentinel.run",
        "alerts.read",
        "alerts.acknowledge",
        "ingestion.upload",
        "audit.read",
        "system.read",
        "jobs.run",
    },
    "risk_manager": {
        "tenant.read",
        "supplier.read",
        "risk.run",
        "evidence.read",
        "evidence.run",
        "evidence.write",
        "scenario.run",
        "sentinel.run",
        "alerts.read",
        "alerts.acknowledge",
        "system.read",
        "jobs.run",
    },
    "analyst": {
        "tenant.read",
        "supplier.read",
        "supplier.write",
        "risk.run",
        "evidence.read",
        "evidence.run",
        "evidence.write",
        "scenario.run",
        "sentinel.run",
        "alerts.read",
        "ingestion.upload",
        "system.read",
    },
    "viewer": {"tenant.read", "supplier.read", "risk.read", "evidence.read", "alerts.read", "system.read"},
    "auditor": {"tenant.read", "supplier.read", "risk.read", "evidence.read", "alerts.read", "audit.read", "system.read"},
}

ROLE_ORDER = {
    "viewer": 1,
    "auditor": 1,
    "analyst": 2,
    "risk_manager": 3,
    "org_admin": 4,
    "platform_admin": 5,
}


@dataclass(slots=True)
class TenantContext:
    tenant_id: str
    username: str
    role: str
    request_id: str = ""

    @property
    def is_platform_admin(self) -> bool:
        return self.role == "platform_admin"


def require_permission(context: TenantContext, permission: str) -> bool:
    permissions = ROLE_PERMISSIONS.get(context.role, set())
    return "*" in permissions or permission in permissions


def role_at_least(role: str, minimum: str) -> bool:
    return ROLE_ORDER.get(role, 0) >= ROLE_ORDER.get(minimum, 0)


def can_grant_role(granter: TenantContext, target_role: str) -> bool:
    if target_role not in ROLE_ORDER:
        return False
    if granter.is_platform_admin:
        return True
    return ROLE_ORDER.get(target_role, 0) <= ROLE_ORDER.get(granter.role, 0)
