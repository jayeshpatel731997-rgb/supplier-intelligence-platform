"""Tenant, membership, and API key repository."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from src.models import AccessReview, BackupRun, Membership, Organization, RetentionPolicy, Tenant, TenantApiKey
from src.security.auth import hash_password, verify_password
from src.tenancy import DEMO_API_KEY, DEMO_PLATFORM_ADMIN, DEMO_TENANT_ID, TenantContext


@dataclass(slots=True)
class TenantSeedResult:
    tenant_id: str
    api_key: str


class TenantRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_tenant(self, tenant_id: str, name: str, status: str = "active") -> Tenant:
        tenant = self.get_tenant(tenant_id)
        if tenant is None:
            tenant = Tenant(tenant_id=tenant_id, name=name, status=status)
            self.session.add(tenant)
            self.session.flush()
        else:
            tenant.name = name
            tenant.status = status
        return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        return self.session.scalar(select(Tenant).where(Tenant.tenant_id == tenant_id))

    def list_tenants(self, active_only: bool = True) -> list[Tenant]:
        stmt = select(Tenant).order_by(Tenant.name)
        if active_only:
            stmt = stmt.where(Tenant.status == "active")
        return list(self.session.scalars(stmt))

    def create_organization(self, tenant_id: str, name: str, domain: str = "") -> Organization:
        org = Organization(tenant_id=tenant_id, name=name, domain=domain)
        self.session.add(org)
        self.session.flush()
        return org

    def create_membership(self, tenant_id: str, username: str, role: str, is_active: bool = True) -> Membership:
        existing = self.session.scalar(
            select(Membership).where(Membership.tenant_id == tenant_id, Membership.username == username)
        )
        if existing is None:
            existing = Membership(tenant_id=tenant_id, username=username, role=role, is_active=is_active)
            self.session.add(existing)
        else:
            existing.role = role
            existing.is_active = is_active
        self.session.flush()
        return existing

    def list_memberships(self, tenant_id: str) -> list[Membership]:
        return list(self.session.scalars(select(Membership).where(Membership.tenant_id == tenant_id).order_by(Membership.username)))

    def get_membership(self, tenant_id: str, username: str) -> Membership | None:
        return self.session.scalar(
            select(Membership).where(
                Membership.tenant_id == tenant_id,
                Membership.username == username,
                Membership.is_active.is_(True),
            )
        )

    def create_api_key(self, tenant_id: str, username: str, role: str, label: str = "", raw_key: str | None = None) -> str:
        key = raw_key or f"sip_{secrets.token_urlsafe(24)}"
        row = TenantApiKey(
            tenant_id=tenant_id,
            username=username,
            role=role,
            label=label,
            key_hash=hash_password(key),
            prefix=key[:12],
            is_active=True,
        )
        self.session.add(row)
        self.create_membership(tenant_id, username, role)
        self.session.flush()
        return key

    def validate_api_key(self, tenant_id: str, api_key: str) -> TenantContext | None:
        if not tenant_id or not api_key:
            return None
        if self.get_tenant(tenant_id) is None:
            return None
        rows = self.session.scalars(
            select(TenantApiKey).where(TenantApiKey.tenant_id == tenant_id, TenantApiKey.is_active.is_(True))
        )
        for row in rows:
            if verify_password(api_key, row.key_hash):
                membership = self.get_membership(tenant_id, row.username)
                if membership is None:
                    return None
                return TenantContext(tenant_id=tenant_id, username=row.username, role=membership.role)
        return None

    def create_access_review(self, tenant_id: str, reviewer: str, notes: str = "") -> AccessReview:
        row = AccessReview(tenant_id=tenant_id, reviewer=reviewer, notes=notes, status="open")
        self.session.add(row)
        self.session.flush()
        return row

    def list_access_reviews(self, tenant_id: str) -> list[AccessReview]:
        return list(self.session.scalars(select(AccessReview).where(AccessReview.tenant_id == tenant_id).order_by(AccessReview.created_at.desc())))

    def create_retention_policy(self, tenant_id: str, data_type: str, retention_days: int) -> RetentionPolicy:
        row = RetentionPolicy(tenant_id=tenant_id, data_type=data_type, retention_days=retention_days)
        self.session.add(row)
        self.session.flush()
        return row

    def create_backup_run(self, tenant_id: str, status: str = "pending", location: str = "", error: str = "") -> BackupRun:
        row = BackupRun(tenant_id=tenant_id, status=status, location=location, error=error)
        self.session.add(row)
        self.session.flush()
        return row

    def ensure_demo_oidc_membership(self, username: str, role: str = "risk_manager") -> None:
        seed_username = username.strip()
        if not seed_username:
            raise ValueError("OIDC staging seed username is required.")
        self._ensure_demo_foundation()
        self.session.execute(
            update(TenantApiKey)
            .where(TenantApiKey.tenant_id == DEMO_TENANT_ID)
            .values(is_active=False)
        )
        self.create_membership(DEMO_TENANT_ID, seed_username, role)

    def ensure_demo_seed(
        self,
        *,
        raw_key: str | None = None,
        username: str | None = None,
        role: str = "platform_admin",
        revoke_usernames: tuple[str, ...] = (),
    ) -> TenantSeedResult:
        seed_key = raw_key or DEMO_API_KEY
        seed_username = username or DEMO_PLATFORM_ADMIN
        self._ensure_demo_foundation()
        if revoke_usernames:
            self.session.execute(
                update(TenantApiKey)
                .where(
                    TenantApiKey.tenant_id == DEMO_TENANT_ID,
                    TenantApiKey.username.in_(revoke_usernames),
                )
                .values(is_active=False)
            )
        if self.validate_api_key(DEMO_TENANT_ID, seed_key) is None:
            self.session.execute(
                update(TenantApiKey)
                .where(
                    TenantApiKey.tenant_id == DEMO_TENANT_ID,
                    TenantApiKey.username == seed_username,
                )
                .values(is_active=False)
            )
            self.create_api_key(
                DEMO_TENANT_ID,
                username=seed_username,
                role=role,
                label="Demo seed API key",
                raw_key=seed_key,
            )
        return TenantSeedResult(tenant_id=DEMO_TENANT_ID, api_key=seed_key)

    def _ensure_demo_foundation(self) -> None:
        self.create_tenant(DEMO_TENANT_ID, "Demo Tenant")
        if not self.list_memberships(DEMO_TENANT_ID):
            self.create_organization(DEMO_TENANT_ID, "Demo Organization", "demo.local")
            self.create_retention_policy(DEMO_TENANT_ID, "audit_logs", 2555)
            self.create_retention_policy(DEMO_TENANT_ID, "news_events", 90)
            self.create_backup_run(DEMO_TENANT_ID, status="configured", location="local-demo")

    @staticmethod
    def tenant_to_dict(row: Tenant) -> dict:
        return {
            "tenant_id": row.tenant_id,
            "name": row.name,
            "status": row.status,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    @staticmethod
    def membership_to_dict(row: Membership) -> dict:
        return {
            "tenant_id": row.tenant_id,
            "username": row.username,
            "role": row.role,
            "is_active": row.is_active,
        }
