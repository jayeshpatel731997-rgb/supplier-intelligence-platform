# RBAC Model

Roles are tenant-scoped through memberships:

- `platform_admin`: cross-tenant admin.
- `org_admin`: tenant user/API key management.
- `risk_manager`: risk, Sentinel, scenarios, alerts.
- `analyst`: ingestion and analysis.
- `viewer`: read-only dashboards.
- `auditor`: read-only audit/system evidence.

Authorization always combines active tenant, authenticated principal, membership role, and requested permission.
