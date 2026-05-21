# Migrations and Tenant Backfill

This project now has an Alembic upgrade path plus SQLite `create_all()` fallback for local/demo mode.

## Local/demo

```powershell
python scripts/migrate.py --create-all-fallback
python scripts/backfill_tenants.py
python scripts/validate_tenant_schema.py
```

Local/demo mode may seed `demo-tenant`, `Demo Organization`, and `demo-api-key`. Production mode never creates unsafe default tenant credentials.

## Production-like Postgres

Set `SUPPLIER_DATABASE_URL` or `DATABASE_URL` to a Postgres URL, then run:

```powershell
alembic upgrade head
python scripts/validate_tenant_schema.py
```

## Backfill rules

- Existing rows with missing or blank `tenant_id` are assigned to `demo-tenant` only outside production.
- `scripts/backfill_tenants.py` raises in production mode.
- `scripts/validate_tenant_schema.py` checks tenant-scoped business tables for `tenant_id`.

## Rollback

The initial Alembic downgrade drops the managed metadata tables. Do not run downgrade against customer data without a verified backup and restore plan.
