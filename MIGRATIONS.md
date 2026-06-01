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
python scripts/migrate.py
python scripts/validate_tenant_schema.py
```

`scripts/migrate.py` runs `alembic upgrade head` after validating the database
configuration. In `SUPPLIER_SECURITY_MODE=production` or
`SUPPLIER_DEPLOYMENT_MODE=render-staging...`, it fails before Alembic when:

- `SUPPLIER_DATABASE_URL` and `DATABASE_URL` are both missing.
- The database URL is invalid.
- The database URL points to SQLite.
- `--create-all-fallback` is requested.

In production/staging, the FastAPI API and background worker do not call
SQLAlchemy `create_all()` at startup. Missing migrations should surface as
`/ready` returning HTTP `503`, not as silent schema creation.

## Backfill rules

- Existing rows with missing or blank `tenant_id` are assigned to `demo-tenant` only outside production.
- `scripts/backfill_tenants.py` raises in production mode.
- `scripts/validate_tenant_schema.py` fails when tenant-scoped business tables are missing or missing `tenant_id`.

## Rollback

The initial Alembic downgrade drops the managed metadata tables. Do not run
downgrade against customer data without a verified backup and restore plan.
For Render staging, prefer Render's service rollback plus a verified Postgres
backup restore/snapshot rollback. After rollback, rerun:

```powershell
python scripts/validate_tenant_schema.py
python scripts/smoke_staging.py --base-url https://supplier-intelligence-api.onrender.com
```
