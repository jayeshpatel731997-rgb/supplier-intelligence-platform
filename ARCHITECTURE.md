# Architecture

## Runtime Shape

The project is now split into three cooperating layers:

- **Streamlit Command Center**: internal executive/operator UI in `app.py`.
- **FastAPI Backend**: integration API in `backend/main.py`.
- **Production Foundation**: reusable config, database, repositories, services, security, scheduler, and observability under `src/`.

The existing analytics remain in place:

- Bayesian supplier risk scoring
- NetworkX centrality
- SIR cascade simulation
- Monte Carlo VaR/CVaR
- Scorecard and TCO logic
- Sentinel/news intelligence
- NASA/PRA-style risk upgrades

## Data Flow

1. Supplier data arrives through Streamlit upload or `POST /ingestion/upload`.
2. `data_ingestion.py` maps and cleans messy procurement columns.
3. `src.services.ingestion_service.IngestionService` stores clean suppliers and ingestion job metadata.
4. `src.services.risk_service.RiskService` recalculates supplier risk and exposure.
5. `src.services.sentinel_service.SentinelService` runs demo, rule-based, or API-backed Sentinel scans.
6. High-risk conditions create alerts in `alerts`.
7. Streamlit and FastAPI both read status from the same database layer.

## Persistence

The production layer uses SQLAlchemy. Default local mode uses:

```text
sqlite:///data/production_app.db
```

Production-like mode should use Postgres:

```text
postgresql+psycopg://supplier_app:<password>@postgres:5432/supplier_intelligence
```

The older pilot auth SQLite layer remains for Streamlit compatibility and gradual migration.

## Multi-Tenancy

The SaaS foundation uses shared-schema tenancy:

- `tenants` and `memberships` define organization context and tenant-scoped roles.
- Protected API calls require `X-Tenant-ID` and `X-API-Key`.
- Business tables carry `tenant_id`.
- Repositories and services accept tenant context and filter all reads/writes by tenant.
- The design leaves an escape hatch for future isolated tenant pods/databases by moving tenant routing above the repository layer.

Local/demo mode seeds `demo-tenant` with `demo-api-key`. Production mode must create real tenants and tenant API keys through admin workflows.

## Background Monitoring

`backend/worker.py` runs an APScheduler process with scheduled jobs:

- `sentinel_scan`
- `risk_recalculate`
- `exposure_recalculate`

For local demo and tests, `LocalJobScheduler.run_job_now()` executes jobs synchronously and records job status.

## API Surface

FastAPI exposes:

- `GET /live`
- `GET /ready`
- `GET /health`
- `GET /suppliers`
- `GET /suppliers/{id}`
- `GET /risk/scores`
- `POST /risk/recalculate`
- `GET /financial/exposure`
- `POST /scenario/run`
- `POST /sentinel/scan`
- `GET /sentinel/events`
- `GET /alerts`
- `POST /alerts/{id}/acknowledge`
- `POST /ingestion/upload`
- `GET /ingestion/jobs`
- `GET /audit/logs`
- `GET /system/status`

## Enterprise Hardening Layer

- Alembic metadata migration plus `scripts/backfill_tenants.py` and `scripts/validate_tenant_schema.py`.
- Celery/Redis-ready `EnterpriseTaskRunner` with local APScheduler fallback.
- Provider abstraction for local/OIDC/SAML/SCIM auth readiness.
- Per-tenant/API-key rate limiting middleware.
- Secrets and KMS provider interfaces.
- Tenant-scoped audit export and compliance evidence collection.
- Backup/restore scripts and retention dry-run service.
- Locust load testing harness.
