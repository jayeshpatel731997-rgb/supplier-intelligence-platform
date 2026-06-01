# Production Readiness

## Now Production-Architecture Ready

The project now has a real production foundation beside the existing Streamlit app:

- FastAPI backend
- SQLAlchemy database layer
- Shared-schema tenant model with tenant-scoped repositories
- Tenant API-key authentication for local/demo protected FastAPI routes
- OIDC JWT/JWKS bearer-token verification foundation for protected FastAPI routes
- Tenant-scoped RBAC roles for SaaS administration and analysis
- SQLite fallback and Postgres-ready URL support
- Repository/service pattern
- Production tables for suppliers, KPIs, risk scores, events, matches, scenarios, exposures, alerts, audit logs, ingestion jobs, health events, and job runs
- Background worker entrypoint with APScheduler
- Local synchronous scheduler fallback
- Alert creation and acknowledgement
- Safer production auth behavior
- System health surfaces in API and Streamlit
- Lifespan-managed FastAPI initialization so imports do not require a live database
- Degraded health/status responses when database-backed status queries fail
- HTTP 503 readiness responses when startup, database, or required production runtime checks fail
- Production startup relies on Alembic instead of automatic table creation
- `scripts/migrate.py` validates staging/production database URL requirements before running Alembic
- Render blueprints start the API/worker with Alembic migrations and use `/live` for process health
- Staging smoke script checks `/live`, `/health`, `/ready`, protected-route auth failure, and optional authenticated supplier read
- Readiness flags wildcard CORS and incomplete OIDC/JWKS/SAML configuration in staging/production mode
- Health/status errors redact database URL credentials and secret-like values
- Bounded FastAPI uploads with configurable file size, extension, and MIME allow-lists
- Upload safety boundary for unsafe filenames/paths, tenant-scoped upload keys, and local/demo upload storage
- Staging/production readiness checks that fail closed when S3-compatible upload storage config is incomplete
- Upload scanner interface/stub with production fail-closed behavior when scanning is required but not configured
- Defensive numeric environment parsing for startup-critical runtime settings
- Tenant admins cannot grant roles above their own privilege level
- Docker Compose stack
- Pytest coverage for API, ingestion, Sentinel failure handling, alerts, jobs, OIDC token validation, auth, deployment readiness, and decision fallback

## Still Pilot/Internal-Company Ready, Not Enterprise SaaS

The hardening phase now adds migration/backfill tooling, Celery/Redis-ready worker structure, auth-provider scaffolding, rate limiting, secrets/KMS abstractions, backup/restore scripts, audit export readiness, compliance evidence stubs, and load testing.

Before selling as a real enterprise SaaS product, still add:

- Real IdP setup, tenant membership sync, and OIDC/SAML operational rollout
- Real MFA enforcement at the identity provider
- SCIM provisioning handlers
- Isolated tenant pods/databases for regulated customers
- Managed secrets and KMS-backed encryption
- Centralized log aggregation and SIEM destination
- Metrics/tracing
- Managed Celery/Redis or cloud-native queue operations
- Operational S3 client dependency/config validation in the target environment
- Real malware scanning and content inspection integration for externally shared uploads
- Render Postgres backup/restore drill
- Render log drain/SIEM setup
- Metrics, tracing, and alert routing
- SOC 2 audit evidence review
- Formal backup/restore drills
- Legal-approved data retention and deletion automation
- Contractual Privacy Policy, Terms, DPA, and security contact process

## Streamlit Role

Streamlit remains the internal command center, not the only production runtime. The backend and database layer are now the production foundation for integrations, scheduled monitoring, and operational state.

## Verification Commands

```bash
python -m compileall .
python -m pytest -v
python -m ruff check .
```
