# Production Readiness

## Now Production-Architecture Ready

The project now has a real production foundation beside the existing Streamlit app:

- FastAPI backend
- SQLAlchemy database layer
- Shared-schema tenant model with tenant-scoped repositories
- Tenant API-key authentication for protected FastAPI routes
- Tenant-scoped RBAC roles for SaaS administration and analysis
- SQLite fallback and Postgres-ready URL support
- Repository/service pattern
- Production tables for suppliers, KPIs, risk scores, events, matches, scenarios, exposures, alerts, audit logs, ingestion jobs, health events, and job runs
- Background worker entrypoint with APScheduler
- Local synchronous scheduler fallback
- Alert creation and acknowledgement
- Safer production auth behavior
- System health surfaces in API and Streamlit
- Docker Compose stack
- Pytest coverage for API, ingestion, Sentinel failure handling, alerts, jobs, auth, and decision fallback

## Still Pilot/Internal-Company Ready, Not Enterprise SaaS

The hardening phase now adds migration/backfill tooling, Celery/Redis-ready worker structure, auth-provider scaffolding, rate limiting, secrets/KMS abstractions, backup/restore scripts, audit export readiness, compliance evidence stubs, and load testing.

Before selling as a real enterprise SaaS product, still add:

- Real OAuth/OIDC or SAML provider setup
- Real MFA enforcement at the identity provider
- SCIM provisioning handlers
- Isolated tenant pods/databases for regulated customers
- Managed secrets and KMS-backed encryption
- Centralized log aggregation and SIEM destination
- Metrics/tracing
- Managed Celery/Redis or cloud-native queue operations
- Object storage for uploads
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
