# Deployment

## Local Demo

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

set SUPPLIER_SECURITY_MODE=local
set SUPPLIER_DEMO_MODE=true
set SUPPLIER_DATABASE_URL=sqlite:///data/production_app.db

streamlit run app.py
```

FastAPI:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Demo API auth:

```bash
curl -H "X-Tenant-ID: demo-tenant" -H "X-API-Key: demo-api-key" http://localhost:8000/suppliers
```

OIDC API auth:

```bash
curl -H "Authorization: Bearer <jwt>" http://localhost:8000/suppliers
```

Worker:

```bash
python -m backend.worker
```

## Docker Compose

```bash
docker compose up --build
```

Services:

- Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000`
- API liveness: `http://localhost:8000/live`
- API health: `http://localhost:8000/health`
- API readiness: `http://localhost:8000/ready`
- Postgres: `localhost:5432`
- Redis: `localhost:6379`

Readiness is stricter than liveness. `/live` is process-only and should stay
usable even when startup dependencies are unavailable. `/ready` returns HTTP
`503` when database initialization/checks fail or when production mode still has
unsafe runtime issues such as SQLite, demo mode, wildcard CORS, incomplete
OIDC/JWKS/SAML configuration, or local auth without an explicit production override.

## Render Staging

The repo includes GitHub-backed Render Blueprints with separate API and UI
services.

`render.yaml` creates:

- `supplier-intelligence-api`: FastAPI using `backend/Dockerfile`
- `supplier-intelligence-ui`: Streamlit using the root `Dockerfile`
- Render Postgres

The API service start command is:

```bash
python scripts/migrate.py && uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

The UI service start command is:

```bash
streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
```

The API Blueprint uses `/live` for Render's process health check; use `/ready`
and `scripts/smoke_staging.py` against the API service URL to verify that
required staging configuration is complete.

`render.full.yaml` keeps the same API/UI split and adds:

- Render Key Value for Redis-compatible Celery broker
- Celery worker
- Cron jobs that enqueue Sentinel/risk/exposure tasks

Runbook:

```bash
git checkout main
git pull origin main
```

Then open:

```text
https://dashboard.render.com/blueprint/new?repo=https://github.com/jayeshpatel731997-rgb/supplier-intelligence-platform
```

Before the first deploy, fill the required manual values in Render from
`RENDER_ENV_CHECKLIST.md`. Missing Postgres, OIDC/SAML, CORS, or upload storage
configuration should leave `/ready` degraded. Do not use demo mode for real
staging.

Required real-staging values:

```bash
SUPPLIER_SECURITY_MODE=production
SUPPLIER_DEPLOYMENT_MODE=render-staging-phase1
SUPPLIER_DEMO_MODE=false
SUPPLIER_DATABASE_URL=<render-postgres-url>
CORS_ALLOW_ORIGINS=https://<your-ui-origin>
AUTH_PROVIDER=oidc
OIDC_ISSUER_URL=<issuer>
OIDC_CLIENT_ID=<client-id>
OIDC_CLIENT_SECRET=<secret>
OIDC_AUDIENCE=<audience-if-needed>
OIDC_JWKS_URL=<jwks-url>
OIDC_ALGORITHMS=RS256
SUPPLIER_UPLOAD_STORAGE_PROVIDER=s3
SUPPLIER_UPLOAD_STORAGE_BUCKET=<bucket>
SUPPLIER_UPLOAD_STORAGE_REGION=<region>
SUPPLIER_UPLOAD_STORAGE_ENDPOINT_URL=<endpoint>
SUPPLIER_UPLOAD_STORAGE_ACCESS_KEY_ID=<access-key-id>
SUPPLIER_UPLOAD_STORAGE_SECRET_ACCESS_KEY=<secret-access-key>
```

After Render deploys, run the smoke test locally:

```bash
set STAGING_API_BASE_URL=https://supplier-intelligence-api.onrender.com
set STAGING_UI_BASE_URL=https://supplier-intelligence-ui.onrender.com
set STAGING_BEARER_TOKEN=<short-lived-oidc-token>
set STAGING_EXPECTED_TENANT_ID=demo-tenant
python scripts/smoke_staging.py
```

`STAGING_API_BASE_URL` must be the `supplier-intelligence-api` URL.
`STAGING_BASE_URL` remains a compatible alias. If either points to
`supplier-intelligence-ui`, Streamlit can serve `200 text/html` for API-like
paths; the smoke test treats that as a failed deployment target.
The default smoke requires staging credentials, the expected tenant, and the
Streamlit URL. Use `python scripts/smoke_staging.py --health-only --skip-ui`
only when intentionally checking public API health and auth rejection without
the authenticated evidence workflow or Streamlit surface.

For an authenticated read check, add either:

```bash
set STAGING_BEARER_TOKEN=<oidc-token>
```

or, only if local auth is explicitly allowed for staging:

```bash
set STAGING_TENANT_ID=<tenant-id>
set STAGING_API_KEY=<tenant-api-key>
```

Seed deterministic demo data before workflow smoke checks:

```bash
python scripts/migrate.py
set SUPPLIER_STAGING_SEED_USERNAME=<oidc-subject-or-verified-email>
python scripts/seed_demo_data.py --tenant-id demo-tenant
```

The seed command does not create staging schema; Alembic migrations must succeed
first. In OIDC mode the staging seed creates a `risk_manager` membership for the
configured token subject or verified email and does not create an API key.

The smoke test checks `/live`, `/health`, `/ready`, verifies `/suppliers`
rejects missing auth, verifies health endpoints return API JSON instead of
Streamlit HTML fallback, and, when auth is configured, checks authenticated
`/suppliers`, connector sync, evidence-chain run, evidence action update, and
scoring config read. In public connector mode, a `skipped` sync is accepted
when an optional source or supplier identifier is not configured. A `failed`
sync fails the smoke test so staging cannot silently ignore connector outages.
The client redacts secret-like values in output.

For separately deployed Streamlit, set:

```bash
SUPPLIER_API_BASE_URL=https://supplier-intelligence-api.onrender.com
```

The Streamlit command center uses this value for API reachability checks and
shows a friendly error if the API is unreachable.

Connector modes for staging:

```bash
SUPPLIER_CONNECTOR_MODE=demo
# or, for optional public data tests:
SUPPLIER_CONNECTOR_MODE=public
SUPPLIER_CONNECTOR_TIMEOUT_SECONDS=10
SUPPLIER_CONNECTOR_RETRY_COUNT=1
SUPPLIER_NEWS_RSS_URLS=https://example.com/feed.xml
SUPPLIER_NEWS_REQUIRE_SUPPLIER_MATCH=true
SUPPLIER_FILINGS_COMPANY_IDENTIFIER=<cik>
SUPPLIER_FILINGS_USER_AGENT=Supplier Intelligence Platform ops@example.com
# Optional override; when blank, the connector uses SEC submissions for the CIK.
SUPPLIER_FILINGS_SOURCE_URLS=
SUPPLIER_HIRING_SOURCE_URLS=https://example.com/jobs.rss
```

If public connector configuration is missing or a source fails, the connector
sync records `skipped` or `failed` and the evidence-chain workflow remains
available. Demo/stub mode remains deterministic and offline. Public news and
hiring inputs must be RSS/Atom-compatible; SEC filings are read from the public
EDGAR submissions JSON endpoint and mapped to financial weak signals.
Routine forms such as 10-K, 10-Q, and Form 4 are not treated as adverse risk
signals; the connector currently maps material 8-K/6-K and late-filing notices.

## Postgres Configuration

Set one managed Postgres connection URL:

```bash
SUPPLIER_DATABASE_URL=postgresql+psycopg://supplier_app:<password>@<host>:5432/supplier_intelligence
```

Provider notes:

- **Render Postgres:** inject the database `connectionString` into
  `SUPPLIER_DATABASE_URL`, as shown in `render.yaml`. A `postgresql://` URL is
  normalized to the Psycopg SQLAlchemy driver.
- **Neon:** copy the pooled connection string into the hosting provider's
  secret store. Keep `sslmode=require`; do not put the URL in Git, shell
  history, screenshots, or logs.
- **Supabase:** use the direct Postgres connection or the transaction/session
  pooler URL appropriate for the service's connection lifetime. Require TLS and
  keep the password only in the deployment secret store.

Production mode does not create or mutate schema at API/worker startup. Run
Alembic before starting production services:

```bash
python scripts/migrate.py
python scripts/validate_tenant_schema.py
```

Run these commands only after verifying that the URL targets the intended
disposable or staging database:

```powershell
$env:SUPPLIER_SECURITY_MODE="production"
$env:SUPPLIER_DEPLOYMENT_MODE="staging"
$env:SUPPLIER_DEMO_MODE="false"
$env:SUPPLIER_DATABASE_URL="<secret from Render, Neon, or Supabase>"
.\venv\Scripts\python.exe scripts\migrate.py
.\venv\Scripts\python.exe scripts\validate_tenant_schema.py
```

For an approved seeded OIDC staging tenant, set the subject or verified email
that will be present in the bearer token and run the seed twice to prove
idempotency:

```powershell
$env:AUTH_PROVIDER="oidc"
$env:SUPPLIER_STAGING_SEED_USERNAME="<oidc-subject-or-verified-email>"
.\venv\Scripts\python.exe scripts\seed_demo_data.py --tenant-id demo-tenant
.\venv\Scripts\python.exe scripts\seed_demo_data.py --tenant-id demo-tenant
```

Do not run these mutating commands against production.

Local/demo mode still supports the SQLAlchemy `create_all()` fallback for SQLite
developer demos.

`scripts/migrate.py` fails before Alembic when staging/production is missing
`SUPPLIER_DATABASE_URL`/`DATABASE_URL`, when the URL is invalid, when SQLite is
used for staging/production, or when `--create-all-fallback` is attempted in
staging/production.

### Manual Managed Postgres Checklist

1. Create an empty staging database or isolated staging project.
2. Restrict network access and create a least-privilege application role.
3. Store `SUPPLIER_DATABASE_URL` in the provider or deployment secret manager.
4. Confirm the hostname and database name out of band before migration.
5. Take or verify a provider snapshot, branch, or backup.
6. Run migration and tenant-schema validation.
7. Run the seed twice only when a seeded demo tenant is approved.
8. Deploy API and Streamlit with `SUPPLIER_API_BASE_URL` pointing to the API.
9. Set `STAGING_API_BASE_URL`, `STAGING_UI_BASE_URL`, a short-lived bearer
   token, and `STAGING_EXPECTED_TENANT_ID`; then run the authenticated smoke.
10. Record revision, sanitized output, backup identifier, and rollback owner.

## NewsAPI / Anthropic / OpenAI

Demo mode does not require paid APIs.

For live Sentinel mode:

```bash
NEWSAPI_KEY=...
OPENAI_API_KEY=...
# or
ANTHROPIC_API_KEY=...
```

If keys are missing or an API fails, Sentinel returns a safe error, records an alert, and does not crash the app.

Supplier evidence narratives remain deterministic by default:

```bash
SUPPLIER_LLM_NARRATIVE_PROVIDER=none
```

`openai` and `anthropic` are future governed modes. Before enabling either,
require evidence-only structured inputs, output schema validation,
claim-to-evidence traceability, tenant-safe logging, evaluation cases,
deterministic fallback, and provider retention review. Current readiness reports
those modes as `interface_only`; do not treat them as live integrations.

## Upload Safety And Storage

The FastAPI ingestion endpoint rejects unsupported extensions and reads at most
one byte over the configured limit before returning `413`. It also rejects
unsafe filenames or paths before writing anything, checks MIME types when the
client provides a useful MIME value, runs the upload scanner hook, and stores
the accepted payload under a tenant-scoped key before passing the same bytes to
the existing CSV/Excel/JSON ingestion parser.

```bash
SUPPLIER_MAX_UPLOAD_BYTES=5000000
SUPPLIER_ALLOWED_UPLOAD_EXTENSIONS=.csv,.xlsx,.xls,.json
SUPPLIER_ALLOWED_UPLOAD_MIME_TYPES=text/csv,application/csv,text/plain,application/json,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
SUPPLIER_UPLOAD_STORAGE_PROVIDER=local
SUPPLIER_UPLOAD_STORAGE_PATH=data/uploads
```

For production mode, use S3-compatible object storage and complete these values
before expecting `/ready` to pass:

```bash
SUPPLIER_UPLOAD_STORAGE_PROVIDER=s3
SUPPLIER_UPLOAD_STORAGE_BUCKET=...
SUPPLIER_UPLOAD_STORAGE_REGION=...
SUPPLIER_UPLOAD_STORAGE_ENDPOINT_URL=...
SUPPLIER_UPLOAD_STORAGE_ACCESS_KEY_ID=...
SUPPLIER_UPLOAD_STORAGE_SECRET_ACCESS_KEY=...
SUPPLIER_UPLOAD_STORAGE_KEY_PREFIX=uploads
```

If upload malware/content scanning is required by policy, production also fails
closed until scanner configuration is present:

```bash
SUPPLIER_UPLOAD_SCANNER_REQUIRED=true
SUPPLIER_UPLOAD_SCANNER_PROVIDER=...
SUPPLIER_UPLOAD_SCANNER_ENDPOINT_URL=...
```

The scanner is currently an integration stub. Connect it to a real malware or
content inspection service before allowing externally shared production uploads.
Do not log file contents, object-storage secrets, tokens, or signed URLs.

## Production Mode

Production mode disables unsafe default admin creation:

```bash
SUPPLIER_SECURITY_MODE=production
SUPPLIER_DEMO_MODE=false
SUPPLIER_DATABASE_URL=postgresql+psycopg://...
CORS_ALLOW_ORIGINS=https://your-ui.example.com
```

If no Streamlit pilot user exists, the login screen shows a first-admin setup form. Use a strong password.
For OIDC production mode, set `OIDC_ISSUER_URL`, `OIDC_CLIENT_ID`,
`OIDC_CLIENT_SECRET`, `OIDC_JWKS_URL`, `OIDC_ALGORITHMS`, and
`OIDC_CLOCK_SKEW_SECONDS`. Set `OIDC_AUDIENCE` when the token audience differs
from `OIDC_CLIENT_ID`. For SAML production mode, set either `SAML_METADATA_URL`
or `SAML_METADATA_FILE`.

OIDC protected routes verify JWT signature, issuer, audience, algorithm, expiry,
and not-before timestamps. After token verification, the platform still requires
an active local membership for the token subject/email in the claimed tenant.
Create or sync tenants and memberships before switching production traffic to
`AUTH_PROVIDER=oidc`.

This OIDC path currently protects FastAPI. Streamlit still uses the pilot/local
login and does not implement a browser OIDC callback, so UI SSO remains a
real-world staging gap.

## Alembic

An Alembic initial metadata migration is included. Local/demo mode still supports automatic table creation for SQLite compatibility.

```bash
python scripts/migrate.py
python scripts/validate_tenant_schema.py
```

Local fallback:

```bash
python scripts/migrate.py --create-all-fallback
```

Celery-ready worker:

```bash
docker compose --profile celery up --build celery-worker
```

Redis/Celery is optional for the first staging gate. Keep `WORKER_MODE=local`
and scheduling disabled for the API + Streamlit + Postgres phase. Promote to
`WORKER_MODE=celery` with `REDIS_URL` only after Redis connectivity, worker
health, retry behavior, tenant-scoped jobs, and cron ownership are tested.

## Staging Observability Plan

Sentry and PostHog are optional integrations, not current staging blockers.

- Sentry: capture unhandled API/worker exceptions with environment and release
  tags; redact headers, tokens, database URLs, supplier payloads, and
  tenant-sensitive data before transport.
- PostHog: capture coarse product events only after consent and data-governance
  review; do not send supplier names, evidence text, credentials, or raw audit
  details.
- Keep application logs and `/ready` useful without either service. Missing
  observability credentials must never crash startup.
- Before rollout, define secret-manager variables, sampling rules, retention
  owners, deletion procedures, and a redaction test.

## Rollback Notes

- Render service rollback: use Render's previous deploy rollback for the API,
  worker, or `supplier-intelligence-ui` service.
- Database rollback: restore from a verified Render Postgres backup/snapshot
  before downgrading schema. The initial Alembic downgrade drops managed tables
  and is not safe for customer data.
- Config rollback: revert the env var change in Render, then redeploy and
  rerun `python scripts/smoke_staging.py`.
