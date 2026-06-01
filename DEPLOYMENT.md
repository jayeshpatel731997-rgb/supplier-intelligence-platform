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

The repo includes a phased GitHub-backed Render Blueprint.

Phase 1 uses `render.yaml` and creates:

- FastAPI backend
- Render Postgres

This is the safest first deployment because it validates Docker, Alembic
migrations, Render Postgres, and health endpoints before adding paid
worker/cron services. The Blueprint uses `/live` for Render's process health
check; use `/ready` and `scripts/smoke_staging.py` to verify that required
staging configuration is complete.

Phase 2 uses `render.full.yaml` after Phase 1 is healthy and adds:

- Streamlit command center
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
set STAGING_BASE_URL=https://supplier-intelligence-api.onrender.com
python scripts/smoke_staging.py
```

For an authenticated read check, add either:

```bash
set STAGING_BEARER_TOKEN=<oidc-token>
```

or, only if local auth is explicitly allowed for staging:

```bash
set STAGING_TENANT_ID=<tenant-id>
set STAGING_API_KEY=<tenant-api-key>
```

The smoke test checks `/live`, `/health`, `/ready`, verifies `/suppliers`
rejects missing auth, and optionally checks authenticated `/suppliers`. It
redacts secret-like values in output.

## Postgres Configuration

Set:

```bash
SUPPLIER_DATABASE_URL=postgresql+psycopg://supplier_app:<password>@<host>:5432/supplier_intelligence
```

Production mode does not create or mutate schema at API/worker startup. Run
Alembic before starting production services:

```bash
python scripts/migrate.py
python scripts/validate_tenant_schema.py
```

Local/demo mode still supports the SQLAlchemy `create_all()` fallback for SQLite
developer demos.

`scripts/migrate.py` fails before Alembic when staging/production is missing
`SUPPLIER_DATABASE_URL`/`DATABASE_URL`, when the URL is invalid, when SQLite is
used for staging/production, or when `--create-all-fallback` is attempted in
staging/production.

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

## Rollback Notes

- Render service rollback: use Render's previous deploy rollback for the API,
  worker, or Streamlit service.
- Database rollback: restore from a verified Render Postgres backup/snapshot
  before downgrading schema. The initial Alembic downgrade drops managed tables
  and is not safe for customer data.
- Config rollback: revert the env var change in Render, then redeploy and
  rerun `python scripts/smoke_staging.py`.
