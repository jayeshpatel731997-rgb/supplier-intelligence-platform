# Render Environment Variable Checklist

Use this checklist for real Render + Postgres staging. Do not paste secrets into
GitHub or docs. Set secret values in the Render Dashboard or a Render-managed
secret source.

## Render-Managed Values

Render injects these from the managed Postgres service:

- `SUPPLIER_DATABASE_URL`
- `DATABASE_URL`

`render.full.yaml` also injects `REDIS_URL` from Render Key Value for the API,
worker, `supplier-intelligence-ui`, and cron jobs.

## Blueprint Defaults

`render.yaml` and `render.full.yaml` create separate web services:

- `supplier-intelligence-api`: `python scripts/migrate.py && uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}`
- `supplier-intelligence-ui`: `streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0`

They set these non-secret values:

- `SUPPLIER_SECURITY_MODE=production`
- `SUPPLIER_DEPLOYMENT_MODE=render-staging-phase1` or `render-staging-full`
- `SUPPLIER_DEMO_MODE=false`
- `AUTH_PROVIDER=oidc`
- `AUTH_ALLOW_LOCAL_IN_PRODUCTION=false`
- `SUPPLIER_APP_ADMIN_USER=staging-admin`
- `RATE_LIMIT_ENABLED=true`
- `RATE_LIMIT_REQUESTS=300`
- `RATE_LIMIT_WINDOW_SECONDS=60`
- `SUPPLIER_MAX_UPLOAD_BYTES=5000000`
- `SUPPLIER_ALLOWED_UPLOAD_EXTENSIONS=.csv,.xlsx,.xls,.json`
- `SUPPLIER_ALLOWED_UPLOAD_MIME_TYPES=text/csv,application/csv,text/plain,application/json,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`
- `SUPPLIER_UPLOAD_STORAGE_PROVIDER=s3`
- `SUPPLIER_UPLOAD_STORAGE_KEY_PREFIX=uploads`
- `SUPPLIER_UPLOAD_SCANNER_REQUIRED=false`
- `SUPPLIER_UPLOAD_SCANNER_PROVIDER=none`
- `RETENTION_ENABLED=false`
- `SECRETS_PROVIDER=env`
- `KMS_PROVIDER=local`
- `WORKER_MODE=local` in `render.yaml`, `celery` for full-stack worker/cron services
- `SUPPLIER_SCHEDULER_ENABLED=false`

Render also generates:

- `SUPPLIER_APP_ADMIN_PASSWORD`

## Required Manual Values Before `/ready` Passes

Set these for real staging:

- `CORS_ALLOW_ORIGINS=https://<your-streamlit-or-ui-origin>`
- `OIDC_ISSUER_URL`
- `OIDC_CLIENT_ID`
- `OIDC_CLIENT_SECRET`
- `OIDC_AUDIENCE` when token `aud` differs from `OIDC_CLIENT_ID`
- `OIDC_JWKS_URL`
- `OIDC_ALGORITHMS=RS256` or the exact algorithms your IdP uses
- `OIDC_CLOCK_SKEW_SECONDS=60`
- `SUPPLIER_UPLOAD_STORAGE_BUCKET`
- `SUPPLIER_UPLOAD_STORAGE_REGION`
- `SUPPLIER_UPLOAD_STORAGE_ENDPOINT_URL`
- `SUPPLIER_UPLOAD_STORAGE_ACCESS_KEY_ID`
- `SUPPLIER_UPLOAD_STORAGE_SECRET_ACCESS_KEY`

If policy requires upload scanning:

- `SUPPLIER_UPLOAD_SCANNER_REQUIRED=true`
- `SUPPLIER_UPLOAD_SCANNER_PROVIDER=<scanner-provider>`
- `SUPPLIER_UPLOAD_SCANNER_ENDPOINT_URL=<scanner-endpoint>`

Optional live intelligence values:

- `NEWSAPI_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

## Local-Auth Staging Exception

OIDC is the recommended staging path. If you deliberately use local API-key auth
for a short staging window, set all of the following explicitly:

- `AUTH_PROVIDER=local`
- `AUTH_ALLOW_LOCAL_IN_PRODUCTION=true`
- Create tenant memberships/API keys through an admin path or controlled seed
  process; do not rely on demo keys in real staging.

## Deploy Validation

Run after the Render deploy:

```powershell
$env:STAGING_BASE_URL="https://supplier-intelligence-api.onrender.com"
python scripts/smoke_staging.py
```

Use the `supplier-intelligence-api` URL. The smoke script fails clearly if the
base URL points at Streamlit and returns HTML fallback instead of FastAPI JSON or
an API auth rejection.

For an authenticated read check, add either:

```powershell
$env:STAGING_BEARER_TOKEN="<oidc-token>"
```

or, only for the local-auth exception:

```powershell
$env:STAGING_TENANT_ID="<tenant-id>"
$env:STAGING_API_KEY="<tenant-api-key>"
```

## Manual Blockers

- Real OIDC/SAML provider configuration and tenant membership sync are manual.
- Real S3-compatible object storage bucket, credentials, and lifecycle policy
  are manual.
- Real scanner integration is manual; the code currently has a scanner
  interface/stub and fail-closed readiness checks.
- Render Postgres backup/restore validation is manual.
- SIEM/log drain, metrics, tracing, and alert routing are later tasks.
