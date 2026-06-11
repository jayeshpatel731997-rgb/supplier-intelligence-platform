# Render Staging Launch Runbook

This runbook describes the repository as it exists. It is not evidence that a
remote staging environment has passed.

## Blueprint Truth

`render.yaml` deploys all three Phase 1 resources:

- `supplier-intelligence-api`: FastAPI web service
- `supplier-intelligence-ui`: Streamlit web service
- `supplier-intelligence-postgres`: managed Postgres

It is therefore an **API + Streamlit + Postgres** Blueprint, not an API-only
Blueprint. Render health-checks the API at `/live` and Streamlit at
`/_stcore/health`.

The environment group sets `AUTH_PROVIDER=oidc`,
`AUTH_ALLOW_LOCAL_IN_PRODUCTION=false`, `SUPPLIER_DEMO_MODE=false`, and
`SUPPLIER_UPLOAD_STORAGE_PROVIDER=s3`. It generates only
`SUPPLIER_APP_ADMIN_PASSWORD` for the current Streamlit pilot login. It does not generate
`SUPPLIER_DEMO_API_KEY`, and real staging must not use the local
`demo-tenant` / `demo-api-key` credential.

OIDC currently protects FastAPI bearer-token routes. Streamlit still uses its
pilot/local login and does not implement an end-user OIDC redirect/callback
flow. The staging smoke therefore validates API OIDC with an externally
obtained short-lived token and checks only Streamlit process health. Do not
claim browser SSO readiness from this Blueprint.

`render.full.yaml` keeps the API, Streamlit, and Postgres resources and adds
Render Key Value, a Celery worker, and two cron services. Do not promote it
until the base Blueprint has passed real OIDC, Postgres, storage, tenant, and
authenticated smoke checks.

## 1. Prepare The Revision

The Blueprint must be present on the Git revision Render deploys. Do not merge
this branch merely to test documentation.

Record:

```powershell
git rev-parse HEAD
git status --short
```

Before deployment, review `render.yaml` in Render and confirm the API and UI
service names and generated hostnames.

## 2. Create The Blueprint

Open:

```text
https://dashboard.render.com/blueprint/new?repo=https://github.com/jayeshpatel731997-rgb/supplier-intelligence-platform
```

Render injects `SUPPLIER_DATABASE_URL` and `DATABASE_URL` from the managed
Postgres resource. Do not copy those values into Git, chat, screenshots, or
shell history.

## 3. Configure Real Staging Values

Set these non-secret values in the `supplier-intelligence-staging` environment
group:

```text
CORS_ALLOW_ORIGINS=https://<streamlit-host>
OIDC_AUDIENCE=<api-audience-or-client-id>
OIDC_ALGORITHMS=RS256
OIDC_CLOCK_SKEW_SECONDS=60
OIDC_ISSUER_URL=https://<issuer>
OIDC_JWKS_URL=https://<issuer>/<jwks-path>
SUPPLIER_API_BASE_URL=https://<api-host>
SUPPLIER_CONNECTOR_MODE=demo
SUPPLIER_LLM_NARRATIVE_PROVIDER=none
SUPPLIER_STAGING_SEED_USERNAME=<oidc-subject-or-verified-email>
SUPPLIER_UPLOAD_STORAGE_BUCKET=<staging-bucket>
SUPPLIER_UPLOAD_STORAGE_ENDPOINT_URL=https://<storage-endpoint>
SUPPLIER_UPLOAD_STORAGE_KEY_PREFIX=uploads
SUPPLIER_UPLOAD_STORAGE_PROVIDER=s3
SUPPLIER_UPLOAD_STORAGE_REGION=<region>
```

Set these secrets through Render or the approved secret manager:

```text
OIDC_CLIENT_ID=<secret-managed-value>
OIDC_CLIENT_SECRET=<secret-managed-value>
SUPPLIER_UPLOAD_STORAGE_ACCESS_KEY_ID=<secret-managed-value>
SUPPLIER_UPLOAD_STORAGE_SECRET_ACCESS_KEY=<secret-managed-value>
```

Optional external intelligence secrets are `NEWSAPI_KEY`, `OPENAI_API_KEY`, and
`ANTHROPIC_API_KEY`. They are not required for deterministic staging smoke.
`OIDC_REDIRECT_URI` is reserved for a future browser login integration and is
not consumed by the current Streamlit UI.

If policy requires content scanning, also configure:

```text
SUPPLIER_UPLOAD_SCANNER_REQUIRED=true
SUPPLIER_UPLOAD_SCANNER_PROVIDER=<scanner-provider>
SUPPLIER_UPLOAD_SCANNER_ENDPOINT_URL=https://<scanner-endpoint>
```

The current scanner is an integration boundary, not proof of malware scanning.

## 4. Apply And Inspect Startup

Apply the Blueprint and wait for:

- Postgres to become available.
- `sh scripts/start_api_render.sh` to complete migrations before Uvicorn starts.
- The API `/live` health check to pass.
- `sh scripts/start_ui_render.sh` to start Streamlit and pass
  `/_stcore/health`.

The Blueprint uses small shell scripts because Render's Docker command override
must not contain a quoted multi-command expression. Both scripts use Render's
`PORT` and fall back to port `10000`; the API launcher replaces itself with
Uvicorn after a successful migration, and the UI launcher replaces itself with
Streamlit.

`/live` proves process availability only. `/ready` must remain HTTP `503` until
Postgres, explicit CORS, OIDC, and S3-compatible storage configuration pass.

## 5. Managed Postgres Migration

The API launcher already runs `python scripts/migrate.py`. For a separate
operator-run migration, first verify out of band that the URL targets the
approved staging database and that a backup or disposable database branch
exists.

Run from a trusted operator environment without printing the URL:

```powershell
$env:SUPPLIER_SECURITY_MODE="production"
$env:SUPPLIER_DEPLOYMENT_MODE="render-staging-phase1"
$env:SUPPLIER_DEMO_MODE="false"
$env:AUTH_PROVIDER="oidc"
$env:SUPPLIER_DATABASE_URL="<managed-staging-postgres-url>"
.\venv\Scripts\python.exe scripts\migrate.py
.\venv\Scripts\python.exe scripts\validate_tenant_schema.py
```

Record the deployed Git revision, Alembic revision, backup identifier, sanitized
command result, and rollback owner.

## 6. Idempotent OIDC Seed

The deterministic seed is optional and supports only `demo-tenant`. In OIDC
staging it creates or updates a least-privilege `risk_manager` membership for
the configured OIDC subject or verified email. It does not create or require a
staging API key and deactivates existing demo-tenant API keys.

Run it twice against the approved staging database:

```powershell
$env:SUPPLIER_SECURITY_MODE="production"
$env:SUPPLIER_DEPLOYMENT_MODE="render-staging-phase1"
$env:SUPPLIER_DEMO_MODE="false"
$env:AUTH_PROVIDER="oidc"
$env:SUPPLIER_DATABASE_URL="<managed-staging-postgres-url>"
$env:SUPPLIER_STAGING_SEED_USERNAME="<oidc-subject-or-verified-email>"
.\venv\Scripts\python.exe scripts\seed_demo_data.py --tenant-id demo-tenant
.\venv\Scripts\python.exe scripts\seed_demo_data.py --tenant-id demo-tenant
```

Compare the sanitized counts and confirm the second run does not add duplicate
signals, evidence runs, actions, or connector syncs.

For local development only, `AUTH_PROVIDER=local` uses
`X-Tenant-ID: demo-tenant` and `X-API-Key: demo-api-key`. A real staging
local-auth exception requires `AUTH_ALLOW_LOCAL_IN_PRODUCTION=true`, an explicit
non-default `SUPPLIER_DEMO_API_KEY`, approval, expiration, audit evidence, and
key revocation after rollback. It is not the default staging path.

## 7. Authenticated Remote Smoke

Obtain a short-lived OIDC token for the seeded membership without printing or
persisting it. Then run:

```powershell
$env:STAGING_API_BASE_URL="https://<api-host>"
$env:STAGING_UI_BASE_URL="https://<streamlit-host>"
$env:STAGING_BEARER_TOKEN="<short-lived-oidc-token>"
$env:STAGING_EXPECTED_TENANT_ID="demo-tenant"
.\venv\Scripts\python.exe scripts\smoke_staging.py
```

The smoke client checks:

- API `/live`, `/health`, and `/ready`
- Streamlit `/_stcore/health`
- missing-auth rejection
- authenticated supplier read
- authenticated tenant identity
- rejection of an `X-Tenant-ID` override in OIDC mode
- connector sync, scoring configuration, evidence run, and action update
- API-versus-Streamlit URL mixups

Use `--health-only --skip-ui` only for an explicitly limited infrastructure
check. That result is not an authenticated staging pass.

## 8. Promote The Worker Stack

After the base Blueprint has passed the real checks above, review
`render.full.yaml`. It adds:

- `supplier-intelligence-redis`
- `supplier-intelligence-celery-worker`
- `supplier-intelligence-sentinel-cron`
- `supplier-intelligence-risk-cron`

Confirm pricing, Redis connectivity, worker health, retry behavior, tenant
scope, and cron ownership before replacing `render.yaml`.

## Rollback

For a bad service deploy, use Render's previous-deploy rollback. For a bad
migration, restore the verified Postgres backup or disposable branch before
rolling application code back. Revoke any emergency local-auth key and rerun
readiness plus authenticated smoke before restoring staging traffic.
