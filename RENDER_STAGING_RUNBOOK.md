# Render Staging Launch Runbook

This deploys a staging SaaS stack from GitHub to Render:

- FastAPI backend
- Streamlit command center
- Render Postgres
- Render Key Value, Redis-compatible
- Celery worker
- Render cron jobs that enqueue Sentinel/risk/exposure tasks

## 1. Merge the deployment PR

`render.yaml` should live on the repository default branch before opening Render. Merge the deployment PR, or run:

```powershell
gh pr merge 1 --merge
```

## 2. Open the Render Blueprint

Use this deeplink after the branch is pushed:

```text
https://dashboard.render.com/blueprint/new?repo=https://github.com/jayeshpatel731997-rgb/supplier-intelligence-platform
```

Render will read `render.yaml` from the repo.

## 3. Fill required secrets

Set these during Blueprint creation:

- `SUPPLIER_DEMO_API_KEY`: generate a long random staging key; do not use `demo-api-key` on public staging.
- `SUPPLIER_APP_ADMIN_PASSWORD`: strong Streamlit staging admin password.

Optional for richer Sentinel behavior:

- `NEWSAPI_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

Reserved for later WorkOS integration:

- `WORKOS_API_KEY`
- `WORKOS_CLIENT_ID`

## 4. Apply and wait

Expected services:

- `supplier-intelligence-api`
- `supplier-intelligence-command-center`
- `supplier-intelligence-celery-worker`
- `supplier-intelligence-sentinel-cron`
- `supplier-intelligence-risk-cron`
- `supplier-intelligence-postgres`
- `supplier-intelligence-redis`

## 5. Verify health

FastAPI:

```powershell
curl https://supplier-intelligence-api.onrender.com/health
curl https://supplier-intelligence-api.onrender.com/live
curl https://supplier-intelligence-api.onrender.com/ready
curl https://supplier-intelligence-api.onrender.com/worker/health
```

Protected API smoke:

```powershell
curl `
  -H "X-Tenant-ID: demo-tenant" `
  -H "X-API-Key: <SUPPLIER_DEMO_API_KEY>" `
  https://supplier-intelligence-api.onrender.com/system/status
```

Streamlit:

```text
https://supplier-intelligence-command-center.onrender.com
```

## 6. Migration and tenant checks

The API and worker run `python scripts/migrate.py` on startup. To inspect manually from a service shell:

```powershell
python scripts/validate_tenant_schema.py
python scripts/backfill_tenants.py
```

Do not run demo backfill in production mode.

## 7. WorkOS later

Keep `AUTH_PROVIDER=local` for this staging launch. When ready:

1. Create a WorkOS application.
2. Add the Render backend URL as an allowed redirect origin.
3. Add WorkOS API key/client id to Render env vars.
4. Implement the real WorkOS/OIDC callback and token validation.
5. Switch `AUTH_PROVIDER` from `local` to `oidc` or `workos` after tests pass.

## 8. Rollback

For a bad deploy:

1. Open the failing service in Render.
2. Go to Deploys.
3. Select the previous successful deploy.
4. Roll back.

For a bad migration, restore Postgres from a verified backup before applying code rollback.
