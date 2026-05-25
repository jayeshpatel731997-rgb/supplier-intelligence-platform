# Render Staging Launch Runbook

This repo now uses a phased Render launch so you do not pay for the full worker
stack before the base API proves it can boot on Render.

## Phase 1: API + Postgres

`render.yaml` deploys only:

- `supplier-intelligence-api`
- `supplier-intelligence-postgres`

This verifies the Docker build, dependency install, Postgres connection,
Alembic/create-all fallback, tenant seed, request middleware, and health
endpoints before adding Streamlit, Redis, Celery, and cron jobs.

## 1. Confirm the Blueprint is on main

`render.yaml` must live on the repository default branch before opening Render.

```powershell
git checkout main
git pull origin main
```

## 2. Open the Render Blueprint

```text
https://dashboard.render.com/blueprint/new?repo=https://github.com/jayeshpatel731997-rgb/supplier-intelligence-platform
```

Render reads `render.yaml` from the repo.

## 3. Review generated secrets

Render can generate shared values in an environment group with `generateValue`.
This is used for:

- `SUPPLIER_DEMO_API_KEY`
- `SUPPLIER_APP_ADMIN_PASSWORD`

After the Blueprint is created, reveal/copy `SUPPLIER_DEMO_API_KEY` from the
`supplier-intelligence-staging` environment group so you can test protected API
routes. Do not paste the value into chat or commit it.

Important: Render does not prompt for `sync: false` values inside environment
groups, so this Blueprint avoids that pattern.

## 4. Apply Phase 1

Click **Apply** and wait for:

- Postgres status: available
- API deploy status: live
- API logs: migration command completes and Uvicorn starts

## 5. Verify Phase 1 health

Replace the hostname if Render assigns a suffix.

```powershell
curl.exe https://supplier-intelligence-api.onrender.com/health
curl.exe https://supplier-intelligence-api.onrender.com/live
curl.exe https://supplier-intelligence-api.onrender.com/ready
curl.exe https://supplier-intelligence-api.onrender.com/worker/health
```

Protected API smoke:

```powershell
$env:SUPPLIER_DEMO_API_KEY="<value copied from Render env group>"
curl.exe `
  -H "X-Tenant-ID: demo-tenant" `
  -H "X-API-Key: $env:SUPPLIER_DEMO_API_KEY" `
  https://supplier-intelligence-api.onrender.com/system/status
```

Expected result: HTTP 200 with database/system status JSON.

## Phase 2: Full staging stack

Only after Phase 1 is healthy, promote the full Blueprint:

```powershell
Copy-Item render.full.yaml render.yaml
git add render.yaml render.full.yaml RENDER_STAGING_RUNBOOK.md RENDER_ENV_CHECKLIST.md DEPLOYMENT.md README.md
git commit -m "Promote Render full staging stack"
git push origin main
```

The full stack adds:

- `supplier-intelligence-command-center`
- `supplier-intelligence-redis`
- `supplier-intelligence-celery-worker`
- `supplier-intelligence-sentinel-cron`
- `supplier-intelligence-risk-cron`

The full stack uses paid worker/cron web-service instance types. Check Render's
review screen before applying.

## Optional keys after Phase 1

Add these manually to the `supplier-intelligence-staging` environment group only
when you are ready to test richer integrations:

- `NEWSAPI_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `WORKOS_API_KEY`
- `WORKOS_CLIENT_ID`

The app is designed to run without these keys in demo mode.

## WorkOS later

Keep `AUTH_PROVIDER=local` for this staging launch. When ready:

1. Create a WorkOS application.
2. Add the Render backend URL as an allowed redirect origin.
3. Add WorkOS API key/client id to Render env vars.
4. Implement the real WorkOS/OIDC callback and token validation.
5. Switch `AUTH_PROVIDER` from `local` to `oidc` or `workos` after tests pass.

## Rollback

For a bad deploy:

1. Open the failing service in Render.
2. Go to Deploys.
3. Select the previous successful deploy.
4. Roll back.

For a bad migration, restore Postgres from a verified backup before applying a
code rollback.
