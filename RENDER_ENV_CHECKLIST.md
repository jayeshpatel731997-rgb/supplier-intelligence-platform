# Render Environment Variable Checklist

## Phase 1 Blueprint

Render sets these from managed services:

- `SUPPLIER_DATABASE_URL`
- `DATABASE_URL`

Render generates these in the shared environment group:

- `SUPPLIER_DEMO_API_KEY`
- `SUPPLIER_APP_ADMIN_PASSWORD`

Configured non-secrets:

- `SUPPLIER_SECURITY_MODE=local`
- `SUPPLIER_DEPLOYMENT_MODE=render-staging-phase1`
- `SUPPLIER_DEMO_MODE=true`
- `AUTH_PROVIDER=local`
- `AUTH_ALLOW_LOCAL_IN_PRODUCTION=false`
- `DEFAULT_TENANT_ID=demo-tenant`
- `SUPPLIER_APP_ADMIN_USER=staging-admin`
- `RATE_LIMIT_ENABLED=true`
- `RATE_LIMIT_REQUESTS=300`
- `RATE_LIMIT_WINDOW_SECONDS=60`
- `RETENTION_ENABLED=false`
- `SECRETS_PROVIDER=env`
- `KMS_PROVIDER=local`
- `CORS_ALLOW_ORIGINS=*`
- `WORKER_MODE=local`
- `SUPPLIER_SCHEDULER_ENABLED=false`

## Phase 2 Full Stack

`render.full.yaml` additionally sets:

- `REDIS_URL` from Render Key Value
- `WORKER_MODE=celery` on the worker and cron services

## Optional Manual Values

Add these only after the first API deploy is healthy:

- `NEWSAPI_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `WORKOS_API_KEY`
- `WORKOS_CLIENT_ID`

Do not put `sync: false` secrets in `envVarGroups`; Render ignores that pattern.
Use `generateValue: true` for generated shared secrets or add secrets manually in
the Render Dashboard.

## Before Sharing Staging Externally

- Reveal/copy `SUPPLIER_DEMO_API_KEY` from Render and test protected routes.
- Rotate `SUPPLIER_DEMO_API_KEY` if it was exposed.
- Use a strong `SUPPLIER_APP_ADMIN_PASSWORD`.
- Restrict CORS from `*` to the Streamlit URL after Phase 2.
- Set up uptime checks against `/health`.
- Add Sentry or another error monitor.
- Decide whether staging should remain in demo mode.
