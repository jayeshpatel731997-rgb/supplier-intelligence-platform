# Render Environment Variable Checklist

## Required for staging

Render sets these from managed services:

- `SUPPLIER_DATABASE_URL`
- `DATABASE_URL`
- `REDIS_URL`

You provide these secrets:

- `SUPPLIER_DEMO_API_KEY`
- `SUPPLIER_APP_ADMIN_PASSWORD`

Configured non-secrets:

- `SUPPLIER_SECURITY_MODE=local`
- `SUPPLIER_DEPLOYMENT_MODE=render-staging`
- `SUPPLIER_DEMO_MODE=true`
- `AUTH_PROVIDER=local`
- `DEFAULT_TENANT_ID=demo-tenant`
- `RATE_LIMIT_ENABLED=true`
- `RETENTION_ENABLED=false`
- `WORKER_MODE=celery` on the worker and cron services

## Optional

- `NEWSAPI_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `WORKOS_API_KEY`
- `WORKOS_CLIENT_ID`

## Before sharing staging externally

- Rotate `SUPPLIER_DEMO_API_KEY`.
- Use a strong `SUPPLIER_APP_ADMIN_PASSWORD`.
- Restrict CORS from `*` to the Streamlit URL.
- Set up uptime checks against `/health`.
- Add Sentry or another error monitor.
- Decide whether staging should remain in demo mode.
