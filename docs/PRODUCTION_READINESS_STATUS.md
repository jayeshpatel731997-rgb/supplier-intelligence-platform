# Production Readiness Status

## Current Label

**Conditional go for managed staging**

This is not production-ready. The project has strong local/CI evidence and a
live Render API health signal, but `/ready` still correctly reports degraded
until external controls are configured and proven.

## Observed Managed Staging Evidence

- Render API is live.
- `/health` reports `status: ok`.
- Database health passes.
- `/health` reported database driver `postgresql+psycopg` and API status
  `ready`.
- The active database URL is a Supabase Postgres pooler host ending in
  `pooler.supabase.com`.
- Current API service: `supplier-intelligence-api-hut2`.
- Current UI service: `supplier-intelligence-ui-hut2`.

No secret values are recorded in this repository.

## Current `/ready` Degraded Items

The remaining degraded readiness items are external configuration controls:

- `CORS_ALLOW_ORIGINS` must be set to explicit trusted origins.
  - Current expected UI origin:
    `https://supplier-intelligence-ui-hut2.onrender.com`
- OIDC configuration is missing:
  - `OIDC_ISSUER_URL`
  - `OIDC_CLIENT_ID`
  - `OIDC_CLIENT_SECRET`
  - `OIDC_AUDIENCE` or `OIDC_CLIENT_ID`
  - `OIDC_JWKS_URL`
- Managed upload storage must be configured. The readiness logic now accepts:
  - `SUPPLIER_UPLOAD_STORAGE_PROVIDER=s3` with complete S3-compatible settings;
  - `SUPPLIER_UPLOAD_STORAGE_PROVIDER=supabase` with:
    - `SUPABASE_EVIDENCE_BUCKET`
    - `SUPABASE_UPLOAD_QUARANTINE_BUCKET`
    - `SUPABASE_UPLOAD_CLEAN_BUCKET`

## What Changed

Supabase is now an accepted managed storage provider for staging/production
readiness, but only when the required Supabase bucket variables exist.

The current Supabase staging bucket names are:

- `supplier-evidence-staging`
- `supplier-uploads-quarantine-staging`
- `supplier-uploads-clean-staging`

This does not weaken production checks:

- local/mock storage still fails staging/production readiness;
- missing Supabase buckets still degrade readiness;
- scanner/quarantine evidence remains a separate control;
- OIDC and CORS remain required external controls.

## Next Validation Step

Set the missing Render/Supabase/OIDC env vars, then run:

```powershell
.\venv\Scripts\python.exe scripts\validate_managed_staging.py
.\venv\Scripts\python.exe scripts\collect_managed_staging_evidence.py
```

Only move to **Staging-ready with external controls validated** after the
artifact includes real API, OIDC tenant, Supabase DB, storage, scanner,
observability, and backup/restore evidence.

Current milestone label:

**Staging API live with Supabase Postgres validated; overall readiness still
degraded pending OIDC, CORS, storage readiness, scanner, backup/restore, and
observability.**
