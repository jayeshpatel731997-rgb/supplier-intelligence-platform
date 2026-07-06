# Supabase Staging Setup

## Scope

This project can use Supabase as the managed Postgres/Auth/Storage provider for
staging. The application still uses SQLAlchemy/Postgres for persistence; it does
not use Supabase client libraries.

## Current Observed Staging Evidence

As of the current staging check, the Render API is live and `/health` reports
`status: ok`. Database health passes using a Supabase Postgres pooler host
ending in `pooler.supabase.com`.

Do not commit or paste the database URL. Keep it in Render/Supabase secret
settings only.

## Required Render Variables

Database:

```text
SUPPLIER_DATABASE_URL=<Supabase pooler Postgres URL>
DATABASE_URL=<same Supabase pooler Postgres URL, if needed by provider tooling>
```

Use the Supabase **Session pooler** for Render runtime. `DATABASE_URL` and
`SUPPLIER_DATABASE_URL` should point to the same redacted Supabase pooler host.
Old Render Postgres is no longer the intended app database.

Runtime posture:

```text
SUPPLIER_SECURITY_MODE=production
SUPPLIER_DEPLOYMENT_MODE=render-staging-phase1
SUPPLIER_DEMO_MODE=false
CORS_ALLOW_ORIGINS=https://supplier-intelligence-ui-hut2.onrender.com
AUTH_PROVIDER=oidc
AUTH_ALLOW_LOCAL_IN_PRODUCTION=false
```

OIDC/Auth remains required before `/ready` can pass:

```text
OIDC_ISSUER_URL=<supabase-or-idp-issuer>
OIDC_CLIENT_ID=<client-id-or-audience>
OIDC_CLIENT_SECRET=<secret-managed-value>
OIDC_AUDIENCE=<api-audience-if-different>
OIDC_JWKS_URL=<jwks-url>
OIDC_ALGORITHMS=RS256
OIDC_CLOCK_SKEW_SECONDS=60
```

Storage readiness accepts either `s3` or `supabase`. For Supabase Storage:

```text
SUPPLIER_UPLOAD_STORAGE_PROVIDER=supabase
SUPABASE_EVIDENCE_BUCKET=supplier-evidence-staging
SUPABASE_UPLOAD_QUARANTINE_BUCKET=supplier-uploads-quarantine-staging
SUPABASE_UPLOAD_CLEAN_BUCKET=supplier-uploads-clean-staging
```

The buckets represent separate evidence, quarantine, and clean-upload storage
zones. They do not prove malware scanning by themselves.

## Scanner/Quarantine Boundary

Supabase Storage readiness is separate from upload malware/content scanning.
Real staging still needs a scanner provider and endpoint before malware scanning
can be claimed:

```text
SUPPLIER_UPLOAD_SCANNER_REQUIRED=true
SUPPLIER_UPLOAD_SCANNER_PROVIDER=<scanner-provider>
SUPPLIER_UPLOAD_SCANNER_ENDPOINT_URL=<scanner-endpoint>
```

Until a real scanner is configured and tested, scanner evidence remains
incomplete even if Supabase buckets are configured.

## Validation Commands

Run non-destructive validation from a trusted local shell:

```powershell
$env:STAGING_API_URL="https://<render-api-host>"
$env:STAGING_API_TOKEN="<short-lived-token>"
$env:STAGING_EXPECTED_TENANT_ID="<tenant-id>"
$env:POSTGRES_URL="<Supabase pooler Postgres URL>"
$env:STAGING_DB_READONLY_APPROVED="true"
$env:SUPPLIER_UPLOAD_STORAGE_PROVIDER="supabase"
$env:SUPABASE_EVIDENCE_BUCKET="supplier-evidence-staging"
$env:SUPABASE_UPLOAD_QUARANTINE_BUCKET="supplier-uploads-quarantine-staging"
$env:SUPABASE_UPLOAD_CLEAN_BUCKET="supplier-uploads-clean-staging"
.\venv\Scripts\python.exe scripts\validate_managed_staging.py
```

The validator redacts secrets and performs read-only database checks only when
`STAGING_DB_READONLY_APPROVED=true` is set.

## UI + CORS Next Step

The next manual Render setting is:

```text
CORS_ALLOW_ORIGINS=https://supplier-intelligence-ui-hut2.onrender.com
```

Manual validation sequence:

1. Resume the Render UI service `supplier-intelligence-ui-hut2`.
2. Open `https://supplier-intelligence-ui-hut2.onrender.com` and confirm the
   Streamlit page loads.
3. Set API service `CORS_ALLOW_ORIGINS` to
   `https://supplier-intelligence-ui-hut2.onrender.com`.
4. Redeploy the API service.
5. Retest `/health` and `/ready`.
6. Run the read-only UI/CORS smoke:

```powershell
$env:STAGING_UI_URL="https://supplier-intelligence-ui-hut2.onrender.com"
$env:STAGING_API_URL="https://supplier-intelligence-api-hut2.onrender.com"
.\venv\Scripts\python.exe scripts\smoke_ui_cors.py
```

The script verifies the UI page, API `/health`, and structured API `/ready`
JSON. It does not require login credentials.

## Optional Supabase Storage Live Check

Bucket configuration is not the same as live object proof. The managed staging
validator can perform a reversible live Supabase Storage check only when
explicitly approved:

```powershell
$env:SUPABASE_STORAGE_WRITE_APPROVED="true"
$env:SUPABASE_URL="<supabase-project-url>"
$env:SUPABASE_SERVICE_ROLE_KEY="<backend-only-service-role-key>"
$env:SUPPLIER_UPLOAD_STORAGE_PROVIDER="supabase"
$env:SUPABASE_EVIDENCE_BUCKET="supplier-evidence-staging"
$env:SUPABASE_UPLOAD_QUARANTINE_BUCKET="supplier-uploads-quarantine-staging"
$env:SUPABASE_UPLOAD_CLEAN_BUCKET="supplier-uploads-clean-staging"
.\venv\Scripts\python.exe scripts\validate_managed_staging.py
```

When approved, the validator uploads a harmless text object under a
`staging-readiness/` prefix, reads it back, and deletes it. This proves live
bucket access only. Malware scanning, quarantine workflow, and clean-bucket
promotion remain separate controls.

## Readiness Label

Use **Conditional go for managed staging** until CORS, OIDC, Supabase storage
buckets, and scanner/observability/backup evidence are complete. Do not call
this production-ready.

Final milestone wording for the current Render/Supabase slice:

**Staging API live with Supabase Postgres validated; overall readiness still
degraded pending OIDC, CORS, storage readiness, scanner, backup/restore, and
observability.**
