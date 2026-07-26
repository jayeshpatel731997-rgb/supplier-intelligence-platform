# Supabase Security Model

## Scope

Supplier Intelligence Platform uses Supabase as the intended managed staging
provider for:

- Postgres via the Supabase Session pooler;
- Supabase Auth or another OIDC/JWT identity layer in a later validation step;
- private Supabase Storage buckets for evidence and upload lifecycle zones.

Render remains the runtime host for the FastAPI API and Streamlit UI services.
Old Render Postgres is no longer the intended application database for managed
staging.

## Database Boundary

Set both database variables on the Render API service to the same Supabase
Session pooler URI:

```text
SUPPLIER_DATABASE_URL=<Supabase Session pooler URI>
DATABASE_URL=<same Supabase Session pooler URI>
```

The application normalizes plain `postgresql://` URLs to
`postgresql+psycopg://` for SQLAlchemy/psycopg. Health and evidence logs must
show only the redacted URL host, never the password or full connection string.

## Key Handling

- `SUPABASE_SERVICE_ROLE_KEY` is backend/API only. Do not expose it to
  Streamlit frontend code, browser bundles, docs, screenshots, or artifacts.
- `SUPABASE_ANON_KEY` or a publishable key may be used only where public client
  access is explicitly intended and RLS/policies are configured accordingly.
- Supabase keys, database URLs, OIDC secrets, JWTs, and Render credentials must
  remain in Render/Supabase secret stores or local environment variables.

## Storage Boundary

Required private staging buckets:

```text
SUPABASE_EVIDENCE_BUCKET=supplier-evidence-staging
SUPABASE_UPLOAD_QUARANTINE_BUCKET=supplier-uploads-quarantine-staging
SUPABASE_UPLOAD_CLEAN_BUCKET=supplier-uploads-clean-staging
```

The readiness check accepts `SUPPLIER_UPLOAD_STORAGE_PROVIDER=supabase` only
when all three bucket variables exist. Bucket configuration proves managed
storage readiness only. It does not prove malware scanning, quarantine
transitions, object lifecycle policy, or retention policy.

## Identity And Tenant Boundary

Supabase Auth can be used later as the identity provider, but authentication is
not authorization. The app must still enforce tenant membership, roles, and
cross-tenant isolation in application code and/or database Row Level Security.

`/ready` should remain degraded until real OIDC/JWT values are configured and
validated:

```text
OIDC_ISSUER_URL=<issuer>
OIDC_CLIENT_ID=<client-id-or-audience>
OIDC_CLIENT_SECRET=<secret-managed-value>
OIDC_AUDIENCE=<api-audience-if-different>
OIDC_JWKS_URL=<jwks-url>
```

## Remaining External Proof

Before claiming external controls are validated, capture evidence for:

- trusted Render UI origin in `CORS_ALLOW_ORIGINS`;
- OIDC login/token validation and tenant sync;
- read-only Supabase Postgres smoke and backup/restore drill;
- Supabase Storage live bucket/object checks;
- real malware/content scanner reject/allow/quarantine behavior;
- Render log drains, metrics, alerts, rollback, and screenshots.
