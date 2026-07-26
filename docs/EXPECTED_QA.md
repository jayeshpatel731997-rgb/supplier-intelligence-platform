# Expected Professor Q&A

## Is this production-ready?

No. The honest status is:

**API + UI live with Supabase Postgres validated; remaining production controls
pending.**

The staging system is live and evidence-backed, but `/ready` is degraded until
production controls such as OIDC/Auth, CORS hardening, scanner proof,
backup/restore, and observability are validated.

## Why does `/ready` return degraded if the API is live?

Because `/health` and `/ready` mean different things. `/health` shows the API
and database are functioning. `/ready` is stricter: it checks whether the
runtime has required production controls. It currently returns degraded because
OIDC/Auth settings are missing.

## What are the five current readiness issues?

The current degraded `/ready` issue count is 5, all related to OIDC/Auth:

- `OIDC_ISSUER_URL`
- `OIDC_CLIENT_ID`
- `OIDC_CLIENT_SECRET`
- `OIDC_AUDIENCE` or `OIDC_CLIENT_ID`
- `OIDC_JWKS_URL`

## What exactly is proven live?

- Render API is live.
- Render UI is live.
- API `/health` returns status OK.
- Supabase Postgres database health passes.
- API `/ready` returns structured degraded JSON.
- CI is green.
- Latest evidence pack includes 161 passing pytest tests.

## Why use evidence chains?

Supplier-risk recommendations need trust. A manager should not just see a risk
score; they should see the weak signals, timestamps, provenance, scoring
inputs, and recommended action path behind that score.

## What is mocked or staging-safe?

- Some demo data is seeded/synthetic.
- OIDC tests use test fixtures unless real IdP values are configured.
- Upload scanner tests use a staging-safe EICAR-style adapter.
- Supabase Storage live object validation is scaffolded but not executed
  without explicit approval and credentials.

## What would you build next?

First priority: real OIDC/Auth or Supabase Auth token validation with tenant
mapping. After that:

1. strict CORS preflight PASS;
2. Supabase Storage upload/read/delete proof;
3. malware scanner/quarantine workflow;
4. backup/restore drill;
5. observability, alerts, and rollback proof.

## How do you prevent cross-tenant data leakage?

The code has tenant-scoped repository patterns and tests around tenant boundary
behavior. The next production control is to connect real identity tokens to
tenant membership and verify that token claims cannot be overridden by headers.

## Why Render and Supabase?

Render gives straightforward API/UI hosting for a staging demo. Supabase gives
managed Postgres now and a path toward Auth and Storage later. This keeps the
architecture practical for a portfolio project while still showing managed
staging discipline.

## What is the biggest technical risk?

The biggest remaining risk is external control validation: identity, storage
scanning, backup/restore, and observability. The app is built to surface these
as readiness gaps instead of hiding them.

## What should I avoid saying?

Avoid:

- "production-ready";
- "fully secured";
- "real malware scanning is complete";
- "OIDC is configured";
- "all production controls are validated."

Say:

"The system is live in staging with API, UI, Supabase Postgres, CI, and tests
validated. Production controls remain pending and are explicitly documented."
