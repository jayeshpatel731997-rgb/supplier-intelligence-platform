# Supabase Auth And OIDC Plan

## Goal

Use a real identity provider for managed staging without weakening tenant
authorization. `/ready` should stay degraded until real identity settings are
configured and validated.

## Option A: Supabase Auth JWT Validation

Supabase Auth can issue JWTs for authenticated users. The API can validate
those JWTs by configuring the OIDC/JWT settings exposed by Supabase:

```text
AUTH_PROVIDER=oidc
OIDC_ISSUER_URL=<supabase-auth-issuer>
OIDC_CLIENT_ID=<api-audience-or-client-id>
OIDC_CLIENT_SECRET=<secret-if-required-by-flow>
OIDC_AUDIENCE=<api-audience-if-different>
OIDC_JWKS_URL=<supabase-jwks-url>
```

Validation evidence needed:

- token signature validates against the Supabase JWKS;
- `aud`, `iss`, expiry, and algorithm checks pass;
- API maps the authenticated subject/email to a tenant membership;
- `X-Tenant-ID` header overrides cannot cross tenants;
- missing/expired/wrong-audience tokens are rejected.

## Option B: External OIDC Provider

An external IdP such as Auth0, Okta, Entra ID, or WorkOS can issue tokens while
Supabase remains the Postgres/Storage platform. The same OIDC variables are
required, using the external IdP issuer and JWKS.

Validation evidence needed is the same: signature, issuer, audience, expiry,
tenant mapping, cross-tenant denial, and negative-token tests.

## Identity Is Not Authorization

OIDC or Supabase Auth proves identity. It does not by itself prove tenant
authorization. Tenant isolation must remain enforced by:

- application tenant membership checks;
- role checks for privileged operations;
- database constraints and query scoping;
- optional Supabase/Postgres Row Level Security for defense in depth.

Do not mark OIDC/Auth ready until real tokens and tenant sync are validated in
managed staging.
