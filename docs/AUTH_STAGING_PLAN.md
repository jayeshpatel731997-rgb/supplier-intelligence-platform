# Staging Authentication Plan

This plan moves the Supplier Intelligence Platform from local pilot credentials
to verified staging OIDC/JWT authentication without breaking local demo mode.

## Current Modes

- Local pilot: `AUTH_PROVIDER=local`, SQLite is allowed, and protected API calls
  use `X-Tenant-ID` plus `X-API-Key`. The seeded
  `demo-tenant` / `demo-api-key` pair is local-only.
- Staging: `SUPPLIER_SECURITY_MODE=production`,
  `SUPPLIER_DEPLOYMENT_MODE=staging`, `SUPPLIER_DEMO_MODE=false`,
  `AUTH_PROVIDER=oidc`, and `AUTH_ALLOW_LOCAL_IN_PRODUCTION=false`.
- Current scope: OIDC verifies FastAPI bearer tokens. Streamlit still uses the
  pilot/local login; browser redirect/callback SSO remains unimplemented.
- Emergency staging exception: local API-key auth requires a time-bounded,
  approved change with a non-default key, named owner, audit record, expiration
  time, and rollback command.

## Required OIDC Configuration

Store values in the deployment secret manager, not Git:

```text
OIDC_ISSUER_URL
OIDC_CLIENT_ID
OIDC_CLIENT_SECRET
OIDC_AUDIENCE
OIDC_JWKS_URL
OIDC_ALGORITHMS=RS256
OIDC_CLOCK_SKEW_SECONDS=60
```

`OIDC_AUDIENCE` may be omitted only when the API audience equals
`OIDC_CLIENT_ID`. The current API verifies bearer-token signature, issuer,
audience, allowed algorithm, expiry, and not-before time against JWKS.

## Tenant And Role Mapping

Map an immutable IdP organization claim to `tenant_id` or `org_id`. Do not
accept `X-Tenant-ID` as an authorization override in OIDC mode.

Map the token subject or verified email to an existing active local membership.
The database membership is authoritative for the role after token validation;
token roles cannot grant access to an unknown tenant or user.

Supported roles are `platform_admin`, `org_admin`, `risk_manager`, `analyst`,
`viewer`, and `auditor`.

Default unmapped identities to no access. Do not auto-provision a tenant from an
untrusted claim. Later SCIM or just-in-time provisioning must require signed
provider events, allowlisted organizations, idempotency, deprovisioning, and
audit evidence.

## Rollout

1. Create a staging IdP application and API audience.
2. Configure the API audience, issuer, JWKS, and trusted CORS origin. Obtain a
   short-lived API bearer token through the IdP's supported operator flow.
3. Create a dedicated staging tenant and least-privilege memberships.
   For the deterministic seed, set `SUPPLIER_STAGING_SEED_USERNAME` to the
   token subject or verified email; OIDC staging does not require an API key.
4. Set OIDC variables and keep local auth disabled.
5. Verify `/live` is `200`; verify `/ready` remains `503` until all database,
   storage, runtime, and auth requirements are complete.
6. Test a valid token for each role and verify tenant-scoped access.
7. Run `scripts/smoke_staging.py` with `STAGING_API_BASE_URL`,
   `STAGING_UI_BASE_URL`, `STAGING_BEARER_TOKEN`, and
   `STAGING_EXPECTED_TENANT_ID`.
8. Review auth failure and protected-action audit events before opening staging.

## Required Tests

- Missing bearer token returns `401`.
- Invalid signature, issuer, audience, algorithm, expiry, or `nbf` returns
  `403` without token details.
- Unknown or inactive tenant, missing membership, and inactive membership are
  denied.
- Cross-tenant claims and `X-Tenant-ID` overrides cannot expose data.
- Role permissions cover supplier, risk, evidence, actions, audit export,
  API-key management, and tenant administration.
- Auth failures and administrative changes create redacted audit events with
  request/correlation IDs.
- `/health`, `/live`, and `/ready` never expose JWTs, client secrets, JWKS
  payloads, API keys, or database credentials.
- Local demo API-key behavior still passes with `AUTH_PROVIDER=local`.

## Audit And Operations

Record provider, subject, tenant, effective database role, action, target,
outcome, request ID, and timestamp. Do not log bearer tokens, raw claims, API
keys, client secrets, or sensitive evidence payloads.

Monitor invalid-token rate, unknown-tenant attempts, denied cross-tenant
requests, membership and role changes, API-key creation, and emergency
local-auth activation.

## Rollback

1. Remove staging traffic or place the environment in maintenance mode.
2. Restore the last known-good IdP and application configuration.
3. Revert deployment configuration and redeploy.
4. Do not enable local auth unless the approved emergency exception is active.
5. If required, create a new non-default staging key, set
   `AUTH_PROVIDER=local` and `AUTH_ALLOW_LOCAL_IN_PRODUCTION=true`, verify
   tenant scope, document expiration, and rotate/revoke the key afterward.
6. Rerun readiness and authenticated smoke tests before restoring traffic.

This plan is a staging control design, not proof that a real IdP has been
configured or tested. It also does not claim Streamlit browser SSO support.
