# Streamlit OIDC Browser Auth Boundary

## Current State

FastAPI protected routes already support OIDC bearer-token validation. The API
verifies issuer, audience, algorithm, expiry/nbf, and JWKS signature, then maps
the principal to a database membership. The database membership remains
authoritative for tenant and role authorization.

Streamlit currently uses the internal pilot/local login. It can call a separate
API with `SUPPLIER_API_BASE_URL`, but it does not yet implement a full browser
OIDC redirect/callback session.

## Staging-Safe Scaffold

Use `.streamlit/secrets.toml.example` as the local template. Real values belong
only in `.streamlit/secrets.toml`, Render environment variables, or the chosen
secret manager.

The intended browser flow is:

1. User clicks "Sign in with IdP" in Streamlit.
2. Streamlit redirects to the IdP authorization endpoint using PKCE.
3. IdP redirects back to `OIDC_REDIRECT_URI`.
4. Streamlit exchanges the authorization code server-side.
5. Streamlit stores only a short-lived session reference, not raw tokens in
   logs or downloadable artifacts.
6. API calls use the bearer token, and FastAPI enforces tenant membership from
   the database.

## Important Security Boundary

OIDC authenticates identity. It does not by itself authorize access to supplier
data.

Tenant authorization must remain enforced by application and database logic:

- Reject missing, inactive, or unknown tenants.
- Ignore `X-Tenant-ID` as an override in OIDC mode.
- Require active tenant membership before returning supplier, evidence, upload,
  alert, job, or audit data.
- Log denied access with request/correlation identifiers without recording
  token values.

## Evidence Status

Current proof is mock/staging-safe auth plumbing:

- `tests/test_oidc_auth.py` proves FastAPI token validation and tenant-boundary
  behavior with test JWKS data.
- No real IdP tenant, browser redirect, token exchange, MFA enforcement, or
  production session store has been validated in this repository.

Do not call the Streamlit browser flow production SSO until a real IdP is
configured and a staged end-to-end login is captured.
