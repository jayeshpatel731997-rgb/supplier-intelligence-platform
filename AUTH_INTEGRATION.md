# Auth Integration

Local auth remains available for demo/dev. Enterprise auth is provider-agnostic and ready to connect to OIDC/SAML/SCIM providers.

## Config

```text
AUTH_PROVIDER=local|oidc|saml
AUTH_ALLOW_LOCAL_IN_PRODUCTION=false
OIDC_ISSUER_URL=
OIDC_CLIENT_ID=
OIDC_CLIENT_SECRET=
OIDC_AUDIENCE=
OIDC_REDIRECT_URI=
SAML_METADATA_URL=
SAML_METADATA_FILE=
SCIM_ENABLED=false
MFA_REQUIRED=false
```

OIDC claim mapping supports `sub`, `email`, `name`, `tenant_id` or `org_id`, and `role` or `roles`/`groups`.

## Provider notes

- Okta/Auth0/Azure AD/Google Workspace/WorkOS can map groups to tenant roles.
- MFA should be enforced at the identity provider first.
- SCIM is a provisioning scaffold; production needs provider-specific lifecycle handlers.

Production mode rejects unsafe local defaults unless explicitly allowed.
