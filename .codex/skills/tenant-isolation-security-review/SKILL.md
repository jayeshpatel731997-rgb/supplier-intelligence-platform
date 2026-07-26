---
name: tenant-isolation-security-review
description: Audit Supplier Intelligence Platform tenant isolation across X-Tenant-ID handling, authentication, tenant-scoped repositories, jobs, uploads, exports, readiness surfaces, and audit logs. Use after API, repository, worker, auth, migration, or multi-tenant changes.
---

# Tenant Isolation Security Review

Assume every externally controlled identifier is hostile until authorization and ownership are proven.

## Workflow

1. Verify repository, branch, diff, tenancy documentation, auth mode, and local demo behavior.
2. Trace `X-Tenant-ID` and authenticated tenant identity from FastAPI dependencies through services and repositories.
3. Confirm the caller cannot select or override another tenant through headers, query parameters, request bodies, job payloads, filenames, export filters, or cached state.
4. Review every affected query, update, delete, aggregate, background job, and audit lookup for mandatory tenant predicates and tenant-owned object validation.
5. Test missing, invalid, mismatched, and cross-tenant identifiers with at least two tenants. Check list, detail, mutation, export, upload, and job-status paths.
6. Verify production mode disables demo bypasses and that audit logs record actor, tenant, action, target, result, and correlation data without sensitive payloads.
7. Report concrete leakage or bypass paths first, then missing tests and residual risks.

## Safety

Use synthetic tenants and data. Do not access real customer records, print credentials, or weaken authentication to make tests pass.
