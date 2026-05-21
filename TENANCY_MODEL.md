# Tenancy Model

V1 uses shared database, shared schema multi-tenancy. Business tables carry `tenant_id`; repositories and services require tenant context and filter by `tenant_id`.

API requests use:

```text
X-Tenant-ID: demo-tenant
X-API-Key: demo-api-key
```

The model is designed so high-value tenants can later move to isolated databases or pods while preserving logical tenant ids.
