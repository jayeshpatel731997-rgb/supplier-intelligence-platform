---
name: postgres-migration-smoke
description: Review and smoke-test Supplier Intelligence Platform migrations against managed or staging Postgres, including Alembic ordering, seed idempotency, rollback notes, tenant schema validation, and staging checks. Use before database deployment or after schema changes.
---

# Postgres Migration Smoke

Verify schema changes without risking production data.

## Workflow

1. Verify repository, branch, target database environment, and migration scope.
2. Inspect `alembic.ini`, `alembic/env.py`, migration revisions, models, repositories, `MIGRATIONS.md`, and deployment commands for consistency.
3. Confirm a fresh database can upgrade to `head` and an existing supported schema can upgrade without destructive surprises.
4. Run the demo seed twice in a disposable database and verify stable counts, identifiers, and tenant ownership.
5. Run tenant schema validation and staging smoke tests against a disposable or explicitly approved staging database.
6. Review downgrade implementations or documented restore/forward-fix procedures for irreversible changes.
7. Record commands, revision before/after, sanitized results, failures, rollback notes, and residual data-migration risks.

## Safety

- Never run migrations against production without explicit approval and a verified backup/restore plan.
- Never print connection strings, passwords, tokens, or customer data.
- Prefer temporary databases or isolated schemas; do not reuse an ambiguous database target.
- Do not install database tools or drivers automatically.
