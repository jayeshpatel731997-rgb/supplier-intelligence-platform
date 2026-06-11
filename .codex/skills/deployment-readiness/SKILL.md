---
name: deployment-readiness
description: Assess Supplier Intelligence Platform readiness for pilot or staging deployment. Use before Render, Docker, or managed Postgres deployment work to check configuration, health endpoints, docs, seed scripts, smoke tests, Streamlit API settings, security, and rollback evidence.
---

# Deployment Readiness

Produce an evidence-backed `go`, `conditional go`, or `no-go` assessment. Do not claim production readiness.

## Workflow

1. Verify repository, branch, release scope, target environment, and deployment files.
2. Compare `.env.example`, `src/config.py`, Streamlit secrets/config, Docker Compose, Render manifests, and deployment docs for consistent variable names and safe defaults.
3. Verify `/live`, `/health`, `/ready`, and `/system/status` semantics. Readiness must fail safely when required dependencies are unavailable and must not expose secrets.
4. Review Postgres configuration, Redis/worker expectations, migration commands, backup/restore notes, seed scripts, and seed idempotency.
5. Verify Streamlit API base URL and authentication configuration support local demo and deployed API modes without hardcoded credentials.
6. Inspect `scripts/smoke_staging.py`, focused tests, CI, logging, rate limits, upload bounds, and rollback instructions.
7. Run build, test, lint, and Docker config checks locally. Before any migration, seed, or database smoke command, require an explicitly named disposable database or isolated schema, verify the target is not production, and obtain approval for remote staging mutations.

## Output

Report blockers, evidence, skipped checks, manual Render/Postgres/secrets setup, remaining gaps, and the next three tasks.

Do not modify live infrastructure, DNS, production data, or secret stores unless explicitly requested.
