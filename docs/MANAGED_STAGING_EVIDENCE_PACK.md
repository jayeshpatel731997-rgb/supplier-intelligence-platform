# Managed Staging Evidence Pack

## Purpose

This document explains how to generate and interpret the timestamped evidence
folder under `artifacts/managed-staging-readiness/`.

Run:

```powershell
.\venv\Scripts\python.exe scripts\collect_managed_staging_evidence.py
```

The script creates:

```text
artifacts/managed-staging-readiness/YYYYMMDD-HHMMSS/
```

with repo verification logs, test/lint/build logs, tool availability, optional
security scan output, local smoke/readiness checks when possible, and a final
Markdown report.

If no `package.json` exists, frontend checks are recorded as skipped. If
Playwright/browser automation is unavailable, screenshot capture is recorded as
skipped and the screenshot checklist in `docs/PROFESSOR_DEMO_GUIDE.md` remains
the manual fallback.

## Report Contents

Each generated `FINAL_REPORT.md` includes:

- verified repo state;
- files changed at collection time;
- tests and checks run;
- what is proven locally or in CI;
- what is mocked/staging-safe;
- what still requires external setup;
- final readiness label.

Allowed readiness labels:

- Conditional go for managed staging
- Staging-ready with mocks for external controls
- Staging-ready with external controls validated

The current default is **Conditional go for managed staging** unless real
external controls and staging smoke evidence are present.

## Real External Validation Inputs

Use `scripts/validate_managed_staging.py` when real staging services exist. It
accepts:

- `STAGING_API_URL` and `STAGING_API_TOKEN` for API health/readiness and
  authenticated status checks.
- `STAGING_EXPECTED_TENANT_ID` to verify the authenticated tenant boundary.
- `POSTGRES_URL`, `DATABASE_URL`, or `SUPPLIER_DATABASE_URL` plus
  `STAGING_DB_READONLY_APPROVED=true` for read-only managed Postgres evidence.
- `STAGING_BACKUP_RESTORE_APPROVED=true`,
  `POSTGRES_RESTORE_TARGET_URL`, and
  `STAGING_RESTORE_TARGET_CONFIRMED_DISPOSABLE=true` before any backup/restore
  drill is considered approved.
- `STAGING_S3_BUCKET`, `STAGING_S3_ENDPOINT_URL`, region/access settings, and
  `STAGING_OBJECT_STORAGE_VALIDATE=true` for a live object-storage reachability
  check.
- `STAGING_UPLOAD_SCANNER_PROVIDER` and
  `STAGING_UPLOAD_SCANNER_ENDPOINT_URL` for scanner configuration evidence.
- `RENDER_API_KEY` and `RENDER_SERVICE_ID`/`STAGING_RENDER_API_SERVICE_ID` for
  non-destructive Render service metadata evidence.

All output is redacted. Missing external credentials are recorded as skipped,
not silently treated as passing.

## Evidence Interpretation

Local passing tests prove repository behavior, not managed-service readiness.
GitHub CI proves a clean remote run for the pushed commit. Render, Postgres,
OIDC, object storage, scanner, observability, and backup/restore evidence must
be captured separately before claiming external controls are validated.
