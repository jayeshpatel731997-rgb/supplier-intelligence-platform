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

## Evidence Interpretation

Local passing tests prove repository behavior, not managed-service readiness.
GitHub CI proves a clean remote run for the pushed commit. Render, Postgres,
OIDC, object storage, scanner, observability, and backup/restore evidence must
be captured separately before claiming external controls are validated.
