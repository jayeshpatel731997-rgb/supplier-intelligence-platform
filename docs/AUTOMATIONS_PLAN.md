# Codex Automation Operations

Recurring Codex checks are advisory. They must not deploy, migrate a live
database, rotate credentials, upload repository contents, or print secrets.

## Weekly Staging Readiness Sweep

Inspect deployment configuration, migrations, health endpoints, smoke coverage,
auth planning, and staging docs. Run local compile, Pytest, Ruff, and
`git diff --check`. Report blockers and manual credential steps without
mutating staging.

## Nightly Secret/Security Scan

Inspect tracked and untracked changes, environment examples, docs, logs,
readiness payloads, tests, and deployment files for secret leakage and concrete
security regressions. Run `python scripts/check_secret_leakage.py`; it reports
only file, line, and secret category. Redact findings and keep repository
content local.

## Release Smoke Test Checklist

Prepare the exact pre-release checklist from current code and docs. Verify local
checks and list approved `STAGING_API_BASE_URL`, auth, migration, backup, seed,
readiness, and smoke commands. Do not run remote mutations without approval.

## Weekly Dependency/Plugin Health Check

Verify trusted Codex skills/plugins needed by this repository still load,
AutoReview can dry-run, and local dependencies have no obvious configuration
breakage. Do not install or update packages automatically.

Real staging tests still require an operator-approved database/API target and
secret-manager values.
