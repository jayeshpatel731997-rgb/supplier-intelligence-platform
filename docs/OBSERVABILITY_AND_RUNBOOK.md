# Observability And Staging Runbook

## Health And Readiness

- `/live` — process liveness; should not require database startup.
- `/health` — API and database health; may report degraded instead of crashing.
- `/ready` — traffic gate; returns `503` when startup, database, migrations, or
  production runtime requirements fail.
- `/system/status` — protected operational status with worker, Sentinel, auth,
  rate-limit, retention, SIEM, connector, and production-configuration posture.

## Logs And Correlation

Use request IDs/correlation IDs for API calls and background work. Logs should
include tenant ID, actor, action, target, result, and request ID when available.
Logs must not include tokens, passwords, API keys, database credentials, upload
contents, or full secret-bearing URLs.

## Metrics And Alerts Still Needed

Managed staging should capture:

- API latency, error rate, and readiness failures.
- Database connection health, migration version, and slow queries.
- Worker job success/failure counts and schedule lag.
- Sentinel connector degradation.
- Upload scanner allow/reject/error counts.
- Auth failures, tenant-boundary denials, and rate-limit events.
- Backup success/failure and restore-drill timestamp.

## Render Dashboard Evidence Needed

Capture screenshots or exported logs for:

- API and UI services deployed from the expected branch/commit.
- Managed Postgres attached to both services.
- Environment group configured without secret exposure.
- `/live`, `/health`, and `/ready` responses.
- Service logs showing clean startup and migration completion.
- Rollback target and previous successful deploy.

## Staging Smoke Checklist

1. Confirm commit SHA and branch.
2. Confirm Render API URL and Streamlit UI URL.
3. Run `scripts/smoke_staging.py` with a short-lived OIDC token.
4. Run `scripts/validate_managed_staging.py` with `STAGING_API_URL`,
   `STAGING_API_TOKEN`, `STAGING_EXPECTED_TENANT_ID`, and any approved Postgres,
   object-storage, scanner, or Render evidence variables.
5. Verify unauthenticated protected routes reject access.
6. Verify authenticated tenant-scoped supplier read.
7. Verify `X-Tenant-ID` override cannot cross tenant boundaries in OIDC mode.
8. Verify connector sync, evidence-chain run, action update, and scoring config.
9. Confirm smoke logs redact secrets.

## Backup/Restore Checklist

1. Confirm target database is staging or disposable, never production.
2. Run logical backup with `pg_dump`.
3. Restore into an isolated local or disposable target.
4. Verify schema, migration revision, tenant count, and key table counts.
5. Record redacted commands, exit codes, and verification output.
6. Document restore duration and any extension/version gaps.

## Rollback Checklist

1. Stop or pause new deploy traffic.
2. Restore previous Render service version or previous image/commit.
3. Verify `/live`, `/health`, and `/ready`.
4. Re-run staging smoke with auth.
5. If database migration is involved, prefer forward-fix or restore from tested
   backup according to `BACKUP_RESTORE_RUNBOOK.md`.
6. Record incident timeline and residual risk.
