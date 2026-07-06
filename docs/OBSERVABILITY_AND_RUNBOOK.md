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
- Render logs screenshot for API startup and recent request handling.
- Render metrics screenshot for API and UI service CPU/memory/restarts.
- Request ID or correlation ID in at least one API response/log pair.
- Log drain or alert-routing plan, including owner and escalation target.
- Rollback proof showing the previous known-good deploy can be selected.

Current observed live-log evidence:

- Render API live logs showed repeated `GET /live` requests returning `200 OK`.
- Treat an older Render "recent deploy failed" banner as historical unless the
  current `/health`, `/ready`, or live logs fail.

## Staging Smoke Checklist

1. Confirm commit SHA and branch.
2. Confirm Render API URL and Streamlit UI URL.
3. Resume `supplier-intelligence-ui-hut2` if it is suspended.
4. Set API `CORS_ALLOW_ORIGINS` to
   `https://supplier-intelligence-ui-hut2.onrender.com` and redeploy the API.
5. Run `scripts/smoke_ui_cors.py` with `STAGING_UI_URL` and `STAGING_API_URL`.
   Current live URLs:
   - `STAGING_UI_URL=https://supplier-intelligence-ui-hut2.onrender.com`
   - `STAGING_API_URL=https://supplier-intelligence-api-hut2.onrender.com`
   Treat `staging_api_cors_preflight=WARN` as a remaining CORS configuration
   gap even when the UI page and API health/readiness checks pass.
6. Run `scripts/smoke_staging.py` with a short-lived OIDC token.
7. Run `scripts/validate_managed_staging.py` with `STAGING_API_URL`,
   `STAGING_API_TOKEN`, `STAGING_EXPECTED_TENANT_ID`, and any approved Postgres,
   object-storage, scanner, or Render evidence variables.
8. Verify unauthenticated protected routes reject access.
9. Verify authenticated tenant-scoped supplier read.
10. Verify `X-Tenant-ID` override cannot cross tenant boundaries in OIDC mode.
11. Verify connector sync, evidence-chain run, action update, and scoring config.
12. Confirm smoke logs redact secrets.

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
