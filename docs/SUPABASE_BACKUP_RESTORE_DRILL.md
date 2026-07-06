# Supabase Backup And Restore Drill

## Goal

Prove that managed staging data can be backed up and restored into an isolated
target without touching production data or leaking credentials.

## Connection Guidance

- Use the Supabase Session pooler for app runtime on Render.
- For `pg_dump`, `pg_restore`, and schema verification, use the Supabase direct
  connection string or the Supabase-recommended maintenance connection where
  appropriate.
- Do not paste database URLs into docs, logs, screenshots, or issue comments.

## Required Safety Gates

Before running a drill:

1. Confirm the source is staging, not production.
2. Create or identify a disposable restore target.
3. Set `STAGING_BACKUP_RESTORE_APPROVED=true`.
4. Set `STAGING_RESTORE_TARGET_CONFIRMED_DISPOSABLE=true`.
5. Redact commands and output before committing evidence.

## Evidence To Capture

- redacted `pg_dump` command and exit code;
- redacted restore command and exit code;
- schema/migration verification query output;
- tenant/table count sanity checks;
- restore duration;
- observed extension/version gaps;
- cleanup confirmation for the disposable target.

## Production-Readiness Boundary

Backup configuration alone is not production readiness. Production readiness
requires a successful restore drill, retention policy, access controls, alerting
on backup failures, and an owner/runbook for restore decisions.
