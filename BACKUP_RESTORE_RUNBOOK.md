# Backup and Restore Runbook

This is readiness documentation, not proof that production backups are operational.

## SQLite local backup

```powershell
python scripts/export_sqlite_backup.py --output-dir backups
python scripts/verify_backup.py backups\your-backup.db
```

## Postgres backup

```powershell
.\scripts\backup_postgres.ps1
```

## Postgres restore

```powershell
.\scripts\restore_postgres.ps1 -BackupPath backups\postgres.dump
```

## Restore drill checklist

- Confirm backup exists and is encrypted at rest.
- Restore into an isolated database.
- Run `python scripts/validate_tenant_schema.py`.
- Run API health checks and tenant isolation tests.
- Record the drill as backup evidence.

Placeholder targets: RPO 24 hours, RTO 4 hours for internal pilot.
