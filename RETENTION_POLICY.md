# Retention Policy

Retention cleanup is disabled by default.

```text
RETENTION_ENABLED=false
RETENTION_DAYS=365
AUDIT_RETENTION_DAYS=2555
BACKUP_RETENTION_DAYS=35
```

Production deletion requires legal review, customer contracts, backup retention alignment, and database-level controls. Local cleanup runs as dry-run only.
