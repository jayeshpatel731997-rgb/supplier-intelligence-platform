# Compliance Readiness

This project does not claim SOC 2 compliance. It now includes evidence and workflow scaffolding for readiness.

## Implemented stubs

- Tenant-scoped audit log export.
- Access review records.
- Backup metadata records.
- Retention dry-run service.
- Evidence collection script.
- SIEM/WORM export placeholders.

## Evidence commands

```powershell
python scripts/export_audit.py --tenant-id demo-tenant --format jsonl
python scripts/collect_evidence.py --tenant-id demo-tenant
```

## Still required

Auditor-approved policies, real WORM storage, SIEM integration, change history integration, vulnerability management, penetration testing, incident exercises, vendor review, legal/privacy/DPA review, and restore drills.
