# Access Review Procedure

1. Export tenant memberships.
2. Export recent auth/audit activity.
3. Open an access review record.
4. Org admin confirms each active account and role.
5. Remove or downgrade stale access.
6. Close review and retain evidence.

Command:

```powershell
python scripts/collect_evidence.py --tenant-id demo-tenant
```
