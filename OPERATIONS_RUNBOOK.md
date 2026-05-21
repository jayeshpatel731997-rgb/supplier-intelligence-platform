# Operations Runbook

## Health Checks

FastAPI:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/system/status
```

Streamlit:

```bash
curl http://localhost:8501/_stcore/health
```

## Background Jobs

Run a job manually:

```bash
curl -X POST http://localhost:8000/background/jobs/sentinel_scan/run
curl -X POST http://localhost:8000/background/jobs/risk_recalculate/run
curl -X POST http://localhost:8000/background/jobs/exposure_recalculate/run
```

Inspect jobs:

```bash
curl http://localhost:8000/background/jobs
```

## Common Alerts

- `supplier_high_risk`: supplier risk exceeded threshold.
- `new_disruption_event`: Sentinel found high-severity or high-exposure event.
- `sentinel_api_failure`: external Sentinel dependency failed or was not configured.
- `background_job_failure`: scheduled job failed.

## Incident Response

1. Check `/system/status`.
2. Inspect Streamlit `Alerts & Health`.
3. Review `audit_logs` and `background_job_runs`.
4. Acknowledge known alerts.
5. Resolve root cause.
6. Record incident notes outside the app until ticketing integration exists.

## Safe Failure Rules

- Missing NewsAPI, OpenAI, or Anthropic keys must not crash the app.
- API failures create alerts and return safe empty event results.
- Production mode must not seed default admin credentials.
- Viewer role is read-only in Streamlit.

## Enterprise Readiness Operations

```bash
python scripts/migrate.py --create-all-fallback
python scripts/backfill_tenants.py
python scripts/validate_tenant_schema.py
python scripts/export_audit.py --tenant-id demo-tenant --format jsonl
python scripts/collect_evidence.py --tenant-id demo-tenant
python scripts/export_sqlite_backup.py --output-dir backups
```

Run Celery-ready worker profile:

```bash
docker compose --profile celery up --build celery-worker
```

Run load tests:

```bash
locust -f load_tests/locustfile.py --host http://localhost:8000
```
