# Worker Architecture

The worker layer is Celery/Redis-ready with local fallback.

## Modes

- `WORKER_MODE=local`: APScheduler-compatible local scheduler and synchronous task runner.
- `WORKER_MODE=celery`: Celery app configured from `REDIS_URL`; run a Celery worker in production-like deployments.

## Tasks

- `sentinel_scan_task`
- `risk_recalculation_task`
- `exposure_recalculation_task`
- `retention_cleanup_task`
- `audit_export_task`
- `backup_metadata_task`

Each task writes `background_job_runs` with tenant, run id, task name, status, timestamps, duration, retry count, and correlation id. Failures create a tenant-scoped alert and audit log.

## Run locally

```powershell
python -m backend.worker
```

## Run Celery

```powershell
docker compose --profile celery up --build celery-worker
```
