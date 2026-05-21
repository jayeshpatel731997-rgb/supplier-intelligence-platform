# Deployment

## Local Demo

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

set SUPPLIER_SECURITY_MODE=local
set SUPPLIER_DEMO_MODE=true
set SUPPLIER_DATABASE_URL=sqlite:///data/production_app.db

streamlit run app.py
```

FastAPI:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Demo API auth:

```bash
curl -H "X-Tenant-ID: demo-tenant" -H "X-API-Key: demo-api-key" http://localhost:8000/suppliers
```

Worker:

```bash
python -m backend.worker
```

## Docker Compose

```bash
docker compose up --build
```

Services:

- Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000`
- API health: `http://localhost:8000/health`
- Postgres: `localhost:5432`
- Redis: `localhost:6379`

## Render Staging

The repo includes `render.yaml` for a GitHub-backed Render Blueprint. It creates:

- FastAPI backend
- Streamlit command center
- Render Postgres
- Render Key Value for Redis-compatible Celery broker
- Celery worker
- Cron jobs that enqueue Sentinel/risk/exposure tasks

Runbook:

```bash
git push origin codex/production-foundation
```

Then open:

```text
https://dashboard.render.com/blueprint/new?repo=https://github.com/jayeshpatel731997-rgb/supplier-intelligence-platform
```

Fill `SUPPLIER_DEMO_API_KEY` and `SUPPLIER_APP_ADMIN_PASSWORD` with strong staging secrets. See `RENDER_STAGING_RUNBOOK.md` and `RENDER_ENV_CHECKLIST.md`.

## Postgres Configuration

Set:

```bash
SUPPLIER_DATABASE_URL=postgresql+psycopg://supplier_app:<password>@<host>:5432/supplier_intelligence
```

The current implementation creates tables automatically with SQLAlchemy. Before enterprise deployment, add Alembic migration management and a controlled migration run step.

## NewsAPI / Anthropic / OpenAI

Demo mode does not require paid APIs.

For live Sentinel mode:

```bash
NEWSAPI_KEY=...
OPENAI_API_KEY=...
# or
ANTHROPIC_API_KEY=...
```

If keys are missing or an API fails, Sentinel returns a safe error, records an alert, and does not crash the app.

## Production Mode

Production mode disables unsafe default admin creation:

```bash
SUPPLIER_SECURITY_MODE=production
SUPPLIER_DEMO_MODE=false
SUPPLIER_DATABASE_URL=postgresql+psycopg://...
```

If no Streamlit pilot user exists, the login screen shows a first-admin setup form. Use a strong password.

## Alembic

An Alembic initial metadata migration is included. Local/demo mode still supports automatic table creation for SQLite compatibility.

```bash
alembic upgrade head
python scripts/validate_tenant_schema.py
```

Local fallback:

```bash
python scripts/migrate.py --create-all-fallback
```

Celery-ready worker:

```bash
docker compose --profile celery up --build celery-worker
```
