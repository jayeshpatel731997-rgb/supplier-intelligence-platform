# AGENTS.md

Guidance for future Codex runs in `supplier-intelligence-platform`.

## Repo Verification

Start every substantial task by confirming the repository and branch:

```powershell
pwd
git remote -v
git branch --show-current
git status --short
git ls-files | Select-Object -First 100
Get-ChildItem -Directory -Recurse -Depth 1 | Sort-Object FullName | Select-Object -First 100 -ExpandProperty FullName
```

Expected platform folders include `backend`, `src`, `tests`, `alembic`, `scripts`, `data`, `docs`, `.github/workflows`, `agents`, and `load_tests`. Stop and report the tree if this is `supplier-decision-intelligence` or a thin repo with only `app.py` and `requirements.txt`.

## Engineering Rules

- Preserve existing Streamlit, FastAPI, agent, model, database, migration, deployment, docs, and test files unless the user explicitly asks for removal.
- Keep changes small, testable, and production-minded. This repo still has research/prototype value; do not rewrite it into a different app.
- Do not hardcode secrets. Use environment variables, Streamlit secrets, or the existing secret-provider abstractions.
- Do not claim production readiness. Prefer "pilot", "staging", or "pre-production" unless enterprise controls have been implemented and verified.
- Keep local demo mode working with SQLite and the seeded `demo-tenant` / `demo-api-key`.
- Prefer existing service/repository patterns under `src/` and existing analytics logic under `models/`, `agents/`, `data_ingestion.py`, and `news_intelligence.py`.
- Update docs when behavior, environment variables, deployment steps, or verification commands change.

## Key Runtime Surfaces

- Streamlit app: `streamlit run app.py`
- FastAPI app: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`
- Worker: `python -m backend.worker`
- Docker Compose: `docker compose up --build`
- FastAPI health: `/live`, `/health`, `/ready`, `/system/status`
- Protected local API headers: `X-Tenant-ID: demo-tenant`, `X-API-Key: demo-api-key`

## Security And Config Notes

- `SUPPLIER_SECURITY_MODE=production` should use Postgres, disable demo mode, and use OIDC/SAML unless local auth is explicitly allowed.
- Uploads are bounded by `SUPPLIER_MAX_UPLOAD_BYTES` and `SUPPLIER_ALLOWED_UPLOAD_EXTENSIONS`.
- Rate limiting is controlled by `RATE_LIMIT_ENABLED`, `RATE_LIMIT_REQUESTS`, and `RATE_LIMIT_WINDOW_SECONDS`.
- Render/Postgres deployments should set `SUPPLIER_DATABASE_URL`; do not rely on SQLite outside local demos.
- Live Sentinel requires `NEWSAPI_KEY` plus either `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`; missing keys must not crash scans.

## Verification

Use the repo virtualenv on Windows when available:

```powershell
.\venv\Scripts\python.exe -m compileall .
.\venv\Scripts\python.exe -m unittest discover -s tests -t . -v
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\ruff.exe check .
docker compose config
```

If the system Python misses dependencies, switch to `.\venv\Scripts\python.exe`. If Docker, secrets, Postgres, Redis, or external APIs are unavailable, report the blocker and still run code-level checks.

## Final Response Format

For production-readiness tasks, include:

1. Correct repo/branch verified
2. Files changed
3. Production gap addressed by each change
4. Commands/tests run
5. Passed and failed checks
6. Remaining production gaps
7. Next 3 recommended Codex tasks
8. Manual Render/Postgres/secrets setup needed
