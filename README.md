# Supplier Intelligence Platform

**Integrated Risk Assessment & Performance Decision Intelligence for Manufacturing Supply Networks**

## Overview

A unified platform that combines quantitative risk modeling with supplier performance evaluation, enabling manufacturing SMEs to:

- Assess multi-tier supply network risk using Bayesian scoring
- Simulate disruption cascades with SIR epidemic-adapted models
- Quantify financial exposure via Monte Carlo VaR/CVaR analysis
- Evaluate supplier performance with weighted multi-criteria scorecards
- Calculate Total Cost of Ownership including hidden quality and delivery costs
- Compare switching scenarios with payback period analysis
- Upload real supplier data and automatically generate a runnable network model
- Monitor portfolio-specific disruption intelligence through the Sentinel tab
- Operate a production-style FastAPI backend with database-backed suppliers, jobs, alerts, audit logs, and system health
- Run near-real-time scheduled Sentinel/risk/exposure refresh jobs with a local scheduler or worker process
- Scope SaaS data by tenant using tenant API keys in local/demo mode or verified OIDC bearer tokens in OIDC mode

## Architecture

| Module | Method | Purpose |
|---|---|---|
| Risk Dashboard | Bayesian posterior probability (6 evidence signals) | Supplier-level disruption risk scoring |
| Network Analysis | Graph centrality (Degree, Betweenness, PageRank) | Identify single points of failure |
| Scenario Engine | SIR propagation (Monte Carlo, 50+ runs) | Disruption cascade prediction |
| Financial Impact | Monte Carlo simulation (5,000+ iterations) | VaR/CVaR financial exposure |
| Performance Scorecard | Weighted multi-criteria scoring | Supplier ranking by strategic priority |
| Decision Intelligence | TCO analysis (COPQ, delivery cost, switching cost) | Financial trade-off quantification |
| Data Upload | Smart schema mapping + generated network topology | Replace demo data with a live supplier portfolio |
| Sentinel Agent | NewsAPI + OpenAI/Claude + local matching | Real-time disruption intelligence without sending supplier rows to LLMs |
| NASA PRA Upgrades | LHS, Weibull beta, Fault Tree Analysis | Stabilize tail-risk estimates and explain root drivers |
| FastAPI Backend | REST API + SQLAlchemy services | Integrations, monitoring, ingestion, alert acknowledgement |
| Background Worker | APScheduler + local fallback | Scheduled Sentinel scans and risk/exposure recalculation |

## Current App Scope

The Streamlit app currently exposes the original analytics tabs plus production command-center surfaces:

1. Risk Dashboard
2. Network Analysis
3. Scenario Engine
4. Financial Impact
5. Performance Scorecard
6. Decision Intelligence
7. Data Upload
8. Sentinel Agent
9. Production Command Center
10. Alerts & Health
11. Admin & Audit (admin role only)

When you upload a supplier file, the platform now uses that uploaded dataset across the full analytics flow. A generated network with inferred upstream links and scenarios is built automatically so the dashboard, graph analysis, cascade simulation, Monte Carlo tab, scorecard, decision tab, and Sentinel tab can all run from the same live portfolio.

## Quick Start

```bash
# Clone or copy project folder
cd supplier-intelligence-platform

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

Opens at **http://localhost:8501**

Run the API backend:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Protected API calls in local/demo mode use:

```bash
curl -H "X-Tenant-ID: demo-tenant" -H "X-API-Key: demo-api-key" http://localhost:8000/suppliers
```

When `AUTH_PROVIDER=oidc`, protected API routes require an OIDC JWT:

```bash
curl -H "Authorization: Bearer <id-or-access-token>" http://localhost:8000/suppliers
```

The token is verified against issuer, audience, algorithm, expiry/nbf with clock
skew, and the configured JWKS. The tenant and user must also exist as an active
local membership, so IdP claims cannot grant access to unknown tenants.

Health endpoints:

- `/live` returns a lightweight process liveness response and does not require database initialization.
- `/health` includes database and API state and reports `degraded` instead of crashing when database-backed status queries fail.
- `/ready` is the traffic gate: it returns HTTP `200` only when startup initialization, database checks, and production runtime checks pass; otherwise it returns HTTP `503` with `status: degraded`. It also reports database/backend mode, auth posture, connector mode, scoring config status, Convex configured/not-configured status, and governed narrative mode without exposing secret values.
- In production mode, readiness also blocks wildcard CORS, incomplete OIDC/JWKS/SAML configuration, missing schema/migrations, and other unsafe runtime defaults. API and worker startup do not auto-create production schema; run Alembic first.
- `/system/status` is protected and adds worker, Sentinel, auth, rate limit, retention, SIEM, and production configuration checks.

Supplier uploads are intentionally bounded for pilot deployments. Configure:

```bash
SUPPLIER_MAX_UPLOAD_BYTES=5000000
SUPPLIER_ALLOWED_UPLOAD_EXTENSIONS=.csv,.xlsx,.xls,.json
SUPPLIER_ALLOWED_UPLOAD_MIME_TYPES=text/csv,application/csv,text/plain,application/json,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
SUPPLIER_UPLOAD_STORAGE_PROVIDER=local
SUPPLIER_UPLOAD_STORAGE_PATH=data/uploads
```

FastAPI uploads are checked for safe filenames, tenant-scoped storage keys,
allowed extensions and MIME types when a useful MIME type is provided, and the
configured size limit before the existing CSV/Excel/JSON ingestion parser runs.
Local/demo mode stores upload objects under `data/uploads/<tenant>/...`.
Production mode fails `/ready` unless upload storage is configured as
S3-compatible object storage with bucket, endpoint, and credentials. If
`SUPPLIER_UPLOAD_SCANNER_REQUIRED=true`, production readiness also fails until a
scanner provider and endpoint are configured. The scanner interface is a safe
stub for now; do not treat it as malware inspection until connected to a real
scanner service.

Run the background worker:

```bash
python -m backend.worker
```

Docker Compose:

```bash
docker compose up --build
```

Streamlit opens at **http://localhost:8501** and FastAPI at **http://localhost:8000**.
The compose stack keeps them separate: `streamlit` builds the root `Dockerfile`
and runs `streamlit run app.py`; `backend` builds `backend/Dockerfile` and runs
`uvicorn backend.main:app`.

Seed deterministic local demo data:

```bash
python scripts/seed_demo_data.py --tenant-id demo-tenant
```

The seed is idempotent and creates demo suppliers, weak signals, connector sync
metadata, scoring config, an evidence-chain run, actions, and historical
outcome examples without requiring external APIs. For OIDC staging, set
`SUPPLIER_STAGING_SEED_USERNAME` to the token subject or verified email before
running the seed; it creates a `risk_manager` membership without creating a
staging API key.

Render staging:

```bash
git checkout main
git pull origin main
```

Then open:

```text
https://dashboard.render.com/blueprint/new?repo=https://github.com/jayeshpatel731997-rgb/supplier-intelligence-platform
```

The default `render.yaml` creates separate Render web services:
`supplier-intelligence-api` runs FastAPI with `uvicorn backend.main:app`, and
`supplier-intelligence-ui` runs Streamlit with `streamlit run app.py`.
`render.full.yaml` keeps the same API/UI split and adds Redis, worker, and cron
services. See `RENDER_STAGING_RUNBOOK.md` for the exact staging launch checklist.
Real Render staging runs with `SUPPLIER_SECURITY_MODE=production`,
`SUPPLIER_DEMO_MODE=false`, Alembic migrations, explicit CORS, complete auth
provider settings, and S3-compatible upload storage. `/live` is the API Render
process health check; `/ready` is the API configuration and database traffic gate.

Smoke test staging after deploy:

```bash
set STAGING_API_BASE_URL=https://supplier-intelligence-api.onrender.com
set STAGING_UI_BASE_URL=https://supplier-intelligence-ui.onrender.com
set STAGING_BEARER_TOKEN=<short-lived-oidc-token>
set STAGING_EXPECTED_TENANT_ID=demo-tenant
python scripts/smoke_staging.py
```

Use the FastAPI service URL for `STAGING_API_BASE_URL`; `STAGING_BASE_URL`
remains a compatible alias. If this accidentally points
at the Streamlit UI service, the smoke script fails when it sees HTML fallback
instead of API JSON/auth responses. With auth configured, the smoke script also
runs tenant-boundary, connector sync, evidence-chain run, action update, and
scoring-config checks. Credentials and the Streamlit URL are required by default
so workflow and UI checks cannot be silently skipped; use
`--health-only --skip-ui` only for an explicit limited infrastructure check.

Streamlit can target a separate local or staging API service with:

```bash
SUPPLIER_API_BASE_URL=https://supplier-intelligence-api.onrender.com
```

If the configured API is unreachable, the command center shows a clear operator
message instead of assuming localhost.

Connector and narrative modes:

- `SUPPLIER_CONNECTOR_MODE=demo` or `stub`: deterministic offline signals.
- `SUPPLIER_CONNECTOR_MODE=public`: optional RSS news/hiring sources and SEC EDGAR submissions; failures are recorded as degraded sync status and do not break scoring.
- Public news entries must mention the configured supplier name or ID by default (`SUPPLIER_NEWS_REQUIRE_SUPPLIER_MATCH=true`) so general feeds are not attributed blindly.
- SEC mapping is conservative: material 8-K/6-K and late-filing notices become weak signals, while routine periodic/ownership forms are skipped rather than accumulated as risk.
- Public requests use `SUPPLIER_CONNECTOR_TIMEOUT_SECONDS` and `SUPPLIER_CONNECTOR_RETRY_COUNT`. Set `SUPPLIER_FILINGS_USER_AGENT` to an application name and monitored contact before using SEC EDGAR.
- `SUPPLIER_LLM_NARRATIVE_PROVIDER=none`: deterministic governed narrative, the default.
- `SUPPLIER_LLM_NARRATIVE_PROVIDER=openai|anthropic`: provider interface and governance boundary are present, but `/ready` reports `interface_only`; real provider calls remain intentionally disabled, tests use a mock, and runtime falls back deterministically.

## Real-Time Sentinel Setup

For company-style deployment, configure API keys as environment variables, a local
`.env` file, or Streamlit secrets. The UI will use server-side values automatically.

```bash
copy .env.example .env
# edit .env with:
# SUPPLIER_APP_ADMIN_USER=...
# SUPPLIER_APP_ADMIN_PASSWORD=...
# NEWSAPI_KEY=...
# OPENAI_API_KEY=...      # optional if using Claude
# ANTHROPIC_API_KEY=...   # optional if using OpenAI
```

Recommended Sentinel mode:

- `Live News + AI`: NewsAPI fetches real articles, OpenAI or Claude classifies only public article text, and supplier matching/exposure is computed locally.
- `NewsAPI Rules`: real articles with local keyword matching, no LLM.
- `AI Scenario Briefing`: synthetic planning scenarios, useful for demos but not verified live news.

## Enterprise Hardening Commands

Local/demo:

```bash
python scripts/migrate.py --create-all-fallback
python scripts/backfill_tenants.py
python scripts/validate_tenant_schema.py
```

Render/Postgres staging:

```bash
python scripts/migrate.py
python scripts/validate_tenant_schema.py
python scripts/seed_demo_data.py --tenant-id demo-tenant
python scripts/smoke_staging.py --base-url https://supplier-intelligence-api.onrender.com
```

Operational exports and evidence:

```bash
python scripts/export_audit.py --tenant-id demo-tenant --format jsonl
python scripts/collect_evidence.py --tenant-id demo-tenant
locust -f load_tests/locustfile.py --host http://localhost:8000
```

For production-like Postgres, run `python scripts/migrate.py` without the
`--create-all-fallback` flag before starting the API or worker.

Readiness docs:

- `ENTERPRISE_SAAS_READINESS.md`
- `MIGRATIONS.md`
- `WORKER_ARCHITECTURE.md`
- `AUTH_INTEGRATION.md`
- `docs/AUTH_STAGING_PLAN.md`
- `docs/AUTOMATIONS_PLAN.md`
- `SECRETS_AND_KMS.md`
- `BACKUP_RESTORE_RUNBOOK.md`
- `LOAD_TESTING.md`
- `COMPLIANCE_READINESS.md`
- `Demo Mode`: no keys required.

## Pilot Security & Deployment

The app now includes a local pilot security layer:

- Login with hashed passwords.
- Roles: `admin`, `analyst`, `viewer`.
- Admin-only user management and audit log.
- SQLite persistence for activated supplier uploads and Sentinel scans.
- Audit events for login, upload activation, simulations, scans, exports, and admin actions.

If no users exist, the app seeds a first admin. For a real pilot, set
`SUPPLIER_APP_ADMIN_USER` and `SUPPLIER_APP_ADMIN_PASSWORD` before first launch.
If you do not, the temporary default is `admin` / `ChangeMe123!`.

Docker run example:

```bash
docker build -t supplier-intelligence-platform .
docker run --env-file .env -p 8501:8501 -v %cd%/data:/app/data supplier-intelligence-platform
```

On Linux/macOS, replace `%cd%/data` with `$(pwd)/data`.
For the API container, build `backend/Dockerfile` and publish port `8000`.

## Verification

```bash
python -m pytest -v
python -m compileall .
ruff check .
```

GitHub Actions runs compile, unit tests, and Ruff on push and pull request.

## Production Documentation

- [Architecture](ARCHITECTURE.md)
- [Deployment](DEPLOYMENT.md)
- [Data Schema](DATA_SCHEMA.md)
- [Operations Runbook](OPERATIONS_RUNBOOK.md)
- [Production Readiness](PRODUCTION_READINESS.md)
- [Security](SECURITY.md)

## Multi-Tenant SaaS Notes

The production foundation uses shared-schema tenancy. Business tables include `tenant_id`, repositories filter by tenant, and FastAPI protected routes use the same `TenantContext` for API-key and OIDC flows. Local/demo mode seeds `demo-tenant` and `demo-api-key`; OIDC mode verifies bearer tokens and then requires an active local tenant membership before RBAC permissions are applied.

## Project Structure

```
supplier-intelligence-platform/
├── app.py                    # Main Streamlit application (8 tabs)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── SECURITY.md               # Security & data handling documentation
├── data_ingestion.py         # Upload parsing, schema mapping, generated network creation
├── news_intelligence.py      # Sentinel intelligence modes and event matching
├── models/
│   ├── bayesian_risk.py      # Bayesian posterior risk scoring
│   ├── sir_propagation.py    # SIR cascade propagation model
│   ├── monte_carlo.py        # Monte Carlo financial simulation
│   ├── graph_metrics.py      # Network centrality analysis
│   └── nasa_upgrades.py      # Aerospace PRA techniques (LHS, Weibull, FTA)
├── agents/
│   ├── sentinel.py           # News monitoring & event classification
│   └── orchestrator.py       # Multi-agent coordination
├── tests/
│   ├── test_data_ingestion.py # Ingestion and uploaded-network tests
│   ├── test_agent_models.py
│   └── test_agents_and_models_core.py
└── data/
    ├── sample_network.json   # 12-node synthetic supply network
    └── lake_cable_network.json # Real-world demo network
```

## References

- Tabachová et al. (2024) — SIR propagation in supply networks
- Hosseini & Ivanov (2020) — Bayesian networks in supply chain risk management
- Chopra & Sodhi (2004) — Managing risk to avoid supply chain breakdown
- Brintrup et al. (2021) — Supply network centrality analysis
- Ellram (1995) — Total Cost of Ownership framework
