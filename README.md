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
- Scope SaaS data by tenant using `X-Tenant-ID` and tenant API keys on protected FastAPI routes

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

Run the background worker:

```bash
python -m backend.worker
```

Docker Compose:

```bash
docker compose up --build
```

Streamlit opens at **http://localhost:8501** and FastAPI at **http://localhost:8000**.

Render staging:

```bash
gh pr merge 1 --merge
```

Then open:

```text
https://dashboard.render.com/blueprint/new?repo=https://github.com/jayeshpatel731997-rgb/supplier-intelligence-platform
```

See `RENDER_STAGING_RUNBOOK.md` for the exact staging launch checklist.

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

```bash
python scripts/migrate.py --create-all-fallback
python scripts/backfill_tenants.py
python scripts/validate_tenant_schema.py
python scripts/export_audit.py --tenant-id demo-tenant --format jsonl
python scripts/collect_evidence.py --tenant-id demo-tenant
locust -f load_tests/locustfile.py --host http://localhost:8000
```

Readiness docs:

- `ENTERPRISE_SAAS_READINESS.md`
- `MIGRATIONS.md`
- `WORKER_ARCHITECTURE.md`
- `AUTH_INTEGRATION.md`
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

The production foundation uses shared-schema tenancy. Business tables include `tenant_id`, repositories filter by tenant, and FastAPI protected routes require `X-Tenant-ID` plus a tenant-scoped API key. Local/demo mode seeds `demo-tenant` and `demo-api-key`; production deployments must create and rotate real tenant keys.

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
