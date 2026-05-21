# Production Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the pilot Streamlit supplier intelligence app with a production-ready backend foundation, persistence layer, scheduled monitoring, alerts, safer auth, and operational docs.

**Architecture:** Keep the existing Streamlit app and analytical modules intact. Add a `src/` production layer with centralized config, SQLAlchemy persistence, repositories, services, scheduler, security helpers, and system health. Add `backend/main.py` as a FastAPI API over those services, and lightly wire Streamlit to expose production command-center panels.

**Tech Stack:** Python 3.11+, Streamlit, FastAPI, SQLAlchemy, APScheduler, SQLite fallback, Postgres-compatible database URL, pytest, Docker Compose.

---

### Task 1: Baseline

**Files:**
- Read: repository root
- Verify: existing tests and lint

- [x] **Step 1: Inspect repository**

Run `Get-ChildItem -Force` and `rg --files`.

- [x] **Step 2: Run baseline verification**

Run:
`.\venv\Scripts\python.exe -m unittest discover -s tests -t . -v`
`.\venv\Scripts\python.exe -m compileall app.py data_ingestion.py news_intelligence.py models agents tests`
`.\venv\Scripts\python.exe -m ruff check .`

### Task 2: Production Tests

**Files:**
- Create: `tests/test_production_foundation.py`
- Create: `tests/test_backend_api.py`

- [ ] **Step 1: Write failing tests for config, database, ingestion, sentinel, decisions, alerts, jobs, and security**

Tests import desired APIs from `src.*` and `backend.main`, expecting safe behavior.

- [ ] **Step 2: Run tests to verify red**

Run `.\venv\Scripts\python.exe -m pytest tests/test_production_foundation.py tests/test_backend_api.py -v`.
Expected: FAIL because `src` and `backend` modules do not exist yet.

### Task 3: Production Foundation

**Files:**
- Create: `src/config.py`
- Create: `src/database.py`
- Create: `src/models.py`
- Create: `src/repositories/*.py`
- Create: `src/services/*.py`
- Create: `src/observability/logging.py`
- Create: `backend/main.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add SQLAlchemy config/database/models**

Provide SQLite fallback and Postgres-ready URL support.

- [ ] **Step 2: Add repository/service layer**

Implement supplier, alert, audit, health, ingestion, risk, sentinel, decision, and scheduler services.

- [ ] **Step 3: Add FastAPI backend**

Expose `/health`, supplier, risk, financial, scenario, sentinel, alerts, ingestion, audit, and system endpoints.

- [ ] **Step 4: Run production tests to green**

Run pytest for new tests and fix failures.

### Task 4: Security Hardening

**Files:**
- Modify: `pilot_security.py`
- Test: `tests/test_pilot_security.py`

- [ ] **Step 1: Add security mode, first-admin setup, password strength, failed login tracking, and lockout**

Production mode must not seed `admin / ChangeMe123!`.

- [ ] **Step 2: Add tests and run them**

Run `.\venv\Scripts\python.exe -m pytest tests/test_pilot_security.py -v`.

### Task 5: Streamlit Command Center

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add alerts and production command center tabs**

Show DB/API/job/Sentinel/security status, refresh indicators, open high-risk alerts, and alert actions.

- [ ] **Step 2: Preserve existing tabs**

Keep all eight analytical tabs working.

### Task 6: Deployment and Docs

**Files:**
- Add/modify: `docker-compose.yml`, `backend/Dockerfile`, `.env.example`, `README.md`, `ARCHITECTURE.md`, `DEPLOYMENT.md`, `DATA_SCHEMA.md`, `OPERATIONS_RUNBOOK.md`, `PRODUCTION_READINESS.md`, `SECURITY.md`

- [ ] **Step 1: Add production-like Docker Compose**

Include Streamlit, FastAPI, Postgres, Redis placeholder, and worker/scheduler services.

- [ ] **Step 2: Update honest documentation**

Document demo mode, Postgres, API keys, scheduler, tests, deployment, and remaining enterprise gaps.

### Task 7: Final Verification

**Files:**
- All

- [ ] **Step 1: Run final checks**

Run:
`.\venv\Scripts\python.exe -m compileall .`
`.\venv\Scripts\python.exe -m pytest -v`
`.\venv\Scripts\python.exe -m ruff check .`
FastAPI health via TestClient.
Streamlit health smoke on localhost.

- [ ] **Step 2: Report exact results and residual gaps**

Summarize production upgrades and known limitations.
