# Evidence Chain Platform Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist supplier weak signals and evidence-chain runs, expose tenant-scoped API workflows, add scoring versions/customer weights, record connector-sourced signals, and generate governed narratives from structured evidence.

**Architecture:** Extend the existing SQLAlchemy/Postgres tenant model rather than introducing a parallel database runtime. Keep the deterministic evidence-chain service as the scoring core, then add a persistence/API service around it. Convex DB is not available as a local plugin or repo dependency, so this phase leaves a documented adapter seam and keeps the working implementation in the platform's current database layer.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic metadata, SQLite local demo, Postgres-compatible models, pytest, Ruff.

---

### Task 1: Persist Evidence Chain Domain

**Files:**
- Modify: `src/models.py`
- Modify: `src/database.py`
- Create: `src/services/supplier_evidence_service.py`
- Test: `tests/test_supplier_evidence_api.py`

- [ ] Add tenant-scoped models for weak signals, scoring versions, customer weights, evidence runs, evidence run suppliers, evidence actions, and connector sync records.
- [ ] Add SQLite compatibility handling for the new tenant tables.
- [ ] Add service methods to seed demo weak signals, ingest connector signals, run evidence scoring, list runs, and update action status.

### Task 2: Add API Workflow

**Files:**
- Modify: `backend/main.py`
- Modify: `src/tenancy.py`
- Test: `tests/test_supplier_evidence_api.py`

- [ ] Add request/response models for evidence runs, connector ingestion, scoring configuration, and action status updates.
- [ ] Add endpoints under `/evidence/*`.
- [ ] Protect read/run/write operations with tenant-scoped permissions.

### Task 3: Governed Narrative

**Files:**
- Modify: `src/services/supplier_risk_evidence_chain.py`
- Modify: `src/services/supplier_evidence_service.py`
- Test: `tests/test_supplier_evidence_api.py`

- [ ] Add deterministic narrative generation from structured evidence only.
- [ ] Add optional provider fields for future LLM use, but do not call paid APIs in this phase.
- [ ] Persist the generated narrative and the structured evidence JSON for auditability.

### Task 4: Verification And Review

**Files:**
- All changed files

- [ ] Run `.\venv\Scripts\python.exe -m compileall .`
- [ ] Run `.\venv\Scripts\python.exe -m pytest -q`
- [ ] Run `.\venv\Scripts\ruff.exe check .`
- [ ] Perform a local code-review pass over persistence, auth, tenant scoping, and API behavior.
