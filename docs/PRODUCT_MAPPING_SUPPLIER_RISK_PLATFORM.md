# Product Mapping: Supplier Risk Prediction Platform

## Product Direction

The Supplier Intelligence & Risk Prediction Platform should act as an early-warning system for supplier failure. The current codebase already contains many of the required pieces: supplier upload and ingestion, deterministic scoring, news/Sentinel intelligence, agentic event mapping, financial impact simulation, Streamlit dashboards, tenant-scoped APIs, roles, audit logs, and production-readiness scaffolding.

The next product step is to connect weak signals across news, financial filings, supplier emails, audit reports, hiring patterns, ERP/vendor master data, supplier performance data, and customer-specific criticality models into explainable supplier-level risk scores, narratives, and recommended actions.

## Existing Modules

| Module | Current role | Product support |
| --- | --- | --- |
| `app.py` | Main Streamlit experience with risk dashboard, network analysis, scenarios, Monte Carlo impact, scorecards, decision intelligence, upload, Sentinel, command center, alerts, and admin/audit views. | Primary pilot and demo surface for the supplier-risk product. |
| `data_ingestion.py` | Parses uploaded supplier files, maps flexible columns, validates required fields, and converts uploads into network data. | Foundation for ERP/vendor master data and supplier performance ingestion. |
| `news_intelligence.py` | Runs Sentinel scans using demo events, NewsAPI rule matching, Live News + AI, or Claude scenario briefings. Keeps supplier matching local for privacy. | Supports news weak signals, LLM classification, recommended actions, and disruption exposure. |
| `agents/sentinel.py` | Agent-style event detection and LLM classification for RSS, NewsAPI, USGS, and other disruption feeds. | Supports future always-on monitoring for news, regulatory, financial, weather, and logistics events. |
| `agents/mapper.py` | Maps detected events to supplier nodes with explicit match reasons. | Supports explainable evidence links between external events and affected suppliers. |
| `agents/propagator.py` | Runs cascade impact analysis and financial propagation from mapped events. | Supports downstream impact estimation when one supplier or tier fails. |
| `agents/strategist.py` | Creates mitigation recommendations and fault-tree style reasoning from cascade outputs. | Supports practical action generation and executive recommendations. |
| `agents/narrator.py` | Deterministically renders executive risk briefs from structured agent outputs. | Supports narrative summaries and can later be swapped to an LLM renderer. |
| `agents/orchestrator.py` | Coordinates Sentinel, Mapper, Propagator, Strategist, and Narrator. | Provides an agent pipeline shape for the new product thesis. |
| `models/bayesian_risk.py` | Computes Bayesian supplier disruption probability from calibrated signals. | Core supplier risk-scoring engine. |
| `models/graph_metrics.py` | Computes centrality, resilience, and single points of failure. | Supports customer-specific supplier criticality and network-risk weighting. |
| `models/monte_carlo.py` | Estimates expected loss, VaR, CVaR, and mitigation ROI. | Supports financial exposure and business-case framing. |
| `models/sir_propagation.py` | Simulates disruption cascade through supply network graph. | Supports tiered supplier failure propagation. |
| `src/services/risk_service.py` | Recalculates tenant supplier risk scores and creates high-risk alerts. | Backend API path for scoring and alerting. |
| `src/services/sentinel_service.py` | Persists Sentinel scan outputs, supplier-event matches, and alerts. | Production-facing Sentinel service wrapper. |
| `src/services/ingestion_service.py` | Tenant-scoped ingestion service backed by repositories and audit logs. | Production path for customer ERP/vendor uploads. |
| `src/services/decision_service.py` | Builds deterministic decision briefs with recommended supplier actions. | Supports TCO/decision intelligence and tradeoff narratives. |
| `src/repositories/*` | Tenant-scoped repository layer for suppliers, alerts, jobs, audit, and tenants. | Supports multi-tenant persistence and product hardening. |
| `src/security/*`, `pilot_security.py` | Authentication, roles, API keys, rate limits, secrets, and local pilot security. | Supports security, roles, and auditability for pilots and staging. |
| `backend/main.py` | FastAPI app exposing health, readiness, suppliers, ingestion, risk, Sentinel, alerts, and tenant admin endpoints. | API surface for dashboards, integrations, and automation. |
| `src/services/system_service.py`, `scheduler.py`, `worker_queue.py` | System status, background job scheduling, and task execution. | Supports recurring risk recalculation, Sentinel scans, and operations monitoring. |

## Product Capability Mapping

| Capability | Current support |
| --- | --- |
| LLM/narrative summaries | `news_intelligence.py` has Claude/OpenAI article classification and Claude scenario briefing; `agents/narrator.py` renders deterministic executive briefs; `src/services/decision_service.py` renders decision briefs. |
| Supplier risk scoring | `models/bayesian_risk.py`, `src/services/risk_service.py`, Streamlit Risk Dashboard, and the new `src/services/supplier_risk_evidence_chain.py`. |
| Sentinel/news intelligence | `news_intelligence.py`, `agents/sentinel.py`, `src/services/sentinel_service.py`, Streamlit Sentinel tab, and production job hooks. |
| Supplier upload/data ingestion | `data_ingestion.py`, `src/services/ingestion_service.py`, `src/repositories/suppliers.py`, FastAPI `/ingestion/upload`, and Streamlit Data Upload. |
| Dashboard views | `app.py` includes the dashboard, network, scenario, financial, scorecard, decision, upload, Sentinel, evidence-chain, command center, alerts, and admin views. |
| Financial impact | `models/monte_carlo.py`, `agents/propagator.py`, `src/services/risk_service.py`, and Streamlit Financial Impact. |
| Supplier scorecards | `app.py` scorecard functions and weighted scoring UI. |
| TCO/decision intelligence | `app.py` decision tab and `src/services/decision_service.py`. |
| Security/roles/audit logs | `pilot_security.py`, `src/security/*`, `src/repositories/audit.py`, `src/repositories/tenants.py`, API-key auth, tenant context, alerts, admin/audit UI, and compliance evidence services. |

## Gaps For The New Product Thesis

- Weak-signal schema is not yet a persisted first-class model across news, filings, audit, email, hiring, ERP, and supplier performance sources.
- Evidence chains were previously scattered across Sentinel match reasons, decision briefs, alerts, and audit logs; the new pilot starts consolidating this at supplier level.
- Customer-specific criticality weights are mostly UI/config driven today; they need tenant-level persisted configuration and model versioning.
- Live integrations for ERP, supplier portals, filings, email, hiring, logistics, and vendor master systems are not implemented.
- LLM outputs do not yet produce governed, source-cited supplier narratives from a common evidence object.
- Human review workflows for accepting, dismissing, annotating, and escalating evidence are limited.
- Alert lifecycle is present but not yet tied to full supplier evidence-chain status, owner assignment, or SLA tracking.
- Model governance needs versioned scoring functions, calibration history, backtesting, drift checks, and explainability exports.

## Recommended MVP Scope

1. Keep the first MVP deterministic and local: use uploaded supplier data plus sample weak signals to produce supplier risk evidence chains.
2. Standardize a weak-signal object with source, signal type, observed date, severity, confidence, summary, and affected supplier.
3. Generate supplier-level risk scores, levels, top drivers, evidence chains, confidence, and recommended actions.
4. Surface the result in Streamlit as a polished Evidence Chains tab alongside Sentinel and existing risk dashboards.
5. Add tests for scoring, risk levels, evidence traceability, recommended actions, and import safety.
6. Avoid live paid APIs until the evidence object, scoring behavior, and dashboard workflow are stable.

## Recommended Next 3 Implementation Phases

### Phase 1: Evidence Data Model And API

- Add tenant-scoped database tables for weak signals, supplier evidence chains, scoring runs, and action recommendations.
- Add FastAPI endpoints to list signals, create manual signals, calculate evidence chains, and acknowledge actions.
- Link Sentinel events and supplier-event matches into the same weak-signal schema.

### Phase 2: Customer-Specific Scoring And Narratives

- Add tenant-configurable criticality weights by category, region, part family, spend, tier, and sole-source status.
- Version scoring functions and persist score explanations for auditability.
- Generate deterministic narrative summaries first, then add optional LLM narrative rendering from structured evidence only.

### Phase 3: Live Signal Connectors And Workflow

- Add connectors for ERP/vendor master data, supplier portal APIs, financial filings, email digests, audit systems, hiring trend feeds, and logistics sources.
- Add alert workflow ownership, SLA tracking, status transitions, and escalation.
- Add backtesting and model monitoring to compare early-warning signals against historical supplier misses, late deliveries, quality events, and disruptions.

## Implemented Backend Evidence Workflow

The next backend slice now adds tenant-scoped persistence and APIs for the evidence-chain product path:

- Weak signals are persisted in `supplier_weak_signals`.
- Evidence-chain runs are persisted in `supplier_evidence_runs` and `supplier_evidence_run_suppliers`.
- Recommended action workflow is persisted in `supplier_evidence_actions`.
- Customer-specific scoring metadata is persisted in `supplier_evidence_scoring_versions`.
- Connector imports are recorded in `supplier_connector_syncs`.
- A governed narrative is generated from structured evidence only and stored with each run.

FastAPI endpoints:

- `GET /evidence/connectors` lists supported normalized connector sources for ERP, supplier portal, financial filings, email, hiring, and logistics feeds.
- `POST /evidence/signals/import` imports normalized weak signals from connector jobs or external systems.
- `GET /evidence/signals` lists tenant weak signals.
- `PUT /evidence/scoring-config` saves and activates a tenant scoring version.
- `GET /evidence/scoring-config/current` returns the active tenant scoring version.
- `POST /evidence/runs` creates and persists an evidence-chain run.
- `GET /evidence/runs` lists evidence-chain runs.
- `GET /evidence/runs/{run_id}` returns a persisted run with suppliers and action workflow.
- `PATCH /evidence/actions/{action_id}` updates recommended-action status and owner.
- `GET /evidence/connectors/syncs` lists tenant-scoped connector sync history.
- `POST /evidence/connectors/{connector_name}/sync` runs `news`, `filings`, or `hiring` in stub/demo/public mode and records completion, skip, or failure status.

## Final MVP Architecture

- FastAPI is the staging API boundary for tenant-scoped suppliers, evidence chains, connector syncs, scoring config, actions, health, readiness, and smoke-test workflows.
- SQLAlchemy/Postgres is the staging persistence path; SQLite remains the local demo path.
- Streamlit remains the demo/control surface and can target a separate API with `SUPPLIER_API_BASE_URL`.
- Connector syncs normalize public or deterministic signals into `supplier_weak_signals`, then the evidence-chain service computes risk scores, source-backed drivers, governed narrative, and actions.
- Readiness reports database/backend mode, auth posture, connector mode, scoring config, Convex status, and narrative provider status without exposing secrets.

## Real Vs Stubbed

- Real: tenant-scoped SQLAlchemy persistence, FastAPI auth/roles, evidence-chain scoring, action workflow, scoring config, connector sync history, staging smoke checks, and deterministic seed/demo data.
- Optional public data: RSS/Atom news and hiring sources plus SEC EDGAR submissions by CIK. These are direct public-source adapters with bounded timeouts, retries, normalized metadata, and persisted sync status; they are not paid data integrations.
- Stubbed or deferred: ERP, email, supplier portal, audit-system integrations, production malware scanning, real Convex runtime writes, and real OpenAI/Anthropic narrative calls.

## LLM Governance

- Deterministic narrative remains default with `SUPPLIER_LLM_NARRATIVE_PROVIDER=none`.
- The provider interface accepts `none|openai|anthropic`, but real provider calls are disabled in this build. Readiness labels non-default providers `interface_only`; tests use mocked providers and runtime falls back deterministically.
- Narrative inputs are structured evidence-chain objects only: supplier id/name, risk score/level, drivers, evidence chain, actions, and confidence.
- Output is constrained to risk summary, top drivers, recommended actions, and confidence caveats. A provider response containing drivers or actions absent from the structured evidence is rejected. Risk-summary and caveat prose are rendered deterministically from the evidence so provider free text cannot introduce unsupported claims.

## Calibration Limitations

- `scripts/calibrate_supplier_risk.py` compares historical outcomes with prior risk score snapshots where data exists.
- Output is review-oriented: coverage, matched examples, and false-positive/false-negative review lists.
- The MVP does not claim predictive accuracy, model validation, or enterprise calibration maturity.

## Demo Script

1. Run the API locally or in staging.
2. Seed with `python scripts/seed_demo_data.py --tenant-id demo-tenant`.
3. Run `python scripts/smoke_staging.py --base-url <api-url>` with tenant auth.
4. In Streamlit, use the Evidence Chains tab to walk through normal supplier, financial distress, logistics disruption, hiring slowdown, compliance issue, and multi-signal high-risk stories.

Convex note: configuration discovery and readiness reporting are present, but this build does not route persistence through Convex. `SUPPLIER_DATA_BACKEND=sqlalchemy` is the supported runtime; selecting Convex produces a clear readiness issue while SQLAlchemy/local mode remains usable.

Next enterprise integrations: ERP/vendor master, supplier portal, email digest, audit systems, quality systems, and procurement workflow sources.
