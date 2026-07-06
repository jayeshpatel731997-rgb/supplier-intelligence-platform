# 30-Minute Professor Presentation Script

## Core Message

This project is a supplier intelligence and risk platform that detects weak
supplier-risk signals, preserves evidence chains, and recommends operational
actions. The current status is:

**API + UI live with Supabase Postgres validated; remaining production controls
pending.**

Do not describe this as production-ready. Describe it as live staging evidence
with an honest readiness boundary.

## 0:00-2:00 — Opening

"Today I am presenting my supplier intelligence platform. The problem I focused
on is that supplier risk often shows up first as weak, fragmented signals:
delivery delays, news events, quality changes, geopolitical signals, financial
stress, or concentration exposure. In real operations, those signals are hard to
trust because they are scattered and often detached from evidence."

"My goal was to build a platform that does three things: detects weak supplier
risk signals, preserves evidence chains behind the risk assessment, and
recommends practical operational actions."

Transition: "I will first explain the problem, then the architecture, then show
the live staging proof, and finally be explicit about what is still pending."

## 2:00-6:00 — Business Problem

"Supplier risk is expensive because teams often react after disruption is
already visible. A supplier can look fine in a dashboard while early warning
signals exist in news, filings, delivery patterns, defects, or regional events."

"The issue is not just prediction. The issue is trust. If a system says
'replace this supplier' or 'dual source this part,' a manager needs to know why.
What evidence led to that recommendation? How recent is it? What signals were
used? What was ignored? What is the financial exposure?"

Emphasize:

- weak signals are noisy;
- procurement decisions need explainability;
- risk scoring without provenance is hard to defend;
- operational recommendations need evidence behind them.

## 6:00-10:00 — Product Solution

"The product is a supplier intelligence and risk platform. It combines supplier
data, risk signals, evidence chains, and recommended actions."

"The important design choice is that it does not just output a score. It keeps
the chain from signal to evidence to score to action. That way, a user can
understand not only which supplier is risky, but why the system thinks so."

Walk through product concepts:

- supplier portfolio and dashboard;
- weak-signal collection;
- evidence scoring;
- recommended operational actions;
- alerts and readiness/health surfaces.

Suggested phrasing:

"The platform is meant to help a supply-chain or procurement user move from
'something may be wrong' to 'here is the evidence and here is the next action.'"

## 10:00-15:00 — Architecture

"The staging architecture is intentionally practical. The API and UI are hosted
on Render. The database is Supabase Postgres. Supabase Auth and Supabase Storage
are planned as the managed identity and storage layers, but I am not claiming
those are fully production-validated yet."

Architecture talking points:

- Streamlit UI for the professor/demo-facing workflow;
- FastAPI backend for health, readiness, supplier data, evidence chains, and
  operational endpoints;
- Supabase Postgres as the managed staging database;
- GitHub Actions for CI quality gates;
- future Supabase Auth or external OIDC for real identity validation;
- future Supabase Storage/S3 plus scanner for upload quarantine and evidence
  retention.

Say clearly:

"The current live proof is staging proof: API live, UI live, Supabase Postgres
validated, CI green, and tests passing. It is not production readiness because
external controls remain."

## 15:00-22:00 — Live Demo

Follow [LIVE_DEMO_RUNBOOK.md](LIVE_DEMO_RUNBOOK.md).

Demo order:

1. Open the UI.
2. Show dashboard.
3. Show evidence chains.
4. Show recommendations.
5. Open API `/health`.
6. Open API `/ready` and explain degraded honestly.
7. Show GitHub CI green.
8. Show evidence pack docs.

When opening `/health`, say:

"This proves the API is live and the database connection is healthy. The driver
is `postgresql+psycopg`, and the database host is the Supabase pooler with
credentials redacted in the evidence logs."

When opening `/ready`, say:

"This returns degraded, and that is expected right now. It is not a bug I want
to hide. It means the app is correctly refusing to call itself fully ready until
OIDC/Auth and other production controls are configured."

## 22:00-25:00 — Testing And Evidence

"For this staging slice, I captured evidence in a timestamped evidence folder.
The local and CI gates are passing."

Mention:

- CI green;
- 161 pytest tests passed in the current evidence pack;
- compile check passed;
- ruff passed;
- secret scan passed;
- local API smoke passed;
- managed staging validator passed;
- UI/API smoke passed;
- `/ready` is degraded with five known production issues.

"The important part is that the project separates what is proven from what is
still external. That makes the readiness statement more credible."

## 25:00-28:00 — Readiness And Roadmap

"The current readiness label is: API + UI live with Supabase Postgres validated;
remaining production controls pending."

Remaining controls:

- OIDC/Auth values and tenant sync;
- CORS hardening, including strict preflight PASS;
- Supabase Storage live object check;
- malware/content scanner and quarantine workflow;
- backup/restore drill;
- observability/log drains/alerts/rollback evidence;
- browser screenshots and authenticated UI/OIDC flow.

"My next engineering slice would be identity: real Supabase Auth or external
OIDC, tenant mapping, and a live negative-token test."

## 28:00-30:00 — Close And Feedback Ask

"What I would like feedback on is whether the evidence-chain framing is strong
enough for a supplier-risk product, and whether the next priority should be
identity/tenant isolation, richer external signal ingestion, or operational
workflow design."

Close with:

"The platform is already live in staging and evidence-backed. The honest next
step is not to call it production-ready, but to validate the missing production
controls one by one."
