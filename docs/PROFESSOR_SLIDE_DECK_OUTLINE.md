# Professor Slide Deck Outline

Target length: 10-12 slides for a 30-minute discussion.

## 1. Title

**Supplier Intelligence Platform: Evidence-Chain Risk Detection**

Subtitle: weak signals, explainable evidence, operational recommendations.

Speaker note: "This is a staging-validated portfolio project, not a
production-ready system yet."

## 2. Business Problem

- Supplier disruptions are detected too late.
- Weak signals are scattered across operational and external sources.
- Risk scores are hard to trust without evidence.
- Managers need actions, not just dashboards.

## 3. Why Evidence Chains Matter

- Risk recommendations must be explainable.
- Evidence chains preserve signal source, timestamp, score, and action.
- Trust improves when a user can inspect "why this recommendation exists."

## 4. Product Walkthrough

- Supplier dashboard.
- Risk and exposure views.
- Evidence-chain view.
- Recommended actions.
- Health/readiness status.

## 5. Architecture

- Render API service.
- Render Streamlit UI service.
- Supabase Postgres staging database.
- GitHub Actions CI.
- Future Supabase Auth/Storage integration.

## 6. Live Staging Proof

- API live.
- UI live.
- `/health` status OK.
- Supabase Postgres database OK.
- `/ready` structured but degraded.
- Readiness phrase:
  **API + UI live with Supabase Postgres validated; remaining production
  controls pending.**

## 7. Testing And CI Evidence

- CI green.
- 161 pytest tests passed in latest evidence pack.
- Compile check passed.
- Ruff passed.
- Secret scan passed.
- Local API smoke passed.
- Managed staging validator passed.

## 8. Readiness Status

- Staging live.
- Not production-ready.
- `/ready` degraded because OIDC/Auth production controls remain.
- Current production issue count: 5.
- Evidence is redacted and timestamped.

## 9. Remaining Production Controls

- OIDC/Auth and tenant sync.
- CORS hardening / strict preflight PASS.
- Supabase Storage live object check.
- Malware/content scanner and quarantine.
- Backup/restore drill.
- Observability, alerts, rollback.

## 10. Roadmap

- Validate Supabase Auth or external OIDC.
- Prove tenant isolation with real tokens.
- Run live Supabase Storage upload/read/delete check.
- Add scanner/quarantine proof.
- Execute backup/restore drill.
- Capture observability and rollback evidence.

## 11. What I Learned

- Production readiness is mostly about evidence, not claims.
- Weak-signal systems need explainability.
- Readiness endpoints should fail honestly.
- Redaction and auditability matter as much as features.

## 12. Professor Feedback Questions

- Is the evidence-chain framing strong enough?
- Which control should be prioritized next: Auth, storage/scanning, or
  observability?
- What would make the demo more convincing for a real procurement stakeholder?
- Is the business value clear enough from the current walkthrough?
