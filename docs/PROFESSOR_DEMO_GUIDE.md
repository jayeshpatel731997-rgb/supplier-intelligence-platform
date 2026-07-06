# Professor Demo Guide

## One-Sentence Framing

This is a supplier intelligence and risk platform that detects weak signals,
preserves evidence chains, and recommends operational actions while clearly
separating demo/staging proof from external production controls.

## Problem It Solves

Manufacturers often discover supplier disruption too late: a quality decline,
regional event, late filing, or logistics signal appears in one place while
spend exposure, alternate sourcing, and decision context live somewhere else.
This project brings those signals into one explainable workflow.

## Five-Minute Demo Flow

1. **Risk Dashboard** — show portfolio risk, supplier ranking, and financial
   exposure.
2. **Sentinel Agent** — explain that weak signals are collected and normalized
   into supplier-level evidence.
3. **Evidence Chains** — show that decisions link back to signals, scores,
   timestamps, and provenance rather than unsupported model claims.
4. **Decision Intelligence** — walk through recommended actions and tradeoffs.
5. **Alerts & Health / Command Center** — show the operational side: readiness,
   auth posture, jobs, audits, and staging checks.

## Talk Track

- "This is a supplier intelligence and risk platform."
- "It detects weak signals before they become expensive disruptions."
- "It preserves evidence chains so recommendations are explainable."
- "It recommends actions using risk, performance, financial exposure, and
  scenario context."
- "It is staging-ready only after external controls are validated; the repo is
  honest about what is proven locally versus what requires managed services."

## What Is Already Proven

- Streamlit analytics flow with seeded/demo data and uploaded supplier data.
- FastAPI backend with tenant-scoped repositories, alerts, jobs, audit logs, and
  health/readiness endpoints.
- Local/demo tenant API-key auth and tested FastAPI OIDC bearer-token plumbing.
- Tenant-boundary tests including OIDC header override rejection.
- Bounded upload handling and staging-safe scanner rejection path.
- Alembic migration structure, seed script, backup/restore scripts, Render
  blueprint, smoke script, and CI release gates.

## What Is Mocked Or Staging-Safe

- Browser OIDC for Streamlit is documented/scaffolded, not a completed real IdP
  login.
- The `eicar-test` upload scanner proves reject behavior but is not malware
  inspection.
- External intelligence keys are optional; missing keys must degrade safely.
- Convex is configuration-reviewed as optional/future unless explicitly
  configured.

## What Still Requires External Setup

- Managed Postgres and a completed backup/restore drill.
- Real IdP, MFA, tenant membership sync, and Streamlit browser SSO.
- S3-compatible object storage plus real scanner/quarantine service.
- Render dashboard/deployment evidence, log drains, metrics, alerting, and
  rollback evidence.
- Managed secrets/KMS and security/compliance review.

## What To Avoid Claiming

Do not say "production-ready." Say:

> "This is a strong managed-staging candidate with local and CI evidence. Real
> production readiness depends on external controls that are explicitly listed."
