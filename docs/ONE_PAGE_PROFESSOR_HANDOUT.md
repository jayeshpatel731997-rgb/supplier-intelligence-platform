# Supplier Intelligence Platform — Professor Handout

## One-Sentence Summary

Supplier Intelligence Platform detects weak supplier-risk signals, preserves
evidence chains, and recommends operational actions for procurement and supply
chain teams.

## Problem

Supplier risk often appears first as weak, fragmented signals: quality drift,
late delivery, regional disruption, negative news, financial stress, or
concentration exposure. These signals are hard to trust unless the system can
show the evidence behind each recommendation.

## Solution

The platform connects supplier data, weak signals, evidence scoring, and
recommended actions. The design goal is explainable operational decision
support, not a black-box risk score.

## Architecture

- Streamlit UI on Render.
- FastAPI backend on Render.
- Supabase Postgres staging database.
- GitHub Actions CI gates.
- Future Supabase Auth/Storage path for managed identity and storage.

## Current Proof

Current status:

**API + UI live with Supabase Postgres validated; remaining production controls
pending.**

Evidence:

- API live.
- UI live.
- `/health` returns status OK.
- Database health OK with Supabase Postgres.
- CI green.
- Latest evidence pack recorded 161 passing pytest tests.
- Compile, Ruff, secret scan, local smoke, and staging validators passed.
- `/ready` returns structured degraded JSON with 5 known production issues.

## Honest Readiness Boundary

This is not production-ready yet. Remaining controls:

- OIDC/Auth configuration and tenant sync.
- CORS hardening / strict preflight PASS.
- Supabase Storage live object check.
- Malware/content scanning and quarantine proof.
- Backup/restore drill.
- Observability, alerting, rollback, and browser-auth evidence.

## What To Look For In The Demo

- Does the product problem feel real?
- Are evidence chains convincing as a trust mechanism?
- Does the architecture show practical staging discipline?
- Are the readiness gaps clearly separated from proven functionality?

## Feedback Requested

1. Is the evidence-chain framing strong enough for supplier risk?
2. Which next control matters most: Auth, storage/scanning, or observability?
3. What would make the product demo more convincing for a real operations team?
