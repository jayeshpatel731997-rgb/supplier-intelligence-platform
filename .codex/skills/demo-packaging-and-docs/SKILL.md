---
name: demo-packaging-and-docs
description: Prepare an honest customer-demo package for the Supplier Intelligence Platform, including demo flow, export checklist, README and DEPLOYMENT updates, verification steps, and clear real-versus-stubbed capability labels. Use before pilot demos, handoffs, or customer-facing walkthroughs.
---

# Demo Packaging And Docs

Make the demo repeatable without overstating the platform.

## Workflow

1. Verify repository, branch, demo environment, seeded tenant, API key placeholder, and local runtime commands.
2. Define a short customer journey through supplier data, risk evidence, narrative, alerts, exports, and operational status using seeded or synthetic data.
3. Create a pre-demo checklist for API, Streamlit, worker, database, health/readiness, seed state, connector mode, and fallback behavior.
4. Create an export checklist covering tenant scope, sensitive fields, timestamps, provenance, filenames, formats, and redaction.
5. Update `README.md`, `DEPLOYMENT.md`, or focused demo docs only when behavior or setup instructions require it.
6. Label every capability as real, seeded/demo, stubbed, optional external integration, or future work. State required keys and network dependencies without exposing values.
7. Run the documented flow and record commands, expected screens/results, known limitations, reset steps, and recovery steps.

## Safety

Use demo or synthetic data only. Do not contact customers, send messages, call paid APIs, deploy, or add real credentials. Prefer `pilot`, `staging`, or `pre-production`; never imply unsupported production readiness.
