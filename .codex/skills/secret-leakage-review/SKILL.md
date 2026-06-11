---
name: secret-leakage-review
description: Scan Supplier Intelligence Platform code, docs, logs, readiness responses, environment examples, fixtures, and tests for API key, password, token, connection-string, or private-data leakage. Use before commits, demos, deployments, exports, or incident closeout.
---

# Secret Leakage Review

Find disclosure risks without reproducing sensitive values.

## Workflow

1. Verify repository and branch, then inspect tracked and untracked changes.
2. Search for credential-shaped assignments, bearer tokens, private keys, database URLs, provider keys, webhook secrets, and suspicious high-entropy strings.
3. Review `.env.example`, Streamlit config, Docker/Render manifests, docs, scripts, fixtures, snapshots, test output, and committed logs.
4. Inspect `/health`, `/ready`, `/system/status`, exception responses, observability fields, audit exports, and connector diagnostics for configuration or secret disclosure.
5. Distinguish documented placeholders such as `demo-api-key` from live-looking credentials, but verify demo credentials cannot enable production access.
6. Report only file, line, secret category, exposure path, and remediation. Redact values completely.
7. If a real secret may be committed, recommend revocation/rotation and history cleanup; do not echo or copy the value.

## Safety

- Do not open `.env` or secret stores unless the user explicitly authorizes a necessary inspection.
- Do not place suspected secrets in commands, patches, reports, screenshots, or test output.
- Do not install scanners or upload repository contents to third parties.
