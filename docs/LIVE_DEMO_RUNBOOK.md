# Live Demo Runbook

## Demo Principle

Show what is live and proven. Do not claim production-ready. Use this phrase:

**API + UI live with Supabase Postgres validated; remaining production controls
pending.**

## Browser Demo Checklist

Open these in separate browser tabs before the meeting:

- UI: <https://supplier-intelligence-ui-hut2.onrender.com>
- API health: <https://supplier-intelligence-api-hut2.onrender.com/health>
- API ready: <https://supplier-intelligence-api-hut2.onrender.com/ready>
- GitHub Actions: open the latest green CI run for branch
  `codex/evidence-chain-platform-hardening`.
- Evidence pack:
  `artifacts/managed-staging-readiness/20260706-210155/FINAL_REPORT.md`
- Readiness status:
  `docs/PRODUCTION_READINESS_STATUS.md`

## Live Demo Flow

### 1. Open UI

Open <https://supplier-intelligence-ui-hut2.onrender.com>.

Say: "This is the live staging UI."

Expected: page loads with HTTP 200 / usable Streamlit page.

### 2. Show Dashboard

Show the supplier-risk dashboard and explain the portfolio view.

Say: "This is where a procurement or operations user starts: which suppliers
need attention and why?"

### 3. Show Evidence Chains

Navigate to evidence-chain or risk-evidence views.

Say: "The platform is designed to preserve the reasoning path from weak signal
to evidence to score to action."

### 4. Show Recommendations

Show recommended operational actions.

Say: "The goal is not only to identify risk, but to recommend operational next
steps such as monitor, dual-source, renegotiate, or escalate."

### 5. Open API `/health`

Open <https://supplier-intelligence-api-hut2.onrender.com/health>.

Point out:

- `status: ok`
- `database.ok: true`
- driver: `postgresql+psycopg`
- API status: `ready`

Say: "This proves the API is live and connected to Supabase Postgres."

### 6. Open API `/ready`

Open <https://supplier-intelligence-api-hut2.onrender.com/ready>.

Point out:

- HTTP 503
- `status: degraded`
- `production_issue_count=5` in evidence logs
- the degraded items are OIDC/Auth configuration controls

Say: "This is intentionally honest. The app is live, but it refuses to call
itself fully ready until production controls are configured."

### 7. Show GitHub CI Green

Show the latest GitHub Actions run.

Say: "The current branch has CI green, including tests, compile, secret scan,
Ruff, and deployment config validation."

### 8. Show Evidence Pack Docs

Open `artifacts/managed-staging-readiness/20260706-210155/FINAL_REPORT.md`.

Mention:

- pytest: 161 passed;
- local API smoke passed;
- managed staging validation passed;
- live UI/API smoke passed;
- strict CORS preflight is WARN, not hidden;
- no secrets are committed.

## Backup Demo Plan If Internet Fails

Use local docs and committed evidence:

1. Open `docs/PROFESSOR_SLIDE_DECK_OUTLINE.md`.
2. Open `docs/ONE_PAGE_PROFESSOR_HANDOUT.md`.
3. Open `artifacts/managed-staging-readiness/20260706-210155/FINAL_REPORT.md`.
4. Show `ui_cors_smoke.log` for live smoke results.
5. Show `pytest.log` for test count.
6. Show `docs/PRODUCTION_READINESS_STATUS.md` for readiness honesty.
7. If local app dependencies are available, run:

```powershell
.\venv\Scripts\python.exe scripts\local_smoke.py
```

If the live sites are unreachable, say:

"The live demo depends on Render network availability. The committed evidence
pack captures the last successful staging proof, and the local smoke test shows
the code-level path still works."

## Known Demo Caveats

- `/ready` is degraded by design until OIDC/Auth is configured.
- Strict CORS preflight is currently WARN until the API echoes the exact UI
  origin.
- Supabase Storage live object check is scaffolded but not executed without
  explicit write approval and credentials.
- Malware scanning, backup/restore, observability, and rollback are pending
  production controls.
