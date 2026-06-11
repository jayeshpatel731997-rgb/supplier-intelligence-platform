# Supplier Intelligence Local Skills

These repository-local skills are instruction-only. They do not install packages, add external code, or grant production access. Invoke one explicitly with `$skill-name` when its workflow matches the task.

| Skill | Use it for |
| --- | --- |
| `codex-goal-writer` | Turn current project status into an executable Codex goal under 4,000 characters. |
| `project-state-recap` | Rebuild project context from docs, tests, Git status, history, and current diffs. |
| `autoreview-closeout` | Close out changes with security, tenancy, tests, secrets, deployment, and docs review. |
| `crabbox-verification` | Decide whether heavy isolated verification is justified and define a safe Crabbox run. |
| `deployment-readiness` | Assess pilot or staging readiness across config, endpoints, seeds, smoke tests, and docs. |
| `postgres-migration-smoke` | Verify Alembic upgrades, seed idempotency, rollback notes, and staging Postgres checks. |
| `convex-integration-review` | Review Convex schema/functions and separation from SQLAlchemy mode. |
| `tenant-isolation-security-review` | Audit tenant identity propagation, scoped queries, bypasses, leakage, and audit logs. |
| `secret-leakage-review` | Find credential or sensitive-value exposure without reproducing secret values. |
| `connector-integration-hardening` | Review RSS, SEC, hiring, logistics, ERP/email, and provider failure behavior. |
| `llm-governance-review` | Enforce structured-evidence narratives, deterministic fallback, and claim traceability. |
| `demo-packaging-and-docs` | Prepare a repeatable demo flow, export checklist, docs, and real-versus-stubbed labels. |
| `maintainer-orchestrator` | Coordinate delegated maintainer work across repository threads and prepare decision-ready queues. |
| `github-project-triage` | Triage GitHub issues, pull requests, CI, blockers, risk, proof, and next actions. |

## Companion Capabilities

- The existing trusted `autoreview` skill/helper can be used by `autoreview-closeout` when already installed.
- The existing `crabbox` skill remains the detailed transport reference; `crabbox-verification` adds project safety boundaries.
- The installed Convex plugin remains available for explicitly requested Convex implementation or official tooling.
- Existing general deployment skills remain available; this pack adds Supplier Intelligence Platform-specific checks.
- `maintainer-orchestrator` and `github-project-triage` were imported from `steipete/agent-scripts`; review their upstream Peter-specific ownership, tooling, and platform assumptions before using them unchanged.

## Rules

Always follow the repository `AGENTS.md`, verify the repository and branch before substantial work, preserve local SQLite demo mode, keep secrets out of output, and avoid production-readiness claims without verified enterprise controls.
