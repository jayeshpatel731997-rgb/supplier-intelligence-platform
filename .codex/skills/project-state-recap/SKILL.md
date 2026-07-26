---
name: project-state-recap
description: Produce an evidence-backed recap of the Supplier Intelligence Platform from repository docs, tests, Git status, recent history, and current changes. Use for handoffs, restart context, status reports, planning, or questions about what works and what remains.
---

# Project State Recap

Summarize the current repository from evidence, not conversation memory.

## Workflow

1. Verify repository, remote, branch, status, tracked files, and expected platform folders using `AGENTS.md`.
2. Read the current plan plus relevant README, architecture, security, deployment, migration, and production-readiness docs.
3. Inspect `git status`, `git diff --stat`, staged and unstaged diffs, and recent commits.
4. Inspect tests and configuration that cover the changed surfaces. Run lightweight non-destructive checks when practical.
5. Separate verified facts, reasonable inferences, blockers, unknowns, and work that is stubbed or demo-only.
6. Report the objective, completed work, work in progress, working behavior, verification evidence, remaining gaps, and next three tasks.

## Safety

- Do not modify files merely to create a recap.
- Do not read or print secret values.
- Do not call local demos, mocks, stubs, or untested integrations production-ready.
- State every command run and every important check skipped.
