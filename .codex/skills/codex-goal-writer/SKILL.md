---
name: codex-goal-writer
description: Convert Supplier Intelligence Platform status, plans, or handoff notes into a clear Codex execution goal under 4,000 characters. Use when drafting a new Codex goal, implementation brief, restart prompt, or bounded follow-up task for this repository.
---

# Codex Goal Writer

Create a compact goal another Codex run can execute without guessing.

## Workflow

1. Remind the operator to verify the repository, project tree, environment, and current branch before doing work.
2. Extract the desired outcome, current evidence, scope, constraints, non-goals, deliverables, and finish conditions.
3. Preserve repository rules: no secrets, no unsupported production-readiness claims, local SQLite demo compatibility, and existing architecture.
4. Include exact paths, commands, or runtime surfaces only when supported by current repository evidence.
5. End with objective validation checks and a request to report changed files, tests, failures, and remaining gaps.
6. Count all characters and compress the final goal below 4,000 characters.

## Output

Use: `Goal`, `Context`, `Scope`, `Constraints`, `Deliverables`, and `Validation`.

Do not include credentials, copied secret values, speculative features, or work unrelated to the stated outcome.
