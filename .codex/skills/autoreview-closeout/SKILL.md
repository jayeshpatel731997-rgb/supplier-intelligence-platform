---
name: autoreview-closeout
description: Perform an AutoReview-style closeout of Supplier Intelligence Platform changes for correctness, security, tenant isolation, tests, secrets, deployment blockers, and documentation. Use before commit, pull request, handoff, staging, or release claims.
---

# AutoReview Closeout

Review the real change set as a skeptical maintainer.

## Workflow

1. Verify repository and branch, then inspect status, diff statistics, staged changes, unstaged changes, and untracked files.
2. Read changed files with adjacent callers, repositories, services, tests, configuration, migrations, and docs.
3. Review for correctness, regression risk, error handling, tenant isolation, authorization bypass, secret disclosure, unsafe defaults, deployment blockers, and missing documentation.
4. Check local demo compatibility and production-mode behavior separately.
5. Rank actionable findings by severity with file and line references. Reject speculative or unrelated rewrites.
6. Apply fixes only when the user requested implementation, then rerun focused checks and review the resulting diff.
7. Report accepted and rejected findings, tests run, failures, residual risks, and whether the closeout is clean.

## Tool Safety

- Use an already installed, trusted AutoReview helper only when available and appropriate.
- Do not download review tools, install packages, push, merge, deploy, or bypass a sandbox.
- Never expose secret values found during review.
- Preserve unrelated user changes.
