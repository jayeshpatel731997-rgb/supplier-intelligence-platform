---
name: crabbox-verification
description: Decide when and how to use an already trusted Crabbox installation for heavy Supplier Intelligence Platform verification. Use for CI-parity, Docker, Postgres, Redis, migration, load, or broad integration checks that are too costly or unavailable locally.
---

# Crabbox Verification

Use Crabbox only for verification that materially benefits from an isolated remote environment.

## Use Crabbox When

- Broad tests, Docker Compose, managed-service parity, migration smoke, load tests, or multi-service checks are impractical locally.
- A clean environment is needed to reproduce CI or dependency behavior.
- The user explicitly requests Crabbox proof.

Prefer local focused tests for ordinary edit loops.

## Safe Workflow

1. Verify the repository, branch, diff, and exact commit or file state to be tested.
2. Confirm a trusted Crabbox command is already installed and authenticated. If unavailable, report the blocker; do not download or install it.
3. Define the smallest remote command, expected artifacts, timeout, provider, and cleanup plan.
4. Run a local secret scan, prepare the exact remote-transfer file manifest, and obtain explicit user approval for that manifest before syncing any repository content.
5. Sync only the approved repository files required for the run. Exclude `.env`, credentials, local databases, caches, and unrelated personal files.
6. Pass secrets only through an approved secret-injection mechanism, only when necessary, and never print, persist, echo, upload, or place them in command history.
7. Run without privileged mode, host mounts, disabled isolation, or dangerous sandbox bypass.
8. Capture redacted logs, exit codes, provider/lease identity, and test results. Stop leases created for the task.

## Boundaries

- Never use `--privileged`, mount sensitive host paths, weaken sandboxing, or request broad cloud credentials.
- Never treat remote access as permission to deploy or mutate production.
- If safe secret injection or isolation cannot be confirmed, do not run; report the missing prerequisite.
