---
name: convex-integration-review
description: Review a Supplier Intelligence Platform Convex integration for schema design, indexes, queries, mutations, authorization, environment fallback, and clean separation from SQLAlchemy mode. Use when Convex files, configuration, or database-mode routing are added or changed.
---

# Convex Integration Review

Treat Convex and SQLAlchemy as explicit operating modes with clear ownership.

## Workflow

1. Verify repository, branch, changed files, and whether Convex is implemented, planned, or only stubbed.
2. Review Convex schema types, indexes, tenant identifiers, audit fields, retention fields, and migration compatibility.
3. Review queries and mutations for authentication, tenant scoping, input validation, bounded reads, idempotency, and safe error responses.
4. Trace environment selection and fallback behavior. Missing Convex configuration must not silently route production traffic to an unsafe local database.
5. Confirm SQLAlchemy repositories remain the owner in SQL mode and Convex functions remain the owner in Convex mode; prevent dual writes or mixed reads unless explicitly designed and tested.
6. Check local demo behavior, test coverage, deployment docs, environment examples, and rollback/mode-switch instructions.
7. Report findings by severity, evidence inspected, tests run, and unresolved mode-separation risks.

## Safety

Use the installed Convex capability or official tooling only when explicitly needed. Do not install packages, create cloud projects, deploy functions, or read secret values during a review.
