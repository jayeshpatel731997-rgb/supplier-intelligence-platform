---
name: connector-integration-hardening
description: Review Supplier Intelligence Platform external data connectors and stubs for resilience, observability, and safe failure. Use for RSS, SEC, hiring, logistics, ERP, email, NewsAPI, or other ingestion and synchronization changes.
---

# Connector Integration Hardening

Review each connector as an unreliable external boundary.

## Workflow

1. Verify repository, branch, connector scope, data ownership, and whether each path is real, demo, stubbed, or planned.
2. Review RSS, SEC, hiring, logistics, ERP/email stubs, and provider clients for explicit connect/read timeouts, bounded retries, exponential backoff, jitter, rate-limit handling, pagination limits, and response-size limits.
3. Validate inputs and parsed records, normalize timestamps and identifiers, and preserve provenance needed for evidence chains.
4. Confirm idempotency, deduplication, checkpoints, sync history, partial-progress recording, and safe replay.
5. Ensure missing keys, malformed data, provider outages, and LLM/provider failures degrade to a clear status without crashing scans or corrupting prior good data.
6. Check tenant scoping, secret handling, logging redaction, audit events, metrics, and operator-visible error summaries.
7. Add or identify tests for success, timeout, retryable failure, permanent failure, partial page, duplicate data, malformed payload, and recovery.

## Safety

Do not call paid or production APIs, send email, mutate ERP systems, or install SDKs unless explicitly approved. Use fixtures or mocks for review and local verification.
