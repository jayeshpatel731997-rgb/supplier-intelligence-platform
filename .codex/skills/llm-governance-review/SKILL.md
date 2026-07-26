---
name: llm-governance-review
description: Review Supplier Intelligence Platform LLM narratives for evidence grounding, deterministic fallback, unsupported claims, prompt/data boundaries, and auditable output. Use when narrator, Sentinel, provider configuration, prompts, evidence chains, or generated risk explanations change.
---

# LLM Governance Review

Require every generated claim to remain traceable to structured evidence.

## Workflow

1. Verify repository, branch, changed files, model/provider configuration, and the user-facing decision impact.
2. Trace structured supplier facts, signals, scores, citations, timestamps, confidence, and provenance into the LLM prompt.
3. Confirm the model cannot invent suppliers, events, causes, probabilities, mitigations, citations, or certainty not represented in the approved evidence payload.
4. Require output validation or constrained structure that preserves evidence identifiers and clearly labels inference, uncertainty, missing data, and stale data.
5. Verify deterministic fallback produces useful, evidence-only narrative when keys are missing, providers fail, outputs are invalid, or LLM use is disabled.
6. Test unsupported-claim prompts, conflicting evidence, empty evidence, stale evidence, provider failure, and deterministic repeatability of fallback output.
7. Review redaction, tenant boundaries, prompt injection resistance, logs, retention, human review, and docs describing limitations.

## Release Rule

Block release when claims cannot be traced to structured evidence or fallback can fail closed only by crashing. Do not send real sensitive data to an external model during review.
