# Data Schema

## Supplier Upload Columns

Required:

- `supplier_name`

Common aliases:

- `supplier`, `vendor`, `company`, `manufacturer`, `partner`

Optional fields:

- `country`
- `tier`
- `category`
- `unit_cost`
- `annual_spend`
- `annual_volume`
- `quality_score`
- `on_time_delivery_pct`
- `defect_rate_pct`
- `lead_time_days`
- `risk_score`
- `financial_health`
- `geopolitical_risk`
- `tariff_exposure`
- `compliance_score`
- `dependency_score`
- `sub_tier_count`
- `criticality`
- `certifications`
- `years_in_business`
- `contact_email`

The ingestion pipeline handles CSV and XLSX files, fuzzy maps aliases, cleans currency and percentage strings, drops duplicate supplier names, and records an ingestion job.

## Production Tables

The SQLAlchemy production schema includes:

- `tenants`
- `organizations`
- `memberships`
- `tenant_api_keys`
- `access_reviews`
- `retention_policies`
- `backup_runs`
- `users`
- `roles`
- `suppliers`
- `supplier_kpis`
- `supplier_risk_scores`
- `news_events`
- `supplier_event_matches`
- `supplier_weak_signals`
- `supplier_evidence_scoring_versions`
- `supplier_evidence_runs`
- `supplier_evidence_run_suppliers`
- `supplier_evidence_actions`
- `supplier_connector_syncs`
- `supplier_historical_outcomes`
- `scenario_runs`
- `financial_exposure_runs`
- `alerts`
- `audit_logs`
- `ingestion_jobs`
- `system_health_events`
- `background_job_runs`

All business tables include `tenant_id` and are intended to be queried through tenant-scoped repositories.

## Supplier Evidence Chain Fields

Weak signals include:

- `signal_id`
- `supplier_id`
- `supplier_name`
- `signal_type`: `news`, `financial`, `operational`, `audit`, `email`, `hiring`, `erp`, or connector-defined type
- `driver`
- `source`
- `source_url`
- `source_system`
- `observed_at`
- `severity`
- `confidence`
- `summary`
- `raw_payload_json`

Evidence-chain runs include:

- `run_id`
- `scoring_version`
- `status`
- `supplier_count`
- `narrative_json`
- `result_json`
- `llm_provider`
- `llm_model`
- `prompt_policy`

Evidence actions include:

- `run_id`
- `supplier_id`
- `action`
- `source_driver`
- `status`: `open`, `in_progress`, `blocked`, `completed`, `dismissed`
- `owner`
- `updated_by`

Connector syncs include:

- `source_system`
- `connector_type`
- `status`: `completed`, `failed`, or `skipped`
- `records_received`
- `records_accepted`
- `started_at`
- `finished_at`
- `error`
- `metadata_json`

Historical outcomes include:

- `supplier_id`
- `event_type`
- `event_date`
- `severity`
- `notes`
- `source`

## Alert Fields

Alerts include:

- `supplier_id`
- `alert_type`
- `severity`
- `message`
- `exposure`
- `status`: `open`, `acknowledged`, `resolved`
- `created_at`
- `updated_at`

## Background Job Run Fields

Job runs include:

- `tenant_id`
- `run_id`
- `job_name`
- `task_name`
- `status`
- `started_at`
- `finished_at`
- `duration_ms`
- `retry_count`
- `error_summary`
- `request_id`
- `correlation_id`
