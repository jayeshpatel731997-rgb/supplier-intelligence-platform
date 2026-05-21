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
- `scenario_runs`
- `financial_exposure_runs`
- `alerts`
- `audit_logs`
- `ingestion_jobs`
- `system_health_events`
- `background_job_runs`

All business tables include `tenant_id` and are intended to be queried through tenant-scoped repositories.

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
