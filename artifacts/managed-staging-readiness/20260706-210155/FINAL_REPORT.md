# Managed Staging Readiness Evidence Report

- Generated UTC: 20260706-210155
- Repository: `C:\Users\jayes\Desktop\supplier-intelligence-platform`
- Final readiness label: **Staging API live with Supabase Postgres validated; UI/CORS and remaining production controls pending.**

## Checks

- PASS: `repo_verification` -> `artifacts\managed-staging-readiness\20260706-210155\repo_verification.log`
- PASS: `latest_commit` -> `artifacts\managed-staging-readiness\20260706-210155\latest_commit.log`
- PASS: `remote_compare_head` -> `artifacts\managed-staging-readiness\20260706-210155\remote_compare_head.log`
- PASS: `remote_compare_origin` -> `artifacts\managed-staging-readiness\20260706-210155\remote_compare_origin.log`
- PASS: `compile` -> `artifacts\managed-staging-readiness\20260706-210155\compile.log`
- PASS: `unittest` -> `artifacts\managed-staging-readiness\20260706-210155\unittest.log`
- PASS: `pytest` -> `artifacts\managed-staging-readiness\20260706-210155\pytest.log`
- PASS: `ruff` -> `artifacts\managed-staging-readiness\20260706-210155\ruff.log`
- PASS: `secret_leakage` -> `artifacts\managed-staging-readiness\20260706-210155\secret_leakage.log`
- PASS: `local_api_smoke` -> `artifacts\managed-staging-readiness\20260706-210155\local_api_smoke.log`
- PASS: `render_yaml_parse` -> `artifacts\managed-staging-readiness\20260706-210155\render_yaml_parse.log`
- PASS: `render_startup_shell_syntax` -> `artifacts\managed-staging-readiness\20260706-210155\render_startup_shell_syntax.log`
- PASS: `github_latest_ci` -> `artifacts\managed-staging-readiness\20260706-210155\github_latest_ci.log`
- PASS: `managed_staging_validation` -> `artifacts\managed-staging-readiness\20260706-210155\managed_staging_validation.log`
- PASS: `observed_render_supabase_health` -> `artifacts\managed-staging-readiness\20260706-210155\observed_render_supabase_health.log`
- SKIPPED: `frontend_checks` -> `artifacts\managed-staging-readiness\20260706-210155\frontend_checks.log`
- PASS: `ui_cors_smoke` -> `artifacts\managed-staging-readiness\20260706-210155\ui_cors_smoke.log`
- PASS: `staging_smoke_health_only` -> `artifacts\managed-staging-readiness\20260706-210155\staging_smoke_health_only.log`
- SKIPPED: `postgres_backup_restore_drill` -> `artifacts\managed-staging-readiness\20260706-210155\postgres_backup_restore_drill.log`
- SKIPPED: `browser_screenshots` -> `artifacts\managed-staging-readiness\20260706-210155\browser_screenshots.log`

## What Is Proven

- Repository-local tests, lint, compile, and secret scan results are captured when commands pass.
- Deployment YAML parsing and Render startup shell syntax are captured.
- Managed staging validation records API, Postgres, object storage, scanner, and Render evidence when corresponding credentials are configured.
- Operator-observed Render/Supabase health can be attached with OBSERVED_RENDER_API_HEALTH_OK=true and OBSERVED_SUPABASE_DB_HEALTH_OK=true without recording URLs or secrets.
- GitHub CI status is captured when GitHub CLI is available.

## What Is Mocked Or Staging-Safe

- FastAPI OIDC tests use test JWKS data, not a real IdP.
- Upload scanner proof uses a staging-safe EICAR-style adapter, not real malware scanning.
- Local smoke checks do not prove Render, managed Postgres, object storage, or scanner services.

## Still Requires External Setup

- Render deployment evidence and dashboard screenshots.
- Managed Postgres backup/restore drill against an approved staging or disposable target.
- Real IdP/MFA/tenant sync and Streamlit browser OIDC callback validation.
- Managed object storage live checks, real scanner/quarantine service, managed secrets/KMS, log drains, metrics, and alerting.
- Browser screenshots and authenticated UI/OIDC flow evidence.

## Failures

- No required local check failures recorded.
