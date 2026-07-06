# Managed Staging Readiness Evidence Report

- Generated UTC: 20260706-195305
- Repository: `C:\Users\jayes\Desktop\supplier-intelligence-platform`
- Final readiness label: **Conditional go for managed staging**

## Checks

- PASS: `repo_verification` -> `artifacts\managed-staging-readiness\20260706-195305\repo_verification.log`
- PASS: `latest_commit` -> `artifacts\managed-staging-readiness\20260706-195305\latest_commit.log`
- PASS: `remote_compare_head` -> `artifacts\managed-staging-readiness\20260706-195305\remote_compare_head.log`
- PASS: `remote_compare_origin` -> `artifacts\managed-staging-readiness\20260706-195305\remote_compare_origin.log`
- PASS: `compile` -> `artifacts\managed-staging-readiness\20260706-195305\compile.log`
- PASS: `unittest` -> `artifacts\managed-staging-readiness\20260706-195305\unittest.log`
- PASS: `pytest` -> `artifacts\managed-staging-readiness\20260706-195305\pytest.log`
- PASS: `ruff` -> `artifacts\managed-staging-readiness\20260706-195305\ruff.log`
- PASS: `secret_leakage` -> `artifacts\managed-staging-readiness\20260706-195305\secret_leakage.log`
- PASS: `local_api_smoke` -> `artifacts\managed-staging-readiness\20260706-195305\local_api_smoke.log`
- PASS: `render_yaml_parse` -> `artifacts\managed-staging-readiness\20260706-195305\render_yaml_parse.log`
- PASS: `render_startup_shell_syntax` -> `artifacts\managed-staging-readiness\20260706-195305\render_startup_shell_syntax.log`
- PASS: `github_latest_ci` -> `artifacts\managed-staging-readiness\20260706-195305\github_latest_ci.log`
- PASS: `managed_staging_validation` -> `artifacts\managed-staging-readiness\20260706-195305\managed_staging_validation.log`
- PASS: `observed_render_supabase_health` -> `artifacts\managed-staging-readiness\20260706-195305\observed_render_supabase_health.log`
- SKIPPED: `frontend_checks` -> `artifacts\managed-staging-readiness\20260706-195305\frontend_checks.log`
- SKIPPED: `staging_smoke_health_only` -> `artifacts\managed-staging-readiness\20260706-195305\staging_smoke_health_only.log`
- SKIPPED: `postgres_backup_restore_drill` -> `artifacts\managed-staging-readiness\20260706-195305\postgres_backup_restore_drill.log`
- SKIPPED: `browser_screenshots` -> `artifacts\managed-staging-readiness\20260706-195305\browser_screenshots.log`

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

## Failures

- No required local check failures recorded.
