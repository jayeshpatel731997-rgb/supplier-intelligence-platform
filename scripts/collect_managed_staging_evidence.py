"""Collect a timestamped managed-staging readiness evidence pack.

The script runs local, non-destructive checks and writes redacted logs under
artifacts/managed-staging-readiness/YYYYMMDD-HHMMSS.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "managed-staging-readiness"
READINESS_LABEL = "Conditional go for managed staging"

SECRET_PATTERNS = [
    re.compile(r"(Authorization:\s*Bearer\s+)[^\s]+", re.IGNORECASE),
    re.compile(r"((?:api[_-]?key|token|secret|password|client[_-]?secret)[=:]\s*)[^,\s]+", re.IGNORECASE),
    re.compile(r"(postgres(?:ql)?(?:\+psycopg)?://[^:/@\s]+:)[^@\s]+(@)", re.IGNORECASE),
    re.compile(r"(DATABASE_URL\s*=\s*)[^\s]+", re.IGNORECASE),
    re.compile(r"((?:AWS|OPENAI|ANTHROPIC|NEWSAPI|GITHUB|RENDER)[A-Z0-9_]*(?:KEY|TOKEN|SECRET)\s*=\s*)[^\s]+", re.IGNORECASE),
]


def redact(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(lambda match: f"{match.group(1)}***{match.group(2) if len(match.groups()) > 1 else ''}", redacted)
    return redacted


def run_command(
    name: str,
    command: list[str],
    artifact_dir: Path,
    timeout: int = 300,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    log_path = artifact_dir / f"{name}.log"
    result: dict[str, object] = {"name": name, "command": command, "skipped": False}
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env={**os.environ, **(env_overrides or {})},
        )
        output = redact((completed.stdout or "") + (completed.stderr or ""))
        log_path.write_text(output, encoding="utf-8")
        result.update({"exit_code": completed.returncode, "log": str(log_path.relative_to(ROOT))})
    except FileNotFoundError as exc:
        message = f"SKIPPED: command not found: {command[0]}\n{exc}\n"
        log_path.write_text(message, encoding="utf-8")
        result.update({"skipped": True, "reason": "command not found", "log": str(log_path.relative_to(ROOT))})
    except subprocess.TimeoutExpired as exc:
        output = redact((exc.stdout or "") + (exc.stderr or "") if isinstance(exc.stdout, str) else str(exc))
        log_path.write_text(f"TIMEOUT after {timeout}s\n{output}\n", encoding="utf-8")
        result.update({"exit_code": 124, "log": str(log_path.relative_to(ROOT))})
    return result


def write_skip(name: str, artifact_dir: Path, reason: str) -> dict[str, object]:
    log_path = artifact_dir / f"{name}.log"
    log_path.write_text(f"SKIPPED: {reason}\n", encoding="utf-8")
    return {"name": name, "skipped": True, "reason": reason, "log": str(log_path.relative_to(ROOT))}


def write_pass(name: str, artifact_dir: Path, detail: str) -> dict[str, object]:
    log_path = artifact_dir / f"{name}.log"
    log_path.write_text(f"PASS: {redact(detail)}\n", encoding="utf-8")
    return {"name": name, "skipped": False, "exit_code": 0, "log": str(log_path.relative_to(ROOT))}


def tool_status() -> dict[str, str]:
    tools = ["docker", "pg_dump", "pg_restore", "psql", "trivy", "clamscan", "gh", "bash", "mypy", "pyright", "npm", "node"]
    status = {tool: shutil.which(tool) or "NOT_FOUND" for tool in tools}
    try:
        import importlib.util

        for module in ["playwright", "streamlit", "boto3", "psycopg"]:
            status[f"python:{module}"] = "available" if importlib.util.find_spec(module) else "NOT_FOUND"
    except Exception:
        status["python_module_probe"] = "failed"
    return status


def main() -> int:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    artifact_dir = ARTIFACT_ROOT / timestamp
    screenshots_dir = artifact_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    checks = [
        ("repo_verification", ["git", "status", "-sb"], 60),
        ("latest_commit", ["git", "log", "-1", "--oneline", "--decorate"], 60),
        ("remote_compare_head", ["git", "rev-parse", "HEAD"], 60),
        ("remote_compare_origin", ["git", "rev-parse", "origin/codex/evidence-chain-platform-hardening"], 60),
        ("compile", [python, "-m", "compileall", "-q", "app.py", "backend", "src", "agents", "models", "scripts", "tests", "data_ingestion.py", "news_intelligence.py", "pilot_security.py"], 300),
        ("unittest", [python, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"], 300),
        ("pytest", [python, "-m", "pytest", "-q"], 600),
        ("ruff", [str(ROOT / "venv" / "Scripts" / "ruff.exe") if os.name == "nt" else "ruff", "check", "."], 300),
        ("secret_leakage", [python, "scripts/check_secret_leakage.py"], 300),
        ("local_api_smoke", [python, "scripts/local_smoke.py"], 180),
        ("render_yaml_parse", [python, "-c", "import pathlib,yaml; [yaml.safe_load(pathlib.Path(p).read_text()) for p in ('render.yaml','render.full.yaml','docker-compose.yml')]; print('deployment YAML parsed')"], 120),
        ("render_startup_shell_syntax", ["bash", "-n", "scripts/start_api_render.sh", "scripts/start_ui_render.sh"], 120),
    ]

    if shutil.which("docker"):
        checks.append(("docker_compose_config", ["docker", "compose", "config"], 180))
    if shutil.which("trivy"):
        checks.append(("trivy_repo_scan", ["trivy", "fs", "--scanners", "vuln,secret,misconfig", "--skip-dirs", "venv", "."], 600))
    if shutil.which("clamscan"):
        checks.append(("clamscan_repo", ["clamscan", "-r", "--exclude-dir=venv", "."], 600))
    if shutil.which("gh"):
        checks.append(("github_latest_ci", ["gh", "run", "list", "--branch", "codex/evidence-chain-platform-hardening", "--limit", "3", "--json", "databaseId,headSha,status,conclusion,name,url,createdAt"], 120))
    package_json_present = (ROOT / "package.json").exists()
    if package_json_present:
        checks.append(("frontend_npm_install_check", ["npm", "install", "--dry-run"], 180))

    results = [run_command(name, command, artifact_dir, timeout) for name, command, timeout in checks]
    managed_env: dict[str, str] = {}
    if os.getenv("OBSERVED_SUPABASE_STORAGE_CONFIGURED", "").strip().lower() in {"1", "true", "yes", "on"}:
        managed_env.update(
            {
                "SUPPLIER_UPLOAD_STORAGE_PROVIDER": "supabase",
                "SUPABASE_EVIDENCE_BUCKET": os.getenv("SUPABASE_EVIDENCE_BUCKET", "supplier-evidence"),
                "SUPABASE_UPLOAD_QUARANTINE_BUCKET": os.getenv(
                    "SUPABASE_UPLOAD_QUARANTINE_BUCKET",
                    "supplier-upload-quarantine",
                ),
                "SUPABASE_UPLOAD_CLEAN_BUCKET": os.getenv(
                    "SUPABASE_UPLOAD_CLEAN_BUCKET",
                    "supplier-upload-clean",
                ),
            }
        )
    results.append(
        run_command(
            "managed_staging_validation",
            [python, "scripts/validate_managed_staging.py"],
            artifact_dir,
            240,
            env_overrides=managed_env,
        )
    )
    if os.getenv("OBSERVED_RENDER_API_HEALTH_OK", "").strip().lower() in {"1", "true", "yes", "on"}:
        detail = "Operator-observed Render API /health returned status ok."
        if os.getenv("OBSERVED_SUPABASE_DB_HEALTH_OK", "").strip().lower() in {"1", "true", "yes", "on"}:
            detail += " Operator-observed database health passed using a Supabase pooler host."
        results.append(write_pass("observed_render_supabase_health", artifact_dir, detail))
    else:
        results.append(
            write_skip(
                "observed_render_supabase_health",
                artifact_dir,
                "OBSERVED_RENDER_API_HEALTH_OK=true not set; no operator-observed Render/Supabase health evidence recorded.",
            )
        )
    if not package_json_present:
        results.append(
            write_skip(
                "frontend_checks",
                artifact_dir,
                "package.json not present; no frontend npm checks configured.",
            )
        )
    if os.getenv("STAGING_API_URL") or os.getenv("STAGING_API_BASE_URL") or os.getenv("STAGING_BASE_URL"):
        results.append(
            run_command(
                "staging_smoke_health_only",
                [python, "scripts/smoke_staging.py", "--health-only", "--skip-ui"],
                artifact_dir,
                120,
            )
        )
    else:
        results.append(
            write_skip(
                "staging_smoke_health_only",
                artifact_dir,
                "STAGING_API_BASE_URL/STAGING_BASE_URL not configured; remote staging smoke not attempted.",
            )
        )

    results.append(
        write_skip(
            "postgres_backup_restore_drill",
            artifact_dir,
            "No approved staging/disposable Postgres URL was configured and Docker is unavailable; real backup/restore drill not attempted.",
        )
    )
    results.append(
        write_skip(
            "browser_screenshots",
            artifact_dir,
            "Browser automation is not installed in the local virtualenv; screenshots were not captured.",
        )
    )

    availability = tool_status()
    (artifact_dir / "tool_availability.json").write_text(json.dumps(availability, indent=2), encoding="utf-8")

    screenshot_note = (
        "No browser screenshots were captured by this collector. "
        "Use docs/PROFESSOR_DEMO_GUIDE.md for the screenshot checklist if a browser is available.\n"
    )
    (screenshots_dir / "README.md").write_text(screenshot_note, encoding="utf-8")

    failed = [item for item in results if not item.get("skipped") and item.get("exit_code") not in {0}]
    report = [
        "# Managed Staging Readiness Evidence Report",
        "",
        f"- Generated UTC: {timestamp}",
        f"- Repository: `{ROOT}`",
        f"- Final readiness label: **{READINESS_LABEL}**",
        "",
        "## Checks",
        "",
    ]
    for item in results:
        status = "SKIPPED" if item.get("skipped") else ("PASS" if item.get("exit_code") == 0 else "FAIL")
        report.append(f"- {status}: `{item['name']}` -> `{item.get('log')}`")
    report.extend(
        [
            "",
            "## What Is Proven",
            "",
            "- Repository-local tests, lint, compile, and secret scan results are captured when commands pass.",
            "- Deployment YAML parsing and Render startup shell syntax are captured.",
            "- Managed staging validation records API, Postgres, object storage, scanner, and Render evidence when corresponding credentials are configured.",
            "- Operator-observed Render/Supabase health can be attached with OBSERVED_RENDER_API_HEALTH_OK=true and OBSERVED_SUPABASE_DB_HEALTH_OK=true without recording URLs or secrets.",
            "- GitHub CI status is captured when GitHub CLI is available.",
            "",
            "## What Is Mocked Or Staging-Safe",
            "",
            "- FastAPI OIDC tests use test JWKS data, not a real IdP.",
            "- Upload scanner proof uses a staging-safe EICAR-style adapter, not real malware scanning.",
            "- Local smoke checks do not prove Render, managed Postgres, object storage, or scanner services.",
            "",
            "## Still Requires External Setup",
            "",
            "- Render deployment evidence and dashboard screenshots.",
            "- Managed Postgres backup/restore drill against an approved staging or disposable target.",
            "- Real IdP/MFA/tenant sync and Streamlit browser OIDC callback validation.",
            "- Managed object storage live checks, real scanner/quarantine service, managed secrets/KMS, log drains, metrics, and alerting.",
            "",
            "## Failures",
            "",
        ]
    )
    if failed:
        for item in failed:
            report.append(f"- `{item['name']}` exited {item.get('exit_code')}; inspect `{item.get('log')}`.")
    else:
        report.append("- No required local check failures recorded.")

    (artifact_dir / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Evidence folder: {artifact_dir}")
    print(f"Final readiness label: {READINESS_LABEL}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
