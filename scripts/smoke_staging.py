"""Smoke test a Render/Postgres staging API without printing secrets."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


SECRET_PATTERNS = [
    re.compile(r"(Authorization:\s*Bearer\s+)[^\s]+", re.IGNORECASE),
    re.compile(r"(X-API-Key=)[^\s]+", re.IGNORECASE),
    re.compile(r"((?:api[_-]?key|token|secret|password|client[_-]?secret)[=:])[^,\s]+", re.IGNORECASE),
    re.compile(r"((?:Cookie|Set-Cookie):\s*)[^\r\n]+", re.IGNORECASE),
    re.compile(r"(postgres(?:ql)?(?:\+psycopg)?://[^:/@\s]+:)[^@\s]+(@)", re.IGNORECASE),
    re.compile(r"(DATABASE_URL=)[^\s]+", re.IGNORECASE),
]


@dataclass(slots=True)
class SmokeResponse:
    status: int
    body: str
    content_type: str = ""


def redact(value: object) -> str:
    text = str(value)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda match: f"{match.group(1)}***{match.group(2) if len(match.groups()) > 1 else ''}", text)
    return text


def normalize_base_url(value: str) -> str:
    base_url = value.strip()
    if not base_url:
        raise ValueError("STAGING_API_BASE_URL, STAGING_BASE_URL, or --base-url is required.")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("Staging base URL must start with http:// or https://.")
    return base_url.rstrip("/") + "/"


def staging_base_url(env: Mapping[str, str]) -> str:
    return env.get("STAGING_API_BASE_URL", "").strip() or env.get("STAGING_BASE_URL", "").strip()


def staging_ui_base_url(env: Mapping[str, str]) -> str:
    return env.get("STAGING_UI_BASE_URL", "").strip()


def auth_headers(env: Mapping[str, str]) -> dict[str, str]:
    token = env.get("STAGING_BEARER_TOKEN", "").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}
    tenant_id = env.get("STAGING_TENANT_ID", "").strip()
    api_key = env.get("STAGING_API_KEY", "").strip()
    if tenant_id and api_key:
        return {"X-Tenant-ID": tenant_id, "X-API-Key": api_key}
    return {}


def expected_tenant_id(env: Mapping[str, str], headers: Mapping[str, str]) -> str:
    configured = env.get("STAGING_EXPECTED_TENANT_ID", "").strip()
    return configured or headers.get("X-Tenant-ID", "").strip()


def configuration_errors(
    env: Mapping[str, str],
    headers: Mapping[str, str],
    *,
    health_only: bool,
    skip_ui: bool,
) -> list[str]:
    errors: list[str] = []
    if not staging_base_url(env):
        errors.append("set STAGING_API_BASE_URL to the FastAPI service URL")
    if not health_only and not headers:
        errors.append("set STAGING_BEARER_TOKEN for OIDC, or STAGING_TENANT_ID and STAGING_API_KEY for an approved local-auth exception")
    if not health_only and "Authorization" in headers and not expected_tenant_id(env, headers):
        errors.append("set STAGING_EXPECTED_TENANT_ID so the OIDC tenant boundary can be verified")
    if not skip_ui and not staging_ui_base_url(env):
        errors.append("set STAGING_UI_BASE_URL to the Streamlit service URL, or pass --skip-ui explicitly")
    return errors


def request_json(
    base_url: str,
    path: str,
    headers: Mapping[str, str] | None = None,
    timeout: int = 10,
    method: str = "GET",
    payload: Mapping[str, object] | None = None,
) -> SmokeResponse:
    url = urljoin(base_url, path.lstrip("/"))
    data = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **dict(headers or {})}
    request = Request(url, data=data, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return SmokeResponse(
                status=response.status,
                body=response.read().decode("utf-8", errors="replace"),
                content_type=response.headers.get("Content-Type", ""),
            )
    except HTTPError as exc:
        return SmokeResponse(
            status=exc.code,
            body=exc.read().decode("utf-8", errors="replace"),
            content_type=exc.headers.get("Content-Type", ""),
        )
    except URLError as exc:
        raise RuntimeError(f"Request failed for {path}: {redact(exc)}") from exc


def _json_payload(response: SmokeResponse) -> object | None:
    try:
        return json.loads(response.body)
    except json.JSONDecodeError:
        return None


def _looks_like_html(response: SmokeResponse) -> bool:
    content_type = response.content_type.lower()
    body_start = response.body.lstrip()[:64].lower()
    return "text/html" in content_type or body_start.startswith(("<!doctype html", "<html"))


def _json_summary(response: SmokeResponse) -> str:
    payload = _json_payload(response)
    if payload is None:
        return ""
    if isinstance(payload, dict):
        summary = {
            key: payload.get(key)
            for key in (
                "status",
                "tenant_id",
                "run_id",
                "version",
                "connector",
                "records_accepted",
                "id",
            )
            if key in payload
        }
        database = payload.get("database")
        if isinstance(database, dict):
            summary["database"] = {
                key: database.get(key)
                for key in ("ok", "driver")
                if key in database
            }
        api = payload.get("api")
        if isinstance(api, dict):
            summary["api"] = {
                key: api.get(key)
                for key in ("ok", "status")
                if key in api
            }
        issues = payload.get("production_issues")
        if isinstance(issues, list):
            summary["production_issue_count"] = len(issues)
        return redact(summary)
    if isinstance(payload, list):
        return f"items={len(payload)}"
    return ""


def _api_json_check(name: str, response: SmokeResponse, expected_status: int = 200) -> tuple[str, bool, str]:
    if response.status != expected_status:
        return (name, False, f"HTTP {response.status} {_json_summary(response)}".strip())
    if _looks_like_html(response):
        return (
            name,
            False,
            f"HTTP {response.status} {response.content_type}; got HTML, likely Streamlit/UI URL instead of FastAPI API URL",
        )
    payload = _json_payload(response)
    if payload is None:
        return (
            name,
            False,
            f"HTTP {response.status} {response.content_type}; expected FastAPI JSON response",
        )
    return (name, True, f"HTTP {response.status} {_json_summary(response)}".strip())


def _protected_route_check(response: SmokeResponse) -> tuple[str, bool, str]:
    if response.status in {401, 403}:
        return ("/suppliers without auth", True, f"HTTP {response.status} {_json_summary(response)}".strip())
    if _looks_like_html(response):
        return (
            "/suppliers without auth",
            False,
            f"HTTP {response.status} {response.content_type}; got HTML, likely Streamlit/UI URL instead of FastAPI API URL",
        )
    return (
        "/suppliers without auth",
        False,
        f"HTTP {response.status} {response.content_type}; expected 401/403 auth rejection",
    )


def _ui_health_check(response: SmokeResponse) -> tuple[str, bool, str]:
    if response.status != 200:
        return ("Streamlit /_stcore/health", False, f"HTTP {response.status}")
    if _looks_like_html(response):
        return ("Streamlit /_stcore/health", False, "got HTML instead of the Streamlit health response")
    return ("Streamlit /_stcore/health", True, f"HTTP {response.status}")


def _payload_check(
    name: str,
    response: SmokeResponse,
    predicate,
    failure_message: str,
) -> tuple[str, bool, str]:
    basic = _api_json_check(name, response)
    if not basic[1]:
        return basic
    payload = _json_payload(response)
    if not isinstance(payload, dict) or not predicate(payload):
        return (name, False, f"{failure_message}; HTTP {response.status} {_json_summary(response)}".strip())
    return (name, True, f"HTTP {response.status} {_json_summary(response)}".strip())


def _workflow_checks(
    base_url: str,
    headers: Mapping[str, str],
    connector_mode: str,
) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    connector = request_json(base_url, "/evidence/connectors/news/sync", headers=headers, method="POST")
    checks.append(
        _payload_check(
            "connector sync",
            connector,
            lambda payload: (
                payload.get("status") == "completed"
                and int(payload.get("records_accepted", 0)) > 0
            )
            or (
                connector_mode == "public"
                and payload.get("status") == "skipped"
            ),
            "Connector sync did not return a valid mode-aware status",
        )
    )

    scoring = request_json(base_url, "/evidence/scoring-config/current", headers=headers)
    checks.append(
        _payload_check(
            "scoring config read",
            scoring,
            lambda payload: bool(payload.get("version")),
            "Scoring config did not return a version",
        )
    )

    run = request_json(
        base_url,
        "/evidence/runs",
        headers=headers,
        method="POST",
        payload={"include_demo_signals": False},
    )
    checks.append(
        _payload_check(
            "evidence-chain run",
            run,
            lambda payload: payload.get("status") == "completed" and bool(payload.get("run_id")),
            "Evidence-chain run did not complete",
        )
    )
    run_payload = _json_payload(run)
    action_id = None
    if isinstance(run_payload, dict):
        actions = run_payload.get("actions") or []
        if actions and isinstance(actions[0], dict):
            action_id = actions[0].get("id")
    if action_id:
        action = request_json(
            base_url,
            f"/evidence/actions/{action_id}",
            headers=headers,
            method="PATCH",
            payload={"status": "in_progress", "owner": "smoke-test"},
        )
        checks.append(
            _payload_check(
                "action update",
                action,
                lambda payload: payload.get("status") == "in_progress",
                "Evidence action was not updated",
            )
        )
    else:
        checks.append(("action update", False, "Evidence run did not return an action id."))
    return checks


def _tenant_isolation_checks(
    base_url: str,
    headers: Mapping[str, str],
    expected_tenant: str,
) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    status = request_json(base_url, "/system/status", headers=headers)
    checks.append(
        _payload_check(
            "authenticated tenant context",
            status,
            lambda payload: bool(expected_tenant) and payload.get("tenant_id") == expected_tenant,
            "Authenticated system status did not confirm the expected tenant",
        )
    )

    hostile_tenant = "cross-tenant-smoke-probe"
    if "Authorization" in headers:
        override_headers = {**headers, "X-Tenant-ID": hostile_tenant}
        override = request_json(base_url, "/system/status", headers=override_headers)
        checks.append(
            _payload_check(
                "OIDC tenant header override denied",
                override,
                lambda payload: payload.get("tenant_id") == expected_tenant,
                "X-Tenant-ID changed the OIDC-authenticated tenant",
            )
        )
    else:
        wrong_tenant_headers = {**headers, "X-Tenant-ID": hostile_tenant}
        wrong_tenant = request_json(base_url, "/system/status", headers=wrong_tenant_headers)
        checks.append(
            (
                "local API key cross-tenant denial",
                wrong_tenant.status in {401, 403},
                f"HTTP {wrong_tenant.status}; expected 401/403 for the wrong tenant",
            )
        )
    return checks


def run_smoke(
    base_url: str,
    headers: Mapping[str, str],
    *,
    health_only: bool = False,
    expected_tenant: str = "",
    ui_base_url: str = "",
    skip_ui: bool = True,
) -> int:
    checks: list[tuple[str, bool, str]] = []

    live = request_json(base_url, "/live")
    checks.append(_api_json_check("/live", live))

    health = request_json(base_url, "/health")
    checks.append(_api_json_check("/health", health))

    ready = request_json(base_url, "/ready")
    checks.append(_api_json_check("/ready", ready))
    ready_payload = _json_payload(ready)
    connector_mode = ""
    if isinstance(ready_payload, dict):
        connectors = ready_payload.get("connectors")
        if isinstance(connectors, dict):
            connector_mode = str(connectors.get("mode") or "")

    protected = request_json(base_url, "/suppliers")
    checks.append(_protected_route_check(protected))

    if skip_ui:
        checks.append(("Streamlit surface", True, "skipped by explicit --skip-ui option"))
    else:
        ui_health = request_json(ui_base_url, "/_stcore/health")
        checks.append(_ui_health_check(ui_health))

    if health_only:
        checks.append(("authenticated workflow", True, "skipped by explicit --health-only option"))
    elif headers:
        suppliers = request_json(base_url, "/suppliers", headers=headers)
        checks.append(_api_json_check("/suppliers with auth", suppliers))
        active_expected_tenant = expected_tenant or headers.get("X-Tenant-ID", "").strip()
        checks.extend(_tenant_isolation_checks(base_url, headers, active_expected_tenant))
        checks.extend(_workflow_checks(base_url, headers, connector_mode))
    else:
        checks.append(
            (
                "authenticated workflow",
                False,
                "missing credentials; set STAGING_BEARER_TOKEN or STAGING_TENANT_ID + STAGING_API_KEY",
            )
        )

    for name, ok, detail in checks:
        state = "PASS" if ok else "FAIL"
        print(f"{state} {name}: {redact(detail)}")
    return 0 if all(ok for _name, ok, _detail in checks) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test a staging Supplier Intelligence API.")
    parser.add_argument("--base-url", default=staging_base_url(os.environ), help="Staging API base URL.")
    parser.add_argument("--ui-base-url", default=staging_ui_base_url(os.environ), help="Staging Streamlit base URL.")
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Check public health/auth rejection only; skip authenticated evidence workflows.",
    )
    parser.add_argument(
        "--skip-ui",
        action="store_true",
        help="Skip the Streamlit health check explicitly.",
    )
    args = parser.parse_args(argv)
    try:
        headers = auth_headers(os.environ)
        config_errors = configuration_errors(
            os.environ,
            headers,
            health_only=args.health_only,
            skip_ui=args.skip_ui,
        )
        if config_errors:
            for error in config_errors:
                print(f"Smoke configuration error: {error}.", file=sys.stderr)
            return 2
        base_url = normalize_base_url(args.base_url)
        ui_base_url = "" if args.skip_ui else normalize_base_url(args.ui_base_url)
        return run_smoke(
            base_url,
            headers,
            health_only=args.health_only,
            expected_tenant=expected_tenant_id(os.environ, headers),
            ui_base_url=ui_base_url,
            skip_ui=args.skip_ui,
        )
    except Exception as exc:
        print(f"Smoke test failed: {redact(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
