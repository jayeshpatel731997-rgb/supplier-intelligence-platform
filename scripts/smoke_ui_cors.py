"""Smoke test staging UI + API readiness without requiring login or secrets."""

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
    re.compile(r"((?:api[_-]?key|token|secret|password|client[_-]?secret)[=:]\s*)[^,\s]+", re.IGNORECASE),
    re.compile(r"((?:Cookie|Set-Cookie):\s*)[^\r\n]+", re.IGNORECASE),
    re.compile(r"(postgres(?:ql)?(?:\+psycopg)?://[^:/@\s]+:)[^@\s]+(@)", re.IGNORECASE),
]


@dataclass(slots=True)
class UiCorsResult:
    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": redact(self.detail)}


def redact(value: object) -> str:
    text = str(value)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda match: f"{match.group(1)}***{match.group(2) if len(match.groups()) > 1 else ''}", text)
    return text


def _env_first(env: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = env.get(name, "").strip()
        if value:
            return value
    return ""


def normalize_url(value: str) -> str:
    if not value:
        raise ValueError("URL is required.")
    if not value.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://.")
    return value.rstrip("/") + "/"


def _request(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: int = 15,
    method: str = "GET",
) -> tuple[int, str, str, dict[str, str]]:
    request = Request(url, headers=dict(headers or {}), method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return (
                response.status,
                response.headers.get("Content-Type", ""),
                response.read().decode("utf-8", errors="replace"),
                dict(response.headers),
            )
    except HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read().decode("utf-8", errors="replace"), dict(exc.headers)


def _json_payload(body: str) -> object | None:
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _api_summary(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    summary: dict[str, object] = {key: payload.get(key) for key in ("status",) if key in payload}
    database = payload.get("database")
    if isinstance(database, dict):
        summary["database"] = {key: database.get(key) for key in ("ok", "driver") if key in database}
    api = payload.get("api")
    if isinstance(api, dict):
        summary["api"] = {key: api.get(key) for key in ("ok", "status") if key in api}
    issues = payload.get("production_issues")
    if isinstance(issues, list):
        summary["production_issue_count"] = len(issues)
    return summary


def run_ui_cors_smoke(env: Mapping[str, str]) -> list[UiCorsResult]:
    ui_url = _env_first(env, "STAGING_UI_URL", "STAGING_UI_BASE_URL")
    api_url = _env_first(env, "STAGING_API_URL", "STAGING_API_BASE_URL", "STAGING_BASE_URL")
    results: list[UiCorsResult] = []
    if not ui_url:
        results.append(UiCorsResult("staging_ui", "SKIP", "STAGING_UI_URL/STAGING_UI_BASE_URL not configured."))
    if not api_url:
        results.append(UiCorsResult("staging_api", "SKIP", "STAGING_API_URL/STAGING_API_BASE_URL not configured."))
    if results:
        return results

    try:
        ui_base = normalize_url(ui_url)
        api_base = normalize_url(api_url)
        ui_status, ui_content_type, ui_body, _ui_headers = _request(ui_base)
        ui_usable = ui_status == 200 and bool(ui_body.strip())
        results.append(
            UiCorsResult(
                "staging_ui_page",
                "PASS" if ui_usable else "FAIL",
                f"HTTP {ui_status}; content_type={ui_content_type}; body_present={bool(ui_body.strip())}",
            )
        )

        health_status, health_content_type, health_body, _health_headers = _request(urljoin(api_base, "health"))
        health_payload = _json_payload(health_body)
        health_ok = health_status == 200 and isinstance(health_payload, dict) and health_payload.get("status") == "ok"
        results.append(
            UiCorsResult(
                "staging_api_health",
                "PASS" if health_ok else "FAIL",
                f"HTTP {health_status}; content_type={health_content_type}; payload={_api_summary(health_payload)}",
            )
        )

        ready_status, ready_content_type, ready_body, _ready_headers = _request(urljoin(api_base, "ready"))
        ready_payload = _json_payload(ready_body)
        ready_structured = ready_status in {200, 503} and isinstance(ready_payload, dict) and "status" in ready_payload
        results.append(
            UiCorsResult(
                "staging_api_ready_structure",
                "PASS" if ready_structured else "FAIL",
                f"HTTP {ready_status}; content_type={ready_content_type}; payload={_api_summary(ready_payload)}",
            )
        )
        cors_status, _cors_content_type, _cors_body, cors_headers = _request(
            urljoin(api_base, "health"),
            method="OPTIONS",
            headers={
                "Origin": ui_base.rstrip("/"),
                "Access-Control-Request-Method": "GET",
            },
        )
        allowed_origin = cors_headers.get("Access-Control-Allow-Origin", "")
        cors_ok = cors_status in {200, 204} and allowed_origin == ui_base.rstrip("/")
        results.append(
            UiCorsResult(
                "staging_api_cors_preflight",
                "PASS" if cors_ok else "WARN",
                f"HTTP {cors_status}; allow_origin_matches_ui={allowed_origin == ui_base.rstrip('/')}",
            )
        )
    except (ValueError, URLError, TimeoutError) as exc:
        results.append(UiCorsResult("ui_cors_smoke", "FAIL", f"UI/CORS smoke failed: {redact(exc)}"))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test staging UI and public API readiness endpoints.")
    parser.add_argument("--ui-url", default=_env_first(os.environ, "STAGING_UI_URL", "STAGING_UI_BASE_URL"))
    parser.add_argument("--api-url", default=_env_first(os.environ, "STAGING_API_URL", "STAGING_API_BASE_URL", "STAGING_BASE_URL"))
    args = parser.parse_args(argv)
    env = {**os.environ, "STAGING_UI_URL": args.ui_url, "STAGING_API_URL": args.api_url}
    results = run_ui_cors_smoke(env)
    payload = {"results": [result.to_dict() for result in results]}
    print(json.dumps(payload, indent=2))
    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
