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
    re.compile(r"(api[_-]?key[=:])[^,\s]+", re.IGNORECASE),
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
        raise ValueError("STAGING_BASE_URL or --base-url is required.")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("Staging base URL must start with http:// or https://.")
    return base_url.rstrip("/") + "/"


def auth_headers(env: Mapping[str, str]) -> dict[str, str]:
    token = env.get("STAGING_BEARER_TOKEN", "").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}
    tenant_id = env.get("STAGING_TENANT_ID", "").strip()
    api_key = env.get("STAGING_API_KEY", "").strip()
    if tenant_id and api_key:
        return {"X-Tenant-ID": tenant_id, "X-API-Key": api_key}
    return {}


def request_json(base_url: str, path: str, headers: Mapping[str, str] | None = None, timeout: int = 10) -> SmokeResponse:
    url = urljoin(base_url, path.lstrip("/"))
    request = Request(url, headers=dict(headers or {}), method="GET")
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
        return redact({key: payload.get(key) for key in ("status", "database", "api", "production_issues") if key in payload})
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


def run_smoke(base_url: str, headers: Mapping[str, str]) -> int:
    checks: list[tuple[str, bool, str]] = []

    live = request_json(base_url, "/live")
    checks.append(_api_json_check("/live", live))

    health = request_json(base_url, "/health")
    checks.append(_api_json_check("/health", health))

    ready = request_json(base_url, "/ready")
    checks.append(_api_json_check("/ready", ready))

    protected = request_json(base_url, "/suppliers")
    checks.append(_protected_route_check(protected))

    if headers:
        suppliers = request_json(base_url, "/suppliers", headers=headers)
        checks.append(_api_json_check("/suppliers with auth", suppliers))
    else:
        checks.append(("optional authenticated /suppliers", True, "skipped; set STAGING_BEARER_TOKEN or STAGING_TENANT_ID + STAGING_API_KEY"))

    for name, ok, detail in checks:
        state = "PASS" if ok else "FAIL"
        print(f"{state} {name}: {redact(detail)}")
    return 0 if all(ok for _name, ok, _detail in checks) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test a staging Supplier Intelligence API.")
    parser.add_argument("--base-url", default=os.getenv("STAGING_BASE_URL", ""), help="Staging API base URL.")
    args = parser.parse_args(argv)
    try:
        base_url = normalize_base_url(args.base_url)
        return run_smoke(base_url, auth_headers(os.environ))
    except Exception as exc:
        print(f"Smoke test failed: {redact(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
