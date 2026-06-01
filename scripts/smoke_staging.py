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
            return SmokeResponse(status=response.status, body=response.read().decode("utf-8", errors="replace"))
    except HTTPError as exc:
        return SmokeResponse(status=exc.code, body=exc.read().decode("utf-8", errors="replace"))
    except URLError as exc:
        raise RuntimeError(f"Request failed for {path}: {redact(exc)}") from exc


def _json_summary(body: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return ""
    if isinstance(payload, dict):
        return redact({key: payload.get(key) for key in ("status", "database", "api", "production_issues") if key in payload})
    return ""


def run_smoke(base_url: str, headers: Mapping[str, str]) -> int:
    checks: list[tuple[str, bool, str]] = []

    live = request_json(base_url, "/live")
    checks.append(("/live", live.status == 200, f"HTTP {live.status}"))

    health = request_json(base_url, "/health")
    checks.append(("/health", health.status == 200, f"HTTP {health.status} {_json_summary(health.body)}".strip()))

    ready = request_json(base_url, "/ready")
    checks.append(("/ready", ready.status == 200, f"HTTP {ready.status} {_json_summary(ready.body)}".strip()))

    protected = request_json(base_url, "/suppliers")
    checks.append(("/suppliers without auth", protected.status in {401, 403}, f"HTTP {protected.status}"))

    if headers:
        suppliers = request_json(base_url, "/suppliers", headers=headers)
        checks.append(("/suppliers with auth", suppliers.status == 200, f"HTTP {suppliers.status}"))
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
