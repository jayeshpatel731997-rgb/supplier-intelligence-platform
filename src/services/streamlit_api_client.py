"""Small API client helper for Streamlit deployment safety."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from src.observability.logging import redact_secret_text


def _secret_or_env(name: str, default: str = "") -> str:
    try:
        import streamlit as st

        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


@dataclass(slots=True)
class StreamlitApiClient:
    base_url: str
    tenant_id: str = ""
    api_key: str = ""
    bearer_token: str = ""
    timeout_seconds: int = 8

    @classmethod
    def from_env(
        cls,
        base_url: str | None = None,
        *,
        tenant_id: str | None = None,
    ) -> "StreamlitApiClient":
        active_base_url = (base_url or _secret_or_env("SUPPLIER_API_BASE_URL", "http://localhost:8000")).strip()
        host = (urlsplit(active_base_url).hostname or "").lower()
        local_default_key = "demo-api-key" if host in {"localhost", "127.0.0.1", "::1"} else ""
        return cls(
            base_url=active_base_url.rstrip("/") + "/",
            tenant_id=tenant_id
            or _secret_or_env("STAGING_TENANT_ID", _secret_or_env("DEFAULT_TENANT_ID", "demo-tenant")),
            api_key=_secret_or_env(
                "STAGING_API_KEY",
                _secret_or_env("SUPPLIER_DEMO_API_KEY", local_default_key),
            ),
            bearer_token=_secret_or_env("STAGING_BEARER_TOKEN", ""),
        )

    def build_url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))

    def headers(self) -> dict[str, str]:
        if self.bearer_token:
            return {"Authorization": f"Bearer {self.bearer_token}"}
        if self.tenant_id and self.api_key:
            return {"X-Tenant-ID": self.tenant_id, "X-API-Key": self.api_key}
        return {}

    def request(self, path: str, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers()}
        request = Request(self.build_url(path), data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API returned HTTP {exc.code}: {redact_secret_text(body)}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(friendly_api_error(exc)) from exc


def friendly_api_error(error: object) -> str:
    safe_error = redact_secret_text(error)
    return (
        "Supplier API is unreachable. Check SUPPLIER_API_BASE_URL, confirm the FastAPI service is running, "
        f"and verify staging network access. Details: {safe_error}"
    )
