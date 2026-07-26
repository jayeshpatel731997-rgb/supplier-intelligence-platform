"""Shared connector contracts and dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.config import Settings
from src.observability.logging import redact_secret_text


@dataclass(slots=True)
class ConnectorRunResult:
    connector: str
    connector_type: str
    status: str
    records_received: int = 0
    records_accepted: int = 0
    signals: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    def finish(self) -> "ConnectorRunResult":
        self.finished_at = datetime.now(UTC)
        self.records_received = len(self.signals) if self.records_received == 0 else self.records_received
        self.records_accepted = len(self.signals)
        self.error = redact_secret_text(self.error)
        return self


def fetch_with_retries(
    get,
    url: str,
    *,
    headers: dict[str, str],
    timeout_seconds: int,
    retry_count: int,
):
    last_error: Exception | None = None
    for _attempt in range(max(0, retry_count) + 1):
        try:
            response = get(url, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
    if last_error is None:
        raise RuntimeError("Connector fetch failed without an error.")
    raise last_error


def run_connector(connector_name: str, settings: Settings, tenant_id: str) -> ConnectorRunResult:
    name = connector_name.strip().lower()
    if name == "news":
        from src.services.connectors.news import run_news_connector

        return run_news_connector(settings, tenant_id)
    if name == "filings":
        from src.services.connectors.filings import run_filings_connector

        return run_filings_connector(settings, tenant_id)
    if name == "hiring":
        from src.services.connectors.hiring import run_hiring_connector

        return run_hiring_connector(settings, tenant_id)
    return ConnectorRunResult(
        connector=name,
        connector_type="unknown",
        status="failed",
        error=f"Unsupported connector: {name}",
    ).finish()
