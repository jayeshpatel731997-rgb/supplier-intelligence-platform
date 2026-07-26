"""Hiring-signal connector."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import requests

from src.config import Settings
from src.observability.logging import redact_secret_text
from src.services.connectors.base import ConnectorRunResult, fetch_with_retries
from src.services.supplier_risk_evidence_chain import load_demo_supplier_signals


def _split_urls(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


def _demo_signals() -> list[dict[str, Any]]:
    return [signal for signal in load_demo_supplier_signals() if signal.get("signal_type") == "hiring"][:3]


def _parse_date(value: str) -> str:
    if not value:
        return datetime.now(UTC).date().isoformat()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        try:
            return parsedate_to_datetime(value).date().isoformat()
        except (TypeError, ValueError):
            return datetime.now(UTC).date().isoformat()


def _entry_to_signal(entry: dict[str, Any], source_name: str, settings: Settings) -> dict[str, Any]:
    title = str(entry.get("title") or "Public hiring signal")
    summary = str(entry.get("summary") or entry.get("description") or title)
    source_url = str(entry.get("link") or "")
    digest = hashlib.sha1(f"{title}|{source_url}".encode("utf-8")).hexdigest()[:16]
    lowered = f"{title} {summary}".lower()
    contraction_terms = ("layoff", "workforce reduction", "hiring freeze", "slowdown", "job cuts")
    growth_terms = ("hiring growth", "expansion", "new jobs", "workforce growth")
    if any(term in lowered for term in contraction_terms):
        driver = "Hiring slowdown or workforce reduction"
        severity = 68
    elif any(term in lowered for term in growth_terms):
        driver = "Hiring growth or capacity expansion"
        severity = 25
    else:
        driver = "Hiring trend signal"
        severity = 45
    return {
        "supplier_id": settings.public_connector_supplier_id,
        "supplier_name": settings.public_connector_supplier_name,
        "signal_id": f"hiring-{digest}",
        "signal_type": "hiring",
        "driver": driver,
        "source": source_name or "Configured public hiring source",
        "source_url": source_url,
        "observed_at": _parse_date(str(entry.get("published") or entry.get("updated") or "")),
        "severity": severity,
        "confidence": 0.6,
        "summary": title,
    }


def fetch_hiring_entries(
    urls: list[str],
    *,
    timeout_seconds: int,
    retry_count: int,
) -> list[tuple[dict[str, Any], str]]:
    entries: list[tuple[dict[str, Any], str]] = []
    for url in urls:
        response = fetch_with_retries(
            requests.get,
            url,
            headers={"User-Agent": "supplier-intelligence-staging/1.0"},
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
        )
        parsed = feedparser.parse(response.content)
        if getattr(parsed, "bozo", False) and not getattr(parsed, "entries", []):
            raise RuntimeError(f"Unable to parse hiring source: {url}")
        source_name = str(getattr(parsed.feed, "title", "") or "Configured public hiring source")
        entries.extend((dict(item), source_name) for item in list(getattr(parsed, "entries", []))[:25])
    return entries


def run_hiring_connector(settings: Settings, tenant_id: str) -> ConnectorRunResult:
    del tenant_id
    result = ConnectorRunResult(connector="hiring", connector_type="labor_market_feed", status="completed")
    if settings.connector_mode in {"stub", "demo"}:
        result.signals = _demo_signals()
        result.metadata = {"mode": settings.connector_mode, "source_names": ["demo weak signals"]}
        return result.finish()
    urls = _split_urls(settings.hiring_source_urls)
    if not urls:
        result.status = "skipped"
        result.error = "At least one public hiring RSS URL is required for public hiring sync."
        result.metadata = {"mode": "public", "source_names": []}
        return result.finish()
    try:
        entries = fetch_hiring_entries(
            urls,
            timeout_seconds=settings.connector_timeout_seconds,
            retry_count=settings.connector_retry_count,
        )
        result.signals = [_entry_to_signal(entry, source_name, settings) for entry, source_name in entries]
        if not result.signals:
            result.status = "skipped"
            result.error = "Configured hiring sources returned no entries."
        result.metadata = {"mode": "public", "source_names": urls}
        return result.finish()
    except Exception as exc:
        result.status = "failed"
        result.error = redact_secret_text(exc)
        result.metadata = {"mode": "public", "source_names": urls}
        return result.finish()
