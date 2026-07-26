"""News/RSS public connector."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from src.config import Settings
from src.observability.logging import redact_secret_text
from src.services.connectors.base import ConnectorRunResult, fetch_with_retries
from src.services.supplier_risk_evidence_chain import load_demo_supplier_signals


def _split_urls(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


def fetch_rss_entries(
    urls: list[str],
    timeout_seconds: int = 10,
    retry_count: int = 0,
) -> list[dict[str, Any]]:
    import feedparser

    entries: list[dict[str, Any]] = []
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
            raise RuntimeError(f"Unable to parse RSS source: {url}")
        for item in list(getattr(parsed, "entries", []))[:25]:
            published = item.get("published") or item.get("updated") or ""
            entries.append(
                {
                    "title": item.get("title", "Untitled public news item"),
                    "url": item.get("link") or "",
                    "published": published,
                    "source": getattr(parsed.feed, "title", "Public RSS"),
                    "summary": item.get("summary", item.get("description", "")),
                    "timeout_seconds": timeout_seconds,
                }
            )
    return entries


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


def _entry_to_signal(entry: dict[str, Any], settings: Settings) -> dict[str, Any] | None:
    title = str(entry.get("title") or "Public news signal")
    url = str(entry.get("url") or "")
    signal_id = "news-" + hashlib.sha1(f"{title}|{url}".encode("utf-8")).hexdigest()[:16]
    summary = str(entry.get("summary") or title)
    lowered = f"{title} {summary}".lower()
    supplier_terms = {
        settings.public_connector_supplier_name.strip().lower(),
        settings.public_connector_supplier_id.strip().lower(),
    }
    supplier_terms.discard("")
    if settings.news_require_supplier_match and not any(term in lowered for term in supplier_terms):
        return None
    severity = 72 if any(term in lowered for term in ("bankruptcy", "strike", "shutdown", "port", "disruption")) else 50
    return {
        "supplier_id": settings.public_connector_supplier_id,
        "supplier_name": settings.public_connector_supplier_name,
        "signal_id": signal_id,
        "signal_type": "news",
        "driver": "Public news disruption signal",
        "source": str(entry.get("source") or "Public RSS"),
        "source_url": url,
        "observed_at": _parse_date(str(entry.get("published") or "")),
        "severity": severity,
        "confidence": 0.62,
        "summary": title,
    }


def _demo_signals() -> list[dict[str, Any]]:
    return [
        {**signal, "source_system": "news"}
        for signal in load_demo_supplier_signals()
        if signal.get("signal_type") in {"news", "operational"}
    ][:3]


def run_news_connector(settings: Settings, tenant_id: str) -> ConnectorRunResult:
    del tenant_id
    result = ConnectorRunResult(connector="news", connector_type="rss_public", status="completed")
    if settings.connector_mode in {"stub", "demo"}:
        result.signals = _demo_signals()
        result.metadata = {"mode": settings.connector_mode, "source_names": ["demo weak signals"]}
        return result.finish()
    urls = _split_urls(settings.news_rss_urls)
    if not urls:
        result.status = "skipped"
        result.error = "At least one public news RSS URL is required for public news sync."
        result.metadata = {"mode": "public", "source_names": []}
        return result.finish()
    try:
        entries = fetch_rss_entries(
            urls,
            timeout_seconds=settings.connector_timeout_seconds,
            retry_count=settings.connector_retry_count,
        )
        result.signals = [
            signal
            for entry in entries
            if (signal := _entry_to_signal(entry, settings)) is not None
        ]
        if not result.signals:
            result.status = "skipped"
            result.error = "Configured news sources returned no supplier-matched entries."
        result.metadata = {"mode": "public", "source_names": urls}
        return result.finish()
    except Exception as exc:
        result.status = "failed"
        result.error = redact_secret_text(exc)
        result.metadata = {"mode": "public", "source_names": urls}
        return result.finish()
