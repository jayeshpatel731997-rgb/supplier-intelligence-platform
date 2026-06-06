"""Public filings connector with SEC EDGAR-compatible submissions mapping."""

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
    return [signal for signal in load_demo_supplier_signals() if signal.get("signal_type") == "financial"][:3]


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


def _sec_submissions_url(identifier: str) -> str:
    cik = "".join(character for character in identifier if character.isdigit())
    if not cik:
        return ""
    return f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"


def _filing_driver(form: str) -> tuple[str, float] | None:
    normalized = form.upper()
    if normalized in {"8-K", "6-K"}:
        return "Material public filing", 62
    if normalized.startswith("NT "):
        return "Late public filing signal", 72
    return None


def _sec_signals(payload: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    recent = payload.get("filings", {}).get("recent", {})
    accessions = list(recent.get("accessionNumber") or [])
    filing_dates = list(recent.get("filingDate") or [])
    forms = list(recent.get("form") or [])
    documents = list(recent.get("primaryDocument") or [])
    cik_digits = "".join(character for character in settings.filings_company_identifier if character.isdigit())
    archive_cik = str(int(cik_digits)) if cik_digits else ""
    signals: list[dict[str, Any]] = []
    for index, accession in enumerate(accessions[:25]):
        form = str(forms[index]) if index < len(forms) else "Filing"
        filing_date = str(filing_dates[index]) if index < len(filing_dates) else ""
        document = str(documents[index]) if index < len(documents) else ""
        accession_compact = str(accession).replace("-", "")
        source_url = (
            f"https://www.sec.gov/Archives/edgar/data/{archive_cik}/{accession_compact}/{document}"
            if archive_cik and document
            else _sec_submissions_url(settings.filings_company_identifier)
        )
        filing_risk = _filing_driver(form)
        if filing_risk is None:
            continue
        driver, severity = filing_risk
        signals.append(
            {
                "supplier_id": settings.public_connector_supplier_id,
                "supplier_name": settings.public_connector_supplier_name,
                "signal_id": f"sec-{accession_compact or hashlib.sha1(source_url.encode()).hexdigest()[:16]}",
                "signal_type": "financial",
                "driver": driver,
                "source": "SEC EDGAR",
                "source_url": source_url,
                "observed_at": _parse_date(filing_date),
                "severity": severity,
                "confidence": 0.72,
                "summary": f"{form} filed with SEC EDGAR on {filing_date or 'an unspecified date'}.",
            }
        )
    return signals


def _rss_signals(content: bytes, source_url: str, settings: Settings) -> list[dict[str, Any]]:
    parsed = feedparser.parse(content)
    if getattr(parsed, "bozo", False) and not getattr(parsed, "entries", []):
        raise RuntimeError(f"Unable to parse filing source: {source_url}")
    source_name = str(getattr(parsed.feed, "title", "") or "Public filing feed")
    signals: list[dict[str, Any]] = []
    for item in list(getattr(parsed, "entries", []))[:25]:
        title = str(item.get("title") or "Public filing")
        item_url = str(item.get("link") or "")
        filing_risk = _filing_driver(title.split(" ", 1)[0])
        if filing_risk is None:
            continue
        driver, severity = filing_risk
        digest = hashlib.sha1(f"{title}|{item_url}".encode("utf-8")).hexdigest()[:16]
        signals.append(
            {
                "supplier_id": settings.public_connector_supplier_id,
                "supplier_name": settings.public_connector_supplier_name,
                "signal_id": f"filing-{digest}",
                "signal_type": "financial",
                "driver": driver,
                "source": source_name,
                "source_url": item_url,
                "observed_at": _parse_date(str(item.get("published") or item.get("updated") or "")),
                "severity": severity,
                "confidence": 0.6,
                "summary": title,
            }
        )
    return signals


def _fetch_filing_sources(urls: list[str], settings: Settings) -> list[tuple[str, Any]]:
    fetched: list[tuple[str, Any]] = []
    for url in urls:
        response = fetch_with_retries(
            requests.get,
            url,
            headers={"User-Agent": settings.filings_user_agent},
            timeout_seconds=settings.connector_timeout_seconds,
            retry_count=settings.connector_retry_count,
        )
        fetched.append((url, response))
    return fetched


def run_filings_connector(settings: Settings, tenant_id: str) -> ConnectorRunResult:
    del tenant_id
    result = ConnectorRunResult(connector="filings", connector_type="filing_feed", status="completed")
    if settings.connector_mode in {"stub", "demo"}:
        result.signals = _demo_signals()
        result.metadata = {"mode": settings.connector_mode, "source_names": ["demo weak signals"]}
        return result.finish()
    if not settings.filings_company_identifier:
        result.status = "skipped"
        result.error = "Company identifier is required for public filings sync."
        result.metadata = {"mode": "public", "source_names": []}
        return result.finish()
    urls = _split_urls(settings.filings_source_urls)
    if not urls:
        sec_url = _sec_submissions_url(settings.filings_company_identifier)
        if sec_url:
            urls = [sec_url]
    if not urls:
        result.status = "skipped"
        result.error = "At least one public filing source URL is required for filings sync."
        result.metadata = {"mode": "public", "source_names": []}
        return result.finish()
    try:
        fetched = _fetch_filing_sources(urls, settings)
        for url, response in fetched:
            if "data.sec.gov/submissions/" in url:
                result.signals.extend(_sec_signals(response.json(), settings))
            else:
                result.signals.extend(_rss_signals(response.content, url, settings))
        if not result.signals:
            result.status = "skipped"
            result.error = "Configured filing sources returned no filing records."
        result.metadata = {"mode": "public", "source_names": urls}
        return result.finish()
    except Exception as exc:
        result.status = "failed"
        result.error = redact_secret_text(exc)
        result.metadata = {"mode": "public", "source_names": urls}
        return result.finish()
