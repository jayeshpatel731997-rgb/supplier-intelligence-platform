from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from tests.test_backend_api import _client, _headers


def test_demo_news_connector_sync_imports_weak_signals(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLIER_CONNECTOR_MODE", "demo")
    client = _client(monkeypatch, tmp_path)

    response = client.post("/evidence/connectors/news/sync", headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["connector"] == "news"
    assert body["status"] == "completed"
    assert body["records_accepted"] >= 1

    signals = client.get("/evidence/signals", headers=_headers())
    assert signals.status_code == 200
    assert any(signal["source_system"] == "news" for signal in signals.json())


def test_public_connector_failure_records_failed_sync_without_breaking_evidence_run(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("SUPPLIER_CONNECTOR_MODE", "public")
    monkeypatch.setenv("SUPPLIER_NEWS_RSS_URLS", "https://feeds.example.invalid/rss")
    client = _client(monkeypatch, tmp_path)

    import src.services.connectors.news as news_connector

    def failing_fetch(*_args, **_kwargs):
        raise TimeoutError("network timeout with token=secret-value")

    monkeypatch.setattr(news_connector, "fetch_rss_entries", failing_fetch)

    sync = client.post("/evidence/connectors/news/sync", headers=_headers())
    run = client.post("/evidence/runs", json={"include_demo_signals": True}, headers=_headers())

    assert sync.status_code == 200
    assert sync.json()["status"] == "failed"
    assert "secret-value" not in sync.json()["error"]
    assert run.status_code == 200
    assert run.json()["status"] == "completed"


def test_connector_error_redaction_removes_secret_urls(monkeypatch):
    from src.observability.logging import redact_secret_text

    error = (
        "GET https://feed.example.test/token-secret/rss?key=also-secret failed; "
        "Max retries exceeded with url: /token-secret/rss?api_key=also-secret"
    )

    safe = redact_secret_text(error)

    assert "token-secret" not in safe
    assert "also-secret" not in safe
    assert "https://feed.example.test" in safe


def test_public_connectors_skip_when_sources_are_not_configured():
    from src.config import Settings
    from src.services.connectors.hiring import run_hiring_connector
    from src.services.connectors.news import run_news_connector

    settings = Settings(connector_mode="public")

    news = run_news_connector(settings, "demo-tenant")
    hiring = run_hiring_connector(settings, "demo-tenant")

    assert news.status == "skipped"
    assert news.signals == []
    assert hiring.status == "skipped"
    assert hiring.signals == []


def test_connector_sync_history_is_tenant_scoped(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    client.post("/tenants", json={"tenant_id": "tenant-b", "name": "Tenant B"}, headers=_headers())
    key = client.post(
        "/tenants/tenant-b/api-keys",
        json={"username": "risk-b@example.com", "role": "risk_manager", "label": "tenant-b key"},
        headers=_headers(),
    ).json()["api_key"]
    tenant_b_headers = _headers(tenant_id="tenant-b", api_key=key)

    assert client.post("/evidence/connectors/hiring/sync", headers=_headers()).status_code == 200
    assert client.post("/evidence/connectors/hiring/sync", headers=tenant_b_headers).status_code == 200

    tenant_a = client.get("/evidence/connectors/syncs", headers=_headers()).json()
    tenant_b = client.get("/evidence/connectors/syncs", headers=tenant_b_headers).json()

    assert {row["tenant_id"] for row in tenant_a} == {"demo-tenant"}
    assert {row["tenant_id"] for row in tenant_b} == {"tenant-b"}


def test_connector_sync_history_redacts_secret_bearing_source_urls(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLIER_CONNECTOR_MODE", "public")
    monkeypatch.setenv("SUPPLIER_HIRING_SOURCE_URLS", "https://feed.example.test/jobs?api_key=secret-token")

    class Response:
        content = b"<rss><channel><title>Empty feed</title></channel></rss>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr("src.services.connectors.hiring.requests.get", lambda *args, **kwargs: Response())
    client = _client(monkeypatch, tmp_path)

    response = client.post("/evidence/connectors/hiring/sync", headers=_headers())
    syncs = client.get("/evidence/connectors/syncs", headers=_headers())

    assert response.status_code == 200
    assert "secret-token" not in str(response.json())
    assert "secret-token" not in str(syncs.json())
    assert syncs.json()[0]["metadata"]["source_names"] == ["https://feed.example.test"]


def test_connector_sync_history_removes_secret_bearing_url_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLIER_CONNECTOR_MODE", "public")
    monkeypatch.setenv("SUPPLIER_HIRING_SOURCE_URLS", "https://feed.example.test/token-secret/jobs.rss")

    class Response:
        content = b"<rss><channel><title>Empty feed</title></channel></rss>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr("src.services.connectors.hiring.requests.get", lambda *args, **kwargs: Response())
    client = _client(monkeypatch, tmp_path)

    client.post("/evidence/connectors/hiring/sync", headers=_headers())
    syncs = client.get("/evidence/connectors/syncs", headers=_headers()).json()

    assert "token-secret" not in str(syncs)
    assert syncs[0]["metadata"]["source_names"] == ["https://feed.example.test"]


def test_news_public_connector_applies_configured_timeout(monkeypatch):
    from src.services.connectors.news import fetch_rss_entries

    observed = {}

    class Response:
        content = b"""
        <rss><channel><title>Demo</title>
        <item><title>Public Source Supplier disruption</title></item>
        </channel></rss>
        """

        def raise_for_status(self):
            return None

    def fake_get(url, headers=None, timeout=None):
        observed["url"] = url
        observed["headers"] = headers
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setattr("src.services.connectors.news.requests.get", fake_get)

    fetch_rss_entries(["https://feed.example.test/rss"], timeout_seconds=3)

    assert observed["timeout"] == 3


def test_news_public_connector_retries_transient_fetch_failure(monkeypatch):
    from src.config import Settings
    from src.services.connectors.news import run_news_connector

    attempts = {"count": 0}

    class Response:
        content = b"""
        <rss><channel><title>Demo</title>
        <item><title>Public Source Supplier disruption</title></item>
        </channel></rss>
        """

        def raise_for_status(self):
            return None

    def flaky_get(*args, **kwargs):
        del args, kwargs
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("temporary timeout")
        return Response()

    monkeypatch.setattr("src.services.connectors.news.requests.get", flaky_get)

    result = run_news_connector(
        Settings(
            connector_mode="public",
            news_rss_urls="https://feed.example.test/rss",
            connector_retry_count=1,
        ),
        "demo-tenant",
    )

    assert result.status == "completed"
    assert attempts["count"] == 2


def test_news_public_connector_preserves_rfc822_publication_date(monkeypatch):
    from src.config import Settings
    from src.services.connectors.news import run_news_connector

    class Response:
        content = b"""
        <rss><channel><title>Demo Feed</title>
        <item><title>Supplier One port disruption</title><link>https://example.test/a</link>
        <pubDate>Tue, 04 Jun 2026 10:00:00 GMT</pubDate></item>
        </channel></rss>
        """

        def raise_for_status(self):
            return None

    monkeypatch.setattr("src.services.connectors.news.requests.get", lambda *args, **kwargs: Response())

    result = run_news_connector(
        Settings(
            connector_mode="public",
            news_rss_urls="https://feed.example.test/rss",
            public_connector_supplier_name="Supplier One",
        ),
        "demo-tenant",
    )

    assert result.status == "completed"
    assert result.signals[0]["observed_at"] == "2026-06-04"


def test_news_connector_does_not_persist_secret_feed_url_as_item_url(monkeypatch):
    from src.config import Settings
    from src.services.connectors.news import run_news_connector

    class Response:
        content = b"""
        <rss><channel><title>Private Feed URL</title>
        <item><title>Supplier One port disruption</title><description>Delay reported.</description></item>
        </channel></rss>
        """

        def raise_for_status(self):
            return None

    monkeypatch.setattr("src.services.connectors.news.requests.get", lambda *args, **kwargs: Response())

    result = run_news_connector(
        Settings(
            connector_mode="public",
            news_rss_urls="https://feed.example.test/token-secret/rss?api_key=also-secret",
            public_connector_supplier_name="Supplier One",
        ),
        "demo-tenant",
    )

    assert result.status == "completed"
    assert result.signals[0]["source_url"] == ""
    assert "secret" not in str(result.signals)


def test_news_connector_skips_entries_without_supplier_match(monkeypatch):
    from src.config import Settings
    from src.services.connectors.news import run_news_connector

    class Response:
        content = b"""
        <rss><channel><title>General News</title>
        <item><title>Unrelated company announces layoffs</title></item>
        </channel></rss>
        """

        def raise_for_status(self):
            return None

    monkeypatch.setattr("src.services.connectors.news.requests.get", lambda *args, **kwargs: Response())

    result = run_news_connector(
        Settings(
            connector_mode="public",
            news_rss_urls="https://feed.example.test/rss",
            public_connector_supplier_name="Supplier One",
        ),
        "demo-tenant",
    )

    assert result.status == "skipped"
    assert result.signals == []


def test_hiring_public_connector_fetches_and_maps_source_entries(monkeypatch):
    from src.config import Settings
    from src.services.connectors.hiring import run_hiring_connector

    class Response:
        content = b"""
        <rss><channel><title>Supplier Jobs</title>
        <item><title>Supplier announces workforce reduction</title>
        <link>https://jobs.example.test/layoff</link>
        <pubDate>Wed, 05 Jun 2026 10:00:00 GMT</pubDate>
        <description>Hiring freeze and layoffs announced.</description></item>
        </channel></rss>
        """

        def raise_for_status(self):
            return None

    monkeypatch.setattr("src.services.connectors.hiring.requests.get", lambda *args, **kwargs: Response())

    result = run_hiring_connector(
        Settings(
            connector_mode="public",
            hiring_source_urls="https://jobs.example.test/rss",
            public_connector_supplier_id="supplier-1",
            public_connector_supplier_name="Supplier One",
        ),
        "demo-tenant",
    )

    assert result.status == "completed"
    assert result.records_accepted == 1
    signal = result.signals[0]
    assert signal["supplier_id"] == "supplier-1"
    assert signal["signal_type"] == "hiring"
    assert signal["driver"] == "Hiring slowdown or workforce reduction"
    assert signal["source_url"] == "https://jobs.example.test/layoff"
    assert signal["observed_at"] == "2026-06-05"


def test_filings_public_connector_validates_source_url_before_accepting(monkeypatch):
    from src.config import Settings
    from src.services.connectors.filings import run_filings_connector

    def failing_get(*args, **kwargs):
        raise TimeoutError("filing source timed out")

    monkeypatch.setattr("src.services.connectors.filings.requests.get", failing_get)

    result = run_filings_connector(
        Settings(
            connector_mode="public",
            filings_company_identifier="0000000000",
            filings_source_urls="https://filings.example.test/feed",
        ),
        "demo-tenant",
    )

    assert result.status == "failed"
    assert result.signals == []
    assert "timed out" in result.error


def test_filings_connector_maps_sec_submissions_to_financial_signals(monkeypatch):
    from src.config import Settings
    from src.services.connectors.filings import run_filings_connector

    class Response:
        content = b"{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "name": "Supplier Public Co",
                "filings": {
                    "recent": {
                        "accessionNumber": ["0001234567-26-000001"],
                        "filingDate": ["2026-06-01"],
                        "form": ["8-K"],
                        "primaryDocument": ["supplier-8k.htm"],
                    }
                },
            }

    monkeypatch.setattr("src.services.connectors.filings.requests.get", lambda *args, **kwargs: Response())

    result = run_filings_connector(
        Settings(
            connector_mode="public",
            filings_company_identifier="1234567",
            public_connector_supplier_id="supplier-1",
            public_connector_supplier_name="Supplier One",
        ),
        "demo-tenant",
    )

    assert result.status == "completed"
    assert result.records_accepted == 1
    signal = result.signals[0]
    assert signal["supplier_id"] == "supplier-1"
    assert signal["signal_type"] == "financial"
    assert signal["driver"] == "Material public filing"
    assert signal["observed_at"] == "2026-06-01"
    assert "supplier-8k.htm" in signal["source_url"]
    assert signal["source"] == "SEC EDGAR"


def test_filings_connector_skips_routine_sec_forms(monkeypatch):
    from src.config import Settings
    from src.services.connectors.filings import run_filings_connector

    class Response:
        content = b"{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "filings": {
                    "recent": {
                        "accessionNumber": ["0001234567-26-000002", "0001234567-26-000003"],
                        "filingDate": ["2026-05-01", "2026-05-02"],
                        "form": ["10-Q", "4"],
                        "primaryDocument": ["quarterly.htm", "ownership.htm"],
                    }
                }
            }

    monkeypatch.setattr("src.services.connectors.filings.requests.get", lambda *args, **kwargs: Response())

    result = run_filings_connector(
        Settings(connector_mode="public", filings_company_identifier="1234567"),
        "demo-tenant",
    )

    assert result.status == "skipped"
    assert result.signals == []


def test_filings_connector_skips_without_company_identifier(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLIER_CONNECTOR_MODE", "public")
    client = _client(monkeypatch, tmp_path)

    response = client.post("/evidence/connectors/filings/sync", headers=_headers())

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    assert "identifier" in response.json()["error"].lower()


def test_mocked_llm_narrative_uses_structured_output_only():
    from src.services.narrative_provider import MockNarrativeProvider, build_evidence_narrative

    reports = [
        {
            "supplier_id": "sup-1",
            "supplier_name": "Apex Components",
            "risk_score": 84.0,
            "risk_level": "critical",
            "top_risk_drivers": [{"driver": "Cash stress", "sources": ["Public filing"]}],
            "evidence_chain": [
                {
                    "signal_id": "sig-1",
                    "source": "Public filing",
                    "source_url": "https://example.test/filing",
                    "summary": "Liquidity warning disclosed in public record.",
                }
            ],
            "recommended_actions": ["Request liquidity evidence."],
            "confidence": 0.8,
        }
    ]
    provider = MockNarrativeProvider(
        {
            "risk_summary": "Apex Components is critical risk based on cited filing evidence.",
            "top_drivers": ["Cash stress"],
            "recommended_actions": ["Request liquidity evidence."],
            "confidence_caveats": ["Single public source."],
            "unsupported_field": "must be removed",
        }
    )

    narrative = build_evidence_narrative(reports, "default-v1", provider=provider)

    assert set(narrative) == {
        "policy",
        "provider",
        "model",
        "scoring_version",
        "risk_summary",
        "top_drivers",
        "recommended_actions",
        "confidence_caveats",
    }
    assert narrative["provider"] == "mock"
    assert "STRUCTURED EVIDENCE ONLY" in narrative["policy"]
    assert narrative["risk_summary"] != provider.response["risk_summary"]
    assert narrative["confidence_caveats"] != provider.response["confidence_caveats"]
    assert "Apex Components is critical risk" in narrative["risk_summary"]


def test_llm_narrative_falls_back_when_claims_are_not_supported():
    from src.services.narrative_provider import MockNarrativeProvider, build_evidence_narrative

    reports = [
        {
            "supplier_id": "sup-1",
            "supplier_name": "Apex Components",
            "risk_score": 84.0,
            "risk_level": "critical",
            "top_risk_drivers": [{"driver": "Cash stress", "sources": ["Public filing"]}],
            "evidence_chain": [{"source": "Public filing", "source_url": "https://example.test/filing"}],
            "recommended_actions": ["Request liquidity evidence."],
            "confidence": 0.8,
        }
    ]
    provider = MockNarrativeProvider(
        {
            "risk_summary": "A hidden source confirms a sanctions violation.",
            "top_drivers": ["Sanctions violation"],
            "recommended_actions": ["Terminate the supplier immediately."],
            "confidence_caveats": [],
        }
    )

    narrative = build_evidence_narrative(reports, "default-v1", provider=provider)

    assert narrative["provider"] == "none"
    assert "Sanctions violation" not in narrative["top_drivers"]
    assert "Terminate the supplier immediately." not in narrative["recommended_actions"]


def test_calibration_evaluates_historical_outcomes_without_accuracy_claims(tmp_path):
    from scripts.calibrate_supplier_risk import HistoricalOutcome, RiskScoreSnapshot, evaluate_calibration

    report = evaluate_calibration(
        outcomes=[
            HistoricalOutcome(
                supplier_id="sup-1",
                event_type="late_delivery",
                event_date=datetime(2026, 5, 1, tzinfo=UTC),
                severity=0.8,
                notes="Customer escalation",
                source="demo",
            )
        ],
        scores=[
            RiskScoreSnapshot(
                supplier_id="sup-1",
                scored_at=datetime(2026, 4, 20, tzinfo=UTC),
                risk_score=82.0,
                risk_level="critical",
            )
        ],
    )

    assert report["coverage"] == 1.0
    assert report["matched_examples"] == 1
    assert report["accuracy_claim"] == "not_claimed"
    assert report["review_lists"]["false_negative_review"] == []


def test_demo_scenarios_load_required_risk_stories():
    from scripts.seed_demo_data import load_demo_scenarios

    scenarios = load_demo_scenarios()
    scenario_keys = {item["scenario"] for item in scenarios}

    assert {
        "normal_supplier",
        "financial_distress",
        "logistics_disruption",
        "hiring_slowdown",
        "compliance_issue",
        "multi_signal_high_risk",
    } <= scenario_keys
