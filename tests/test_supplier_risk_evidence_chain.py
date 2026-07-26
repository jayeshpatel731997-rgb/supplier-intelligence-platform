from __future__ import annotations

import importlib

from src.services.supplier_risk_evidence_chain import (
    build_supplier_risk_evidence_chains,
    load_demo_supplier_signals,
)


def test_risk_score_orders_supplier_with_multiple_severe_signals_first():
    reports = build_supplier_risk_evidence_chains(load_demo_supplier_signals())

    assert len(reports) >= 3
    assert reports[0]["supplier_id"] == "sup-vn-tech"
    assert reports[0]["risk_score"] > reports[-1]["risk_score"]
    assert reports[0]["risk_level"] in {"critical", "high"}


def test_demo_dataset_has_multiple_signals_per_supplier():
    signals = load_demo_supplier_signals()
    counts: dict[str, int] = {}
    for signal in signals:
        counts[signal["supplier_id"]] = counts.get(signal["supplier_id"], 0) + 1

    assert 3 <= len(counts) <= 5
    assert all(count >= 2 for count in counts.values())


def test_risk_level_classification_uses_score_bands():
    reports = build_supplier_risk_evidence_chains(load_demo_supplier_signals())
    by_supplier = {item["supplier_id"]: item for item in reports}

    assert by_supplier["sup-vn-tech"]["risk_level"] == "critical"
    assert by_supplier["sup-mx-apex"]["risk_level"] == "high"
    assert by_supplier["sup-us-machining"]["risk_level"] == "low"


def test_evidence_chain_links_each_top_driver_to_source_signal():
    report = next(
        item
        for item in build_supplier_risk_evidence_chains(load_demo_supplier_signals())
        if item["supplier_id"] == "sup-vn-tech"
    )

    assert report["top_risk_drivers"]
    assert report["evidence_chain"]
    driver_names = {driver["driver"] for driver in report["top_risk_drivers"]}
    evidence_drivers = {item["driver"] for item in report["evidence_chain"]}
    assert driver_names <= evidence_drivers
    for item in report["evidence_chain"]:
        assert item["signal_id"]
        assert item["source"]
        assert item["summary"]


def test_recommended_actions_reflect_risk_level_and_drivers():
    reports = build_supplier_risk_evidence_chains(load_demo_supplier_signals())
    high_risk = next(item for item in reports if item["supplier_id"] == "sup-vn-tech")
    low_risk = next(item for item in reports if item["supplier_id"] == "sup-us-machining")

    assert any("qualify alternate supplier" in action.lower() for action in high_risk["recommended_actions"])
    assert any("increase safety stock" in action.lower() for action in high_risk["recommended_actions"])
    assert any("hiring" in action.lower() for action in high_risk["recommended_actions"])
    assert any("standard monitoring" in action.lower() for action in low_risk["recommended_actions"])


def test_service_module_import_is_safe_without_live_api_keys(monkeypatch):
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    module = importlib.reload(importlib.import_module("src.services.supplier_risk_evidence_chain"))

    assert module.load_demo_supplier_signals()
