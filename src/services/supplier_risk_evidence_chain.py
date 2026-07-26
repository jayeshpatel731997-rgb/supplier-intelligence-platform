"""Deterministic supplier weak-signal evidence chain pilot."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEMO_SIGNAL_PATH = Path(__file__).resolve().parents[2] / "data" / "supplier_risk_weak_signals.json"

SIGNAL_TYPE_WEIGHTS = {
    "financial": 1.25,
    "audit": 1.15,
    "operational": 1.1,
    "news": 1.0,
    "email": 0.95,
    "hiring": 1.05,
}


def load_demo_supplier_signals(path: str | Path = DEMO_SIGNAL_PATH) -> list[dict[str, Any]]:
    """Load deterministic pilot signals from the local sample dataset."""
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    return list(data)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _risk_level(score: float) -> str:
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def _signal_points(signal: dict[str, Any], signal_type_weights: dict[str, float] | None = None) -> float:
    severity = _clamp(float(signal.get("severity", 0)), 0, 100)
    confidence = _clamp(float(signal.get("confidence", 0.5)), 0, 1)
    signal_type = str(signal.get("signal_type", "")).lower()
    active_weights = signal_type_weights or SIGNAL_TYPE_WEIGHTS
    return severity * confidence * float(active_weights.get(signal_type, 1.0))


def _build_top_drivers(
    signals: list[dict[str, Any]],
    signal_type_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for signal in signals:
        grouped[str(signal.get("driver") or "Unclassified signal")].append(signal)

    drivers = []
    for driver, driver_signals in grouped.items():
        contribution = sum(_signal_points(signal, signal_type_weights) for signal in driver_signals)
        strongest = max(driver_signals, key=lambda signal: _signal_points(signal, signal_type_weights))
        drivers.append(
            {
                "driver": driver,
                "contribution": round(contribution, 1),
                "signal_ids": [str(signal["signal_id"]) for signal in driver_signals],
                "sources": sorted({str(signal.get("source", "")) for signal in driver_signals}),
                "explanation": strongest.get("summary", ""),
            }
        )
    return sorted(drivers, key=lambda item: item["contribution"], reverse=True)


def _build_evidence_chain(
    signals: list[dict[str, Any]],
    top_drivers: list[dict[str, Any]],
    signal_type_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    top_signal_ids = {signal_id for driver in top_drivers for signal_id in driver["signal_ids"]}
    evidence = []
    for signal in sorted(signals, key=lambda item: _signal_points(item, signal_type_weights), reverse=True):
        if signal["signal_id"] not in top_signal_ids:
            continue
        evidence.append(
            {
                "driver": signal.get("driver", "Unclassified signal"),
                "signal_id": signal["signal_id"],
                "signal_type": signal.get("signal_type", ""),
                "source": signal.get("source", ""),
                "source_url": signal.get("source_url", ""),
                "observed_at": signal.get("observed_at", ""),
                "severity": int(signal.get("severity", 0)),
                "confidence": round(float(signal.get("confidence", 0)), 2),
                "summary": signal.get("summary", ""),
            }
        )
    return evidence


def _recommended_actions(level: str, top_drivers: list[dict[str, Any]]) -> list[str]:
    driver_text = " ".join(driver["driver"].lower() for driver in top_drivers)
    actions: list[str] = []

    if level in {"critical", "high"}:
        actions.append("Qualify alternate supplier capacity for critical parts within 14 days.")
        actions.append("Contact the supplier for a written recovery plan, current backlog, and shipment status.")
        actions.append("Review contract terms for service-level, force-majeure, and recovery-time protections.")
    elif level == "medium":
        actions.append("Move the supplier to heightened monitoring with a monthly risk review.")
    else:
        actions.append("Continue standard monitoring cadence unless new signals emerge.")

    if "hiring" in driver_text:
        actions.append("Review hiring contraction with the supplier and confirm staffing for constrained work centers.")
    if "cash" in driver_text or "financial" in driver_text:
        actions.append("Request updated liquidity, insurance, and continuity evidence before new volume awards.")
    if "logistics" in driver_text or "delivery" in driver_text or "port" in driver_text:
        actions.append("Increase safety stock where cover is thin and expedite open POs for exposed SKUs.")
        actions.append("Monitor port and geographic disruption until lead times normalize.")
    if "quality" in driver_text or "audit" in driver_text:
        actions.append("Audit supplier controls and require dated corrective-action closure evidence.")

    return list(dict.fromkeys(actions))[:5]


def build_supplier_risk_evidence_chains(
    signals: list[dict[str, Any]],
    signal_type_weights: dict[str, float] | None = None,
    supplier_criticality: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate weak signals into explainable supplier risk reports."""
    by_supplier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for signal in signals:
        by_supplier[str(signal["supplier_id"])].append(signal)

    reports = []
    for supplier_id, supplier_signals in by_supplier.items():
        supplier_name = str(supplier_signals[0].get("supplier_name", supplier_id))
        raw_points = sum(_signal_points(signal, signal_type_weights) for signal in supplier_signals)
        diversity_bonus = max(0, len({signal.get("signal_type") for signal in supplier_signals}) - 1) * 4
        criticality_factor = float((supplier_criticality or {}).get(supplier_id, 1.0))
        risk_score = round(_clamp((raw_points / 1.8 + diversity_bonus) * criticality_factor, 0, 100), 1)
        level = _risk_level(risk_score)
        top_drivers = _build_top_drivers(supplier_signals, signal_type_weights)[:3]
        confidence = sum(float(signal.get("confidence", 0.5)) for signal in supplier_signals) / len(supplier_signals)

        reports.append(
            {
                "supplier_id": supplier_id,
                "supplier_name": supplier_name,
                "risk_score": risk_score,
                "risk_level": level,
                "top_risk_drivers": top_drivers,
                "evidence_chain": _build_evidence_chain(supplier_signals, top_drivers, signal_type_weights),
                "recommended_actions": _recommended_actions(level, top_drivers),
                "confidence": round(confidence, 2),
            }
        )

    return sorted(reports, key=lambda item: item["risk_score"], reverse=True)


def build_governed_narrative(reports: list[dict[str, Any]], scoring_version: str) -> dict[str, Any]:
    """Build a deterministic narrative from structured evidence only."""
    from src.services.narrative_provider import build_evidence_narrative

    narrative = build_evidence_narrative(reports, scoring_version)
    narrative["summary"] = narrative["risk_summary"]
    narrative["evidence_supplier_count"] = len(reports)
    return narrative
