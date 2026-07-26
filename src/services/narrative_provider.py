"""Governed narrative generation from structured evidence only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.config import Settings


ALLOWED_NARRATIVE_FIELDS = {
    "risk_summary",
    "top_drivers",
    "recommended_actions",
    "confidence_caveats",
}
PROMPT_POLICY = "STRUCTURED EVIDENCE ONLY; NO UNSUPPORTED CLAIMS; CITE PROVIDED SOURCES"


class NarrativeProvider(Protocol):
    name: str
    model: str

    def generate(self, evidence_input: dict[str, Any]) -> dict[str, Any]:
        """Return structured narrative fields."""


@dataclass(slots=True)
class MockNarrativeProvider:
    response: dict[str, Any]
    name: str = "mock"
    model: str = "mocked"

    def generate(self, evidence_input: dict[str, Any]) -> dict[str, Any]:
        del evidence_input
        return dict(self.response)


@dataclass(slots=True)
class UnavailableNarrativeProvider:
    name: str
    model: str = ""

    def generate(self, evidence_input: dict[str, Any]) -> dict[str, Any]:
        del evidence_input
        raise RuntimeError(f"{self.name} narrative provider is not configured.")


def provider_from_settings(settings: Settings | None) -> NarrativeProvider | None:
    if settings is None or settings.llm_narrative_provider == "none":
        return None
    if settings.llm_narrative_provider == "openai" and settings.openai_api_key:
        return UnavailableNarrativeProvider(name="openai", model=settings.openai_model)
    if settings.llm_narrative_provider == "anthropic" and settings.anthropic_api_key:
        return UnavailableNarrativeProvider(name="anthropic", model=settings.anthropic_model)
    return None


def _deterministic_narrative(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {
            "risk_summary": "No weak signals were available for this evidence-chain run.",
            "top_drivers": [],
            "recommended_actions": [],
            "confidence_caveats": ["No evidence inputs were present."],
        }
    sorted_reports = sorted(reports, key=lambda item: float(item.get("risk_score", 0)), reverse=True)
    top = sorted_reports[:3]
    fragments: list[str] = []
    top_drivers: list[str] = []
    recommended_actions: list[str] = []
    caveats: list[str] = []
    for report in top:
        driver = report.get("top_risk_drivers", [{}])[0].get("driver", "No dominant driver")
        source = report.get("evidence_chain", [{}])[0].get("source", "No source")
        fragments.append(
            f"{report.get('supplier_name', report.get('supplier_id'))} is {report.get('risk_level')} risk "
            f"({float(report.get('risk_score', 0)):.1f}/100) driven by {driver} from {source}."
        )
        if driver and driver not in top_drivers:
            top_drivers.append(driver)
        for action in report.get("recommended_actions", []):
            if action not in recommended_actions:
                recommended_actions.append(action)
    if any(float(report.get("confidence", 0)) < 0.6 for report in sorted_reports):
        caveats.append("Some suppliers have limited or lower-confidence evidence.")
    return {
        "risk_summary": " ".join(fragments),
        "top_drivers": top_drivers[:5],
        "recommended_actions": recommended_actions[:5],
        "confidence_caveats": caveats or ["Confidence is based only on structured evidence inputs."],
    }


def _structured_input(reports: list[dict[str, Any]], scoring_version: str) -> dict[str, Any]:
    return {
        "policy": PROMPT_POLICY,
        "scoring_version": scoring_version,
        "suppliers": [
            {
                "supplier_id": report.get("supplier_id"),
                "supplier_name": report.get("supplier_name"),
                "risk_score": report.get("risk_score"),
                "risk_level": report.get("risk_level"),
                "top_risk_drivers": report.get("top_risk_drivers", []),
                "evidence_chain": report.get("evidence_chain", []),
                "recommended_actions": report.get("recommended_actions", []),
                "confidence": report.get("confidence"),
            }
            for report in reports
        ],
    }


def _sanitize_provider_output(raw: dict[str, Any]) -> dict[str, Any]:
    sanitized = {key: raw.get(key) for key in ALLOWED_NARRATIVE_FIELDS}
    sanitized["top_drivers"] = list(sanitized.get("top_drivers") or [])
    sanitized["recommended_actions"] = list(sanitized.get("recommended_actions") or [])
    sanitized["confidence_caveats"] = list(sanitized.get("confidence_caveats") or [])
    sanitized["risk_summary"] = str(sanitized.get("risk_summary") or "")
    return sanitized


def _is_supported_output(body: dict[str, Any], reports: list[dict[str, Any]]) -> bool:
    supported_drivers = {
        str(driver.get("driver"))
        for report in reports
        for driver in report.get("top_risk_drivers", [])
        if driver.get("driver")
    }
    supported_actions = {
        str(action)
        for report in reports
        for action in report.get("recommended_actions", [])
        if action
    }
    return set(body["top_drivers"]).issubset(supported_drivers) and set(
        body["recommended_actions"]
    ).issubset(supported_actions)


def build_evidence_narrative(
    reports: list[dict[str, Any]],
    scoring_version: str,
    *,
    settings: Settings | None = None,
    provider: NarrativeProvider | None = None,
) -> dict[str, Any]:
    active_provider = provider or provider_from_settings(settings)
    if active_provider is None:
        body = _deterministic_narrative(reports)
        provider_name = "none"
        model = ""
    else:
        try:
            body = _sanitize_provider_output(active_provider.generate(_structured_input(reports, scoring_version)))
            if not _is_supported_output(body, reports):
                raise ValueError("Narrative output contains claims not supported by structured evidence.")
            deterministic = _deterministic_narrative(reports)
            body["risk_summary"] = deterministic["risk_summary"]
            body["confidence_caveats"] = deterministic["confidence_caveats"]
            provider_name = active_provider.name
            model = active_provider.model
        except Exception:
            body = _deterministic_narrative(reports)
            provider_name = "none"
            model = ""
    return {
        "policy": PROMPT_POLICY,
        "provider": provider_name,
        "model": model,
        "scoring_version": scoring_version,
        **body,
    }
