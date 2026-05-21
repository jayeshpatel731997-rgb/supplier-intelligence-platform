"""Decision intelligence brief generation with deterministic fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class DecisionBrief:
    supplier_name: str
    risk_drivers: list[str]
    financial_exposure: float
    confidence: str
    recommended_action: str
    tradeoff_explanation: str
    alternatives: list[str]
    final_decision: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_decision_brief(
    supplier_name: str,
    risk_score: float,
    risk_drivers: list[str],
    financial_exposure: float,
    confidence: str = "medium",
    llm_api_key: str = "",
) -> DecisionBrief:
    del llm_api_key
    if risk_score >= 0.85 or financial_exposure >= 500_000:
        decision = "replace supplier"
        action = "Escalate to procurement leadership and start replacement qualification."
    elif risk_score >= 0.70 or financial_exposure >= 250_000:
        decision = "dual-source"
        action = "Launch dual-source plan and renegotiate service-level protections."
    elif risk_score >= 0.50 or financial_exposure >= 100_000:
        decision = "renegotiate"
        action = "Renegotiate resilience terms, buffers, and recovery-time commitments."
    elif risk_score >= 0.30:
        decision = "monitor"
        action = "Keep supplier active with heightened monitoring and monthly review."
    else:
        decision = "continue"
        action = "Continue with standard monitoring cadence."

    drivers = ", ".join(risk_drivers or ["no dominant driver identified"])
    tradeoff = (
        f"{supplier_name} shows risk drivers around {drivers}. "
        f"The estimated exposure is ${financial_exposure:,.0f}, so the recommendation balances continuity cost "
        "against downside protection."
    )
    alternatives = [
        "continue with monitoring",
        "renegotiate commercial and service-level terms",
        "dual-source critical parts",
        "replace supplier after qualification",
    ]
    return DecisionBrief(
        supplier_name=supplier_name,
        risk_drivers=risk_drivers or [],
        financial_exposure=float(financial_exposure or 0.0),
        confidence=confidence,
        recommended_action=action,
        tradeoff_explanation=tradeoff,
        alternatives=alternatives,
        final_decision=decision,
    )
