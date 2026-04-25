"""
Strategist Agent — Mitigation Recommendation
=============================================

The fourth agent in the 5-agent pipeline. Takes cascade results from the
Propagator and produces a ranked menu of mitigation options for each
significant event, with cost / timeline / risk-reduction estimates.

Pipeline: Sentinel → Mapper → Propagator → [Strategist] → Narrator

Game-theoretic framing
----------------------
Every mitigation is a *bet* against an uncertain disruption. The
Strategist's job is to assemble a payoff table:

   Action          | Cost      | Risk Reduction | Time-to-effect
   ----------------+-----------+----------------+----------------
   Dual-source     | $$        | high           | months
   Buffer stock    | $         | medium         | days
   Nearshore       | $$$       | very high      | year+
   Contract terms  | $         | low            | weeks

The decision-maker (the human PM) chooses the policy. The Strategist is
deliberately *not* picking one — it surfaces the menu with consistent
cost/benefit framing so trade-offs are explicit. This avoids the common
LLM failure mode of confidently recommending a single action without
showing the alternatives that were rejected.
"""

from dataclasses import dataclass, field

from agents.propagator import CascadeResult
from models.nasa_upgrades import build_supplier_fault_tree


@dataclass
class MitigationOption:
    """One concrete action the buyer could take."""
    action: str
    description: str
    cost: str               # human-readable range, not a calibrated number
    timeline: str
    risk_reduction: str     # "10-20%", "40-60%", etc.
    priority: str           # HIGH | MEDIUM | LOW


@dataclass
class MitigationRecommendation:
    """All options proposed for one event, plus a top pick."""
    event: str
    options: list[MitigationOption] = field(default_factory=list)
    recommended_action: str = ""
    fault_tree: dict | None = None

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "options": [
                {
                    "action": o.action,
                    "description": o.description,
                    "cost": o.cost,
                    "timeline": o.timeline,
                    "risk_reduction": o.risk_reduction,
                    "priority": o.priority,
                }
                for o in self.options
            ],
            "recommended_action": self.recommended_action,
            "fault_tree": self.fault_tree,
        }


class StrategistAgent:
    """
    Generates mitigation menus from cascade results using rule-based
    heuristics calibrated against common procurement playbooks.

    Future: replace rules with a learned policy or LLM prompt that
    consumes a CascadeResult + buyer constraints (budget, time horizon)
    and outputs a ranked menu.
    """

    # ─── INDIVIDUAL OPTION GENERATORS ───────────────────────────────

    def __init__(self, network_data: dict | None = None):
        self.network_data = network_data or {}
        self._node_lookup = {
            node.get("id"): node
            for node in self.network_data.get("nodes", [])
            if node.get("id")
        }

    def _option_dual_source(self, severity: str) -> MitigationOption:
        return MitigationOption(
            action="Dual-Source Critical Components",
            description="Qualify alternative supplier for affected components",
            cost="$15,000-$50,000 (qualification)",
            timeline="60-90 days",
            risk_reduction="40-60%",
            priority="HIGH" if severity in ("critical", "high") else "MEDIUM",
        )

    def _option_buffer(self, expected_loss: float) -> MitigationOption:
        buffer_cost = expected_loss * 0.3
        return MitigationOption(
            action="Increase Safety Stock Buffer",
            description="Build 21-day safety stock for affected SKUs",
            cost=f"${buffer_cost:,.0f} (one-time inventory investment)",
            timeline="Immediate",
            risk_reduction="25-35%",
            priority="HIGH",
        )

    def _option_nearshore(self) -> MitigationOption:
        return MitigationOption(
            action="Nearshore/Reshore Assessment",
            description="Evaluate Mexico/US alternative for tariff-exposed suppliers",
            cost="$5,000-$15,000 (assessment) + implementation",
            timeline="6-12 months",
            risk_reduction="50-80%",
            priority="MEDIUM",
        )

    def _option_contract(self, severity: str) -> MitigationOption:
        return MitigationOption(
            action="Strengthen Contractual Protections",
            description="Add force majeure clauses, SLA penalties, and backup sourcing requirements",
            cost="$2,000-$5,000 (legal)",
            timeline="30 days",
            risk_reduction="10-20%",
            priority="MEDIUM" if severity != "low" else "LOW",
        )

    # ─── BATCH RECOMMENDATION ───────────────────────────────────────

    def _option_fault_tree_focus(self, driver_name: str) -> MitigationOption:
        return MitigationOption(
            action=f"Mitigate Root Driver: {driver_name}",
            description="Use fault-tree importance to focus the next supplier action on the largest modeled cause of supply loss.",
            cost="$2,000-$20,000 (targeted corrective action)",
            timeline="2-6 weeks",
            risk_reduction="10-35%",
            priority="HIGH",
        )

    def _fault_tree_summary(self, result: CascadeResult) -> dict | None:
        if not result.affected_nodes:
            return None

        primary_id = result.affected_nodes[0]
        node = self._node_lookup.get(primary_id)
        if not node:
            return None

        risk_signals = {
            "financial_health": node.get("financial_health", 0.5),
            "geopolitical_risk": node.get("geopolitical_risk", 0.3),
            "weather_risk": node.get("weather_risk", 0.2),
            "concentration_risk": node.get("concentration_risk", 0.3),
            "on_time_rate": node.get("on_time_rate", 0.9),
            "tariff_exposure": node.get("tariff_exposure", 0.3),
        }
        tree = build_supplier_fault_tree(node.get("name", primary_id), risk_signals)
        top_drivers = list(tree.importance_measure().values())[:3]

        return {
            "supplier_id": primary_id,
            "supplier_name": node.get("name", primary_id),
            "top_event_probability": tree.compute_probability(),
            "top_drivers": top_drivers,
        }

    def recommend(
        self,
        cascade_results: list[CascadeResult],
        verbose: bool = True,
    ) -> list[MitigationRecommendation]:
        """Produce a ranked mitigation menu per cascade result."""
        recommendations: list[MitigationRecommendation] = []

        for result in cascade_results:
            severity = result.event_severity
            financial = result.financial_impact or {}
            expected_loss = financial.get("expected_loss", 0)
            fault_tree = self._fault_tree_summary(result)

            options: list[MitigationOption] = [self._option_dual_source(severity)]

            if fault_tree and fault_tree.get("top_drivers"):
                options.append(self._option_fault_tree_focus(fault_tree["top_drivers"][0]["name"]))

            if expected_loss > 10_000:
                options.append(self._option_buffer(expected_loss))

            if severity in ("critical", "high"):
                options.append(self._option_nearshore())

            options.append(self._option_contract(severity))

            recommendations.append(MitigationRecommendation(
                event=result.event_title,
                options=options,
                recommended_action=options[0].action,
                fault_tree=fault_tree,
            ))

            if verbose:
                print(f"  {result.event_title[:50]}... → {len(options)} mitigation options")

        return recommendations
