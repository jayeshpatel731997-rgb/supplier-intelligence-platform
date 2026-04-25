"""
Narrator Agent — Executive Brief Generation
=============================================

The fifth and final agent in the pipeline. Takes the structured outputs
from all upstream agents and renders a human-readable executive brief.

Pipeline: Sentinel → Mapper → Propagator → Strategist → [Narrator]

Why a separate agent
--------------------
Generation is a different problem than analysis. The first four agents
produce *structured data*; the Narrator's job is to compress that data
into the smallest readable form an executive can act on. Keeping it
separate means we can swap the renderer (text → Slack-formatted →
HTML email → Streamlit panel) without touching analysis logic.

The current implementation is a deterministic template. A future version
can call an LLM with the structured data + a brand voice spec to produce
a more natural narrative — but the structured data must remain the
source of truth.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from agents.sentinel import DetectedEvent, EventSeverity
from agents.propagator import CascadeResult
from agents.strategist import MitigationRecommendation


@dataclass
class ExecutiveBrief:
    """A rendered executive brief plus the underlying summary metrics."""
    text: str
    generated_at: str
    n_events: int
    n_critical: int
    n_high: int
    total_cvar_95: float
    max_oem_risk: float
    resilience_score: Optional[float] = None


class NarratorAgent:
    """
    Renders an executive brief from upstream agent outputs.

    Parameters
    ----------
    resilience : optional dict from compute_network_resilience_score().
                 If supplied, the brief includes the network resilience line.
    """

    def __init__(self, resilience: Optional[dict] = None):
        self.resilience = resilience

    # ─── HEADLINE METRICS ───────────────────────────────────────────

    @staticmethod
    def _summarize(
        events: list[DetectedEvent],
        cascade_results: list[CascadeResult],
    ) -> dict:
        n_critical = sum(1 for e in events if e.severity == EventSeverity.CRITICAL)
        n_high = sum(1 for e in events if e.severity == EventSeverity.HIGH)
        total_exposure = sum(
            (r.financial_impact or {}).get("cvar_95", 0)
            for r in cascade_results
        )
        max_oem_risk = max(
            (r.sir_cascade.get("oem_infection_rate", 0) for r in cascade_results),
            default=0,
        )
        return {
            "n_events": len(events),
            "n_critical": n_critical,
            "n_high": n_high,
            "total_cvar_95": total_exposure,
            "max_oem_risk": max_oem_risk,
        }

    # ─── RENDER ─────────────────────────────────────────────────────

    def render(
        self,
        events: list[DetectedEvent],
        cascade_results: list[CascadeResult],
        recommendations: list[MitigationRecommendation],
        verbose: bool = True,
    ) -> ExecutiveBrief:
        summary = self._summarize(events, cascade_results)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        resilience_line = ""
        if self.resilience is not None:
            resilience_line = (
                f"Network Resilience: {self.resilience['resilience_score']:.0f}/100 "
                f"({self.resilience['interpretation']})\n"
            )

        brief = (
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  SUPPLY CHAIN RISK INTELLIGENCE BRIEF                       ║\n"
            f"║  Generated: {ts}                              ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n"
            "\n"
            "EXECUTIVE SUMMARY\n"
            "─────────────────\n"
            f"Events Detected:    {summary['n_events']} "
            f"({summary['n_critical']} critical, {summary['n_high']} high)\n"
            f"{resilience_line}"
            f"Max OEM Exposure:   {summary['max_oem_risk']:.0%} probability of direct impact\n"
            f"Total CVaR₉₅:      ${summary['total_cvar_95']:,.0f} "
            f"(worst-case 5% tail risk)\n"
            "\nTOP RISKS\n─────────\n"
        )

        for i, event in enumerate(events[:5]):
            brief += (
                f"{i+1}. [{event.severity.value.upper():8s}] {event.title[:60]}\n"
                f"   Regions: {', '.join(event.affected_regions[:3])}\n"
            )

        brief += "\nCASCADE ANALYSIS\n─────────────────\n"
        for result in cascade_results[:3]:
            sir = result.sir_cascade
            brief += (
                f"• {result.event_title[:50]}...\n"
                f"  Cascade: {sir.get('avg_infected', 0):.0f} nodes, "
                f"OEM risk: {sir.get('oem_infection_rate', 0):.0%}\n"
            )
            if result.financial_impact:
                fi = result.financial_impact
                brief += (
                    f"  Financial: Expected ${fi['expected_loss']:,.0f} | "
                    f"CVaR₉₅ ${fi['cvar_95']:,.0f}\n"
                )

        brief += "\nRECOMMENDED ACTIONS\n───────────────────\n"
        for rec in recommendations[:3]:
            brief += (
                f"• {rec.recommended_action}\n"
                f"  For: {rec.event[:50]}...\n"
            )

        brief += (
            "\n"
            "───────────────────────────────────────────────────────────────\n"
            "Generated by: Agentic AI Supplier Risk Intelligence System\n"
            "Models: SIR Propagation · Bayesian Risk · Monte Carlo · Graph Centrality\n"
            "\n"
            "DISCLAIMER: This brief is for informational purposes only. Risk\n"
            "scores are statistical estimates, not forecasts. Financial impact\n"
            "figures are Monte Carlo simulations based on assumed distributions.\n"
            "Do not use as sole basis for procurement or financial decisions.\n"
            "Verify all critical findings with qualified professionals.\n"
        )

        if verbose:
            print(brief)

        return ExecutiveBrief(
            text=brief,
            generated_at=ts,
            n_events=summary["n_events"],
            n_critical=summary["n_critical"],
            n_high=summary["n_high"],
            total_cvar_95=summary["total_cvar_95"],
            max_oem_risk=summary["max_oem_risk"],
            resilience_score=(
                self.resilience["resilience_score"] if self.resilience else None
            ),
        )
