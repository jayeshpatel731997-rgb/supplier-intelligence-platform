"""
Propagator Agent — Cascade & Financial Simulation
===================================================

The third agent in the 5-agent pipeline. Takes event-to-node mappings
and runs two simulations per affected event:

  1. SIR cascade (Monte Carlo over the supplier graph)  — physical spread
  2. Triangular-delay Monte Carlo (per primary node)    — financial impact

Pipeline: Sentinel → Mapper → [Propagator] → Strategist → Narrator

Why two models, not one
-----------------------
The SIR model answers "how far does the shock spread?" — it operates on
*topology* (who depends on whom) and gives infection probabilities per
node. The financial Monte Carlo answers "what does the worst 5% of
outcomes cost us?" — it operates on *quantities* (delay days, units,
margin) and gives Expected Loss + VaR + CVaR in dollars.

Combining a graph-spread model with a tail-risk dollar model is a common
pattern in catastrophic-risk modeling (think reinsurance + epidemiology).
"""

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from models.sir_propagation import run_monte_carlo_sir, PropagationParams
from models.nasa_upgrades import monte_carlo_with_lhs


# Severity → (min, mode, max) delay days for the triangular distribution.
# Calibration intuition: critical events (Taiwan strait, sanctions) take
# weeks-to-months to recover from; low events (a single late shipment) days.
SEVERITY_DELAY_PROFILE: dict[str, tuple[float, float, float]] = {
    "critical": (14, 45, 120),
    "high":     (7,  21, 60),
    "medium":   (3,  14, 30),
    "low":      (1,  7,  14),
}

# Severity → SIR transmission rate β.  High-severity events transmit
# faster across supplier dependencies.
SEVERITY_BETA: dict[str, float] = {
    "critical": 0.35,
    "high":     0.35,
    "medium":   0.25,
    "low":      0.20,
}


@dataclass
class CascadeResult:
    """Combined output of SIR + financial Monte Carlo for one event."""
    event_title: str
    event_severity: str
    affected_nodes: list[str]
    sir_cascade: dict
    financial_impact: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "event_title": self.event_title,
            "event_severity": self.event_severity,
            "affected_nodes": self.affected_nodes,
            "sir_cascade": self.sir_cascade,
            "financial_impact": self.financial_impact,
        }


class PropagatorAgent:
    """
    Runs cascade + financial simulations per affected event.

    Parameters
    ----------
    G                 : pre-built NetworkX directed graph of the network
    network_data      : raw dict (used to look up node metadata like spend)
    n_sir_runs        : how many SIR replicates per event (default 50)
    n_mc_iterations   : LHS iterations per financial sim (default 2000;
                        LHS accuracy ≈ plain MC at 30 000 — keep this low)
    max_shocked_nodes : cap on how many top-matched nodes seed the cascade
    """

    def __init__(
        self,
        G: nx.DiGraph,
        network_data: dict,
        n_sir_runs: int = 50,
        n_mc_iterations: int = 2000,
        max_shocked_nodes: int = 3,
    ):
        self.G = G
        self.network_data = network_data
        self.n_sir_runs = n_sir_runs
        self.n_mc_iterations = n_mc_iterations
        self.max_shocked_nodes = max_shocked_nodes

    # ─── PER-EVENT SIMULATION ───────────────────────────────────────

    def _run_sir(self, severity: str, shocked_nodes: list[str]) -> dict:
        """Run Monte Carlo SIR cascade."""
        params = PropagationParams(
            beta=SEVERITY_BETA.get(severity, 0.25),
            gamma=0.08,
            time_steps=30,
            use_weibull_beta=True,
            weibull_time_scale=1.0,
        )
        return run_monte_carlo_sir(
            self.G, shocked_nodes, params, n_runs=self.n_sir_runs
        )

    def _run_financial(
        self,
        severity: str,
        primary_node_id: str,
    ) -> Optional[dict]:
        """
        Run NASA LHS financial Monte Carlo for the primary affected node.

        Uses Latin Hypercube Sampling (nasa_upgrades.monte_carlo_with_lhs)
        instead of plain independent draws.  With the default 2 000 iterations,
        LHS gives convergence equivalent to ~30 000 plain-MC samples — which
        means our CVaR₉₅ headline number is far more stable run-to-run.

        Why this matters (game theory): the procurement leader is making a
        one-shot decision under uncertainty.  A noisy CVaR that swings ±40%
        between runs destroys trust.  LHS tightens the confidence interval
        without increasing compute time.
        """
        node_data = next(
            (n for n in self.network_data["nodes"] if n["id"] == primary_node_id),
            {},
        )
        if node_data.get("spend", 0) <= 0:
            return None

        annual_spend = node_data["spend"]
        daily_demand = annual_spend / 365 / 50   # units/day (unit_cost = $50)

        # Build plain-dict profiles expected by monte_carlo_with_lhs.
        supplier_dict = {
            "daily_demand":       daily_demand,
            "unit_cost":          50.0,
            "holding_rate":       0.25,          # 25 % annual inventory carrying cost
            "freight_cost":       2_000.0,        # $ per expedite shipment
            "expedite_multiplier": 3.0,           # expedite vs. standard freight
            "margin":             20.0,           # $ profit margin per unit
            "order_value":        annual_spend / 52,  # typical weekly PO value
            "defect_rate":        0.05,
        }

        min_d, mode_d, max_d = SEVERITY_DELAY_PROFILE.get(severity, (7, 21, 60))
        disruption_dict = {
            "min_delay":          min_d,
            "mode_delay":         mode_d,
            "max_delay":          max_d,
            "duration_months":    max(mode_d / 30, 0.5),
            "expedite_threshold": 10,   # days before we start expediting
            "stockout_threshold": 21,   # days before we incur stockout losses
            "quality_prob":       0.15,
        }

        return monte_carlo_with_lhs(supplier_dict, disruption_dict, self.n_mc_iterations)

    # ─── BATCH SIMULATION ───────────────────────────────────────────

    def simulate(
        self,
        mappings: list,         # list[EventNodeMapping] (kept loose to avoid circular import)
        verbose: bool = True,
    ) -> list[CascadeResult]:
        """
        Run cascade + financial sims for every event with at least one
        affected node. Events with no mapped nodes are skipped silently.
        """
        results: list[CascadeResult] = []

        for mapping in mappings:
            if not mapping.affected_nodes:
                continue

            event = mapping.event
            severity = event["severity"]
            top_nodes = mapping.affected_nodes[:self.max_shocked_nodes]
            shocked_ids = [n.node_id for n in top_nodes]

            sir_result = self._run_sir(severity, shocked_ids)

            primary_node = mapping.affected_nodes[0]
            mc_result = self._run_financial(severity, primary_node.node_id)

            sir_summary = {
                "avg_infected": sir_result["cascade_stats"]["avg_total_infected"],
                "avg_depth": sir_result["cascade_stats"]["avg_cascade_depth"],
                "oem_infection_rate": sir_result["per_node"].get("OEM", {}).get("infection_rate", 0),
                "method": "SIR with Weibull time-dependent beta",
                "per_node": {
                    nid: {
                        "infection_rate": stats["infection_rate"],
                        "avg_risk": stats["avg_risk_score"],
                    }
                    for nid, stats in sir_result["per_node"].items()
                    if stats["infection_rate"] > 0.1
                },
            }

            financial_summary = None
            if mc_result is not None:
                # mc_result is already a dict from monte_carlo_with_lhs;
                # pull the keys the rest of the pipeline expects, plus the
                # richer breakdown and convergence note for the output JSON.
                financial_summary = {
                    "expected_loss": mc_result["expected_loss"],
                    "var_95":        mc_result["var_95"],
                    "cvar_95":       mc_result["cvar_95"],
                    "p99":           mc_result["p99"],
                    "breakdown":     mc_result["breakdown"],
                    "method":        mc_result["method"],
                }

            cascade = CascadeResult(
                event_title=event["title"],
                event_severity=severity,
                affected_nodes=shocked_ids,
                sir_cascade=sir_summary,
                financial_impact=financial_summary,
            )
            results.append(cascade)

            if verbose:
                avg_inf = sir_result["cascade_stats"]["avg_total_infected"]
                oem_r = sir_result["per_node"].get("OEM", {}).get("infection_rate", 0)
                print(f"  Cascade: {avg_inf:.1f} nodes infected, "
                      f"OEM risk: {oem_r:.0%}")
                if mc_result is not None:
                    print(f"  Financial (LHS): Expected ${mc_result['expected_loss']:,.0f} | "
                          f"CVaR₉₅ ${mc_result['cvar_95']:,.0f}")

        return results
