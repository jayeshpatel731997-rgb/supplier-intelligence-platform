"""
SIR-Adapted Risk Propagation Model for Supply Chain Networks
=============================================================

Adapts the epidemiological Susceptible-Infected-Recovered (SIR) model
for modeling how disruptions cascade through multi-tier supplier networks.

Mathematical Foundation:
- P(infection) = 1 - (1 - β·δ^tier)^k
  where β = transmission rate, δ = tier damping factor, k = infected neighbors
- Recovery follows exponential: P(recover) = 1 - exp(-γ·t)

References:
- Tabachová et al. 2024: Graph-based cascade models (epidemic-style propagation)
- Sun & Liao 2025: Multi-tier network disruption amplification
- Berger et al. 2023: 33%+ disruptions from beyond Tier-1
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
class NodeState(Enum):
    SUSCEPTIBLE = "S"
    INFECTED = "I"
    RECOVERED = "R"
@dataclass
class PropagationParams:
    """Parameters for SIR propagation model."""
    beta: float = 0.30          # Base transmission rate per edge per timestep
    gamma: float = 0.10         # Recovery rate
    tier_damping: float = 0.85  # Amplification damping per tier distance
    time_steps: int = 30        # Simulation horizon
    edge_weight_factor: float = 1.0  # How much edge weights affect transmission
@dataclass
class NodeResult:
    """Result for a single node after simulation."""
    node_id: str
    name: str
    tier: int
    final_state: NodeState
    infected_at: Optional[int] = None
    recovered_at: Optional[int] = None
    risk_score: float = 0.0
    time_to_impact: Optional[int] = None
    max_neighbors_infected: int = 0
@dataclass
class PropagationResult:
    """Full result of a single SIR simulation run."""
    node_results: list[NodeResult] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    total_infected: int = 0
    peak_infection_time: int = 0
    cascade_depth: int = 0
def build_networkx_graph(network_data: dict) -> nx.DiGraph:
    """Convert network data dict to NetworkX directed graph."""
    G = nx.DiGraph()

    for node in network_data["nodes"]:
        G.add_node(node["id"], **node)

    for edge in network_data["edges"]:
        G.add_edge(edge["source"], edge["target"],
                    weight=edge.get("weight", 1.0),
                    edge_type=edge.get("type", "supply"))
    return G
def run_sir_simulation(
    G: nx.DiGraph,
    shocked_nodes: list[str],
    params: PropagationParams = PropagationParams(),
    seed: Optional[int] = None,
) -> PropagationResult:
    """
    Run a single SIR propagation simulation on a supply network.

    Args:
        G: NetworkX graph representing the supply network
        shocked_nodes: List of node IDs where disruption originates
        params: Simulation parameters
        seed: Random seed for reproducibility

    Returns:
        PropagationResult with per-node outcomes and time history
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize states
    states = {}
    infected_at = {}
    recovered_at = {}

    for node_id in G.nodes():
        if node_id in shocked_nodes:
            states[node_id] = NodeState.INFECTED
            infected_at[node_id] = 0
        else:
            states[node_id] = NodeState.SUSCEPTIBLE

    # History tracking: count S, I, R at each timestep
    history = []
    history.append(_snapshot(states, 0))

    for t in range(1, params.time_steps + 1):
        new_states = dict(states)

        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            tier = node_data.get("tier", 0)

            if states[node_id] == NodeState.SUSCEPTIBLE:
                # Find infected neighbors (predecessors and successors in directed graph)
                neighbors = set(G.predecessors(node_id)) | set(G.successors(node_id))
                infected_neighbors = [
                    n for n in neighbors if states[n] == NodeState.INFECTED
                ]

                if infected_neighbors:
                    # Compute infection probability
                    # P(infect) = 1 - product over infected neighbors of (1 - β * δ^tier * w_edge)
                    p_safe = 1.0
                    for inf_neighbor in infected_neighbors:
                        # Get edge weight
                        if G.has_edge(inf_neighbor, node_id):
                            w = G[inf_neighbor][node_id].get("weight", 1.0)
                        elif G.has_edge(node_id, inf_neighbor):
                            w = G[node_id][inf_neighbor].get("weight", 1.0)
                        else:
                            w = 0.5

                        tier_mult = params.tier_damping ** abs(tier)
                        p_transmit = params.beta * tier_mult * w * params.edge_weight_factor
                        p_transmit = min(p_transmit, 0.95)  # Cap at 95%
                        p_safe *= (1 - p_transmit)

                    p_infect = 1 - p_safe

                    if np.random.random() < p_infect:
                        new_states[node_id] = NodeState.INFECTED
                        infected_at[node_id] = t

            elif states[node_id] == NodeState.INFECTED:
                # Recovery: exponential CDF
                time_since = t - infected_at.get(node_id, 0)
                p_recover = 1 - np.exp(-params.gamma * time_since)

                if np.random.random() < p_recover:
                    new_states[node_id] = NodeState.RECOVERED
                    recovered_at[node_id] = t

        states = new_states
        history.append(_snapshot(states, t))

    # Compile results
    node_results = []
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        risk_score = _compute_risk_score(
            states[node_id], infected_at.get(node_id),
            params.time_steps, node_data.get("tier", 0), params.tier_damping
        )
        node_results.append(NodeResult(
            node_id=node_id,
            name=node_data.get("name", node_id),
            tier=node_data.get("tier", 0),
            final_state=states[node_id],
            infected_at=infected_at.get(node_id),
            recovered_at=recovered_at.get(node_id),
            risk_score=risk_score,
            time_to_impact=infected_at.get(node_id),
        ))

    total_infected = sum(1 for s in states.values() if s != NodeState.SUSCEPTIBLE)
    peak_time = max(
        (h["t"] for h in history if h["I"] == max(hh["I"] for hh in history)),
        default=0
    )
    cascade_depth = max(
        (G.nodes[nid].get("tier", 0)
         for nid, s in states.items()
         if s != NodeState.SUSCEPTIBLE),
        default=0
    )

    return PropagationResult(
        node_results=node_results,
        history=history,
        total_infected=total_infected,
        peak_infection_time=peak_time,
        cascade_depth=cascade_depth,
    )
def run_monte_carlo_sir(
    G: nx.DiGraph,
    shocked_nodes: list[str],
    params: PropagationParams = PropagationParams(),
    n_runs: int = 100,
) -> dict:
    """
    Run multiple SIR simulations and aggregate results.

    Returns:
        Dict with per-node average infection rates, risk scores,
        and confidence intervals.
    """
    all_results = []
    for i in range(n_runs):
        result = run_sir_simulation(G, shocked_nodes, params, seed=None)
        all_results.append(result)

    # Aggregate per node
    node_ids = list(G.nodes())
    aggregated = {}

    for node_id in node_ids:
        infection_count = 0
        risk_scores = []
        times_to_impact = []

        for result in all_results:
            node_result = next(
                (nr for nr in result.node_results if nr.node_id == node_id), None
            )
            if node_result:
                if node_result.final_state != NodeState.SUSCEPTIBLE:
                    infection_count += 1
                risk_scores.append(node_result.risk_score)
                if node_result.time_to_impact is not None:
                    times_to_impact.append(node_result.time_to_impact)

        aggregated[node_id] = {
            "infection_rate": infection_count / n_runs,
            "avg_risk_score": np.mean(risk_scores) if risk_scores else 0,
            "risk_score_std": np.std(risk_scores) if risk_scores else 0,
            "avg_time_to_impact": np.mean(times_to_impact) if times_to_impact else None,
            "median_time_to_impact": np.median(times_to_impact) if times_to_impact else None,
            "ci_95_lower": np.percentile(risk_scores, 2.5) if risk_scores else 0,
            "ci_95_upper": np.percentile(risk_scores, 97.5) if risk_scores else 0,
        }

    # Aggregate cascade stats
    cascade_stats = {
        "avg_total_infected": np.mean([r.total_infected for r in all_results]),
        "avg_peak_time": np.mean([r.peak_infection_time for r in all_results]),
        "avg_cascade_depth": np.mean([r.cascade_depth for r in all_results]),
        "max_cascade_depth": max(r.cascade_depth for r in all_results),
    }

    return {
        "per_node": aggregated,
        "cascade_stats": cascade_stats,
        "n_runs": n_runs,
    }
def _snapshot(states: dict, t: int) -> dict:
    """Create a timestep snapshot of S/I/R counts."""
    return {
        "t": t,
        "S": sum(1 for s in states.values() if s == NodeState.SUSCEPTIBLE),
        "I": sum(1 for s in states.values() if s == NodeState.INFECTED),
        "R": sum(1 for s in states.values() if s == NodeState.RECOVERED),
    }
def _compute_risk_score(
    state: NodeState,
    infected_at: Optional[int],
    total_steps: int,
    tier: int,
    damping: float,
) -> float:
    """Compute normalized risk score for a node."""
    if state == NodeState.SUSCEPTIBLE:
        return 0.0

    # Earlier infection = higher risk
    time_factor = 1.0 - (infected_at / total_steps) if infected_at is not None else 0.5

    # Closer to focal firm (lower tier) = higher impact
    tier_factor = damping ** abs(tier)

    # Combine
    return min(time_factor * tier_factor, 1.0)
# ─── SENSITIVITY ANALYSIS ────────────────────────────────────────

def sensitivity_analysis(
    G: nx.DiGraph,
    shocked_nodes: list[str],
    param_name: str,
    param_range: list[float],
    base_params: PropagationParams = PropagationParams(),
    n_runs: int = 50,
) -> list[dict]:
    """
    Run sensitivity analysis by varying one parameter.

    Returns list of {param_value, avg_infection_rate, avg_cascade_depth}
    """
    results = []

    for val in param_range:
        params = PropagationParams(
            beta=base_params.beta,
            gamma=base_params.gamma,
            tier_damping=base_params.tier_damping,
            time_steps=base_params.time_steps,
        )
        setattr(params, param_name, val)

        mc = run_monte_carlo_sir(G, shocked_nodes, params, n_runs=n_runs)

        avg_infection = np.mean([
            v["infection_rate"] for v in mc["per_node"].values()
        ])

        results.append({
            "param_value": val,
            "avg_infection_rate": avg_infection,
            "avg_cascade_depth": mc["cascade_stats"]["avg_cascade_depth"],
            "avg_total_infected": mc["cascade_stats"]["avg_total_infected"],
        })

    return results
