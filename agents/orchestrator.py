"""
Agent Orchestrator — Coordinates the 5-Agent Pipeline
=======================================================

Pipeline: Sentinel → Mapper → Propagator → Strategist → Narrator

The Orchestrator is intentionally thin. Each step delegates to a real
agent class living in its own module. The orchestrator's responsibility
is wiring, sequencing, and persistence — not analysis.

Usage
-----
    from agents.orchestrator import Orchestrator

    orch = Orchestrator(network_path="data/sample_network.json")
    output = orch.run_full_pipeline(use_llm=True)
    # → JSON written to outputs/pipeline_output_<timestamp>.json
"""

import json
import os
from datetime import datetime

from agents.sentinel import SentinelAgent, DetectedEvent
from agents.mapper import MapperAgent, EventNodeMapping
from agents.propagator import PropagatorAgent, CascadeResult
from agents.strategist import StrategistAgent, MitigationRecommendation
from agents.narrator import NarratorAgent, ExecutiveBrief

from models.sir_propagation import build_networkx_graph
from models.graph_metrics import (
    compute_centrality_metrics, compute_network_resilience_score,
)


class Orchestrator:
    """Coordinates the multi-agent pipeline for supply chain risk analysis."""

    def __init__(self, network_path: str = "data/sample_network.json"):
        # Load network
        with open(network_path) as f:
            self.network_data = json.load(f)
        self.G = build_networkx_graph(self.network_data)

        # Pre-compute graph metrics (used by Narrator + downstream consumers)
        self.centralities = compute_centrality_metrics(self.G)
        self.resilience = compute_network_resilience_score(self.G)

        # Instantiate the five agents
        self.sentinel = SentinelAgent()
        self.mapper = MapperAgent(self.network_data)
        self.propagator = PropagatorAgent(self.G, self.network_data)
        self.strategist = StrategistAgent(self.network_data)
        self.narrator = NarratorAgent(resilience=self.resilience)

        print(f"[Orchestrator] Initialized with network: "
              f"{self.network_data.get('network_name')}")
        print(f"[Orchestrator] Nodes: {len(self.network_data['nodes'])} | "
              f"Edges: {len(self.network_data['edges'])}")

    # ─── INDIVIDUAL STEPS ───────────────────────────────────────────

    def detect_events(self, use_llm: bool = True, **kwargs) -> list[DetectedEvent]:
        print("\n" + "=" * 60)
        print("[Step 1] SENTINEL — Event Detection")
        print("=" * 60)
        return self.sentinel.scan(classify_with_llm=use_llm, **kwargs)

    def map_events_to_nodes(
        self, events: list[DetectedEvent]
    ) -> list[EventNodeMapping]:
        print("\n" + "=" * 60)
        print("[Step 2] MAPPER — Event-to-Node Mapping")
        print("=" * 60)
        return self.mapper.map_events(events)

    def simulate_cascades(
        self, mappings: list[EventNodeMapping]
    ) -> list[CascadeResult]:
        print("\n" + "=" * 60)
        print("[Step 3] PROPAGATOR — Cascade & Financial Simulation")
        print("=" * 60)
        return self.propagator.simulate(mappings)

    def recommend_mitigations(
        self, cascade_results: list[CascadeResult]
    ) -> list[MitigationRecommendation]:
        print("\n" + "=" * 60)
        print("[Step 4] STRATEGIST — Mitigation Recommendations")
        print("=" * 60)
        return self.strategist.recommend(cascade_results)

    def generate_brief(
        self,
        events: list[DetectedEvent],
        cascade_results: list[CascadeResult],
        recommendations: list[MitigationRecommendation],
    ) -> ExecutiveBrief:
        print("\n" + "=" * 60)
        print("[Step 5] NARRATOR — Executive Brief")
        print("=" * 60)
        return self.narrator.render(events, cascade_results, recommendations)

    # ─── FULL PIPELINE ──────────────────────────────────────────────

    def run_full_pipeline(
        self,
        use_llm: bool = True,
        output_dir: str = "outputs",
    ) -> dict:
        """Run the complete 5-agent pipeline and persist results."""
        os.makedirs(output_dir, exist_ok=True)
        start = datetime.now()

        print("\n" + "╔" + "═" * 58 + "╗")
        print("║  AGENTIC AI SUPPLIER RISK INTELLIGENCE — FULL PIPELINE  ║")
        print("╚" + "═" * 58 + "╝")

        events = self.detect_events(use_llm=use_llm)
        mappings = self.map_events_to_nodes(events)
        cascades = self.simulate_cascades(mappings)
        recommendations = self.recommend_mitigations(cascades)
        brief = self.generate_brief(events, cascades, recommendations)

        output = {
            "timestamp": start.isoformat(),
            "duration_seconds": (datetime.now() - start).total_seconds(),
            "events_detected": len(events),
            "cascades_simulated": len(cascades),
            "recommendations": len(recommendations),
            "events": [e.to_dict() for e in events],
            "mappings": [m.to_dict() for m in mappings],
            "cascades": [c.to_dict() for c in cascades],
            "recommendations_full": [r.to_dict() for r in recommendations],
            "brief": brief.text,
        }

        out_path = os.path.join(
            output_dir,
            f"pipeline_output_{start.strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n[Orchestrator] Pipeline complete in {output['duration_seconds']:.1f}s")
        print(f"[Orchestrator] Output saved to {out_path}")
        return output


# ─── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full agent pipeline")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls")
    parser.add_argument("--network", default="data/sample_network.json")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--scenario", help="Inject a specific scenario by name")
    args = parser.parse_args()

    orch = Orchestrator(network_path=args.network)

    if args.scenario:
        # Inject a pre-built scenario from the network file
        from agents.mapper import EventNodeMapping, AffectedNode

        scenarios = orch.network_data.get("scenarios", [])
        scenario = next((s for s in scenarios if s["id"] == args.scenario), None)
        if scenario:
            event = orch.sentinel.inject_event(
                title=scenario["name"],
                description=scenario["description"],
                severity="high",
                category="geopolitical",
                regions=[],
            )
            mapping = EventNodeMapping(
                event=event.to_dict(),
                affected_nodes=[
                    AffectedNode(
                        node_id=nid, name=nid,
                        match_score=1.0, match_reasons=["scenario"],
                    )
                    for nid in scenario["affected_nodes"]
                ],
                total_affected=len(scenario["affected_nodes"]),
            )
            cascades = orch.simulate_cascades([mapping])
            recs = orch.recommend_mitigations(cascades)
            orch.generate_brief([event], cascades, recs)
        else:
            print(f"Scenario '{args.scenario}' not found")
    else:
        orch.run_full_pipeline(
            use_llm=not args.no_llm, output_dir=args.output
        )
