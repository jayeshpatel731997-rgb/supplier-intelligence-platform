"""
Agent Orchestrator — Coordinates the 5-Agent Pipeline
=======================================================

Pipeline: Sentinel → Mapper → Propagator → Strategist → Narrator

This orchestrator:
1. Runs Sentinel to detect events
2. Maps events to affected supplier nodes
3. Runs SIR cascade + Monte Carlo simulations
4. Generates mitigation recommendations
5. Produces executive brief
"""

import json
import os
from datetime import datetime
from pathlib import Path

from agents.sentinel import SentinelAgent, DetectedEvent, EventSeverity
from models.sir_propagation import (
    build_networkx_graph, run_monte_carlo_sir, PropagationParams
)
from models.bayesian_risk import compute_bayesian_risk, SupplierSignals
from models.monte_carlo import (
    run_monte_carlo, SupplierCostProfile, DisruptionProfile
)
from models.graph_metrics import (
    compute_centrality_metrics, compute_network_resilience_score
)
class Orchestrator:
    """
    Coordinates the multi-agent pipeline for supply chain risk analysis.
    """

    def __init__(self, network_path: str = "data/sample_network.json"):
        # Load network
        with open(network_path) as f:
            self.network_data = json.load(f)
        self.G = build_networkx_graph(self.network_data)

        # Initialize agents
        self.sentinel = SentinelAgent()

        # Pre-compute graph metrics
        self.centralities = compute_centrality_metrics(self.G)
        self.resilience = compute_network_resilience_score(self.G)

        print("[Orchestrator] Initialized with network:", self.network_data.get("network_name"))
        print(f"[Orchestrator] Nodes: {len(self.network_data['nodes'])} | Edges: {len(self.network_data['edges'])}")

    # ─── STEP 1: EVENT DETECTION (SENTINEL) ──────────────────────

    def detect_events(self, use_llm: bool = True, **kwargs) -> list[DetectedEvent]:
        """Run Sentinel Agent to detect supply chain events."""
        print("\n" + "="*60)
        print("[Step 1] SENTINEL — Event Detection")
        print("="*60)
        return self.sentinel.scan(classify_with_llm=use_llm, **kwargs)

    # ─── STEP 2: EVENT → NODE MAPPING (MAPPER) ──────────────────

    def map_events_to_nodes(self, events: list[DetectedEvent]) -> list[dict]:
        """
        Map detected events to affected supplier nodes.

        Uses region matching, industry matching, and material matching
        to identify which suppliers in our network are impacted.
        """
        print("\n" + "="*60)
        print("[Step 2] MAPPER — Event-to-Node Mapping")
        print("="*60)

        mappings = []
        for event in events:
            affected_nodes = []

            for node in self.network_data["nodes"]:
                if node.get("type") != "supplier":
                    continue

                match_score = 0.0
                match_reasons = []

                # Region matching
                node_region = node.get("region", "").lower()
                for region in event.affected_regions:
                    region_lower = region.lower()
                    if region_lower in node_region or node_region in region_lower:
                        match_score += 0.5
                        match_reasons.append(f"Region: {region}")
                    # Country-level match
                    elif region_lower.split("-")[0] in node_region.split("-")[0]:
                        match_score += 0.3
                        match_reasons.append(f"Country: {region.split('-')[0]}")

                # Keyword matching against node name/industry
                node_text = (node.get("name", "") + " " + node.get("region", "")).lower()
                for keyword in event.keywords:
                    if keyword.lower() in node_text:
                        match_score += 0.3
                        match_reasons.append(f"Keyword: {keyword}")

                # Category-specific matching
                if event.category.value == "tariff_trade" and node.get("tariff_exposure", 0) > 0.5:
                    match_score += 0.4
                    match_reasons.append("High tariff exposure")

                if event.category.value == "natural_disaster" and node.get("weather_risk", 0) > 0.5:
                    match_score += 0.3
                    match_reasons.append("High weather risk")

                if match_score > 0.2:
                    affected_nodes.append({
                        "node_id": node["id"],
                        "name": node["name"],
                        "match_score": min(match_score, 1.0),
                        "match_reasons": match_reasons,
                    })

            mappings.append({
                "event": event.to_dict(),
                "affected_nodes": sorted(affected_nodes, key=lambda x: -x["match_score"]),
                "total_affected": len(affected_nodes),
            })

            if affected_nodes:
                print(f"  [{event.severity.value.upper()}] {event.title[:50]}... → "
                      f"{len(affected_nodes)} nodes affected")
            else:
                print(f"  [{event.severity.value.upper()}] {event.title[:50]}... → No network match")

        return mappings

    # ─── STEP 3: CASCADE SIMULATION (PROPAGATOR) ────────────────

    def simulate_cascades(
        self,
        mappings: list[dict],
        n_sir_runs: int = 50,
        n_mc_iters: int = 2000,
    ) -> list[dict]:
        """
        Run SIR cascade + Monte Carlo for each event with affected nodes.
        """
        print("\n" + "="*60)
        print("[Step 3] PROPAGATOR — Cascade & Financial Simulation")
        print("="*60)

        results = []
        for mapping in mappings:
            if not mapping["affected_nodes"]:
                continue

            event = mapping["event"]
            shocked_nodes = [n["node_id"] for n in mapping["affected_nodes"][:3]]

            # SIR Cascade
            sir_params = PropagationParams(
                beta=0.35 if event["severity"] in ["critical", "high"] else 0.25,
                gamma=0.08,
                time_steps=30,
            )
            sir_result = run_monte_carlo_sir(
                self.G, shocked_nodes, sir_params, n_runs=n_sir_runs
            )

            # Monte Carlo Financial for primary affected node
            primary_node = mapping["affected_nodes"][0]
            node_data = next(
                (n for n in self.network_data["nodes"] if n["id"] == primary_node["node_id"]),
                {}
            )

            mc_result = None
            if node_data.get("spend", 0) > 0:
                supplier_profile = SupplierCostProfile(
                    annual_spend=node_data["spend"],
                    daily_demand_units=node_data["spend"] / 365 / 50,
                    unit_cost=50.0,
                    profit_margin_per_unit=20.0,
                )
                severity_delays = {
                    "critical": (14, 45, 120),
                    "high": (7, 21, 60),
                    "medium": (3, 14, 30),
                    "low": (1, 7, 14),
                }
                delays = severity_delays.get(event["severity"], (7, 21, 60))
                disruption = DisruptionProfile(
                    min_delay_days=delays[0],
                    mode_delay_days=delays[1],
                    max_delay_days=delays[2],
                )
                mc_result = run_monte_carlo(supplier_profile, disruption, n_mc_iters)

            results.append({
                "event_title": event["title"],
                "event_severity": event["severity"],
                "affected_nodes": shocked_nodes,
                "sir_cascade": {
                    "avg_infected": sir_result["cascade_stats"]["avg_total_infected"],
                    "avg_depth": sir_result["cascade_stats"]["avg_cascade_depth"],
                    "oem_infection_rate": sir_result["per_node"].get("OEM", {}).get("infection_rate", 0),
                    "per_node": {
                        nid: {
                            "infection_rate": stats["infection_rate"],
                            "avg_risk": stats["avg_risk_score"],
                        }
                        for nid, stats in sir_result["per_node"].items()
                        if stats["infection_rate"] > 0.1
                    },
                },
                "financial_impact": {
                    "expected_loss": mc_result.expected_loss if mc_result else 0,
                    "var_95": mc_result.var_95 if mc_result else 0,
                    "cvar_95": mc_result.cvar_95 if mc_result else 0,
                } if mc_result else None,
            })

            print(f"  Cascade: {sir_result['cascade_stats']['avg_total_infected']:.1f} nodes infected, "
                  f"OEM risk: {sir_result['per_node'].get('OEM', {}).get('infection_rate', 0):.0%}")
            if mc_result:
                print(f"  Financial: Expected ${mc_result.expected_loss:,.0f} | "
                      f"CVaR₉₅ ${mc_result.cvar_95:,.0f}")

        return results

    # ─── STEP 4: MITIGATION (STRATEGIST) ────────────────────────

    def recommend_mitigations(self, cascade_results: list[dict]) -> list[dict]:
        """Generate mitigation recommendations based on cascade analysis."""
        print("\n" + "="*60)
        print("[Step 4] STRATEGIST — Mitigation Recommendations")
        print("="*60)

        recommendations = []
        for result in cascade_results:
            options = []
            severity = result["event_severity"]
            financial = result.get("financial_impact", {})

            # Option 1: Dual-source
            options.append({
                "action": "Dual-Source Critical Components",
                "description": "Qualify alternative supplier for affected components",
                "cost": "$15,000-$50,000 (qualification)",
                "timeline": "60-90 days",
                "risk_reduction": "40-60%",
                "priority": "HIGH" if severity in ["critical", "high"] else "MEDIUM",
            })

            # Option 2: Buffer inventory
            if financial and financial.get("expected_loss", 0) > 10000:
                buffer_cost = financial["expected_loss"] * 0.3
                options.append({
                    "action": "Increase Safety Stock Buffer",
                    "description": f"Build {21}-day safety stock for affected SKUs",
                    "cost": f"${buffer_cost:,.0f} (one-time inventory investment)",
                    "timeline": "Immediate",
                    "risk_reduction": "25-35%",
                    "priority": "HIGH",
                })

            # Option 3: Nearshore
            if severity in ["critical", "high"]:
                options.append({
                    "action": "Nearshore/Reshore Assessment",
                    "description": "Evaluate Mexico/US alternative for tariff-exposed suppliers",
                    "cost": "$5,000-$15,000 (assessment) + implementation",
                    "timeline": "6-12 months",
                    "risk_reduction": "50-80%",
                    "priority": "MEDIUM",
                })

            # Option 4: Contract renegotiation
            options.append({
                "action": "Strengthen Contractual Protections",
                "description": "Add force majeure clauses, SLA penalties, and backup sourcing requirements",
                "cost": "$2,000-$5,000 (legal)",
                "timeline": "30 days",
                "risk_reduction": "10-20%",
                "priority": "MEDIUM" if severity != "low" else "LOW",
            })

            recommendations.append({
                "event": result["event_title"],
                "options": options,
                "recommended_action": options[0]["action"],  # Top priority
            })

            print(f"  {result['event_title'][:50]}... → {len(options)} mitigation options")

        return recommendations

    # ─── STEP 5: EXECUTIVE BRIEF (NARRATOR) ─────────────────────

    def generate_brief(
        self,
        events: list[DetectedEvent],
        cascade_results: list[dict],
        recommendations: list[dict],
    ) -> str:
        """Generate executive brief summarizing the full analysis."""
        print("\n" + "="*60)
        print("[Step 5] NARRATOR — Executive Brief")
        print("="*60)

        n_critical = sum(1 for e in events if e.severity == EventSeverity.CRITICAL)
        n_high = sum(1 for e in events if e.severity == EventSeverity.HIGH)
        total_exposure = sum(
            r.get("financial_impact", {}).get("cvar_95", 0)
            for r in cascade_results if r.get("financial_impact")
        )
        max_oem_risk = max(
            (r["sir_cascade"]["oem_infection_rate"] for r in cascade_results),
            default=0
        )

        brief = f"""
╔══════════════════════════════════════════════════════════════╗
║  SUPPLY CHAIN RISK INTELLIGENCE BRIEF                       ║
║  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}                              ║
╚══════════════════════════════════════════════════════════════╝

EXECUTIVE SUMMARY
─────────────────
Events Detected:    {len(events)} ({n_critical} critical, {n_high} high)
Network Resilience: {self.resilience['resilience_score']:.0f}/100 ({self.resilience['interpretation']})
Max OEM Exposure:   {max_oem_risk:.0%} probability of direct impact
Total CVaR₉₅:      ${total_exposure:,.0f} (worst-case 5% tail risk)

TOP RISKS
─────────
"""
        for i, event in enumerate(events[:5]):
            brief += f"{i+1}. [{event.severity.value.upper():8s}] {event.title[:60]}\n"
            brief += f"   Regions: {', '.join(event.affected_regions[:3])}\n"

        brief += "\nCASCADE ANALYSIS\n─────────────────\n"
        for result in cascade_results[:3]:
            brief += f"• {result['event_title'][:50]}...\n"
            brief += f"  Cascade: {result['sir_cascade']['avg_infected']:.0f} nodes, "
            brief += f"OEM risk: {result['sir_cascade']['oem_infection_rate']:.0%}\n"
            if result.get("financial_impact"):
                brief += f"  Financial: Expected ${result['financial_impact']['expected_loss']:,.0f} | "
                brief += f"CVaR₉₅ ${result['financial_impact']['cvar_95']:,.0f}\n"

        brief += "\nRECOMMENDED ACTIONS\n───────────────────\n"
        for rec in recommendations[:3]:
            brief += f"• {rec['recommended_action']}\n"
            brief += f"  For: {rec['event'][:50]}...\n"

        brief += f"""
───────────────────────────────────────────────────────────────
Generated by: Agentic AI Supplier Risk Intelligence System
Models: SIR Propagation · Bayesian Risk · Monte Carlo · Graph Centrality

DISCLAIMER: This brief is for informational purposes only. Risk
scores are statistical estimates, not forecasts. Financial impact
figures are Monte Carlo simulations based on assumed distributions.
Do not use as sole basis for procurement or financial decisions.
Verify all critical findings with qualified professionals.
"""
        print(brief)
        return brief

    # ─── FULL PIPELINE ───────────────────────────────────────────

    def run_full_pipeline(
        self,
        use_llm: bool = True,
        output_dir: str = "outputs",
    ) -> dict:
        """
        Run the complete 5-agent pipeline.

        Returns dict with all intermediate and final results.
        """
        os.makedirs(output_dir, exist_ok=True)
        start = datetime.now()

        print("\n" + "╔" + "═"*58 + "╗")
        print("║  AGENTIC AI SUPPLIER RISK INTELLIGENCE — FULL PIPELINE  ║")
        print("╚" + "═"*58 + "╝")

        # Step 1: Detect
        events = self.detect_events(use_llm=use_llm)

        # Step 2: Map
        mappings = self.map_events_to_nodes(events)

        # Step 3: Simulate
        cascades = self.simulate_cascades(mappings)

        # Step 4: Recommend
        recommendations = self.recommend_mitigations(cascades)

        # Step 5: Brief
        brief = self.generate_brief(events, cascades, recommendations)

        # Save outputs
        output = {
            "timestamp": start.isoformat(),
            "duration_seconds": (datetime.now() - start).total_seconds(),
            "events_detected": len(events),
            "cascades_simulated": len(cascades),
            "recommendations": len(recommendations),
            "events": [e.to_dict() for e in events],
            "cascades": cascades,
            "recommendations": recommendations,
            "brief": brief,
        }

        output_path = os.path.join(output_dir, f"pipeline_output_{start.strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n[Orchestrator] Pipeline complete in {output['duration_seconds']:.1f}s")
        print(f"[Orchestrator] Output saved to {output_path}")

        return output
# ─── CLI ─────────────────────────────────────────────────────────

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
        # Inject a scenario manually
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
            events = [event]
            mappings = [{"event": event.to_dict(), "affected_nodes": [
                {"node_id": nid, "name": nid, "match_score": 1.0, "match_reasons": ["scenario"]}
                for nid in scenario["affected_nodes"]
            ], "total_affected": len(scenario["affected_nodes"])}]
            cascades = orch.simulate_cascades(mappings)
            recs = orch.recommend_mitigations(cascades)
            orch.generate_brief(events, cascades, recs)
        else:
            print(f"Scenario '{args.scenario}' not found")
    else:
        orch.run_full_pipeline(use_llm=not args.no_llm, output_dir=args.output)
