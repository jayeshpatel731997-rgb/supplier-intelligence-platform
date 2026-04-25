import json
import unittest

from agents.mapper import AffectedNode, EventNodeMapping
from agents.propagator import PropagatorAgent
from models.sir_propagation import (
    PropagationParams,
    _effective_beta,
    build_networkx_graph,
)


class WeibullSirTests(unittest.TestCase):
    def test_static_beta_remains_default(self):
        params = PropagationParams(beta=0.33)
        node_data = {"type": "supplier", "region": "US-Ohio"}

        self.assertAlmostEqual(_effective_beta("T1-A", node_data, params, 5), 0.33)

    def test_weibull_beta_changes_with_supplier_profile(self):
        params = PropagationParams(beta=0.33, use_weibull_beta=True)
        node_data = {
            "type": "supplier",
            "region": "China-Shenzhen",
            "tariff_exposure": 0.9,
        }

        beta_t = _effective_beta("T1-B", node_data, params, 5)

        self.assertGreater(beta_t, 0)
        self.assertLessEqual(beta_t, 0.95)
        self.assertNotAlmostEqual(beta_t, 0.33)

    def test_propagator_uses_weibull_sir_and_lhs_financials(self):
        with open("data/sample_network.json") as f:
            network_data = json.load(f)
        graph = build_networkx_graph(network_data)
        propagator = PropagatorAgent(
            graph,
            network_data,
            n_sir_runs=2,
            n_mc_iterations=128,
            max_shocked_nodes=1,
        )
        mapping = EventNodeMapping(
            event={"title": "Tariff shock", "severity": "high"},
            affected_nodes=[
                AffectedNode(
                    node_id="T1-B",
                    name="ElectroParts International",
                    match_score=1.0,
                    match_reasons=["test"],
                )
            ],
            total_affected=1,
        )

        results = propagator.simulate([mapping], verbose=False)

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].sir_cascade["method"],
            "SIR with Weibull time-dependent beta",
        )
        self.assertIn("Latin Hypercube", results[0].financial_impact["method"])
        self.assertIn("breakdown", results[0].financial_impact)


if __name__ == "__main__":
    unittest.main()
