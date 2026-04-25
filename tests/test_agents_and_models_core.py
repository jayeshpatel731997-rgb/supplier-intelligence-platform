import json
import unittest

from agents.mapper import MapperAgent
from agents.propagator import CascadeResult
from agents.sentinel import DetectedEvent, EventCategory, EventSeverity
from agents.strategist import StrategistAgent
from models.bayesian_risk import SupplierSignals, compute_bayesian_risk
from models.graph_metrics import (
    compute_centrality_metrics,
    compute_network_resilience_score,
)
from models.sir_propagation import build_networkx_graph


class MapperAgentTests(unittest.TestCase):
    def test_mapper_matches_region_and_tariff_exposure(self):
        network_data = {
            "nodes": [
                {
                    "id": "T1-CN",
                    "name": "Shenzhen Electronics",
                    "type": "supplier",
                    "region": "China-Shenzhen",
                    "tariff_exposure": 0.9,
                },
                {
                    "id": "OEM",
                    "name": "Focal",
                    "type": "focal",
                    "region": "US-Illinois",
                },
            ]
        }
        event = DetectedEvent(
            event_id="e1",
            timestamp="2026-04-25",
            title="China tariff escalation",
            summary="",
            source="test",
            url="",
            severity=EventSeverity.HIGH,
            category=EventCategory.TARIFF_TRADE,
            confidence=0.9,
            affected_regions=["China"],
            keywords=["electronics"],
        )

        mapping = MapperAgent(network_data).map_events([event], verbose=False)[0]

        self.assertEqual(mapping.total_affected, 1)
        self.assertEqual(mapping.affected_nodes[0].node_id, "T1-CN")
        self.assertGreaterEqual(mapping.affected_nodes[0].match_score, 0.7)


class StrategistAgentTests(unittest.TestCase):
    def test_strategist_adds_fault_tree_summary_when_network_data_available(self):
        with open("data/sample_network.json") as f:
            network_data = json.load(f)
        result = CascadeResult(
            event_title="Tariff shock",
            event_severity="high",
            affected_nodes=["T1-B"],
            sir_cascade={"oem_infection_rate": 0.4},
            financial_impact={"expected_loss": 50000, "cvar_95": 120000},
        )

        rec = StrategistAgent(network_data).recommend([result], verbose=False)[0]

        self.assertIsNotNone(rec.fault_tree)
        self.assertIn("top_event_probability", rec.fault_tree)
        self.assertTrue(rec.fault_tree["top_drivers"])
        self.assertTrue(any(o.action.startswith("Mitigate Root Driver") for o in rec.options))
        self.assertIn("fault_tree", rec.to_dict())


class ModelCoreTests(unittest.TestCase):
    def test_bayesian_risk_orders_high_risk_above_low_risk(self):
        low = compute_bayesian_risk(
            SupplierSignals(
                financial_health=0.95,
                geopolitical_risk=0.05,
                weather_risk=0.05,
                concentration_risk=0.1,
                historical_reliability=0.98,
                tariff_exposure=0.05,
            )
        )
        high = compute_bayesian_risk(
            SupplierSignals(
                financial_health=0.15,
                geopolitical_risk=0.9,
                weather_risk=0.8,
                concentration_risk=0.9,
                historical_reliability=0.65,
                tariff_exposure=0.9,
            )
        )

        self.assertLess(low.posterior_probability, high.posterior_probability)
        self.assertEqual(high.risk_level, "CRITICAL")

    def test_graph_metrics_return_resilience_and_rankings(self):
        with open("data/sample_network.json") as f:
            network_data = json.load(f)
        graph = build_networkx_graph(network_data)

        centralities = compute_centrality_metrics(graph)
        resilience = compute_network_resilience_score(graph)

        self.assertEqual(len(centralities), graph.number_of_nodes())
        self.assertGreaterEqual(resilience["resilience_score"], 0)
        self.assertIn(resilience["interpretation"], {"LOW", "MEDIUM", "HIGH"})


if __name__ == "__main__":
    unittest.main()
