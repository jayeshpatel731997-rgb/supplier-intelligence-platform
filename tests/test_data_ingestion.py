import unittest

import pandas as pd

from data_ingestion import ingest_file, dataframe_to_network_data


class DataIngestionTests(unittest.TestCase):
    def test_ingest_file_maps_aliases_and_applies_defaults(self):
        csv_bytes = (
            b"Vendor Name,Nation,Cost,Annual Volume,OTD,Risk Rating\n"
            b"Apex Manufacturing,Mexico,12.5,10000,0.92,35\n"
            b"Bright Star,USA,8.2,5000,88,20\n"
        )

        result = ingest_file(csv_bytes, "suppliers.csv")

        self.assertTrue(result.success)
        self.assertEqual(result.row_count, 2)
        self.assertIn("supplier_name", result.df.columns)
        self.assertIn("country", result.df.columns)
        self.assertIn("unit_cost", result.df.columns)
        self.assertAlmostEqual(result.df.loc[0, "on_time_delivery_pct"], 92.0)
        self.assertEqual(result.df.loc[1, "country"], "USA")

    def test_dataframe_to_network_data_builds_edges_and_scenarios(self):
        df = pd.DataFrame(
            [
                {
                    "supplier_name": "Tier 1 Copper",
                    "country": "USA",
                    "tier": 1,
                    "category": "Copper",
                    "annual_spend": 500000,
                    "annual_volume": 10000,
                    "unit_cost": 50,
                    "on_time_delivery_pct": 96,
                    "risk_score": 20,
                    "tariff_exposure": 0.05,
                },
                {
                    "supplier_name": "Tier 1 Resin",
                    "country": "Mexico",
                    "tier": 1,
                    "category": "Resin",
                    "annual_spend": 300000,
                    "annual_volume": 15000,
                    "unit_cost": 20,
                    "on_time_delivery_pct": 90,
                    "risk_score": 35,
                    "tariff_exposure": 0.40,
                },
                {
                    "supplier_name": "Tier 2 Mine",
                    "country": "Chile",
                    "tier": 2,
                    "category": "Copper",
                    "annual_spend": 0,
                    "annual_volume": 8000,
                    "unit_cost": 10,
                    "on_time_delivery_pct": 82,
                    "risk_score": 55,
                    "geopolitical_risk": 0.55,
                    "tariff_exposure": 0.25,
                },
            ]
        )

        network = dataframe_to_network_data(df, company_name="Test OEM")

        self.assertEqual(network["network_name"], "Uploaded Supplier Network")
        self.assertEqual(len(network["nodes"]), len(df) + 3)
        self.assertEqual(len(network["scenarios"]), 5)

        supplier_nodes = [node for node in network["nodes"] if node.get("type") == "supplier"]
        tier1_ids = {node["id"] for node in supplier_nodes if node["tier"] == 1}
        tier2_ids = {node["id"] for node in supplier_nodes if node["tier"] == 2}

        oem_edges = [edge for edge in network["edges"] if edge["target"] == "OEM" and edge["source"] in tier1_ids]
        upstream_edges = [edge for edge in network["edges"] if edge["source"] in tier2_ids and edge["target"] in tier1_ids]

        self.assertTrue(oem_edges)
        self.assertTrue(upstream_edges)
        self.assertIn("DIST", {node["id"] for node in network["nodes"]})
        self.assertIn("CUST", {node["id"] for node in network["nodes"]})

        tier2_node = next(node for node in supplier_nodes if node["tier"] == 2)
        self.assertGreater(tier2_node["spend"], 0)
        self.assertGreaterEqual(tier2_node["financial_health"], 0.1)
        self.assertLessEqual(tier2_node["financial_health"], 0.95)


if __name__ == "__main__":
    unittest.main()
