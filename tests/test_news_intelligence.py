import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from news_intelligence import (
    NewsArticle,
    SupplierNewsImpact,
    build_public_article_analysis_prompt,
    run_sentinel_scan,
)


class NewsIntelligenceTests(unittest.TestCase):
    def test_supplier_news_impact_normalizes_severity_to_lowercase(self):
        impact = SupplierNewsImpact(
            article=NewsArticle(
                article_id="a1",
                title="Port strike expands",
                description="",
                url="https://example.com",
                source="Example",
                published_at=datetime.utcnow(),
            ),
            disruption_type="Logistics/Transport",
            severity="High",
            severity_score=75,
            affected_suppliers=[],
            affected_countries=[],
            affected_categories=[],
            estimated_exposure_usd=0.0,
            summary="",
            recommended_actions=[],
            confidence="Medium",
            analysis_method="test",
        )

        self.assertEqual(impact.severity, "high")
        self.assertEqual(impact.confidence, "medium")

    def test_live_llm_prompt_contains_only_public_article_text(self):
        article = NewsArticle(
            article_id="a1",
            title="Typhoon disrupts southern China manufacturing",
            description="Ports and factories report delays.",
            url="https://example.com/news",
            source="Example News",
            published_at=datetime(2026, 4, 25),
            content_snippet="Public article snippet.",
        )

        prompt = build_public_article_analysis_prompt(article)

        self.assertIn("Typhoon disrupts southern China manufacturing", prompt)
        self.assertNotIn("Apex Secret Supplier", prompt)
        self.assertNotIn("annual_spend", prompt)
        self.assertNotIn("supplier_name", prompt)

    @patch("news_intelligence.classify_article_with_llm")
    @patch("news_intelligence.fetch_news_local")
    def test_live_news_ai_scan_matches_suppliers_locally(self, mock_fetch, mock_classify):
        mock_fetch.return_value = (
            [
                NewsArticle(
                    article_id="a1",
                    title="China electronics exports face delay",
                    description="Electronics shipments from China are delayed.",
                    url="https://example.com/news",
                    source="Example News",
                    published_at=datetime(2026, 4, 25),
                )
            ],
            "",
        )
        mock_classify.return_value = {
            "is_relevant": True,
            "disruption_type": "Logistics/Transport",
            "severity": "high",
            "severity_score": 78,
            "affected_countries": ["China"],
            "affected_categories": ["electronics"],
            "summary": "Electronics shipments from China may face delays.",
            "recommended_actions": ["Contact exposed suppliers."],
            "confidence": "high",
        }
        suppliers = pd.DataFrame(
            [
                {
                    "supplier_name": "Apex Electronics",
                    "country": "China",
                    "category": "Electronics",
                    "annual_spend": 100000.0,
                }
            ]
        )

        impacts, mode, error = run_sentinel_scan(
            news_api_key="news-key",
            supplier_df=suppliers,
            openai_api_key="openai-key",
            llm_provider="openai",
            mode="live_ai",
        )

        self.assertEqual(mode, "Live News + AI (OpenAI)")
        self.assertEqual(error, "")
        self.assertEqual(impacts[0].affected_suppliers, ["Apex Electronics"])
        self.assertGreater(impacts[0].estimated_exposure_usd, 0)


if __name__ == "__main__":
    unittest.main()
