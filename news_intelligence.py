"""
news_intelligence.py
====================
Sentinel Agent: Supply Chain News Intelligence

Connects to NewsAPI to fetch real-world supply chain disruption news,
then uses AI (Claude) to:
  1. Classify disruption type and severity
  2. Match affected companies/regions to YOUR suppliers
  3. Estimate financial exposure
  4. Generate actionable recommendations

Runs WITHOUT the Anthropic API if no key is available (uses rule-based fallback).
"""

import os
import re
import time
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import requests
import pandas as pd


# ─── DATA CLASSES ─────────────────────────────────────────────────

@dataclass
class NewsArticle:
    """A single news article from NewsAPI."""
    article_id: str
    title: str
    description: str
    url: str
    source: str
    published_at: datetime
    content_snippet: str = ""


@dataclass
class SupplierNewsImpact:
    """Result of analyzing a news article against a supplier portfolio."""
    article: NewsArticle
    disruption_type: str          # e.g. "Trade/Tariff", "Natural Disaster", "Geopolitical"
    severity: str                 # "Critical", "High", "Medium", "Low"
    severity_score: int           # 0-100
    affected_suppliers: list      # list of supplier names from your portfolio
    affected_countries: list      # countries mentioned in article
    affected_categories: list     # product categories mentioned
    estimated_exposure_usd: float # rough financial exposure estimate
    summary: str                  # 2-3 sentence AI summary
    recommended_actions: list     # list of action strings
    confidence: str               # "High", "Medium", "Low" - how sure we are of match
    analysis_method: str          # "AI" or "Rule-based"


# ─── NEWS API CLIENT ──────────────────────────────────────────────

SUPPLY_CHAIN_QUERIES = [
    "supply chain disruption",
    "factory shutdown manufacturing",
    "port congestion freight delay",
    "tariff trade war import",
    "supplier bankruptcy insolvency",
    "natural disaster factory flood earthquake",
    "semiconductor chip shortage",
    "geopolitical risk trade restriction",
]

DISRUPTION_KEYWORDS = {
    "Trade/Tariff": [
        "tariff", "trade war", "sanctions", "import duty", "export ban",
        "trade restriction", "embargo", "customs", "section 301", "section 232"
    ],
    "Natural Disaster": [
        "earthquake", "flood", "typhoon", "hurricane", "tsunami", "tornado",
        "wildfire", "drought", "monsoon", "storm damage", "natural disaster"
    ],
    "Geopolitical": [
        "war", "conflict", "invasion", "coup", "political instability",
        "civil unrest", "protest", "border closure", "diplomatic", "sanctions"
    ],
    "Logistics/Transport": [
        "port congestion", "shipping delay", "freight", "container shortage",
        "logistics disruption", "rail strike", "trucking shortage", "canal blockage"
    ],
    "Labor/Industrial": [
        "strike", "labor dispute", "union", "worker shortage", "walkout",
        "factory closure", "plant shutdown", "workforce disruption"
    ],
    "Financial/Bankruptcy": [
        "bankruptcy", "insolvency", "financial distress", "debt default",
        "credit downgrade", "receivership", "liquidation", "restructuring"
    ],
    "Quality/Safety": [
        "recall", "quality issue", "safety concern", "contamination",
        "product defect", "compliance failure", "regulatory action", "FDA warning"
    ],
    "Cyber/Technology": [
        "cyberattack", "ransomware", "data breach", "system outage",
        "IT disruption", "hack", "malware"
    ],
    "Pandemic/Health": [
        "pandemic", "outbreak", "covid", "lockdown", "quarantine",
        "health crisis", "epidemic"
    ],
}

SEVERITY_KEYWORDS = {
    "Critical": [
        "complete shutdown", "total disruption", "catastrophic", "plant closure",
        "major earthquake", "severe flood", "war", "invasion", "sanctions imposed",
        "banned", "factory destroyed"
    ],
    "High": [
        "significant disruption", "major delay", "production halted",
        "supply shortage", "strike", "bankruptcy", "heavy flooding"
    ],
    "Medium": [
        "delay", "slowdown", "reduced capacity", "shortage", "concern",
        "warning", "monitoring", "risk of", "potential impact"
    ],
    "Low": [
        "minor", "slight", "recovering", "resolved", "minimal impact", "watch"
    ],
}


def fetch_news(api_key: str, query: str = None, days_back: int = 7,
               page_size: int = 20) -> list[NewsArticle]:
    """
    Fetch supply chain news from NewsAPI.
    Returns list of NewsArticle objects.
    """
    if not api_key or api_key.strip() == "":
        return []

    base_url = "https://newsapi.org/v2/everything"
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # Default to supply chain query if none provided
    search_query = query or (
        "supply chain disruption OR tariff OR factory shutdown OR freight delay"
    )

    params = {
        "q": search_query,
        "from": from_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": min(page_size, 100),
        "apiKey": api_key,
    }

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            return []

        articles = []
        for art in data.get("articles", []):
            # Create stable ID from URL hash
            art_id = hashlib.md5((art.get("url", "") + art.get("title", "")).encode()).hexdigest()[:12]

            pub_at = art.get("publishedAt", "")
            try:
                pub_dt = datetime.fromisoformat(pub_at.replace("Z", "+00:00"))
            except Exception:
                pub_dt = datetime.utcnow()

            articles.append(NewsArticle(
                article_id=art_id,
                title=art.get("title", "") or "",
                description=art.get("description", "") or "",
                url=art.get("url", "") or "",
                source=art.get("source", {}).get("name", "Unknown"),
                published_at=pub_dt,
                content_snippet=art.get("content", "") or "",
            ))

        return articles

    except requests.exceptions.ConnectionError:
        return []
    except Exception:
        return []


# ─── RULE-BASED CLASSIFIER (no API key needed) ────────────────────

def classify_disruption_type(title: str, description: str) -> str:
    """Classify disruption type using keyword matching."""
    text = (title + " " + description).lower()
    scores = {}
    for dtype, keywords in DISRUPTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text)
        if score > 0:
            scores[dtype] = score
    return max(scores, key=scores.get) if scores else "General Supply Chain"


def classify_severity(title: str, description: str) -> tuple[str, int]:
    """Classify severity level. Returns (level, score 0-100)."""
    text = (title + " " + description).lower()
    for level in ["Critical", "High", "Medium", "Low"]:
        keywords = SEVERITY_KEYWORDS[level]
        if any(kw.lower() in text for kw in keywords):
            score_map = {"Critical": 90, "High": 70, "Medium": 45, "Low": 20}
            return level, score_map[level]
    return "Medium", 40


def extract_countries_from_text(text: str) -> list:
    """Extract country mentions from text."""
    KNOWN_COUNTRIES = [
        "China", "India", "Vietnam", "Mexico", "Taiwan", "South Korea", "Japan",
        "Germany", "USA", "United States", "UK", "United Kingdom", "France",
        "Brazil", "Malaysia", "Thailand", "Indonesia", "Philippines", "Bangladesh",
        "Turkey", "Poland", "Czech Republic", "Hungary", "Romania",
    ]
    text_lower = text.lower()
    found = []
    for country in KNOWN_COUNTRIES:
        if country.lower() in text_lower:
            found.append(country)
    return list(dict.fromkeys(found))  # Deduplicate while preserving order


def match_suppliers_to_article(
    article: NewsArticle,
    supplier_df: pd.DataFrame
) -> tuple[list, list, float]:
    """
    Match news article to suppliers in your portfolio.

    Returns:
      - matched_supplier_names: list of supplier names affected
      - affected_categories: product categories affected
      - estimated_exposure: rough USD exposure estimate
    """
    if supplier_df is None or len(supplier_df) == 0:
        return [], [], 0.0

    text = (article.title + " " + article.description).lower()
    affected_suppliers = []
    total_exposure = 0.0

    # 1. Direct name match
    for _, row in supplier_df.iterrows():
        name = str(row.get("supplier_name", ""))
        if len(name) > 3 and name.lower() in text:
            affected_suppliers.append(name)
            spend = float(row.get("annual_spend", 0))
            total_exposure += spend

    # 2. Country match
    article_countries = extract_countries_from_text(
        article.title + " " + article.description
    )
    if "country" in supplier_df.columns:
        for country in article_countries:
            country_suppliers = supplier_df[
                supplier_df["country"].str.lower() == country.lower()
            ]
            for _, row in country_suppliers.iterrows():
                name = row["supplier_name"]
                if name not in affected_suppliers:
                    affected_suppliers.append(name)
                    spend = float(row.get("annual_spend", 0))
                    # Country-level match = partial exposure (maybe 30-50% of spend)
                    total_exposure += spend * 0.4

    # 3. Category keyword match
    CATEGORY_KEYWORDS = {
        "Electronics": ["semiconductor", "chip", "electronic", "pcb", "display", "sensor"],
        "Metals": ["steel", "aluminum", "copper", "metal", "foundry", "casting"],
        "Plastics": ["plastic", "polymer", "resin", "injection molding"],
        "Textiles": ["textile", "fabric", "apparel", "garment"],
        "Auto Parts": ["automotive", "vehicle", "car", "auto", "iatf"],
        "Chemicals": ["chemical", "pharmaceutical", "reagent"],
        "Precision Parts": ["precision", "machined", "cnc", "aerospace"],
    }
    affected_categories = []
    if "category" in supplier_df.columns:
        for cat, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                cat_suppliers = supplier_df[
                    supplier_df["category"].str.lower().str.contains(cat.lower(), na=False)
                ]
                if len(cat_suppliers) > 0:
                    affected_categories.append(cat)
                    for _, row in cat_suppliers.iterrows():
                        name = row["supplier_name"]
                        if name not in affected_suppliers:
                            affected_suppliers.append(name)
                            spend = float(row.get("annual_spend", 0))
                            total_exposure += spend * 0.2

    return affected_suppliers, affected_categories, total_exposure


def generate_recommendations(
    disruption_type: str,
    severity: str,
    affected_suppliers: list,
    estimated_exposure: float,
) -> list:
    """Generate actionable recommendations based on disruption analysis."""
    recs = []

    # Universal recommendation
    if affected_suppliers:
        recs.append(
            f"Contact {', '.join(affected_suppliers[:3])} "
            f"immediately to assess current inventory levels and order status."
        )

    # Type-specific
    type_recs = {
        "Trade/Tariff": [
            "Review tariff classification for affected goods — reclassification may reduce duty.",
            "Evaluate bonded warehouse or FTZ use to defer tariff payments.",
            "Request suppliers to explore alternative HTS codes or origin routing.",
        ],
        "Natural Disaster": [
            "Activate backup supplier qualification process immediately.",
            "Check current inventory against demand; increase safety stock for affected SKUs.",
            "Request force majeure notification from affected suppliers.",
        ],
        "Geopolitical": [
            "Begin dual-sourcing qualification for suppliers in affected region.",
            "Review contract terms for geopolitical force majeure clauses.",
            "Consult legal counsel on trade compliance implications.",
        ],
        "Logistics/Transport": [
            "Identify alternative routing (air freight, alternate ports) for critical shipments.",
            "Review in-transit inventory — may need expedited clearance.",
            "Adjust lead times and safety stock for next 60-90 days.",
        ],
        "Labor/Industrial": [
            "Request production status from affected suppliers.",
            "Assess finished goods buffer stock at supplier facilities.",
            "Prepare purchase orders for backup suppliers.",
        ],
        "Financial/Bankruptcy": [
            "Place orders only against confirmed letters of credit.",
            "Retrieve tooling, dies, and IP from supplier facilities immediately.",
            "Begin emergency supplier qualification for all sourced parts.",
        ],
        "Quality/Safety": [
            "Issue stop-ship/quarantine for in-transit product from affected supplier.",
            "Initiate incoming quality inspection for recent receipts.",
            "Review product liability and insurance coverage.",
        ],
        "Cyber/Technology": [
            "Request supplier confirmation that no customer data was compromised.",
            "Assess whether your EDI/ERP integration was affected.",
            "Prepare for potential order processing delays of 2-4 weeks.",
        ],
    }

    recs.extend(type_recs.get(disruption_type, [
        "Monitor situation closely over next 48-72 hours.",
        "Alert procurement leadership and supply chain risk team.",
    ]))

    # Severity escalation
    if severity == "Critical":
        recs.insert(0, "⚠️ ESCALATE TO EXECUTIVE LEADERSHIP — Critical supply chain event detected.")
    elif severity == "High":
        recs.insert(0, "Escalate to supply chain leadership within 24 hours.")

    # Financial threshold
    if estimated_exposure > 1_000_000:
        recs.append(
            f"Estimated exposure of ${estimated_exposure:,.0f} exceeds $1M threshold — "
            f"engage CFO and insurance broker."
        )

    return recs[:6]  # Limit to top 6 recommendations


def analyze_article_rule_based(
    article: NewsArticle,
    supplier_df: pd.DataFrame,
) -> SupplierNewsImpact:
    """Analyze a news article without AI — pure rule-based classification."""
    disruption_type = classify_disruption_type(article.title, article.description)
    severity, severity_score = classify_severity(article.title, article.description)
    affected_countries = extract_countries_from_text(article.title + " " + article.description)
    affected_suppliers, affected_categories, exposure = match_suppliers_to_article(article, supplier_df)

    recommendations = generate_recommendations(
        disruption_type, severity, affected_suppliers, exposure
    )

    confidence = "High" if affected_suppliers else ("Medium" if affected_countries else "Low")

    summary = (
        f"{article.title}. "
        f"Classified as {disruption_type} event with {severity} severity. "
    )
    if affected_suppliers:
        summary += f"Potentially affects: {', '.join(affected_suppliers[:3])}."
    elif affected_countries:
        summary += f"Affects suppliers in: {', '.join(affected_countries[:3])}."

    return SupplierNewsImpact(
        article=article,
        disruption_type=disruption_type,
        severity=severity,
        severity_score=severity_score,
        affected_suppliers=affected_suppliers,
        affected_countries=affected_countries,
        affected_categories=affected_categories,
        estimated_exposure_usd=exposure,
        summary=summary,
        recommended_actions=recommendations,
        confidence=confidence,
        analysis_method="Rule-based",
    )


def analyze_article_with_ai(
    article: NewsArticle,
    supplier_df: pd.DataFrame,
    anthropic_api_key: str,
) -> SupplierNewsImpact:
    """
    Analyze article using Claude AI for richer classification.
    Falls back to rule-based if API call fails.
    """
    try:
        import anthropic

        # Build supplier context for AI
        supplier_context = ""
        if supplier_df is not None and len(supplier_df) > 0:
            for _, row in supplier_df.iterrows():
                supplier_context += (
                    f"- {row['supplier_name']} | {row.get('country', 'Unknown')} | "
                    f"Tier {row.get('tier', 1)} | {row.get('category', 'Unknown')} | "
                    f"Spend: ${float(row.get('annual_spend', 0)):,.0f}\n"
                )

        prompt = f"""You are a supply chain risk intelligence analyst.

NEWS ARTICLE:
Title: {article.title}
Source: {article.source}
Published: {article.published_at.strftime('%Y-%m-%d')}
Description: {article.description}
{f'Content: {article.content_snippet[:500]}' if article.content_snippet else ''}

SUPPLIER PORTFOLIO:
{supplier_context if supplier_context else 'No supplier data available.'}

Analyze this article for supply chain risk. Respond ONLY with a JSON object (no markdown):
{{
  "disruption_type": "<one of: Trade/Tariff, Natural Disaster, Geopolitical, Logistics/Transport, Labor/Industrial, Financial/Bankruptcy, Quality/Safety, Cyber/Technology, Pandemic/Health, General Supply Chain>",
  "severity": "<one of: Critical, High, Medium, Low>",
  "severity_score": <integer 0-100>,
  "affected_suppliers": ["<exact supplier names from the portfolio that are directly or indirectly affected>"],
  "affected_countries": ["<countries mentioned or implied>"],
  "affected_categories": ["<product categories at risk>"],
  "estimated_exposure_fraction": <float 0-1, fraction of matched supplier annual spend at risk>,
  "summary": "<2-3 sentence plain-English summary of the risk and portfolio impact>",
  "recommended_actions": ["<up to 5 specific, actionable steps for a procurement manager>"],
  "confidence": "<High/Medium/Low — how confident you are this affects the portfolio>"
}}"""

        client = anthropic.Anthropic(api_key=anthropic_api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # Fast and cheap for classification
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        import json
        text = response.content[0].text.strip()
        # Strip markdown code blocks if present
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        ai_result = json.loads(text)

        # Calculate exposure from fraction
        total_spend = supplier_df["annual_spend"].sum() if supplier_df is not None else 0
        exposure = total_spend * float(ai_result.get("estimated_exposure_fraction", 0))

        # Match supplier names back to portfolio (AI may have slight name differences)
        matched_suppliers = ai_result.get("affected_suppliers", [])
        if supplier_df is not None:
            portfolio_names = supplier_df["supplier_name"].tolist()
            validated = []
            for ai_name in matched_suppliers:
                # Find closest match in portfolio
                for pname in portfolio_names:
                    if (ai_name.lower() in pname.lower() or
                            pname.lower() in ai_name.lower() or
                            _name_similarity(ai_name, pname) > 0.7):
                        if pname not in validated:
                            validated.append(pname)
                        break
            matched_suppliers = validated if validated else matched_suppliers

        return SupplierNewsImpact(
            article=article,
            disruption_type=ai_result.get("disruption_type", "General Supply Chain"),
            severity=ai_result.get("severity", "Medium"),
            severity_score=int(ai_result.get("severity_score", 40)),
            affected_suppliers=matched_suppliers,
            affected_countries=ai_result.get("affected_countries", []),
            affected_categories=ai_result.get("affected_categories", []),
            estimated_exposure_usd=exposure,
            summary=ai_result.get("summary", article.description),
            recommended_actions=ai_result.get("recommended_actions", []),
            confidence=ai_result.get("confidence", "Medium"),
            analysis_method="AI (Claude)",
        )

    except Exception:
        # Fallback to rule-based
        return analyze_article_rule_based(article, supplier_df)


def _name_similarity(a: str, b: str) -> float:
    """Simple character-level similarity for name matching."""
    a, b = a.lower(), b.lower()
    if not a or not b:
        return 0.0
    common = sum(1 for c in set(a) if c in b)
    return common / max(len(set(a)), len(set(b)))


def run_sentinel_scan(
    news_api_key: str,
    supplier_df: pd.DataFrame,
    anthropic_api_key: str = "",
    days_back: int = 7,
    max_articles: int = 20,
    custom_query: str = "",
) -> list[SupplierNewsImpact]:
    """
    Main entry point: fetch news + analyze impact on supplier portfolio.

    Returns list of SupplierNewsImpact, sorted by severity score descending.
    """
    # Build targeted query from supplier portfolio
    if not custom_query and supplier_df is not None and len(supplier_df) > 0:
        countries = supplier_df["country"].dropna().unique().tolist()[:5] if "country" in supplier_df.columns else []
        categories = supplier_df["category"].dropna().unique().tolist()[:3] if "category" in supplier_df.columns else []

        query_parts = ["supply chain disruption"]
        if countries:
            query_parts.append(" OR ".join(countries[:4]))
        if categories:
            query_parts.append(" OR ".join(categories[:3]))

        query = " OR ".join(query_parts)
    else:
        query = custom_query or "supply chain disruption tariff factory"

    # Fetch articles
    articles = fetch_news(news_api_key, query=query, days_back=days_back, page_size=max_articles)

    if not articles:
        return []

    # Analyze each article
    use_ai = bool(anthropic_api_key and anthropic_api_key.strip())
    impacts = []

    for article in articles:
        if use_ai:
            impact = analyze_article_with_ai(article, supplier_df, anthropic_api_key)
        else:
            impact = analyze_article_rule_based(article, supplier_df)

        impacts.append(impact)
        if use_ai:
            time.sleep(0.3)  # Rate limit courtesy pause

    # Sort by severity score
    impacts.sort(key=lambda x: x.severity_score, reverse=True)
    return impacts


# ─── SEVERITY STYLING HELPERS ─────────────────────────────────────

SEVERITY_COLORS = {
    "Critical": "#ef4444",
    "High": "#f59e0b",
    "Medium": "#3b82f6",
    "Low": "#10b981",
}

SEVERITY_ICONS = {
    "Critical": "🔴",
    "High": "🟠",
    "Medium": "🔵",
    "Low": "🟢",
}

DISRUPTION_ICONS = {
    "Trade/Tariff": "💱",
    "Natural Disaster": "🌪️",
    "Geopolitical": "🌍",
    "Logistics/Transport": "🚢",
    "Labor/Industrial": "👷",
    "Financial/Bankruptcy": "💸",
    "Quality/Safety": "⚠️",
    "Cyber/Technology": "💻",
    "Pandemic/Health": "🏥",
    "General Supply Chain": "📦",
}
