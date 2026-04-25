"""
news_intelligence.py — FIXED VERSION
=====================================
ROOT CAUSE OF ORIGINAL FAILURE:
  1. newsapi.org is BLOCKED by Streamlit Cloud's network proxy (403 Forbidden)
  2. ALL external news APIs are blocked — Reuters, BBC, GNews, NYT, MediaStack
  3. Original code silently swallowed errors with `except Exception: return []`
     so you never saw the real error in the UI
  4. Free NewsAPI tier also restricts /v2/everything endpoint anyway

SOLUTION:
  - Mode 1 (PRIMARY): Claude AI generates supply chain intelligence briefings
    based on your supplier portfolio. Uses Anthropic API which IS reachable.
    No external news key needed.
  - Mode 2 (DEMO): Realistic pre-built scenarios for presentation/demo use
    when no API key is available at all.
  - Mode 3 (NEWSAPI - LOCAL ONLY): Works when running locally on your machine.
    Streamlit Cloud blocks it, your laptop doesn't.

This architecture means the Sentinel tab ALWAYS shows useful output.
"""

import json
import os
import re
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


# ─── DATA CLASSES ─────────────────────────────────────────────────

@dataclass
class NewsArticle:
    article_id: str
    title: str
    description: str
    url: str
    source: str
    published_at: datetime
    content_snippet: str = ""


@dataclass
class SupplierNewsImpact:
    article: NewsArticle
    disruption_type: str
    severity: str
    severity_score: int
    affected_suppliers: list
    affected_countries: list
    affected_categories: list
    estimated_exposure_usd: float
    summary: str
    recommended_actions: list
    confidence: str
    analysis_method: str

    def __post_init__(self):
        self.severity = normalize_severity(self.severity)
        self.confidence = str(self.confidence or "medium").lower()


# ─── SEVERITY / DISRUPTION STYLING ────────────────────────────────

SEVERITY_COLORS = {
    "critical": "#ef4444",
    "high": "#f59e0b",
    "medium": "#3b82f6",
    "low": "#10b981",
    "info": "#94a3b8",
    "Critical": "#ef4444",
    "High": "#f59e0b",
    "Medium": "#3b82f6",
    "Low": "#10b981",
}

SEVERITY_ICONS = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🔵",
    "low": "🟢",
    "info": "⚪",
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

DISRUPTION_KEYWORDS = {
    "Trade/Tariff": ["tariff", "trade war", "sanctions", "import duty", "export ban", "embargo"],
    "Natural Disaster": ["earthquake", "flood", "typhoon", "hurricane", "tsunami", "wildfire"],
    "Geopolitical": ["war", "conflict", "invasion", "sanctions", "political instability"],
    "Logistics/Transport": ["port congestion", "shipping delay", "freight", "container shortage"],
    "Labor/Industrial": ["strike", "labor dispute", "factory closure", "plant shutdown"],
    "Financial/Bankruptcy": ["bankruptcy", "insolvency", "financial distress", "default"],
    "Quality/Safety": ["recall", "quality issue", "contamination", "product defect"],
    "Cyber/Technology": ["cyberattack", "ransomware", "data breach", "system outage"],
    "Pandemic/Health": ["pandemic", "outbreak", "lockdown", "health crisis"],
}

VALID_SEVERITIES = {"critical", "high", "medium", "low", "info"}


def normalize_severity(value: str) -> str:
    severity = str(value or "medium").strip().lower()
    return severity if severity in VALID_SEVERITIES else "medium"


def severity_label(value: str) -> str:
    return normalize_severity(value).title()


def provider_label(provider: str) -> str:
    provider_key = (provider or "").strip().lower()
    if provider_key == "openai":
        return "OpenAI"
    if provider_key in {"anthropic", "claude"}:
        return "Anthropic"
    return provider_key.title()


def _extract_json_payload(raw_text: str):
    """Parse the first JSON object/array from an LLM response."""
    raw = (raw_text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start_candidates = [idx for idx in [raw.find("{"), raw.find("[")] if idx != -1]
        if not start_candidates:
            raise
        start = min(start_candidates)
        end = max(raw.rfind("}"), raw.rfind("]"))
        if end <= start:
            raise
        return json.loads(raw[start : end + 1])


# ─── MODE 1: CLAUDE AI INTELLIGENCE BRIEFING ─────────────────────
# Uses Anthropic API (which IS reachable from Streamlit Cloud)
# Claude generates current supply chain intelligence based on your portfolio

def run_ai_intelligence_briefing(
    supplier_df: pd.DataFrame,
    anthropic_api_key: str,
    n_events: int = 10,
    custom_query: str = "",
) -> list[SupplierNewsImpact]:
    """
    Uses Claude to generate a supply chain intelligence briefing
    based on your supplier portfolio and current geopolitical context.
    
    This WORKS on Streamlit Cloud because Anthropic API is reachable.
    No external news API needed.
    """
    try:
        import anthropic

        # Build privacy-safe aggregate context. Do not send supplier names,
        # spend, pricing, delivery metrics, or network topology to the LLM.
        portfolio_lines = []
        if supplier_df is not None and len(supplier_df) > 0:
            if "country" in supplier_df.columns:
                country_counts = supplier_df["country"].fillna("Unknown").value_counts().head(12)
                portfolio_lines.append(
                    "Supplier countries: "
                    + ", ".join(f"{country} ({count})" for country, count in country_counts.items())
                )
            if "category" in supplier_df.columns:
                category_counts = supplier_df["category"].fillna("Unknown").value_counts().head(12)
                portfolio_lines.append(
                    "Supplier categories: "
                    + ", ".join(f"{category} ({count})" for category, count in category_counts.items())
                )
            if "tier" in supplier_df.columns:
                tier_counts = supplier_df["tier"].fillna("Unknown").value_counts().sort_index()
                portfolio_lines.append(
                    "Supplier tiers: "
                    + ", ".join(f"Tier {tier} ({count})" for tier, count in tier_counts.items())
                )

        portfolio_text = "\n".join(portfolio_lines) if portfolio_lines else "No supplier data provided."
        focus = f"Focus on: {custom_query}." if custom_query else ""
        today = datetime.utcnow().strftime("%B %Y")

        prompt = f"""You are a senior supply chain risk intelligence analyst. Today is {today}.

AGGREGATED SUPPLIER PORTFOLIO (no supplier names or spend):
{portfolio_text}

{focus}

Generate {n_events} realistic, specific supply chain disruption intelligence briefings that are 
DIRECTLY RELEVANT to this portfolio. Base these on real patterns happening in supply chains 
right now (tariff escalations, regional instability, logistics bottlenecks, etc.).

For each event, respond with a JSON array. Each element must have exactly these fields:
{{
  "title": "Specific, realistic headline (max 100 chars)",
  "source": "Realistic news source name",
  "published_days_ago": <integer 0-6>,
  "description": "2-3 sentence description of the event and its supply chain impact",
  "disruption_type": "<one of: Trade/Tariff, Natural Disaster, Geopolitical, Logistics/Transport, Labor/Industrial, Financial/Bankruptcy, Quality/Safety, Cyber/Technology, Pandemic/Health, General Supply Chain>",
  "severity": "<one of: critical, high, medium, low, info>",
  "severity_score": <integer 0-100>,
  "affected_countries": ["countries affected"],
  "affected_categories": ["product categories affected"],
  "exposure_fraction": <float 0.0-1.0, fraction of matched supplier annual spend at risk>,
  "summary": "2-3 sentence analyst summary of risk and recommended response",
  "actions": [
    "Specific action 1 for procurement team",
    "Specific action 2",
    "Specific action 3"
  ],
  "confidence": "<High/Medium/Low>"
}}

Make events SPECIFIC and REALISTIC. Reference real trade tensions, real regions, real logistics 
bottlenecks that affect this aggregate portfolio. Mix severities. Ensure at least 2 critical/high events.
Return ONLY the JSON array, no markdown, no preamble."""

        client = anthropic.Anthropic(api_key=anthropic_api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        events = json.loads(raw)

        # Convert to SupplierNewsImpact objects
        total_spend = float(supplier_df["annual_spend"].sum()) if (supplier_df is not None and "annual_spend" in supplier_df.columns) else 0
        impacts = []
        now = datetime.utcnow()

        for ev in events:
            pub_dt = now - timedelta(days=int(ev.get("published_days_ago", 1)))

            validated_suppliers, matched_exposure = _match_supplier_impacts_locally(
                supplier_df=supplier_df,
                affected_countries=ev.get("affected_countries", []),
                affected_categories=ev.get("affected_categories", []),
                severity=ev.get("severity", "medium"),
            )
            exposure = matched_exposure or total_spend * float(ev.get("exposure_fraction", 0.1))

            article = NewsArticle(
                article_id=f"ai_{len(impacts):03d}",
                title=ev.get("title", "Supply Chain Intelligence Event"),
                description=ev.get("description", ""),
                url="#",  # AI-generated, no URL
                source=f"AI Intelligence Briefing ({ev.get('source', 'Analyst Synthesis')})",
                published_at=pub_dt,
                content_snippet="",
            )

            impacts.append(SupplierNewsImpact(
                article=article,
                disruption_type=ev.get("disruption_type", "General Supply Chain"),
                severity=ev.get("severity", "Medium"),
                severity_score=int(ev.get("severity_score", 40)),
                affected_suppliers=validated_suppliers,
                affected_countries=ev.get("affected_countries", []),
                affected_categories=ev.get("affected_categories", []),
                estimated_exposure_usd=exposure,
                summary=ev.get("summary", ev.get("description", "")),
                recommended_actions=ev.get("actions", []),
                confidence=ev.get("confidence", "Medium"),
                analysis_method="AI Intelligence Briefing (Claude)",
            ))

        impacts.sort(key=lambda x: x.severity_score, reverse=True)
        return impacts

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Claude returned malformed JSON: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"AI briefing failed: {str(e)}")


# ─── MODE 2: DEMO MODE (no API key needed) ────────────────────────
# Realistic pre-built events for demos and presentations

def get_demo_impacts(supplier_df: pd.DataFrame) -> list[SupplierNewsImpact]:
    """
    Returns realistic demo intelligence events.
    Used when no API key is available — always shows something useful.
    Tailored to the supplier portfolio when possible.
    """
    now = datetime.utcnow()

    # Extract portfolio context
    countries = []
    categories = []
    supplier_names = []
    if supplier_df is not None and len(supplier_df) > 0:
        if "country" in supplier_df.columns:
            countries = supplier_df["country"].dropna().unique().tolist()
        if "category" in supplier_df.columns:
            categories = supplier_df["category"].dropna().unique().tolist()
        if "supplier_name" in supplier_df.columns:
            supplier_names = supplier_df["supplier_name"].tolist()

    total_spend = float(supplier_df["annual_spend"].sum()) if (
        supplier_df is not None and "annual_spend" in supplier_df.columns
    ) else 5_000_000

    # Build events — reference actual countries/categories from portfolio
    china_suppliers = [s for s in supplier_names if "china" in str(
        supplier_df[supplier_df["supplier_name"] == s]["country"].values[0] if len(supplier_df[supplier_df["supplier_name"] == s]) > 0 else ""
    ).lower()]

    events_data = [
        {
            "title": "US Announces 25% Tariff Escalation on Chinese Electronics Components",
            "source": "Reuters",
            "days_ago": 1,
            "description": "The Biden administration announced an additional 25% tariff on Chinese-manufactured electronics components effective 90 days from today, affecting semiconductors, PCBs, and sensors.",
            "disruption_type": "Trade/Tariff",
            "severity": "Critical",
            "severity_score": 92,
            "affected_countries": ["China"],
            "affected_categories": ["Electronics", "Semiconductors"],
            "exposure_fraction": 0.35,
            "summary": "25% tariff escalation on Chinese electronics creates immediate cost pressure. Suppliers in China face significant margin compression. Recommend dual-sourcing qualification in Vietnam or Taiwan within 60 days.",
            "actions": [
                "Immediately calculate total tariff exposure across all Chinese-origin suppliers",
                "Request HTS code review from customs broker — reclassification may reduce duty",
                "Begin RFQ process with Vietnam and Taiwan alternatives within 2 weeks",
                "Evaluate bonded warehouse or FTZ options to defer tariff payments",
                "Brief CFO on estimated annual cost increase before next earnings call",
            ],
            "confidence": "High",
            "affected_supplier_names": china_suppliers[:3] if china_suppliers else [],
        },
        {
            "title": "Typhoon Saola Causes Severe Port Disruptions in Taiwan Strait",
            "source": "Bloomberg",
            "days_ago": 2,
            "description": "Category 4 Typhoon Saola has forced closure of major ports in Taiwan and southeastern China, with shipping delays of 14-21 days expected across Kaohsiung, Keelung, and Xiamen.",
            "disruption_type": "Natural Disaster",
            "severity": "High",
            "severity_score": 78,
            "affected_countries": ["Taiwan", "China"],
            "affected_categories": ["Electronics", "Precision Parts"],
            "exposure_fraction": 0.20,
            "summary": "Port closures affecting Taiwan Strait create 2-3 week shipping delay for electronics and precision components. In-transit inventory at risk. Safety stock depletion likely within 30 days for fast-moving SKUs.",
            "actions": [
                "Check current in-transit shipments from Taiwan and SE China — contact freight forwarder",
                "Calculate days of safety stock remaining for affected SKUs at current demand",
                "Identify critical items below 30-day cover — trigger emergency air freight if needed",
                "Request force majeure notification from affected suppliers",
                "Adjust demand planning horizon to 90 days for affected categories",
            ],
            "confidence": "High",
            "affected_supplier_names": [],
        },
        {
            "title": "Indian Logistics Strike Disrupts Freight Movement in Maharashtra",
            "source": "Financial Times",
            "days_ago": 3,
            "description": "A nationwide trucking strike in India has disrupted freight movement from manufacturing hubs including Pune and Mumbai. Strike expected to last 7-10 days, affecting shipment clearance and port operations.",
            "disruption_type": "Labor/Industrial",
            "severity": "High",
            "severity_score": 71,
            "affected_countries": ["India"],
            "affected_categories": ["Metals", "Auto Parts", "Plastics"],
            "exposure_fraction": 0.15,
            "summary": "Indian trucking strike affecting major manufacturing corridors. Suppliers in Maharashtra face production bottlenecks even if factories are running. Export shipments delayed 1-2 weeks minimum.",
            "actions": [
                "Contact all Indian suppliers to assess current inventory and shipment status",
                "Check if in-transit goods from India are held at customs or port",
                "Request suppliers to pre-position finished goods at port once strike resolves",
                "Evaluate air freight for critical components with < 3 weeks stock cover",
            ],
            "confidence": "High",
            "affected_supplier_names": [],
        },
        {
            "title": "Vietnam Manufacturing PMI Drops to 3-Year Low Amid Export Slowdown",
            "source": "Wall Street Journal",
            "days_ago": 4,
            "description": "Vietnam's manufacturing Purchasing Managers Index fell to 46.8 in the latest reading, signaling contraction. Several key industrial zones near Ho Chi Minh City report order backlogs and reduced capacity utilization.",
            "disruption_type": "General Supply Chain",
            "severity": "Medium",
            "severity_score": 48,
            "affected_countries": ["Vietnam"],
            "affected_categories": ["Plastics", "Textiles", "Electronics"],
            "exposure_fraction": 0.10,
            "summary": "Vietnam manufacturing contraction may indicate quality or delivery reliability risks. Suppliers facing reduced orders may cut corners on quality control or delay investments in capacity.",
            "actions": [
                "Schedule quality audit visits to Vietnamese suppliers in next 90 days",
                "Request updated capacity confirmation for your next 2 forecast periods",
                "Monitor on-time delivery metrics closely — early warning of performance decline",
            ],
            "confidence": "Medium",
            "affected_supplier_names": [],
        },
        {
            "title": "German Industrial Output Falls 4.2% — Auto Supplier Chain Under Pressure",
            "source": "Der Spiegel",
            "days_ago": 5,
            "description": "German industrial production fell sharply in Q4, with automotive supply chain output down 4.2% year-over-year. Several Tier-2 suppliers have announced capacity reductions and workforce layoffs.",
            "disruption_type": "Financial/Bankruptcy",
            "severity": "Medium",
            "severity_score": 52,
            "affected_countries": ["Germany"],
            "affected_categories": ["Auto Parts", "Precision Parts", "Metals"],
            "exposure_fraction": 0.08,
            "summary": "German industrial decline creates financial stress for precision manufacturers. Watch for supplier financial distress signals — delayed payments, workforce cuts, facility consolidations.",
            "actions": [
                "Request latest financial statements from German Tier-1 suppliers",
                "Review payment terms — consider early payment programs for financially stressed suppliers",
                "Assess tooling and IP ownership — recover if supplier shows distress signals",
            ],
            "confidence": "Medium",
            "affected_supplier_names": [],
        },
        {
            "title": "Red Sea Shipping Crisis: Freight Rates Up 180% as Carriers Reroute",
            "source": "Lloyd's List",
            "days_ago": 2,
            "description": "Ongoing Houthi attacks in the Red Sea have caused major container carriers to reroute around Cape of Good Hope, adding 10-14 days to Asia-Europe shipping times and driving freight rates up 180% since Q3.",
            "disruption_type": "Logistics/Transport",
            "severity": "High",
            "severity_score": 75,
            "affected_countries": ["India", "Vietnam", "Bangladesh"],
            "affected_categories": ["Textiles", "Electronics", "Plastics"],
            "exposure_fraction": 0.18,
            "summary": "Red Sea rerouting significantly increases lead times and freight costs for Asia-Europe lanes. Suppliers quoting FOB prices may face margin pressure from freight surcharges. Plan for 3-4 week additional lead time.",
            "actions": [
                "Update all Asia-sourced supplier lead times by +14 days in planning systems",
                "Renegotiate freight terms — shift to DDP or CIF where possible to lock in costs",
                "Review safety stock levels — increase by 30% for Asia-origin suppliers on Europe lanes",
                "Consider air freight for high-value, low-weight components to bridge the gap",
            ],
            "confidence": "High",
            "affected_supplier_names": [],
        },
        {
            "title": "Semiconductor Equipment Export Restrictions Extended to Malaysia and Thailand",
            "source": "Nikkei Asia",
            "days_ago": 0,
            "description": "US Commerce Department extended advanced semiconductor equipment export restrictions to include Malaysia and Thailand, impacting chip packaging and testing facilities that serve global electronics OEMs.",
            "disruption_type": "Trade/Tariff",
            "severity": "Medium",
            "severity_score": 58,
            "affected_countries": ["Malaysia", "Thailand"],
            "affected_categories": ["Electronics", "Semiconductors"],
            "exposure_fraction": 0.12,
            "summary": "Extended export restrictions may limit capacity expansion at Southeast Asian chip facilities. Medium-term supply risk for electronics components sourced from Malaysia/Thailand packaging houses.",
            "actions": [
                "Map which of your electronics components pass through Malaysia/Thailand for packaging",
                "Request capacity confirmation from affected suppliers for next 6-12 months",
                "Begin qualification of alternative packaging sources in Japan or South Korea",
            ],
            "confidence": "Medium",
            "affected_supplier_names": [],
        },
        {
            "title": "Mexican Nearshoring Boom Creates Skilled Labor Shortage in Monterrey",
            "source": "Bloomberg",
            "days_ago": 6,
            "description": "Rapid expansion of manufacturing capacity in Nuevo León and Tamaulipas has created a severe skilled labor shortage, with wages up 35% YoY. Several established suppliers report difficulty maintaining quality staffing.",
            "disruption_type": "Labor/Industrial",
            "severity": "Low",
            "severity_score": 32,
            "affected_countries": ["Mexico"],
            "affected_categories": ["Auto Parts", "Electronics", "Metals"],
            "exposure_fraction": 0.05,
            "summary": "Nearshoring demand driving wage inflation and quality staffing challenges in Mexico. Longer-term quality and reliability risk if suppliers can't retain skilled workers. Monitor quality KPIs closely.",
            "actions": [
                "Conduct quality trend review for all Mexican suppliers over last 6 months",
                "Include labor stability questions in next supplier business review",
                "Consider wage/benefit benchmarking as part of supplier development program",
            ],
            "confidence": "Low",
            "affected_supplier_names": [],
        },
    ]

    # Match suppliers from portfolio to events by country
    impacts = []
    for ev in events_data:
        affected = []
        if supplier_df is not None and "country" in supplier_df.columns:
            for country in ev["affected_countries"]:
                matches = supplier_df[supplier_df["country"].str.lower() == country.lower()]
                for _, row in matches.iterrows():
                    name = row["supplier_name"]
                    if name not in affected:
                        affected.append(name)

        # Also use pre-set names
        affected.extend(ev.get("affected_supplier_names", []))
        affected = list(dict.fromkeys(affected))  # deduplicate

        exposure = total_spend * ev["exposure_fraction"] if affected else total_spend * ev["exposure_fraction"] * 0.3

        pub_dt = now - timedelta(days=ev["days_ago"])
        article = NewsArticle(
            article_id=f"demo_{len(impacts):03d}",
            title=ev["title"],
            description=ev["description"],
            url="#",
            source=f"DEMO — {ev['source']}",
            published_at=pub_dt,
        )

        impacts.append(SupplierNewsImpact(
            article=article,
            disruption_type=ev["disruption_type"],
            severity=ev["severity"],
            severity_score=ev["severity_score"],
            affected_suppliers=affected,
            affected_countries=ev["affected_countries"],
            affected_categories=ev["affected_categories"],
            estimated_exposure_usd=exposure,
            summary=ev["summary"],
            recommended_actions=ev["actions"],
            confidence=ev["confidence"],
            analysis_method="Demo Mode (Example Events)",
        ))

    impacts.sort(key=lambda x: x.severity_score, reverse=True)
    return impacts


# ─── MODE 3: NEWSAPI (LOCAL ONLY) ─────────────────────────────────
# Works on your laptop. Blocked by Streamlit Cloud proxy.

def fetch_news_local(api_key: str, query: str = None, days_back: int = 7,
                     page_size: int = 20) -> tuple[list, str]:
    """
    Fetch from NewsAPI — ONLY works when running locally.
    Returns (articles, error_message).
    error_message is empty string if successful.
    """
    try:
        import requests

        base_url = "https://newsapi.org/v2/everything"
        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        search_query = query or "supply chain disruption OR tariff OR factory shutdown"

        params = {
            "q": search_query,
            "from": from_date,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": min(page_size, 100),
            "apiKey": api_key,
        }

        resp = requests.get(base_url, params=params, timeout=10)
        data = resp.json()

        if data.get("status") != "ok":
            error = data.get("message", "Unknown API error")
            if "apiKey" in error.lower() or "401" in str(resp.status_code):
                return [], "❌ Invalid NewsAPI key. Check your key at newsapi.org"
            if "rateLimited" in data.get("code", ""):
                return [], "⏱️ NewsAPI rate limit hit. Wait 1 hour or upgrade your plan."
            return [], f"❌ NewsAPI error: {error}"

        articles = []
        import hashlib
        for art in data.get("articles", []):
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

        return articles, ""

    except Exception as e:
        err_str = str(e)
        if "ProxyError" in err_str or "403" in err_str or "Forbidden" in err_str:
            return [], (
                "🚫 NewsAPI is blocked by Streamlit Cloud's network proxy. "
                "Use **AI Intelligence Briefing mode** instead (works on Streamlit Cloud), "
                "or run locally on your laptop where NewsAPI is accessible."
            )
        return [], f"❌ Network error: {err_str[:200]}"


def analyze_article_rule_based(article: NewsArticle, supplier_df: pd.DataFrame) -> SupplierNewsImpact:
    """Rule-based analysis for articles from local NewsAPI fetch."""
    text = (article.title + " " + article.description).lower()

    # Classify type
    scores = {dtype: sum(1 for kw in kws if kw.lower() in text)
              for dtype, kws in DISRUPTION_KEYWORDS.items()}
    disruption_type = max(scores, key=scores.get) if any(scores.values()) else "General Supply Chain"

    # Severity
    severity, severity_score = "Medium", 40
    if any(kw in text for kw in ["complete shutdown", "catastrophic", "war", "invasion", "banned"]):
        severity, severity_score = "Critical", 88
    elif any(kw in text for kw in ["significant", "major", "halted", "strike", "bankruptcy"]):
        severity, severity_score = "High", 70
    elif any(kw in text for kw in ["minor", "slight", "recovering", "minimal"]):
        severity, severity_score = "Low", 20

    # Countries
    COUNTRIES = ["China", "India", "Vietnam", "Mexico", "Taiwan", "South Korea", "Japan",
                 "Germany", "USA", "United States", "Malaysia", "Thailand", "Indonesia"]
    countries = [c for c in COUNTRIES if c.lower() in text]

    # Match suppliers
    affected = []
    exposure = 0.0
    if supplier_df is not None:
        for _, row in supplier_df.iterrows():
            name = str(row.get("supplier_name", ""))
            country = str(row.get("country", ""))
            if (len(name) > 3 and name.lower() in text) or country.lower() in [c.lower() for c in countries]:
                affected.append(name)
                exposure += float(row.get("annual_spend", 0)) * 0.3

    return SupplierNewsImpact(
        article=article,
        disruption_type=disruption_type,
        severity=severity,
        severity_score=severity_score,
        affected_suppliers=affected,
        affected_countries=countries,
        affected_categories=[],
        estimated_exposure_usd=exposure,
        summary=f"{article.description or article.title}",
        recommended_actions=[
            f"Monitor {disruption_type.lower()} situation closely over next 48-72 hours.",
            "Contact affected suppliers to assess current inventory and order status.",
            "Alert supply chain leadership if situation escalates.",
        ],
        confidence="Medium" if affected else "Low",
        analysis_method="Rule-based (NewsAPI)",
    )


def build_public_article_analysis_prompt(article: NewsArticle) -> str:
    """
    Build the LLM prompt for live news classification.

    Privacy boundary: this prompt contains only public article text. Supplier
    names, spend, volumes, and uploaded company data are matched locally after
    the LLM returns a generic supply-chain classification.
    """
    article_text = "\n".join(
        part for part in [article.title, article.description, article.content_snippet] if part
    )[:6000]
    return f"""You are a supply-chain risk analyst. Classify this public news article for manufacturing supply-chain disruption risk.

PUBLIC ARTICLE:
Source: {article.source}
Published: {article.published_at.isoformat()}
URL: {article.url}
Text: {article_text}

Return ONLY one JSON object with exactly these fields:
{{
  "is_relevant": true,
  "disruption_type": "Trade/Tariff | Natural Disaster | Geopolitical | Logistics/Transport | Labor/Industrial | Financial/Bankruptcy | Quality/Safety | Cyber/Technology | Pandemic/Health | General Supply Chain",
  "severity": "critical | high | medium | low | info",
  "severity_score": 0,
  "affected_countries": ["country or region"],
  "affected_categories": ["component/material/industry"],
  "summary": "2 sentence operational supply-chain impact summary",
  "recommended_actions": ["action 1", "action 2", "action 3"],
  "confidence": "high | medium | low"
}}

Use lowercase severity. If the article is not supply-chain relevant, set "is_relevant": false and severity "info"."""


def _call_openai_for_json(prompt: str, api_key: str, model: str = "") -> dict:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK is not installed. Run: pip install openai") from exc

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Return strict JSON only. Do not include markdown or commentary.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return _extract_json_payload(response.choices[0].message.content or "")


def _call_anthropic_for_json(prompt: str, api_key: str, model: str = "") -> dict:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("Anthropic SDK is not installed. Run: pip install anthropic") from exc

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
        max_tokens=1200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_json_payload(response.content[0].text)


def classify_article_with_llm(
    article: NewsArticle,
    provider: str,
    api_key: str,
    model: str = "",
) -> dict:
    """Classify a public news article with OpenAI or Claude."""
    provider_key = (provider or "").strip().lower()
    prompt = build_public_article_analysis_prompt(article)
    if provider_key == "openai":
        data = _call_openai_for_json(prompt, api_key, model)
    elif provider_key in {"anthropic", "claude"}:
        data = _call_anthropic_for_json(prompt, api_key, model)
    else:
        raise RuntimeError(f"Unsupported LLM provider: {provider}")

    data["severity"] = normalize_severity(data.get("severity", "medium"))
    data["severity_score"] = int(max(0, min(100, data.get("severity_score", 40))))
    data["affected_countries"] = list(data.get("affected_countries") or [])
    data["affected_categories"] = list(data.get("affected_categories") or [])
    data["recommended_actions"] = list(data.get("recommended_actions") or [])
    data["confidence"] = str(data.get("confidence", "medium")).lower()
    return data


def _match_supplier_impacts_locally(
    supplier_df: pd.DataFrame,
    affected_countries: list[str],
    affected_categories: list[str],
    severity: str,
) -> tuple[list[str], float]:
    """Match generic article risk to uploaded suppliers without sending supplier data to an LLM."""
    if supplier_df is None or len(supplier_df) == 0:
        return [], 0.0

    countries = {str(c).lower() for c in affected_countries}
    categories = {str(c).lower() for c in affected_categories}
    severity_fraction = {
        "critical": 0.60,
        "high": 0.35,
        "medium": 0.20,
        "low": 0.08,
        "info": 0.02,
    }.get(normalize_severity(severity), 0.20)

    affected = []
    exposure = 0.0
    for _, row in supplier_df.iterrows():
        supplier_name = str(row.get("supplier_name", "")).strip()
        country = str(row.get("country", "")).lower()
        category = str(row.get("category", "")).lower()

        country_match = country and any(country in c or c in country for c in countries)
        category_match = category and any(category in c or c in category for c in categories)
        if country_match or category_match:
            affected.append(supplier_name or "Unknown Supplier")
            exposure += float(row.get("annual_spend", 0) or 0) * severity_fraction

    return list(dict.fromkeys(affected)), exposure


def analyze_article_with_llm(
    article: NewsArticle,
    supplier_df: pd.DataFrame,
    provider: str,
    api_key: str,
    model: str = "",
) -> SupplierNewsImpact | None:
    """Analyze real NewsAPI article with LLM classification plus local supplier matching."""
    classification = classify_article_with_llm(article, provider, api_key, model)
    if not classification.get("is_relevant", True):
        return None

    affected_suppliers, exposure = _match_supplier_impacts_locally(
        supplier_df=supplier_df,
        affected_countries=classification["affected_countries"],
        affected_categories=classification["affected_categories"],
        severity=classification["severity"],
    )

    actions = classification["recommended_actions"] or [
        "Review open orders and inventory buffers for potentially exposed suppliers.",
        "Contact matched suppliers for current production and shipment status.",
        "Prepare an alternate-source or expedite plan if severity escalates.",
    ]

    return SupplierNewsImpact(
        article=article,
        disruption_type=classification.get("disruption_type", "General Supply Chain"),
        severity=classification["severity"],
        severity_score=classification["severity_score"],
        affected_suppliers=affected_suppliers,
        affected_countries=classification["affected_countries"],
        affected_categories=classification["affected_categories"],
        estimated_exposure_usd=exposure,
        summary=classification.get("summary", article.description or article.title),
        recommended_actions=actions,
        confidence=classification["confidence"],
        analysis_method=f"Live News + AI ({provider_label(provider)})",
    )


def run_live_news_ai_scan(
    news_api_key: str,
    supplier_df: pd.DataFrame,
    llm_provider: str,
    llm_api_key: str,
    days_back: int = 7,
    max_articles: int = 20,
    custom_query: str = "",
    llm_model: str = "",
) -> tuple[list[SupplierNewsImpact], str]:
    """Fetch real NewsAPI articles and classify them with OpenAI or Claude."""
    articles, error = fetch_news_local(news_api_key, custom_query, days_back, max_articles)
    if error:
        return [], error
    if not articles:
        return [], "NewsAPI returned no articles for the selected query/window."

    impacts = []
    errors = []
    for article in articles[:max_articles]:
        try:
            impact = analyze_article_with_llm(
                article=article,
                supplier_df=supplier_df,
                provider=llm_provider,
                api_key=llm_api_key,
                model=llm_model,
            )
            if impact is not None:
                impacts.append(impact)
        except Exception as exc:
            errors.append(str(exc))
            fallback = analyze_article_rule_based(article, supplier_df)
            fallback.analysis_method = "Rule-based (AI fallback)"
            impacts.append(fallback)

    impacts.sort(key=lambda x: x.severity_score, reverse=True)
    warning = ""
    if errors:
        warning = f"AI analysis failed for {len(errors)} article(s); rule-based fallback was used."
    return impacts, warning


# ─── MAIN ENTRY POINT ─────────────────────────────────────────────

def run_sentinel_scan(
    news_api_key: str,
    supplier_df: pd.DataFrame,
    anthropic_api_key: str = "",
    openai_api_key: str = "",
    llm_provider: str = "anthropic",
    llm_model: str = "",
    days_back: int = 7,
    max_articles: int = 20,
    custom_query: str = "",
    mode: str = "auto",  # "auto", "live_ai", "ai", "demo", "newsapi"
) -> tuple[list[SupplierNewsImpact], str, str]:
    """
    Main entry point for Sentinel scans.

    Returns: (impacts, mode_used, error_message)
    - impacts: list of SupplierNewsImpact
    - mode_used: which mode ran ("AI Briefing", "Demo", "NewsAPI")  
    - error_message: empty if success, explanation if fallback used
    """

    # ── AUTO MODE: pick best available ─────────────────────────────
    if mode == "auto":
        provider = (llm_provider or "anthropic").strip().lower()
        llm_key = openai_api_key if provider == "openai" else anthropic_api_key
        if news_api_key and news_api_key.strip() and llm_key and llm_key.strip():
            mode = "live_ai"
        elif anthropic_api_key and anthropic_api_key.strip():
            mode = "ai"
        elif news_api_key and news_api_key.strip():
            mode = "newsapi"
        else:
            mode = "demo"

    if mode == "live_ai":
        provider = (llm_provider or "anthropic").strip().lower()
        llm_key = openai_api_key if provider == "openai" else anthropic_api_key
        if not news_api_key or not news_api_key.strip():
            impacts = get_demo_impacts(supplier_df)
            return impacts, "Demo (missing NewsAPI key)", "NewsAPI key is required for Live News + AI mode."
        if not llm_key or not llm_key.strip():
            impacts = get_demo_impacts(supplier_df)
            return impacts, "Demo (missing LLM key)", "OpenAI or Anthropic key is required for Live News + AI mode."

        impacts, warning = run_live_news_ai_scan(
            news_api_key=news_api_key,
            supplier_df=supplier_df,
            llm_provider=provider,
            llm_api_key=llm_key,
            days_back=days_back,
            max_articles=max_articles,
            custom_query=custom_query,
            llm_model=llm_model,
        )
        if impacts:
            return impacts, f"Live News + AI ({provider_label(provider)})", warning

        fallback = get_demo_impacts(supplier_df)
        return fallback, "Demo (Live News fallback)", warning or "No relevant live news impacts were returned."

    # ── AI BRIEFING MODE ───────────────────────────────────────────
    if mode == "ai":
        try:
            impacts = run_ai_intelligence_briefing(
                supplier_df=supplier_df,
                anthropic_api_key=anthropic_api_key,
                n_events=max_articles,
                custom_query=custom_query,
            )
            return impacts, "AI Intelligence Briefing", ""
        except Exception as e:
            err = str(e)
            # Fall through to demo if AI fails
            impacts = get_demo_impacts(supplier_df)
            return impacts, "Demo (AI fallback)", (
                f"⚠️ AI briefing failed: {err}. Showing demo events instead."
            )

    # ── NEWSAPI MODE (local only) ──────────────────────────────────
    if mode == "newsapi":
        articles, error = fetch_news_local(news_api_key, custom_query, days_back, max_articles)
        if error:
            # Fall back to demo with explanation
            impacts = get_demo_impacts(supplier_df)
            return impacts, "Demo (NewsAPI fallback)", error
        if not articles:
            impacts = get_demo_impacts(supplier_df)
            return impacts, "Demo (no articles)", "⚠️ NewsAPI returned no articles. Showing demo events."

        impacts = [analyze_article_rule_based(a, supplier_df) for a in articles]
        impacts.sort(key=lambda x: x.severity_score, reverse=True)
        return impacts, "NewsAPI (Live)", ""

    # ── DEMO MODE ──────────────────────────────────────────────────
    impacts = get_demo_impacts(supplier_df)
    return impacts, "Demo Mode", ""
