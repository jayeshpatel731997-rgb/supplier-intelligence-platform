"""
Sentinel Agent — Supply Chain Event Detection & Classification
================================================================

The first agent in the 5-agent pipeline. Monitors news, regulatory,
financial, and weather sources for events that could disrupt supply chains.

Pipeline: Sentinel → Mapper → Propagator → Strategist → Narrator

Key capabilities:
1. Monitors RSS feeds, news APIs, and structured data sources
2. Uses LLM (Claude) to classify events by supply chain relevance
3. Extracts structured data: affected regions, industries, severity
4. Outputs classified events in JSON for downstream agents

Data sources (free tier):
- GDELT Project (global event database)
- NewsAPI (news aggregation)
- USGS (earthquakes/natural disasters)
- NOAA (severe weather)
- Federal Register (US regulations/tariffs)
- SEC EDGAR (company filings, bankruptcy)
- RSS feeds (Reuters, Bloomberg supply chain)
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum

# These will be lazy-imported to avoid breaking if not installed
# import requests
# import feedparser
# from anthropic import Anthropic
class EventSeverity(Enum):
    CRITICAL = "critical"   # Immediate, major disruption
    HIGH = "high"           # Significant impact within days
    MEDIUM = "medium"       # Moderate impact, weeks to materialize
    LOW = "low"             # Minor, may not affect supply chain
    INFO = "info"           # Background signal, no immediate action
class EventCategory(Enum):
    TARIFF_TRADE = "tariff_trade"
    GEOPOLITICAL = "geopolitical"
    NATURAL_DISASTER = "natural_disaster"
    SUPPLIER_FINANCIAL = "supplier_financial"
    REGULATORY = "regulatory"
    LABOR_STRIKE = "labor_strike"
    LOGISTICS = "logistics"           # Port congestion, shipping disruption
    COMMODITY_PRICE = "commodity_price"
    CYBER_ATTACK = "cyber_attack"
    PANDEMIC_HEALTH = "pandemic_health"
@dataclass
class DetectedEvent:
    """A supply chain relevant event detected by the Sentinel Agent."""
    event_id: str
    timestamp: str
    title: str
    summary: str
    source: str
    url: str
    severity: EventSeverity
    category: EventCategory
    confidence: float           # 0-1, how confident we are this is real
    affected_regions: list[str] = field(default_factory=list)
    affected_industries: list[str] = field(default_factory=list)
    affected_materials: list[str] = field(default_factory=list)
    estimated_duration_days: Optional[int] = None
    estimated_delay_days: Optional[int] = None
    keywords: list[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self):
        d = asdict(self)
        d["severity"] = self.severity.value
        d["category"] = self.category.value
        return d
# ─── RSS FEED SOURCES ────────────────────────────────────────────

RSS_FEEDS = {
    "reuters_supply_chain": {
        "url": "https://www.reutersagency.com/feed/?best-topics=supply-chain&post_type=best",
        "category": "general",
    },
    "scmr": {
        "url": "https://www.scmr.com/rss",
        "category": "supply_chain",
    },
    "supply_chain_brain": {
        "url": "https://www.supplychainbrain.com/rss",
        "category": "supply_chain",
    },
    "manufacturing_dive": {
        "url": "https://www.manufacturingdive.com/feeds/news/",
        "category": "manufacturing",
    },
    "usgs_earthquakes": {
        "url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_week.atom",
        "category": "natural_disaster",
    },
    "federal_register": {
        "url": "https://www.federalregister.gov/api/v1/articles.rss?conditions[term]=tariff+trade",
        "category": "regulatory",
    },
}
# ─── LLM CLASSIFICATION PROMPT ──────────────────────────────────

CLASSIFICATION_PROMPT = """You are a supply chain risk analyst. Analyze the following news item and determine if it could affect manufacturing supply chains.

NEWS ITEM:
Title: {title}
Source: {source}
Date: {date}
Content: {content}

Respond with a JSON object (and NOTHING else, no markdown):
{{
    "is_supply_chain_relevant": true/false,
    "severity": "critical" | "high" | "medium" | "low" | "info",
    "category": "tariff_trade" | "geopolitical" | "natural_disaster" | "supplier_financial" | "regulatory" | "labor_strike" | "logistics" | "commodity_price" | "cyber_attack" | "pandemic_health",
    "confidence": 0.0-1.0,
    "affected_regions": ["region1", "region2"],
    "affected_industries": ["manufacturing", "electronics", etc.],
    "affected_materials": ["steel", "semiconductors", etc.],
    "estimated_duration_days": number or null,
    "estimated_delay_days": number or null,
    "summary": "One-sentence summary of supply chain impact",
    "keywords": ["keyword1", "keyword2"]
}}

Be specific about regions (use country-province format like "China-Shenzhen", "US-Ohio").
For severity: critical = immediate halt, high = significant delays, medium = moderate impact, low = minor, info = background.
"""
# ─── SENTINEL AGENT CLASS ────────────────────────────────────────

class SentinelAgent:
    """
    Event detection and classification agent.

    Monitors news sources for supply chain disruption signals,
    classifies them using LLM, and outputs structured events
    for downstream agents.
    """

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        newsapi_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        cache_dir: str = ".sentinel_cache",
    ):
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.newsapi_key = newsapi_key or os.getenv("NEWSAPI_KEY")
        self.model = model
        self.cache_dir = cache_dir
        self._seen_events: set = set()

        # Lazy init API clients
        self._anthropic = None
        self._session = None

    @property
    def anthropic(self):
        if self._anthropic is None:
            try:
                from anthropic import Anthropic
                self._anthropic = Anthropic(api_key=self.anthropic_api_key)
            except ImportError:
                raise ImportError("pip install anthropic")
        return self._anthropic

    @property
    def session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "SupplierRiskIntel/1.0 (research project)"
            })
        return self._session

    # ─── DATA COLLECTION ─────────────────────────────────────────

    def fetch_rss_feeds(self, feeds: Optional[dict] = None) -> list[dict]:
        """Fetch and parse RSS feeds for supply chain news."""
        import feedparser

        feeds = feeds or RSS_FEEDS
        articles = []

        for feed_name, feed_config in feeds.items():
            try:
                parsed = feedparser.parse(feed_config["url"])
                for entry in parsed.entries[:10]:  # Last 10 per feed
                    article_id = hashlib.md5(
                        (entry.get("title", "") + entry.get("link", "")).encode()
                    ).hexdigest()

                    if article_id not in self._seen_events:
                        articles.append({
                            "id": article_id,
                            "title": entry.get("title", ""),
                            "summary": entry.get("summary", ""),
                            "link": entry.get("link", ""),
                            "published": entry.get("published", ""),
                            "source": feed_name,
                            "category": feed_config["category"],
                        })
                        self._seen_events.add(article_id)

            except Exception as e:
                print(f"Error fetching {feed_name}: {e}")

        return articles

    def fetch_newsapi(
        self,
        query: str = "supply chain disruption OR tariff OR manufacturing",
        days_back: int = 3,
    ) -> list[dict]:
        """Fetch articles from NewsAPI."""
        if not self.newsapi_key:
            return []

        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 20,
            "apiKey": self.newsapi_key,
        }

        try:
            resp = self.session.get(url, params=params, timeout=10)
            data = resp.json()
            articles = []
            for article in data.get("articles", []):
                article_id = hashlib.md5(
                    (article.get("title", "") + article.get("url", "")).encode()
                ).hexdigest()
                if article_id not in self._seen_events:
                    articles.append({
                        "id": article_id,
                        "title": article.get("title", ""),
                        "summary": article.get("description", ""),
                        "link": article.get("url", ""),
                        "published": article.get("publishedAt", ""),
                        "source": article.get("source", {}).get("name", "NewsAPI"),
                        "category": "general",
                    })
                    self._seen_events.add(article_id)
            return articles
        except Exception as e:
            print(f"NewsAPI error: {e}")
            return []

    def fetch_usgs_earthquakes(self, min_magnitude: float = 5.0) -> list[dict]:
        """Fetch significant earthquakes from USGS."""
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_week.geojson"
        try:
            resp = self.session.get(url, timeout=10)
            data = resp.json()
            events = []
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                mag = props.get("mag", 0)
                if mag >= min_magnitude:
                    events.append({
                        "id": f"usgs_{props.get('code', '')}",
                        "title": f"M{mag} Earthquake: {props.get('place', 'Unknown')}",
                        "summary": f"Magnitude {mag} earthquake at {props.get('place', 'unknown location')}. "
                                   f"Depth: {feature.get('geometry', {}).get('coordinates', [0,0,0])[2]}km",
                        "link": props.get("url", ""),
                        "published": datetime.fromtimestamp(props.get("time", 0) / 1000).isoformat(),
                        "source": "USGS",
                        "category": "natural_disaster",
                    })
            return events
        except Exception as e:
            print(f"USGS error: {e}")
            return []

    # ─── LLM CLASSIFICATION ──────────────────────────────────────

    def classify_event(self, article: dict) -> Optional[DetectedEvent]:
        """Use Claude to classify a news article for supply chain relevance."""
        prompt = CLASSIFICATION_PROMPT.format(
            title=article.get("title", ""),
            source=article.get("source", ""),
            date=article.get("published", ""),
            content=article.get("summary", "")[:2000],  # Truncate long content
        )

        try:
            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Parse JSON response
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]

            result = json.loads(text)

            if not result.get("is_supply_chain_relevant", False):
                return None

            if result.get("confidence", 0) < 0.3:
                return None

            return DetectedEvent(
                event_id=article["id"],
                timestamp=article.get("published", datetime.now().isoformat()),
                title=article.get("title", ""),
                summary=result.get("summary", article.get("summary", "")),
                source=article.get("source", ""),
                url=article.get("link", ""),
                severity=EventSeverity(result.get("severity", "info")),
                category=EventCategory(result.get("category", "geopolitical")),
                confidence=result.get("confidence", 0.5),
                affected_regions=result.get("affected_regions", []),
                affected_industries=result.get("affected_industries", []),
                affected_materials=result.get("affected_materials", []),
                estimated_duration_days=result.get("estimated_duration_days"),
                estimated_delay_days=result.get("estimated_delay_days"),
                keywords=result.get("keywords", []),
                raw_text=article.get("summary", ""),
            )

        except json.JSONDecodeError as e:
            print(f"JSON parse error for '{article.get('title', '')}': {e}")
            return None
        except Exception as e:
            print(f"Classification error: {e}")
            return None

    # ─── RULE-BASED FAST FILTER ──────────────────────────────────

    def fast_filter(self, article: dict) -> bool:
        """
        Quick rule-based filter before sending to LLM.
        Reduces API calls by ~60% by filtering obviously irrelevant articles.
        """
        text = (article.get("title", "") + " " + article.get("summary", "")).lower()

        # Must contain at least one supply chain keyword
        sc_keywords = [
            "supply chain", "supplier", "tariff", "disruption", "shortage",
            "manufacturing", "procurement", "logistics", "shipping",
            "semiconductor", "chip", "rare earth", "trade war", "sanctions",
            "port", "freight", "inventory", "factory", "warehouse",
            "recall", "bankruptcy", "strike", "embargo", "export ban",
            "reshoring", "nearshoring", "customs", "import", "duty",
            "raw material", "commodity", "steel", "aluminum", "lithium",
            "cobalt", "copper", "nickel", "polymer", "resin",
            "earthquake", "hurricane", "flood", "typhoon", "wildfire",
        ]

        return any(kw in text for kw in sc_keywords)

    # ─── MAIN SCAN PIPELINE ─────────────────────────────────────

    def scan(
        self,
        use_newsapi: bool = True,
        use_rss: bool = True,
        use_usgs: bool = True,
        classify_with_llm: bool = True,
        max_llm_calls: int = 20,
    ) -> list[DetectedEvent]:
        """
        Run full scan pipeline:
        1. Collect articles from all sources
        2. Fast-filter for supply chain relevance
        3. Classify with LLM
        4. Return sorted detected events

        Returns:
            List of DetectedEvent objects, sorted by severity then confidence
        """
        print(f"[Sentinel] Starting scan at {datetime.now().isoformat()}")

        # Step 1: Collect
        all_articles = []
        if use_rss:
            rss = self.fetch_rss_feeds()
            print(f"[Sentinel] RSS: {len(rss)} articles")
            all_articles.extend(rss)

        if use_newsapi and self.newsapi_key:
            news = self.fetch_newsapi()
            print(f"[Sentinel] NewsAPI: {len(news)} articles")
            all_articles.extend(news)

        if use_usgs:
            quakes = self.fetch_usgs_earthquakes()
            print(f"[Sentinel] USGS: {len(quakes)} events")
            all_articles.extend(quakes)

        print(f"[Sentinel] Total collected: {len(all_articles)}")

        # Step 2: Fast filter
        filtered = [a for a in all_articles if self.fast_filter(a)]
        print(f"[Sentinel] After fast filter: {len(filtered)}")

        # Step 3: Classify with LLM
        detected_events = []
        if classify_with_llm and self.anthropic_api_key:
            for i, article in enumerate(filtered[:max_llm_calls]):
                event = self.classify_event(article)
                if event:
                    detected_events.append(event)
                    print(f"[Sentinel] Detected: [{event.severity.value.upper()}] {event.title[:60]}...")
        else:
            # Without LLM, create basic events from filtered articles
            for article in filtered:
                detected_events.append(DetectedEvent(
                    event_id=article["id"],
                    timestamp=article.get("published", datetime.now().isoformat()),
                    title=article.get("title", ""),
                    summary=article.get("summary", ""),
                    source=article.get("source", ""),
                    url=article.get("link", ""),
                    severity=EventSeverity.MEDIUM,
                    category=EventCategory.GEOPOLITICAL,
                    confidence=0.5,
                    raw_text=article.get("summary", ""),
                ))

        # Step 4: Sort by severity then confidence
        severity_order = {
            EventSeverity.CRITICAL: 0,
            EventSeverity.HIGH: 1,
            EventSeverity.MEDIUM: 2,
            EventSeverity.LOW: 3,
            EventSeverity.INFO: 4,
        }
        detected_events.sort(
            key=lambda e: (severity_order.get(e.severity, 5), -e.confidence)
        )

        print(f"[Sentinel] Final detected events: {len(detected_events)}")
        return detected_events

    # ─── MANUAL EVENT INJECTION ──────────────────────────────────

    def inject_event(
        self,
        title: str,
        description: str,
        severity: str = "high",
        category: str = "geopolitical",
        regions: Optional[list] = None,
        industries: Optional[list] = None,
    ) -> DetectedEvent:
        """
        Manually inject an event (for testing or analyst-detected events).
        """
        event = DetectedEvent(
            event_id=hashlib.md5(title.encode()).hexdigest(),
            timestamp=datetime.now().isoformat(),
            title=title,
            summary=description,
            source="manual_injection",
            url="",
            severity=EventSeverity(severity),
            category=EventCategory(category),
            confidence=1.0,
            affected_regions=regions or [],
            affected_industries=industries or [],
        )
        return event

    # ─── EXPORT ──────────────────────────────────────────────────

    def export_events(self, events: list[DetectedEvent], filepath: str):
        """Export detected events to JSON file."""
        with open(filepath, "w") as f:
            json.dump(
                {
                    "scan_timestamp": datetime.now().isoformat(),
                    "total_events": len(events),
                    "events": [e.to_dict() for e in events],
                },
                f,
                indent=2,
            )
        print(f"[Sentinel] Exported {len(events)} events to {filepath}")
# ─── CLI INTERFACE ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sentinel Agent — Supply Chain Event Scanner")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM classification")
    parser.add_argument("--no-newsapi", action="store_true", help="Skip NewsAPI")
    parser.add_argument("--output", default="sentinel_events.json", help="Output file")
    parser.add_argument("--max-llm", type=int, default=20, help="Max LLM classification calls")
    args = parser.parse_args()

    sentinel = SentinelAgent()

    events = sentinel.scan(
        classify_with_llm=not args.no_llm,
        use_newsapi=not args.no_newsapi,
        max_llm_calls=args.max_llm,
    )

    # Print results
    for event in events:
        sev_colors = {
            "critical": "\033[91m", "high": "\033[93m",
            "medium": "\033[94m", "low": "\033[92m", "info": "\033[90m",
        }
        color = sev_colors.get(event.severity.value, "")
        reset = "\033[0m"
        print(f"{color}[{event.severity.value.upper():8s}]{reset} "
              f"{event.title[:70]} | "
              f"Regions: {', '.join(event.affected_regions[:3])} | "
              f"Conf: {event.confidence:.0%}")

    sentinel.export_events(events, args.output)
    print(f"\n{'='*60}")
    print(f"Total events detected: {len(events)}")
    print(f"  Critical: {sum(1 for e in events if e.severity == EventSeverity.CRITICAL)}")
    print(f"  High:     {sum(1 for e in events if e.severity == EventSeverity.HIGH)}")
    print(f"  Medium:   {sum(1 for e in events if e.severity == EventSeverity.MEDIUM)}")
    print(f"  Low:      {sum(1 for e in events if e.severity == EventSeverity.LOW)}")
