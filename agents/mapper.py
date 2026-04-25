"""
Mapper Agent — Event-to-Node Mapping
======================================

The second agent in the 5-agent pipeline. Takes events detected by Sentinel
and identifies which suppliers in the network are affected, with a match
score and reasoning trail.

Pipeline: Sentinel → [Mapper] → Propagator → Strategist → Narrator

First-principles approach
-------------------------
A real-world event (a tariff, an earthquake, a strike) only matters to *us*
if it touches *our* suppliers. That requires three signals — geographic
proximity, keyword/material overlap, and category-specific exposure flags
already on each supplier node. We score each (event, supplier) pair across
those signals, threshold, and rank.

The match score is intentionally a sum (not a probability) — it is a
relevance ranking, not a calibrated forecast. Calibrated risk lives
downstream in Bayesian and Monte Carlo modules.
"""

from dataclasses import dataclass, field

from agents.sentinel import DetectedEvent


@dataclass
class AffectedNode:
    """A single supplier flagged as potentially affected by an event."""
    node_id: str
    name: str
    match_score: float          # [0.0, 1.0] — saturation of relevance signals
    match_reasons: list[str] = field(default_factory=list)


@dataclass
class EventNodeMapping:
    """Mapping between one event and the suppliers it touches."""
    event: dict                 # DetectedEvent.to_dict() form (JSON-friendly)
    affected_nodes: list[AffectedNode] = field(default_factory=list)
    total_affected: int = 0

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "affected_nodes": [
                {
                    "node_id": n.node_id,
                    "name": n.name,
                    "match_score": n.match_score,
                    "match_reasons": n.match_reasons,
                }
                for n in self.affected_nodes
            ],
            "total_affected": self.total_affected,
        }


class MapperAgent:
    """
    Maps detected events to affected supplier nodes in the network.

    Scoring signals (additive, capped at 1.0):
      * Region match (exact substring)        → +0.5
      * Country-level match (region prefix)   → +0.3
      * Keyword in node name/region            → +0.3
      * Tariff event × high tariff_exposure   → +0.4
      * Disaster event × high weather_risk    → +0.3

    A supplier is included in the mapping if its score exceeds
    `min_match_score` (default 0.2 — i.e., at least one weak signal).
    """

    def __init__(self, network_data: dict, min_match_score: float = 0.2):
        self.network_data = network_data
        self.min_match_score = min_match_score

    # ─── PER-EVENT SCORING ───────────────────────────────────────────

    def _score_node(self, event: DetectedEvent, node: dict) -> tuple[float, list[str]]:
        """Score a single (event, node) pair. Returns (score, reasons)."""
        score = 0.0
        reasons: list[str] = []

        # Region matching (substring, both directions)
        node_region = node.get("region", "").lower()
        for region in event.affected_regions:
            region_lower = region.lower()
            if region_lower in node_region or node_region in region_lower:
                score += 0.5
                reasons.append(f"Region: {region}")
            elif region_lower.split("-")[0] in node_region.split("-")[0]:
                score += 0.3
                reasons.append(f"Country: {region.split('-')[0]}")

        # Keyword matching against node name + region
        node_text = (node.get("name", "") + " " + node.get("region", "")).lower()
        for keyword in event.keywords:
            if keyword.lower() in node_text:
                score += 0.3
                reasons.append(f"Keyword: {keyword}")

        # Category-specific exposure
        category_value = event.category.value
        if category_value == "tariff_trade" and node.get("tariff_exposure", 0) > 0.5:
            score += 0.4
            reasons.append("High tariff exposure")
        if category_value == "natural_disaster" and node.get("weather_risk", 0) > 0.5:
            score += 0.3
            reasons.append("High weather risk")

        return score, reasons

    # ─── BATCH MAPPING ───────────────────────────────────────────────

    def map_events(
        self,
        events: list[DetectedEvent],
        verbose: bool = True,
    ) -> list[EventNodeMapping]:
        """
        Map a list of detected events to affected suppliers.

        Only nodes with type='supplier' are considered (focal firm,
        distributor, and customer nodes are excluded by design).
        """
        mappings: list[EventNodeMapping] = []

        for event in events:
            affected: list[AffectedNode] = []

            for node in self.network_data["nodes"]:
                if node.get("type") != "supplier":
                    continue

                score, reasons = self._score_node(event, node)
                if score > self.min_match_score:
                    affected.append(AffectedNode(
                        node_id=node["id"],
                        name=node["name"],
                        match_score=min(score, 1.0),
                        match_reasons=reasons,
                    ))

            affected.sort(key=lambda a: -a.match_score)

            mapping = EventNodeMapping(
                event=event.to_dict(),
                affected_nodes=affected,
                total_affected=len(affected),
            )
            mappings.append(mapping)

            if verbose:
                tag = f"[{event.severity.value.upper()}]"
                if affected:
                    print(f"  {tag} {event.title[:50]}... → "
                          f"{len(affected)} nodes affected")
                else:
                    print(f"  {tag} {event.title[:50]}... → No network match")

        return mappings
