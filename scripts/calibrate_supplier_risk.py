"""Offline calibration review for supplier risk scores.

This script produces review metrics only. It does not claim predictive accuracy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class HistoricalOutcome:
    supplier_id: str
    event_type: str
    event_date: datetime
    severity: float
    notes: str = ""
    source: str = ""


@dataclass(slots=True)
class RiskScoreSnapshot:
    supplier_id: str
    scored_at: datetime
    risk_score: float
    risk_level: str


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _load_json(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8") as handle:
        return list(json.load(handle))


def load_outcomes(path: str | Path) -> list[HistoricalOutcome]:
    return [
        HistoricalOutcome(
            supplier_id=str(item["supplier_id"]),
            event_type=str(item["event_type"]),
            event_date=_parse_dt(str(item["event_date"])),
            severity=float(item.get("severity", 0)),
            notes=str(item.get("notes", "")),
            source=str(item.get("source", "")),
        )
        for item in _load_json(path)
    ]


def load_scores(path: str | Path) -> list[RiskScoreSnapshot]:
    return [
        RiskScoreSnapshot(
            supplier_id=str(item["supplier_id"]),
            scored_at=_parse_dt(str(item["scored_at"])),
            risk_score=float(item.get("risk_score", 0)),
            risk_level=str(item.get("risk_level", "")).lower(),
        )
        for item in _load_json(path)
    ]


def _prior_score(outcome: HistoricalOutcome, scores: list[RiskScoreSnapshot]) -> RiskScoreSnapshot | None:
    candidates = [
        score
        for score in scores
        if score.supplier_id == outcome.supplier_id and score.scored_at <= outcome.event_date
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.scored_at, reverse=True)[0]


def evaluate_calibration(
    outcomes: list[HistoricalOutcome],
    scores: list[RiskScoreSnapshot],
) -> dict[str, Any]:
    matched = []
    false_negative_review = []
    high_without_outcome_review = []
    for outcome in outcomes:
        score = _prior_score(outcome, scores)
        if score is None:
            continue
        item = {
            "supplier_id": outcome.supplier_id,
            "event_type": outcome.event_type,
            "event_date": outcome.event_date.isoformat(),
            "severity": outcome.severity,
            "prior_risk_score": score.risk_score,
            "prior_risk_level": score.risk_level,
        }
        matched.append(item)
        if outcome.severity >= 0.6 and score.risk_score < 50:
            false_negative_review.append(item)
    outcome_supplier_ids = {outcome.supplier_id for outcome in outcomes}
    for score in scores:
        if score.risk_score >= 70 and score.supplier_id not in outcome_supplier_ids:
            high_without_outcome_review.append(
                {
                    "supplier_id": score.supplier_id,
                    "scored_at": score.scored_at.isoformat(),
                    "risk_score": score.risk_score,
                    "risk_level": score.risk_level,
                }
            )
    coverage = len(matched) / len(outcomes) if outcomes else 0.0
    return {
        "accuracy_claim": "not_claimed",
        "coverage": round(coverage, 3),
        "historical_outcomes": len(outcomes),
        "score_snapshots": len(scores),
        "matched_examples": len(matched),
        "examples": matched[:20],
        "review_lists": {
            "false_negative_review": false_negative_review,
            "high_risk_without_recorded_outcome_review": high_without_outcome_review[:20],
        },
        "limitations": [
            "This is an offline review helper, not a validated predictive-accuracy claim.",
            "Coverage depends on available historical outcomes and prior score snapshots.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Review supplier risk calibration coverage.")
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--scores", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_calibration(load_outcomes(args.outcomes), load_scores(args.scores)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

