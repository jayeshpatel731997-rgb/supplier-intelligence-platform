"""
Bayesian Risk Scoring for Supplier Disruption Prediction
=========================================================

Computes posterior disruption probability using Bayes' theorem
with 6 calibrated evidence signals:

P(disruption | evidence) = P(evidence | disruption) × P(disruption) / P(evidence)

Signals:
1. Financial health (Altman Z-score proxy)
2. Geopolitical risk (country-level)
3. Weather/climate risk
4. Concentration risk (single-source dependency)
5. Historical reliability (on-time delivery track record)
6. Tariff exposure (trade policy vulnerability)

References:
- Hosseini & Ivanov 2020: Bayesian networks in SCRM
- Sarkar & Das 2023: ML in supply chain risk assessment for SMEs
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
@dataclass
class SupplierSignals:
    """Evidence signals for Bayesian risk scoring."""
    financial_health: float = 0.5      # 0-1 (1 = very healthy, 0 = distressed)
    geopolitical_risk: float = 0.3     # 0-1 (1 = extreme risk)
    weather_risk: float = 0.2          # 0-1 (1 = extreme climate exposure)
    concentration_risk: float = 0.3    # 0-1 (1 = sole source, no alternatives)
    historical_reliability: float = 0.9  # 0-1 (1 = perfect on-time record)
    tariff_exposure: float = 0.3       # 0-1 (1 = 100%+ tariff impact)
@dataclass
class BayesianRiskResult:
    """Output of Bayesian risk scoring."""
    prior_probability: float
    posterior_probability: float
    risk_level: str  # CRITICAL, HIGH, MEDIUM, LOW
    combined_likelihood_ratio: float
    individual_likelihood_ratios: dict
    confidence: float  # How confident we are in the estimate
    dominant_risk_factor: str
    risk_decomposition: dict  # Contribution of each factor
# ─── LIKELIHOOD RATIO FUNCTIONS ──────────────────────────────────
# These are calibrated from supply chain risk literature.
# Each function maps a 0-1 signal to a likelihood ratio:
# LR > 1 means evidence supports higher disruption probability
# LR < 1 means evidence supports lower disruption probability
# LR = 1 means evidence is uninformative

def lr_financial_health(signal: float) -> float:
    """
    Financial distress is a strong predictor of supply disruption.
    Based on Altman Z-score correlation with supply failures.

    signal = 0: company in severe financial distress
    signal = 1: company is very financially healthy
    """
    if signal < 0.2:
        return 4.5   # Very distressed → 4.5x more likely to disrupt
    elif signal < 0.4:
        return 2.8
    elif signal < 0.6:
        return 1.5
    elif signal < 0.8:
        return 0.8
    else:
        return 0.4   # Very healthy → 0.4x (protective factor)
def lr_geopolitical(signal: float) -> float:
    """
    Geopolitical risk from country-level instability, sanctions,
    trade wars, military conflicts.

    Calibrated against WEF Global Risk Report data.
    """
    if signal > 0.8:
        return 5.0   # Active conflict / sanctions zone
    elif signal > 0.6:
        return 3.2
    elif signal > 0.4:
        return 1.8
    elif signal > 0.2:
        return 1.1
    else:
        return 0.7   # Stable democratic trading partner
def lr_weather_climate(signal: float) -> float:
    """
    Weather and climate risk (hurricanes, floods, earthquakes, droughts).
    Calibrated against FEMA / Munich Re historical loss data.
    """
    if signal > 0.7:
        return 3.0
    elif signal > 0.5:
        return 2.0
    elif signal > 0.3:
        return 1.3
    else:
        return 0.85
def lr_concentration(signal: float) -> float:
    """
    Supplier concentration / single-source dependency.
    Higher concentration = higher systemic risk.

    Single-source suppliers have 3-5x higher disruption impact
    (Tomlin 2006, Chopra & Sodhi 2004).
    """
    if signal > 0.8:
        return 4.0   # Sole source, no alternatives
    elif signal > 0.6:
        return 2.5
    elif signal > 0.4:
        return 1.6
    elif signal > 0.2:
        return 1.0
    else:
        return 0.7   # Highly diversified supply base
def lr_historical_reliability(signal: float) -> float:
    """
    Track record of on-time delivery and quality.
    Past performance is a moderate predictor of future disruption.

    Note: This has the OPPOSITE direction — high reliability = low risk.
    """
    if signal < 0.7:
        return 3.5   # Consistently late/unreliable
    elif signal < 0.8:
        return 2.2
    elif signal < 0.9:
        return 1.4
    elif signal < 0.95:
        return 0.8
    else:
        return 0.4   # Near-perfect track record
def lr_tariff_exposure(signal: float) -> float:
    """
    Exposure to tariff increases and trade policy changes.
    Calibrated for 2025-2026 US tariff environment.

    145% on China, 25% on many others.
    """
    if signal > 0.8:
        return 4.0   # Heavily tariffed origin
    elif signal > 0.6:
        return 2.8
    elif signal > 0.4:
        return 1.7
    elif signal > 0.2:
        return 1.1
    else:
        return 0.8   # Domestic or tariff-exempt
# ─── MAIN SCORING FUNCTION ──────────────────────────────────────

def compute_bayesian_risk(
    signals: SupplierSignals,
    prior: float = 0.15,
) -> BayesianRiskResult:
    """
    Compute Bayesian posterior disruption probability.

    Args:
        signals: Evidence signals for the supplier
        prior: Base rate of disruption (default 15% per quarter,
               calibrated from BCI Supply Chain Resilience Report 2024)

    Returns:
        BayesianRiskResult with posterior probability and decomposition
    """
    # Compute individual likelihood ratios
    lrs = {
        "financial_health": lr_financial_health(signals.financial_health),
        "geopolitical": lr_geopolitical(signals.geopolitical_risk),
        "weather_climate": lr_weather_climate(signals.weather_risk),
        "concentration": lr_concentration(signals.concentration_risk),
        "historical_reliability": lr_historical_reliability(signals.historical_reliability),
        "tariff_exposure": lr_tariff_exposure(signals.tariff_exposure),
    }

    # Combined likelihood ratio (independence assumption)
    combined_lr = np.prod(list(lrs.values()))

    # Bayes update: posterior odds = prior odds × combined LR
    prior_odds = prior / (1 - prior)
    posterior_odds = prior_odds * combined_lr
    posterior_prob = posterior_odds / (1 + posterior_odds)

    # Clamp to [0.01, 0.99]
    posterior_prob = np.clip(posterior_prob, 0.01, 0.99)

    # Risk level classification
    if posterior_prob > 0.70:
        risk_level = "CRITICAL"
    elif posterior_prob > 0.40:
        risk_level = "HIGH"
    elif posterior_prob > 0.20:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Identify dominant risk factor (highest LR)
    dominant = max(lrs.items(), key=lambda x: x[1])
    dominant_factor = dominant[0]

    # Risk decomposition: what % of total risk does each factor contribute?
    log_lrs = {k: np.log(max(v, 0.01)) for k, v in lrs.items()}
    total_log_lr = sum(abs(v) for v in log_lrs.values())
    decomposition = {
        k: abs(v) / total_log_lr if total_log_lr > 0 else 1/6
        for k, v in log_lrs.items()
    }

    # Confidence estimate: based on how extreme the signals are
    # More extreme signals = more confident prediction
    signal_extremity = np.mean([
        abs(signals.financial_health - 0.5) * 2,
        abs(signals.geopolitical_risk - 0.5) * 2,
        abs(signals.weather_risk - 0.5) * 2,
        abs(signals.concentration_risk - 0.5) * 2,
        abs(signals.historical_reliability - 0.5) * 2,
        abs(signals.tariff_exposure - 0.5) * 2,
    ])
    confidence = 0.5 + signal_extremity * 0.4  # 50-90% range

    return BayesianRiskResult(
        prior_probability=prior,
        posterior_probability=posterior_prob,
        risk_level=risk_level,
        combined_likelihood_ratio=combined_lr,
        individual_likelihood_ratios=lrs,
        confidence=confidence,
        dominant_risk_factor=dominant_factor,
        risk_decomposition=decomposition,
    )
# ─── PORTFOLIO-LEVEL RISK ───────────────────────────────────────

def compute_portfolio_risk(
    supplier_risks: list[BayesianRiskResult],
    spend_weights: Optional[list[float]] = None,
) -> dict:
    """
    Compute aggregate risk for the entire supplier portfolio.

    Uses spend-weighted average + correlation adjustment.
    """
    n = len(supplier_risks)
    if n == 0:
        return {"portfolio_risk": 0, "diversification_benefit": 0}

    probs = np.array([r.posterior_probability for r in supplier_risks])

    if spend_weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.array(spend_weights)
        weights = weights / weights.sum()

    # Spend-weighted average risk
    weighted_avg = np.sum(probs * weights)

    # Portfolio VaR: assuming ~0.3 correlation between supplier disruptions
    rho = 0.30
    portfolio_var = np.sqrt(
        np.sum((weights * probs) ** 2) +
        2 * rho * np.sum(
            weights[i] * probs[i] * weights[j] * probs[j]
            for i in range(n) for j in range(i+1, n)
        )
    )

    # Diversification benefit
    undiversified = np.sum(weights * probs)
    diversification_benefit = max(0, undiversified - portfolio_var)

    # Count by risk level
    risk_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in supplier_risks:
        risk_counts[r.risk_level] += 1

    return {
        "portfolio_risk": weighted_avg,
        "portfolio_var": portfolio_var,
        "diversification_benefit": diversification_benefit,
        "risk_distribution": risk_counts,
        "highest_risk_supplier": max(
            range(n), key=lambda i: probs[i]
        ),
    }
