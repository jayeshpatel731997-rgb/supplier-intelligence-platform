"""
Monte Carlo Financial Impact Simulation
=========================================

Quantifies the financial exposure of supply chain disruptions using
Monte Carlo simulation with:
- Triangular distribution for delay duration
- Log-normal distribution for cost multipliers (fat right tail)
- Outputs: VaR, CVaR, expected loss, full distribution

Cost components:
1. Safety stock holding cost (inventory buffer)
2. Expedite freight cost (emergency shipping)
3. Stockout/lost sales cost (unmet demand)
4. Quality defect cost (disruption-related quality degradation)

References:
- Chopra & Sodhi 2004: Managing risk to avoid supply chain breakdown
- Simchi-Levi et al. 2015: Identifying risks and mitigating disruptions
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
@dataclass
class SupplierCostProfile:
    """Financial parameters for a supplier."""
    annual_spend: float = 1_000_000      # Annual procurement spend
    daily_demand_units: float = 100       # Units consumed per day
    unit_cost: float = 50.0               # Cost per unit
    profit_margin_per_unit: float = 20.0  # Margin per unit (for stockout cost)
    holding_cost_rate: float = 0.25       # Annual holding cost as % of value
    avg_freight_cost_per_order: float = 2_000  # Normal freight cost
    expedite_multiplier: float = 3.0      # Air vs ocean freight ratio
    orders_per_year: float = 52           # Weekly ordering assumed
    avg_order_value: float = 20_000       # Average PO value
    defect_cost_rate: float = 0.05        # Cost of defects as % of order value
    contractual_penalty_per_day: float = 0  # SLA penalty if applicable
@dataclass
class DisruptionProfile:
    """Characterization of a disruption event."""
    min_delay_days: float = 5
    mode_delay_days: float = 14
    max_delay_days: float = 60
    expedite_threshold_days: float = 10  # Delay triggers expediting
    stockout_threshold_days: float = 21  # Delay triggers stockout
    quality_impact_probability: float = 0.15  # P(quality degrades due to disruption)
    duration_months: float = 3           # How long the disruption persists
@dataclass
class MonteCarloResult:
    """Output of Monte Carlo financial simulation."""
    # Summary statistics
    expected_loss: float          # Mean total loss
    median_loss: float            # P50
    p75_loss: float
    p90_loss: float
    p95_loss: float               # Value at Risk (95%)
    p99_loss: float
    var_95: float                 # Same as p95
    cvar_95: float                # Conditional VaR (expected loss in worst 5%)
    max_loss: float

    # Breakdown by cost component
    avg_safety_stock_cost: float
    avg_expedite_cost: float
    avg_stockout_cost: float
    avg_quality_cost: float
    avg_contractual_penalty: float

    # Distribution data
    loss_distribution: np.ndarray  # Full array of simulated losses
    delay_distribution: np.ndarray

    # Risk metrics
    probability_of_loss_over: dict  # P(loss > threshold) for various thresholds
    annualized_expected_loss: float
def triangular_sample(min_val: float, mode_val: float, max_val: float, n: int) -> np.ndarray:
    """Sample from triangular distribution."""
    # Ensure valid parameters
    min_val = max(0, min_val)
    mode_val = max(min_val, mode_val)
    max_val = max(mode_val + 0.01, max_val)
    return np.random.triangular(min_val, mode_val, max_val, n)
def lognormal_sample(mu: float, sigma: float, n: int) -> np.ndarray:
    """Sample from log-normal distribution (for fat-tailed cost multipliers)."""
    return np.random.lognormal(mu, sigma, n)
def run_monte_carlo(
    supplier: SupplierCostProfile,
    disruption: DisruptionProfile,
    n_iterations: int = 5000,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """
    Run Monte Carlo financial impact simulation.

    Simulates n_iterations scenarios, each with:
    1. Random delay duration (triangular distribution)
    2. Random cost multiplier (log-normal for tail risk)
    3. Deterministic cost calculations given the delay

    Returns comprehensive risk metrics including VaR and CVaR.
    """
    if seed is not None:
        np.random.seed(seed)

    n = n_iterations

    # Sample delay durations
    delays = triangular_sample(
        disruption.min_delay_days,
        disruption.mode_delay_days,
        disruption.max_delay_days,
        n
    )

    # Sample cost multipliers (log-normal captures extreme events)
    cost_multipliers = lognormal_sample(0, 0.4, n)

    # ─── COMPUTE COST COMPONENTS ─────────────────────────────────

    # 1. Safety Stock Holding Cost
    # Extra inventory = delay_days × daily_demand
    # Annual holding cost = extra_inventory × unit_cost × holding_rate / 365
    # But disruption lasts for duration_months, so:
    safety_stock_units = delays * supplier.daily_demand_units
    safety_stock_value = safety_stock_units * supplier.unit_cost
    safety_stock_cost = (
        safety_stock_value *
        supplier.holding_cost_rate *
        disruption.duration_months / 12
    )

    # 2. Expedite Freight Cost
    # Triggered when delay > expedite_threshold
    # Cost = number of expedited orders × (expedite_cost - normal_cost)
    needs_expedite = delays > disruption.expedite_threshold_days
    expedite_orders = np.ceil(delays / 7) * needs_expedite  # Weekly orders affected
    expedite_cost = (
        expedite_orders *
        supplier.avg_freight_cost_per_order *
        (supplier.expedite_multiplier - 1)
    )

    # 3. Stockout / Lost Sales Cost
    # Triggered when delay > stockout_threshold
    # Lost sales = (delay - threshold) × daily_demand × margin
    stockout_days = np.maximum(delays - disruption.stockout_threshold_days, 0)
    stockout_cost = (
        stockout_days *
        supplier.daily_demand_units *
        supplier.profit_margin_per_unit
    )

    # 4. Quality Defect Cost
    # Probabilistic: disruptions sometimes cause quality issues
    # (rushed production, alternative materials, etc.)
    quality_events = np.random.random(n) < disruption.quality_impact_probability
    quality_cost = (
        quality_events *
        supplier.avg_order_value *
        supplier.defect_cost_rate *
        cost_multipliers  # Fat tail for quality disasters
    )

    # 5. Contractual Penalties
    penalty_cost = delays * supplier.contractual_penalty_per_day

    # ─── TOTAL IMPACT ────────────────────────────────────────────

    total_losses = (
        safety_stock_cost +
        expedite_cost +
        stockout_cost +
        quality_cost +
        penalty_cost
    )

    # Sort for percentile calculations
    sorted_losses = np.sort(total_losses)

    # ─── RISK METRICS ────────────────────────────────────────────

    p95_idx = int(n * 0.95)
    var_95 = sorted_losses[p95_idx]
    cvar_95 = np.mean(sorted_losses[p95_idx:])

    # P(loss > threshold) for various thresholds
    thresholds = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
    prob_over = {
        t: np.mean(total_losses > t) for t in thresholds
    }

    # Annualized: expected loss × (12 / duration_months) × disruption frequency
    # Assume ~1 significant disruption per year for high-risk suppliers
    annualized = np.mean(total_losses) * (12 / max(disruption.duration_months, 1))

    return MonteCarloResult(
        expected_loss=np.mean(total_losses),
        median_loss=np.median(total_losses),
        p75_loss=np.percentile(total_losses, 75),
        p90_loss=np.percentile(total_losses, 90),
        p95_loss=var_95,
        p99_loss=np.percentile(total_losses, 99),
        var_95=var_95,
        cvar_95=cvar_95,
        max_loss=np.max(total_losses),
        avg_safety_stock_cost=np.mean(safety_stock_cost),
        avg_expedite_cost=np.mean(expedite_cost),
        avg_stockout_cost=np.mean(stockout_cost),
        avg_quality_cost=np.mean(quality_cost),
        avg_contractual_penalty=np.mean(penalty_cost),
        loss_distribution=total_losses,
        delay_distribution=delays,
        probability_of_loss_over=prob_over,
        annualized_expected_loss=annualized,
    )
# ─── SCENARIO COMPARISON ────────────────────────────────────────

def compare_scenarios(
    supplier: SupplierCostProfile,
    scenarios: list[tuple[str, DisruptionProfile]],
    n_iterations: int = 5000,
) -> list[dict]:
    """
    Compare financial impact across multiple disruption scenarios.

    Args:
        supplier: Supplier cost profile
        scenarios: List of (name, DisruptionProfile) tuples
        n_iterations: MC iterations per scenario

    Returns:
        List of dicts with scenario comparisons
    """
    results = []
    for name, disruption in scenarios:
        mc = run_monte_carlo(supplier, disruption, n_iterations)
        results.append({
            "scenario": name,
            "expected_loss": mc.expected_loss,
            "var_95": mc.var_95,
            "cvar_95": mc.cvar_95,
            "p90": mc.p90_loss,
            "max_loss": mc.max_loss,
            "annualized": mc.annualized_expected_loss,
            "breakdown": {
                "safety_stock": mc.avg_safety_stock_cost,
                "expedite": mc.avg_expedite_cost,
                "stockout": mc.avg_stockout_cost,
                "quality": mc.avg_quality_cost,
            },
        })

    return sorted(results, key=lambda x: x["cvar_95"], reverse=True)
# ─── MITIGATION ROI CALCULATOR ──────────────────────────────────

def mitigation_roi(
    baseline_mc: MonteCarloResult,
    mitigated_mc: MonteCarloResult,
    mitigation_cost: float,
    time_horizon_years: float = 3.0,
) -> dict:
    """
    Calculate ROI of a mitigation action.

    Compares expected losses before and after mitigation,
    factoring in the cost of the mitigation itself.
    """
    annual_savings = baseline_mc.annualized_expected_loss - mitigated_mc.annualized_expected_loss
    total_savings = annual_savings * time_horizon_years
    net_benefit = total_savings - mitigation_cost
    roi = net_benefit / mitigation_cost if mitigation_cost > 0 else float('inf')
    payback_years = mitigation_cost / annual_savings if annual_savings > 0 else float('inf')

    return {
        "annual_savings": annual_savings,
        "total_savings_3yr": total_savings,
        "mitigation_cost": mitigation_cost,
        "net_benefit": net_benefit,
        "roi_percentage": roi * 100,
        "payback_years": payback_years,
        "var_reduction": baseline_mc.var_95 - mitigated_mc.var_95,
        "cvar_reduction": baseline_mc.cvar_95 - mitigated_mc.cvar_95,
    }
