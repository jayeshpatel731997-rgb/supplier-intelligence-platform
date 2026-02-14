"""
Supplier Intelligence Platform
================================
Unified system for supplier risk assessment and performance decision-making.

Modules:
- Risk Dashboard: Bayesian risk scoring across supplier network
- Network Analysis: Graph centrality and single points of failure
- Scenario Engine: SIR cascade disruption simulation
- Financial Impact: Monte Carlo VaR/CVaR simulation
- Performance Scorecard: Weighted multi-criteria supplier evaluation
- Decision Intelligence: Financial trade-off analysis for supplier selection
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path

# Import risk models
from models.sir_propagation import (
    build_networkx_graph, run_monte_carlo_sir, PropagationParams, sensitivity_analysis
)
from models.bayesian_risk import (
    compute_bayesian_risk, compute_portfolio_risk, SupplierSignals
)
from models.monte_carlo import (
    run_monte_carlo, compare_scenarios, SupplierCostProfile, DisruptionProfile, mitigation_roi
)
from models.graph_metrics import (
    compute_centrality_metrics, identify_single_points_of_failure,
    compute_network_resilience_score
)

# ─── PAGE CONFIG ─────────────────────────────────────────────────

st.set_page_config(
    page_title="Supplier Intelligence Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background-color: #0a0e17;
    }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        color: #e2e8f0 !important;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(99,102,241,0.02));
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .risk-critical { color: #ef4444; }
    .risk-high { color: #f59e0b; }
    .risk-medium { color: #3b82f6; }
    .risk-low { color: #10b981; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 8px 16px;
        color: #94a3b8;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(99,102,241,0.15) !important;
        color: #a5b4fc !important;
    }

    div[data-testid="stSidebar"] {
        background-color: #0d1117;
    }

    .agent-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

@st.cache_data
def load_network():
    with open("data/sample_network.json") as f:
        return json.load(f)

@st.cache_data
def compute_all_risks(network_data):
    """Compute Bayesian risk for all suppliers."""
    results = []
    for node in network_data["nodes"]:
        if node.get("type") == "supplier":
            signals = SupplierSignals(
                financial_health=node.get("financial_health", 0.5),
                geopolitical_risk=node.get("geopolitical_risk", 0.3),
                weather_risk=node.get("weather_risk", 0.2),
                concentration_risk=node.get("concentration_risk", 0.3),
                historical_reliability=node.get("on_time_rate", 0.9),
                tariff_exposure=node.get("tariff_exposure", 0.3),
            )
            risk = compute_bayesian_risk(signals)
            results.append({
                "id": node["id"],
                "name": node["name"],
                "tier": node["tier"],
                "region": node["region"],
                "spend": node.get("spend", 0),
                "on_time": node.get("on_time_rate", 0),
                "risk_prob": risk.posterior_probability,
                "risk_level": risk.risk_level,
                "dominant_factor": risk.dominant_risk_factor,
                "combined_lr": risk.combined_likelihood_ratio,
                "confidence": risk.confidence,
                "lr_financial": risk.individual_likelihood_ratios["financial_health"],
                "lr_geopolitical": risk.individual_likelihood_ratios["geopolitical"],
                "lr_weather": risk.individual_likelihood_ratios["weather_climate"],
                "lr_concentration": risk.individual_likelihood_ratios["concentration"],
                "lr_historical": risk.individual_likelihood_ratios["historical_reliability"],
                "lr_tariff": risk.individual_likelihood_ratios["tariff_exposure"],
            })
    return pd.DataFrame(results).sort_values("risk_prob", ascending=False)

@st.cache_data
def get_graph_metrics(network_data):
    G = build_networkx_graph(network_data)
    centralities = compute_centrality_metrics(G)
    resilience = compute_network_resilience_score(G)
    spofs = identify_single_points_of_failure(G)
    return centralities, resilience, spofs


# ═══════════════════════════════════════════════════════════════════
# DECISION INTELLIGENCE: SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════

def get_scorecard_data() -> pd.DataFrame:
    """
    Sample supplier data for decision intelligence scorecard.
    In production, this would connect to ERP/procurement systems.
    """
    data = {
        "Supplier": [
            "Apex Manufacturing (Mexico)",
            "Precision Parts India",
            "Vietnam Tech Components",
            "Taiwan Quality Corp",
            "Midwest US Machining"
        ],
        "Country": ["Mexico", "India", "Vietnam", "Taiwan", "USA"],
        "Unit_Cost": [12.50, 8.75, 9.20, 15.00, 22.00],
        "Quality_Score": [82, 71, 68, 94, 97],
        "On_Time_Delivery_Pct": [88.0, 72.0, 75.0, 95.0, 98.0],
        "Defect_Rate_Pct": [1.8, 4.2, 3.5, 0.5, 0.3],
        "Risk_Score": [35, 55, 60, 25, 15],
        "Certifications": [
            "ISO 9001, IATF 16949",
            "ISO 9001",
            "ISO 9001",
            "ISO 9001, AS9100, ISO 14001",
            "ISO 9001, AS9100, NADCAP"
        ],
        "Years_In_Business": [10, 8, 6, 18, 25],
        "Annual_Volume": [50000, 80000, 60000, 30000, 20000],
    }
    return pd.DataFrame(data)


def normalize_metric(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Normalize metric to 0-100 scale using min-max normalization."""
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return pd.Series([50.0] * len(series))
    if higher_is_better:
        return ((series - min_val) / (max_val - min_val) * 100).round(1)
    else:
        return ((max_val - series) / (max_val - min_val) * 100).round(1)


def calculate_weighted_score(df, w_cost, w_quality, w_delivery, w_risk):
    """Calculate weighted overall score for each supplier."""
    result = df.copy()
    result["Norm_Cost"] = normalize_metric(df["Unit_Cost"], higher_is_better=False)
    result["Norm_Quality"] = normalize_metric(df["Quality_Score"], higher_is_better=True)
    result["Norm_Delivery"] = normalize_metric(df["On_Time_Delivery_Pct"], higher_is_better=True)
    result["Norm_Risk"] = normalize_metric(df["Risk_Score"], higher_is_better=False)

    result["Overall_Score"] = (
        w_cost * result["Norm_Cost"]
        + w_quality * result["Norm_Quality"]
        + w_delivery * result["Norm_Delivery"]
        + w_risk * result["Norm_Risk"]
    ).round(1)

    result["Rank"] = result["Overall_Score"].rank(ascending=False).astype(int)
    return result.sort_values("Rank")


def calculate_copq(defect_rate, annual_volume, unit_cost, rework_pct=0.6, scrap_pct=0.4):
    """
    Cost of Poor Quality calculation.
    - rework_pct: fraction of defects that can be reworked (at 40% of unit cost)
    - scrap_pct: fraction of defects that are scrapped (100% loss)
    """
    total_defects = annual_volume * (defect_rate / 100)
    rework_cost = total_defects * rework_pct * (unit_cost * 0.4)
    scrap_cost = total_defects * scrap_pct * unit_cost
    warranty_cost = total_defects * 0.1 * (unit_cost * 1.5)  # 10% warranty claims at 1.5x
    return rework_cost + scrap_cost + warranty_cost


def calculate_delivery_cost(otd_pct, annual_volume, unit_cost, safety_stock_days=14):
    """
    Cost of delivery variability.
    - Late deliveries trigger expediting and safety stock holding costs.
    """
    late_rate = (100 - otd_pct) / 100
    daily_demand = annual_volume / 250  # working days
    expediting_cost = late_rate * annual_volume * (unit_cost * 0.15)  # 15% premium
    safety_stock_units = daily_demand * safety_stock_days * late_rate
    holding_cost = safety_stock_units * unit_cost * 0.25  # 25% annual holding cost
    return expediting_cost + holding_cost


def calculate_switching_cost(annual_volume, unit_cost):
    """
    Estimated cost of switching to a new supplier.
    Includes qualification, ramp-up, and learning curve.
    """
    qualification = 15000  # audit, samples, testing
    ramp_up = annual_volume * unit_cost * 0.05  # 5% efficiency loss first year
    learning_curve = annual_volume * unit_cost * 0.02  # 2% quality dip
    return qualification + ramp_up + learning_curve


# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════

data = load_network()
risk_df = compute_all_risks(data)
G = build_networkx_graph(data)
centralities, resilience, spofs = get_graph_metrics(data)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🔬 Supplier Intelligence Platform")
    st.markdown(
        "<span style='font-size:0.75rem;color:#64748b;'>"
        "Risk Assessment · Decision Intelligence</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Network summary
    n_suppliers = len([n for n in data["nodes"] if n.get("type") == "supplier"])
    n_critical = len(risk_df[risk_df["risk_level"] == "CRITICAL"])
    n_high = len(risk_df[risk_df["risk_level"] == "HIGH"])
    total_spend = risk_df["spend"].sum()

    st.metric("Suppliers Monitored", n_suppliers)
    st.metric("Critical Risk", n_critical, delta=f"+{n_high} High", delta_color="inverse")
    st.metric("Total Spend at Risk", f"${total_spend/1e6:.1f}M")
    st.metric("Network Resilience", f"{resilience['resilience_score']:.0f}/100")

    st.divider()
    st.markdown(
        "<span style='font-size:0.65rem;color:#334155;font-family:monospace;'>"
        "Models: SIR Propagation · Bayesian Risk<br>"
        "Monte Carlo · Graph Centrality<br>"
        "Decision Intelligence · TCO Analysis</span>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='font-size:1.8rem;font-weight:300;margin-bottom:0;'>"
    "Supplier Intelligence Platform</h1>"
    "<p style='color:#64748b;font-size:0.85rem;margin-top:4px;'>"
    "Integrated Risk Assessment & Performance Decision Intelligence for Manufacturing Supply Networks</p>",
    unsafe_allow_html=True,
)

# ─── TABS ────────────────────────────────────────────────────────

tab_dashboard, tab_network, tab_scenarios, tab_monte_carlo, tab_scorecard, tab_decision = st.tabs([
    "📊 Risk Dashboard",
    "🕸️ Network Analysis",
    "⚡ Scenario Engine",
    "💰 Financial Impact",
    "📋 Performance Scorecard",
    "🎯 Decision Intelligence",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1: RISK DASHBOARD
# ═══════════════════════════════════════════════════════════════════

with tab_dashboard:
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value risk-critical'>{n_critical}</div>"
            f"<div class='metric-label'>Critical Risk Suppliers</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        avg_risk = risk_df["risk_prob"].mean()
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value risk-high'>{avg_risk:.0%}</div>"
            f"<div class='metric-label'>Avg Disruption Probability</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value risk-medium'>{resilience['resilience_score']:.0f}</div>"
            f"<div class='metric-label'>Network Resilience (0-100)</div></div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value risk-low'>{len(spofs)}</div>"
            f"<div class='metric-label'>Single Points of Failure</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Risk heatmap
    col_chart, col_table = st.columns([1.2, 1])

    with col_chart:
        st.markdown("#### Bayesian Risk Scoring — All Suppliers")
        st.caption("Posterior P(disruption) computed from 6 evidence signals × calibrated likelihood ratios")

        fig = go.Figure()
        colors = {
            "CRITICAL": "#ef4444", "HIGH": "#f59e0b",
            "MEDIUM": "#3b82f6", "LOW": "#10b981",
        }
        for _, row in risk_df.iterrows():
            fig.add_trace(go.Bar(
                y=[row["name"]],
                x=[row["risk_prob"]],
                orientation="h",
                marker_color=colors.get(row["risk_level"], "#64748b"),
                text=f"{row['risk_prob']:.0%}",
                textposition="auto",
                hovertemplate=(
                    f"<b>{row['name']}</b><br>"
                    f"Risk: {row['risk_prob']:.1%}<br>"
                    f"Level: {row['risk_level']}<br>"
                    f"Dominant: {row['dominant_factor']}<br>"
                    f"Tier: {row['tier']}<br>"
                    f"Region: {row['region']}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

        fig.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            xaxis=dict(
                title="P(Disruption)", range=[0, 1],
                gridcolor="rgba(255,255,255,0.05)",
                tickformat=".0%",
            ),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=200, r=20, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("#### Likelihood Ratio Breakdown")
        st.caption("LR > 1 = increases risk · LR < 1 = decreases risk")

        display_df = risk_df[[
            "name", "tier", "risk_level", "risk_prob",
            "lr_financial", "lr_geopolitical", "lr_tariff",
            "lr_concentration", "lr_historical", "lr_weather",
        ]].copy()
        display_df.columns = [
            "Supplier", "Tier", "Risk", "P(D)",
            "Financial", "Geopolitical", "Tariff",
            "Concentration", "History", "Weather",
        ]
        display_df["P(D)"] = display_df["P(D)"].apply(lambda x: f"{x:.0%}")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400,
            column_config={
                "Financial": st.column_config.NumberColumn(format="%.1f×"),
                "Geopolitical": st.column_config.NumberColumn(format="%.1f×"),
                "Tariff": st.column_config.NumberColumn(format="%.1f×"),
                "Concentration": st.column_config.NumberColumn(format="%.1f×"),
                "History": st.column_config.NumberColumn(format="%.1f×"),
                "Weather": st.column_config.NumberColumn(format="%.1f×"),
            },
        )

    # Key insights
    st.markdown("---")
    st.markdown("#### Key Findings")
    c1, c2, c3 = st.columns(3)
    with c1:
        top_risk = risk_df.iloc[0]
        st.error(
            f"**Highest Risk:** {top_risk['name']}\n\n"
            f"P(disruption) = {top_risk['risk_prob']:.0%} — "
            f"driven by {top_risk['dominant_factor'].replace('_', ' ')}"
        )
    with c2:
        tier2_critical = risk_df[(risk_df["tier"] >= 2) & (risk_df["risk_level"].isin(["CRITICAL", "HIGH"]))]
        st.warning(
            f"**Hidden Tier-2/3 Risks:** {len(tier2_critical)} suppliers\n\n"
            f"33%+ of disruptions originate beyond Tier-1 (Berger et al. 2023)"
        )
    with c3:
        st.info(
            f"**Network Resilience: {resilience['resilience_score']:.0f}/100 ({resilience['interpretation']})**\n\n"
            f"Density: {resilience['density']:.3f} · "
            f"Articulation points: {resilience['articulation_points']}"
        )


# ═══════════════════════════════════════════════════════════════════
# TAB 2: NETWORK ANALYSIS
# ═══════════════════════════════════════════════════════════════════

with tab_network:
    st.markdown("#### Supply Network Topology & Centrality Analysis")
    st.caption("Node size = criticality score · Color = risk level · Edge thickness = dependency weight")

    # Build network visualization
    pos = {}
    tier_x = {-2: 5, -1: 4, 0: 3, 1: 2, 2: 1, 3: 0}
    tier_counts = {}

    for node in data["nodes"]:
        tier = node.get("tier", 0)
        tier_counts[tier] = tier_counts.get(tier, 0)
        y_pos = tier_counts[tier] * 1.5
        pos[node["id"]] = (tier_x.get(tier, 3), y_pos)
        tier_counts[tier] += 1

    # Center each tier vertically
    for tier in tier_counts:
        nodes_in_tier = [(nid, p) for nid, p in pos.items()
                         if any(n["id"] == nid and n.get("tier", 0) == tier for n in data["nodes"])]
        if nodes_in_tier:
            max_y = max(p[1] for _, p in nodes_in_tier)
            offset = max_y / 2
            for nid, p in nodes_in_tier:
                pos[nid] = (p[0], p[1] - offset)

    # Build plotly figure
    fig = go.Figure()

    # Edges
    for edge in data["edges"]:
        if edge["source"] in pos and edge["target"] in pos:
            x0, y0 = pos[edge["source"]]
            x1, y1 = pos[edge["target"]]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=edge.get("weight", 0.5) * 2.5, color="rgba(148,163,184,0.2)"),
                hoverinfo="skip",
                showlegend=False,
            ))

    # Nodes
    risk_colors = {"CRITICAL": "#ef4444", "HIGH": "#f59e0b", "MEDIUM": "#3b82f6", "LOW": "#10b981"}

    for node in data["nodes"]:
        if node["id"] not in pos:
            continue
        x, y = pos[node["id"]]

        risk_row = risk_df[risk_df["id"] == node["id"]]
        if not risk_row.empty:
            color = risk_colors.get(risk_row.iloc[0]["risk_level"], "#64748b")
            size = 15 + risk_row.iloc[0]["risk_prob"] * 35
            hover = (
                f"<b>{node['name']}</b><br>"
                f"Tier: {node.get('tier', '?')}<br>"
                f"Risk: {risk_row.iloc[0]['risk_prob']:.0%} ({risk_row.iloc[0]['risk_level']})<br>"
                f"Region: {node.get('region', '?')}<br>"
                f"Dominant: {risk_row.iloc[0]['dominant_factor']}"
            )
        else:
            color = "#475569"
            size = 20
            hover = f"<b>{node['name']}</b><br>Tier: {node.get('tier', '?')}"

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=size, color=color, line=dict(width=1, color="rgba(255,255,255,0.1)")),
            text=node["id"],
            textposition="top center",
            textfont=dict(size=9, color="#94a3b8"),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        ))

    # Tier labels
    for tier, x in tier_x.items():
        label = {-2: "Customers", -1: "Distribution", 0: "OEM", 1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}
        fig.add_annotation(
            x=x, y=-3.5, text=label.get(tier, f"Tier {tier}"),
            showarrow=False, font=dict(size=11, color="#6366f1", family="JetBrains Mono"),
        )

    fig.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        margin=dict(l=20, r=20, t=20, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Centrality table
    st.markdown("#### Graph Centrality Rankings")
    st.caption("Criticality = 0.15·Degree + 0.35·Betweenness + 0.20·Eigenvector + 0.30·PageRank")

    cent_data = []
    for c in centralities:
        cent_data.append({
            "Node": c.name,
            "Tier": c.tier,
            "Criticality": c.criticality_score,
            "Betweenness": c.betweenness_centrality,
            "PageRank": c.pagerank,
            "Eigenvector": c.eigenvector_centrality,
            "Degree": c.degree_centrality,
            "Risk Amp": c.risk_amplification_factor,
        })

    cent_df = pd.DataFrame(cent_data)
    st.dataframe(
        cent_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Criticality": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "Betweenness": st.column_config.NumberColumn(format="%.3f"),
            "PageRank": st.column_config.NumberColumn(format="%.3f"),
            "Eigenvector": st.column_config.NumberColumn(format="%.3f"),
            "Degree": st.column_config.NumberColumn(format="%.3f"),
            "Risk Amp": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # SPOFs
    if spofs:
        st.markdown("#### ⚠️ Single Points of Failure")
        for spof in spofs:
            st.error(f"**{spof['name']}** (Tier {spof['tier']}) — Removing this node disconnects {spof['suppliers_disconnected']} suppliers")


# ═══════════════════════════════════════════════════════════════════
# TAB 3: SCENARIO ENGINE
# ═══════════════════════════════════════════════════════════════════

with tab_scenarios:
    st.markdown("#### Disruption Cascade Simulator")
    st.caption("SIR epidemic-adapted propagation model · Monte Carlo simulation runs")

    col_config, col_results = st.columns([1, 1.5])

    with col_config:
        scenarios = data.get("scenarios", [])
        scenario_names = [s["name"] for s in scenarios]
        selected_name = st.selectbox("Select Disruption Scenario", scenario_names)
        scenario = next(s for s in scenarios if s["name"] == selected_name)

        st.info(f"**{scenario['name']}**\n\n{scenario['description']}")
        st.markdown(f"**Affected nodes:** {', '.join(scenario['affected_nodes'])}")
        st.markdown(f"**Delay range:** {scenario['min_delay']}–{scenario['max_delay']} days (mode: {scenario['mode_delay']})")

        st.markdown("---")
        st.markdown("**Model Parameters**")
        beta = st.slider("Transmission rate (β)", 0.1, 0.8, 0.35, 0.05,
                          help="Probability of disruption spreading per edge per timestep")
        gamma = st.slider("Recovery rate (γ)", 0.01, 0.3, 0.08, 0.01,
                           help="Rate at which disrupted suppliers recover")
        n_runs = st.slider("Simulation runs", 20, 200, 50, 10)

        run_sim = st.button("🚀 Run Cascade Simulation", type="primary", use_container_width=True)

    with col_results:
        if run_sim:
            with st.spinner(f"Running {n_runs} SIR simulations..."):
                params = PropagationParams(beta=beta, gamma=gamma, time_steps=30)
                mc_sir = run_monte_carlo_sir(G, scenario["affected_nodes"], params, n_runs=n_runs)

            st.success(f"Completed {n_runs} simulations")

            st.markdown("##### Cascade Propagation Results")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Avg Nodes Infected",
                           f"{mc_sir['cascade_stats']['avg_total_infected']:.1f} / {len(data['nodes'])}")
            with m2:
                st.metric("Avg Cascade Depth",
                           f"{mc_sir['cascade_stats']['avg_cascade_depth']:.1f} tiers")
            with m3:
                oem_rate = mc_sir['per_node'].get('OEM', {}).get('infection_rate', 0)
                st.metric("P(OEM Impacted)", f"{oem_rate:.0%}")

            node_results = []
            for nid, stats in mc_sir["per_node"].items():
                node_info = next((n for n in data["nodes"] if n["id"] == nid), {})
                if node_info.get("type") in ["supplier", "focal"]:
                    node_results.append({
                        "Supplier": node_info.get("name", nid),
                        "Tier": node_info.get("tier", "?"),
                        "Infection Rate": stats["infection_rate"],
                        "Avg Risk Score": stats["avg_risk_score"],
                        "Avg Time to Impact": stats.get("avg_time_to_impact"),
                    })

            result_df = pd.DataFrame(node_results).sort_values("Infection Rate", ascending=False)

            fig = px.bar(
                result_df, x="Infection Rate", y="Supplier",
                orientation="h", color="Infection Rate",
                color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                range_color=[0, 1],
            )
            fig.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat=".0%"),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=200, r=20, t=10, b=40),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Infection Rate": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.0%%"),
                    "Avg Risk Score": st.column_config.NumberColumn(format="%.3f"),
                    "Avg Time to Impact": st.column_config.NumberColumn(format="%.1f days"),
                },
            )
        else:
            st.markdown(
                "<div style='text-align:center;padding:80px 0;color:#475569;'>"
                "← Configure parameters and click <b>Run Cascade Simulation</b></div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════
# TAB 4: FINANCIAL IMPACT (Monte Carlo)
# ═══════════════════════════════════════════════════════════════════

with tab_monte_carlo:
    st.markdown("#### Monte Carlo Financial Impact Simulation")
    st.caption("2,000+ iteration simulation with triangular delay + log-normal cost distributions")

    col_params, col_output = st.columns([1, 1.5])

    with col_params:
        suppliers_with_spend = risk_df[risk_df["spend"] > 0]
        selected_supplier = st.selectbox(
            "Select Supplier",
            suppliers_with_spend["name"].tolist(),
        )
        sup_row = suppliers_with_spend[suppliers_with_spend["name"] == selected_supplier].iloc[0]

        st.markdown(f"**{selected_supplier}** — Tier {sup_row['tier']} · {sup_row['region']}")
        st.markdown(f"Risk: **{sup_row['risk_level']}** ({sup_row['risk_prob']:.0%})")

        st.markdown("---")
        st.markdown("**Cost Parameters**")

        annual_spend = st.number_input("Annual Spend ($)", value=int(sup_row["spend"]), step=100000)
        daily_demand = st.number_input("Daily Demand (units)", value=200, step=50)
        unit_cost = st.number_input("Unit Cost ($)", value=55.0, step=5.0)
        profit_margin = st.number_input("Profit Margin/Unit ($)", value=22.0, step=2.0)
        holding_rate = st.slider("Holding Cost Rate (%)", 15, 40, 25) / 100

        st.markdown("---")
        st.markdown("**Disruption Parameters**")
        min_delay = st.number_input("Min Delay (days)", value=7, step=1)
        mode_delay = st.number_input("Most Likely Delay (days)", value=21, step=1)
        max_delay = st.number_input("Max Delay (days)", value=90, step=5)
        mc_iters = st.slider("MC Iterations", 1000, 10000, 5000, 500)

        run_mc = st.button("💰 Run Financial Simulation", type="primary", use_container_width=True)

    with col_output:
        if run_mc:
            supplier_profile = SupplierCostProfile(
                annual_spend=annual_spend,
                daily_demand_units=daily_demand,
                unit_cost=unit_cost,
                profit_margin_per_unit=profit_margin,
                holding_cost_rate=holding_rate,
                avg_freight_cost_per_order=annual_spend * 0.03 / 52,
                expedite_multiplier=3.0,
            )
            disruption_profile = DisruptionProfile(
                min_delay_days=min_delay,
                mode_delay_days=mode_delay,
                max_delay_days=max_delay,
                expedite_threshold_days=14,
                stockout_threshold_days=30,
                quality_impact_probability=0.15,
                duration_months=3,
            )

            with st.spinner(f"Running {mc_iters} Monte Carlo iterations..."):
                mc = run_monte_carlo(supplier_profile, disruption_profile, mc_iters, seed=None)

            st.markdown("##### Risk Metrics")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Expected Loss", f"${mc.expected_loss:,.0f}")
            with m2:
                st.metric("P90 (Likely Worst)", f"${mc.p90_loss:,.0f}")
            with m3:
                st.metric("VaR₉₅", f"${mc.var_95:,.0f}")
            with m4:
                st.metric("CVaR₉₅ (Tail Risk)", f"${mc.cvar_95:,.0f}")

            st.caption("⚠️ Simulated estimates based on assumed distributions, not financial forecasts.")

            st.markdown("##### Loss Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=mc.loss_distribution,
                nbinsx=60,
                marker_color="rgba(99,102,241,0.5)",
                marker_line=dict(color="rgba(99,102,241,0.8)", width=0.5),
            ))
            fig.add_vline(x=mc.var_95, line_dash="dash", line_color="#ef4444",
                          annotation_text=f"VaR₉₅ = ${mc.var_95:,.0f}",
                          annotation_font_color="#ef4444")
            fig.add_vline(x=mc.expected_loss, line_dash="dash", line_color="#10b981",
                          annotation_text=f"Expected = ${mc.expected_loss:,.0f}",
                          annotation_font_color="#10b981")

            fig.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                xaxis=dict(title="Total Financial Impact ($)", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(l=60, r=20, t=20, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### Cost Breakdown (Average)")
            breakdown_data = {
                "Safety Stock": mc.avg_safety_stock_cost,
                "Expedite Freight": mc.avg_expedite_cost,
                "Stockout Loss": mc.avg_stockout_cost,
                "Quality Defects": mc.avg_quality_cost,
            }
            fig_pie = go.Figure(go.Pie(
                labels=list(breakdown_data.keys()),
                values=list(breakdown_data.values()),
                marker_colors=["#6366f1", "#f59e0b", "#ef4444", "#a855f7"],
                textinfo="label+percent",
                hole=0.4,
            ))
            fig_pie.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("##### Probability of Exceeding Thresholds")
            prob_data = []
            for threshold, prob in mc.probability_of_loss_over.items():
                prob_data.append({"Threshold": f"${threshold:,}", "P(Loss > Threshold)": prob})
            st.dataframe(pd.DataFrame(prob_data), use_container_width=True, hide_index=True)

        else:
            st.markdown(
                "<div style='text-align:center;padding:80px 0;color:#475569;'>"
                "← Set parameters and click <b>Run Financial Simulation</b></div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════
# TAB 5: PERFORMANCE SCORECARD (NEW)
# ═══════════════════════════════════════════════════════════════════

with tab_scorecard:
    st.markdown("#### Supplier Performance Scorecard")
    st.caption("Multi-criteria weighted evaluation · Adjust weights to match your procurement strategy")

    # Weight controls in columns
    st.markdown("**Strategic Weights** (must sum to 1.0)")
    wc1, wc2, wc3, wc4, wc5 = st.columns([1, 1, 1, 1, 0.5])
    with wc1:
        w_cost = st.slider("💰 Cost", 0.0, 1.0, 0.30, 0.05, key="sc_cost")
    with wc2:
        w_quality = st.slider("✅ Quality", 0.0, 1.0, 0.30, 0.05, key="sc_qual")
    with wc3:
        w_delivery = st.slider("🚚 Delivery", 0.0, 1.0, 0.25, 0.05, key="sc_del")
    with wc4:
        w_risk = st.slider("⚠️ Risk", 0.0, 1.0, 0.15, 0.05, key="sc_risk")
    with wc5:
        total_w = w_cost + w_quality + w_delivery + w_risk
        if abs(total_w - 1.0) > 0.01:
            st.error(f"Sum: {total_w:.2f}")
        else:
            st.success(f"Sum: {total_w:.2f}")

    weights_valid = abs(total_w - 1.0) <= 0.01

    # Load and score
    df_raw = get_scorecard_data()
    if weights_valid:
        df_scored = calculate_weighted_score(df_raw, w_cost, w_quality, w_delivery, w_risk)
    else:
        df_scored = calculate_weighted_score(df_raw, 0.25, 0.25, 0.25, 0.25)

    # Top supplier highlight
    top = df_scored.iloc[0]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🏆 Top Supplier", top["Supplier"].split("(")[0].strip())
    m2.metric("Score", f"{top['Overall_Score']:.1f}/100")
    m3.metric("Unit Cost", f"${top['Unit_Cost']:.2f}")
    m4.metric("Quality", f"{top['Quality_Score']}/100")
    m5.metric("On-Time", f"{top['On_Time_Delivery_Pct']}%")

    st.markdown("---")

    # Scorecard table
    display_sc = df_scored[[
        "Rank", "Supplier", "Country", "Unit_Cost", "Quality_Score",
        "On_Time_Delivery_Pct", "Defect_Rate_Pct", "Risk_Score",
        "Overall_Score", "Certifications"
    ]].copy()

    display_sc = display_sc.rename(columns={
        "Unit_Cost": "Cost ($)",
        "Quality_Score": "Quality",
        "On_Time_Delivery_Pct": "OTD %",
        "Defect_Rate_Pct": "Defect %",
        "Risk_Score": "Risk",
        "Overall_Score": "Score"
    })

    st.dataframe(
        display_sc,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Cost ($)": st.column_config.NumberColumn("Cost ($)", format="$%.2f"),
            "Quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100),
            "OTD %": st.column_config.ProgressColumn("OTD %", min_value=0, max_value=100),
            "Defect %": st.column_config.NumberColumn("Defect %", format="%.1f%%"),
            "Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100),
            "Score": st.column_config.NumberColumn("Score", format="%.1f"),
        }
    )

    # Normalized breakdown chart
    st.markdown("---")
    st.markdown("#### Normalized Score Breakdown")
    norm_df = df_scored[["Supplier", "Norm_Cost", "Norm_Quality", "Norm_Delivery", "Norm_Risk"]].copy()
    norm_df = norm_df.rename(columns={
        "Norm_Cost": f"Cost (w={w_cost:.2f})",
        "Norm_Quality": f"Quality (w={w_quality:.2f})",
        "Norm_Delivery": f"Delivery (w={w_delivery:.2f})",
        "Norm_Risk": f"Risk (w={w_risk:.2f})"
    })
    norm_df = norm_df.set_index("Supplier")
    st.bar_chart(norm_df, height=350)


# ═══════════════════════════════════════════════════════════════════
# TAB 6: DECISION INTELLIGENCE (NEW)
# ═══════════════════════════════════════════════════════════════════

with tab_decision:
    st.markdown("#### Total Cost of Ownership & Decision Intelligence")
    st.caption("Quantify the hidden costs behind each supplier choice — beyond unit price")

    df_raw = get_scorecard_data()

    # Calculate hidden costs for each supplier
    tco_rows = []
    for _, row in df_raw.iterrows():
        copq = calculate_copq(
            row["Defect_Rate_Pct"], row["Annual_Volume"],
            row["Unit_Cost"]
        )
        delivery_cost = calculate_delivery_cost(
            row["On_Time_Delivery_Pct"], row["Annual_Volume"],
            row["Unit_Cost"]
        )
        switching_cost = calculate_switching_cost(
            row["Annual_Volume"], row["Unit_Cost"]
        )
        direct_cost = row["Annual_Volume"] * row["Unit_Cost"]
        total_tco = direct_cost + copq + delivery_cost

        tco_rows.append({
            "Supplier": row["Supplier"],
            "Country": row["Country"],
            "Direct Cost": direct_cost,
            "Cost of Poor Quality": copq,
            "Delivery Variability Cost": delivery_cost,
            "Switching Cost": switching_cost,
            "Total TCO": total_tco,
            "Unit Cost": row["Unit_Cost"],
            "Hidden Cost %": ((copq + delivery_cost) / direct_cost * 100),
        })

    tco_df = pd.DataFrame(tco_rows).sort_values("Total TCO")

    # TCO Comparison metrics
    cheapest_unit = tco_df.loc[tco_df["Unit Cost"].idxmin()]
    cheapest_tco = tco_df.iloc[0]

    st.markdown("##### Key Insight")
    if cheapest_unit["Supplier"] != cheapest_tco["Supplier"]:
        st.warning(
            f"**The cheapest per-unit supplier ({cheapest_unit['Supplier']}) "
            f"is NOT the cheapest on total cost of ownership.**\n\n"
            f"Lowest unit cost: **{cheapest_unit['Supplier']}** at ${cheapest_unit['Unit Cost']:.2f}/unit\n\n"
            f"Lowest TCO: **{cheapest_tco['Supplier']}** at ${cheapest_tco['Total TCO']:,.0f}/year\n\n"
            f"Hidden costs add **{cheapest_unit['Hidden Cost %']:.1f}%** to {cheapest_unit['Supplier']}'s true cost."
        )
    else:
        st.success(
            f"**{cheapest_tco['Supplier']}** has both the lowest unit cost "
            f"and the lowest total cost of ownership at ${cheapest_tco['Total TCO']:,.0f}/year."
        )

    st.markdown("---")

    # TCO Stacked bar chart
    st.markdown("##### Total Cost of Ownership Comparison")

    fig_tco = go.Figure()
    fig_tco.add_trace(go.Bar(
        name="Direct Cost",
        x=tco_df["Supplier"],
        y=tco_df["Direct Cost"],
        marker_color="#6366f1",
    ))
    fig_tco.add_trace(go.Bar(
        name="Cost of Poor Quality",
        x=tco_df["Supplier"],
        y=tco_df["Cost of Poor Quality"],
        marker_color="#ef4444",
    ))
    fig_tco.add_trace(go.Bar(
        name="Delivery Variability",
        x=tco_df["Supplier"],
        y=tco_df["Delivery Variability Cost"],
        marker_color="#f59e0b",
    ))

    fig_tco.update_layout(
        barmode="stack",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        yaxis=dict(title="Annual Cost ($)", gridcolor="rgba(255,255,255,0.05)", tickformat="$,.0f"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=80, r=20, t=20, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_tco, use_container_width=True)

    # Detailed TCO table
    st.markdown("##### Cost Breakdown by Supplier")
    tco_display = tco_df[[
        "Supplier", "Country", "Direct Cost", "Cost of Poor Quality",
        "Delivery Variability Cost", "Switching Cost", "Total TCO", "Hidden Cost %"
    ]].copy()

    st.dataframe(
        tco_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Direct Cost": st.column_config.NumberColumn(format="$%,.0f"),
            "Cost of Poor Quality": st.column_config.NumberColumn(format="$%,.0f"),
            "Delivery Variability Cost": st.column_config.NumberColumn(format="$%,.0f"),
            "Switching Cost": st.column_config.NumberColumn(format="$%,.0f"),
            "Total TCO": st.column_config.NumberColumn(format="$%,.0f"),
            "Hidden Cost %": st.column_config.NumberColumn(format="%.1f%%"),
        }
    )

    # Scenario comparison
    st.markdown("---")
    st.markdown("##### Scenario: What If You Switch Suppliers?")

    col_from, col_to = st.columns(2)
    with col_from:
        current = st.selectbox("Current Supplier", tco_df["Supplier"].tolist(), key="from_sup")
    with col_to:
        alternatives = [s for s in tco_df["Supplier"].tolist() if s != current]
        target = st.selectbox("Alternative Supplier", alternatives, key="to_sup")

    current_row = tco_df[tco_df["Supplier"] == current].iloc[0]
    target_row = tco_df[tco_df["Supplier"] == target].iloc[0]

    annual_savings = current_row["Total TCO"] - target_row["Total TCO"]
    switch_cost = target_row["Switching Cost"]
    payback_months = (switch_cost / (annual_savings / 12)) if annual_savings > 0 else float('inf')

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Current TCO", f"${current_row['Total TCO']:,.0f}/yr")
    r2.metric("Alternative TCO", f"${target_row['Total TCO']:,.0f}/yr")
    r3.metric(
        "Annual Savings",
        f"${annual_savings:,.0f}",
        delta=f"{'Save' if annual_savings > 0 else 'Lose'} ${abs(annual_savings):,.0f}/yr",
        delta_color="normal" if annual_savings > 0 else "inverse",
    )
    if annual_savings > 0 and payback_months < 999:
        r4.metric("Payback Period", f"{payback_months:.1f} months")
    else:
        r4.metric("Payback Period", "N/A — no savings")


# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")

with st.expander("⚖️ Legal Disclaimer & Data Notice", expanded=False):
    st.markdown("""
    **DISCLAIMER OF WARRANTIES & LIMITATION OF LIABILITY**

    This system provides supply chain risk intelligence for **informational and decision-support purposes only**.
    All outputs — including risk probabilities, financial impact estimates (VaR, CVaR), cascade predictions,
    and mitigation recommendations — are **statistical models based on available data and mathematical simulations**.

    **They are not guarantees, forecasts, or professional advice.**

    - **Risk scores** are Bayesian posterior probabilities derived from calibrated likelihood ratios.
      They represent model-estimated disruption probability, not certainty of future events.
    - **Financial impact figures** are Monte Carlo simulation outputs based on assumed distributions.
      Actual losses may differ materially from simulated values.
    - **Cascade predictions** use SIR epidemic-adapted models with stochastic elements.
      Results vary across simulation runs and depend on parameter assumptions.
    - **TCO calculations** use industry-standard cost models with configurable parameters.
      Actual costs depend on company-specific factors not captured in default assumptions.

    **DATA PROTECTION NOTICE**

    - Supplier data uploaded to this system is processed locally and is not shared with third parties.
    - AI-powered event classification (Sentinel Agent) sends news article text to the Anthropic API
      for analysis. No proprietary supplier data is included in these API calls.
    - Users are responsible for ensuring they have authorization to upload and process
      any supplier data entered into this system.

    **LIMITATION OF LIABILITY**

    The creators of this system shall not be liable for any direct, indirect, incidental,
    consequential, or special damages arising from the use of or inability to use this system.

    **USE AT YOUR OWN RISK. ALWAYS VERIFY CRITICAL DECISIONS WITH QUALIFIED PROFESSIONALS.**

    ---
    *Model Documentation: SIR Propagation (Tabachová et al. 2024) · Bayesian Risk (Hosseini & Ivanov 2020)
    · Monte Carlo VaR/CVaR (Chopra & Sodhi 2004) · Graph Centrality (Brintrup et al. 2021)
    · TCO Analysis (Ellram 1995)*
    """)

st.markdown(
    "<div style='text-align:center;'>"
    "<span style='font-size:0.65rem;color:#334155;font-family:JetBrains Mono,monospace;'>"
    "SUPPLIER INTELLIGENCE PLATFORM — v2.0<br>"
    "SIR Propagation · Bayesian Risk · Monte Carlo · Graph Centrality · Decision Intelligence · TCO Analysis"
    "</span></div>",
    unsafe_allow_html=True,
)
