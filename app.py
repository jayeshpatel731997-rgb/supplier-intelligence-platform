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
import os
from pathlib import Path
from datetime import datetime, timedelta

from data_ingestion import (
    ingest_file, generate_sample_template,
    dataframe_to_network_data,
)
from news_intelligence import (
    run_sentinel_scan,
    SEVERITY_COLORS, SEVERITY_ICONS, DISRUPTION_ICONS,
)
from src.services.supplier_risk_evidence_chain import (
    build_supplier_risk_evidence_chains,
    load_demo_supplier_signals,
)
from pilot_security import (
    authenticate_user,
    change_user_password,
    create_initial_admin,
    create_user,
    has_mutation_access,
    init_pilot_database,
    is_admin,
    list_audit_logs,
    list_sentinel_scans,
    list_users,
    log_audit,
    save_sentinel_scan,
    save_supplier_upload,
    set_user_active,
)

try:
    from src.config import get_settings as get_production_settings
    from src.database import create_session_factory as create_production_session_factory
    from src.database import init_database as init_production_database
    from src.database import seed_demo_tenant as seed_production_demo_tenant
    from src.repositories.alerts import AlertRepository as ProductionAlertRepository
    from src.repositories.tenants import TenantRepository as ProductionTenantRepository
    from src.services.audit_export import AuditExportService as ProductionAuditExportService
    from src.services.compliance_evidence import EvidenceService as ProductionEvidenceService
    from src.services.migration_service import validate_tenant_schema as validate_production_tenant_schema
    from src.services.scheduler import LocalJobScheduler
    from src.services.system_service import system_status as get_production_system_status
    from src.services.worker_queue import EnterpriseTaskRunner, get_worker_mode
    from src.tenancy import DEMO_TENANT_ID, DEMO_PLATFORM_ADMIN, TenantContext

    PRODUCTION_FOUNDATION_AVAILABLE = True
except Exception:
    PRODUCTION_FOUNDATION_AVAILABLE = False

try:
    from src.services.streamlit_api_client import StreamlitApiClient, friendly_api_error

    STREAMLIT_API_CLIENT_AVAILABLE = True
except Exception:
    STREAMLIT_API_CLIENT_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_secret_or_env(name: str, default: str = "") -> str:
    """Read deployment secrets from Streamlit secrets first, then environment."""
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)

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

NETWORK_OPTIONS = {
    "🏭  Lake Cable LLC (Live Demo)": "data/lake_cable_network.json",
    "🔬  Sample Network (Synthetic)": "data/sample_network.json",
}

@st.cache_data
def load_network(path: str = "data/lake_cable_network.json"):
    with open(path) as f:
        return json.load(f)


@st.cache_data
def build_uploaded_network(df: pd.DataFrame):
    return dataframe_to_network_data(df)

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


@st.cache_data
def get_supplier_risk_evidence_chain_reports():
    return build_supplier_risk_evidence_chains(load_demo_supplier_signals())


@st.cache_data
def load_demo_scenario_profiles():
    scenario_path = Path(get_secret_or_env("SUPPLIER_DEMO_SCENARIO_PATH", "data/demo_supplier_scenarios.json"))
    if not scenario_path.exists():
        return []
    with scenario_path.open(encoding="utf-8") as handle:
        return list(json.load(handle))


# ═══════════════════════════════════════════════════════════════════
# DECISION INTELLIGENCE: SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════

def get_scorecard_data() -> pd.DataFrame:
    """
    Sample supplier data for decision intelligence scorecard.
    In production, this would connect to ERP/procurement systems.
    Returns uploaded data when available, otherwise demo data.
    """
    if st.session_state.get("using_uploaded_data") and st.session_state.get("uploaded_supplier_df") is not None:
        df = st.session_state.uploaded_supplier_df.copy()
        rename_map = {
            "supplier_name":       "Supplier",
            "country":             "Country",
            "unit_cost":           "Unit_Cost",
            "quality_score":       "Quality_Score",
            "on_time_delivery_pct":"On_Time_Delivery_Pct",
            "defect_rate_pct":     "Defect_Rate_Pct",
            "risk_score":          "Risk_Score",
            "certifications":      "Certifications",
            "years_in_business":   "Years_In_Business",
            "annual_volume":       "Annual_Volume",
        }
        df = df.rename(columns=rename_map)
        defaults = {
            "Supplier":             "Unknown",
            "Country":              "Unknown",
            "Unit_Cost":            10.0,
            "Quality_Score":        75.0,
            "On_Time_Delivery_Pct": 85.0,
            "Defect_Rate_Pct":      2.0,
            "Risk_Score":           40.0,
            "Certifications":       "",
            "Years_In_Business":    5.0,
            "Annual_Volume":        10000,
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
        return df

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


# ─── SESSION STATE ───────────────────────────────────────────────

if "uploaded_supplier_df" not in st.session_state:
    st.session_state.uploaded_supplier_df = None
if "using_uploaded_data" not in st.session_state:
    st.session_state.using_uploaded_data = False
if "sentinel_results" not in st.session_state:
    st.session_state.sentinel_results = []
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None


# ─── PILOT AUTHENTICATION GATE ──────────────────────────────────

pilot_db_info = init_pilot_database()
SESSION_TIMEOUT_MINUTES = int(get_secret_or_env("SUPPLIER_APP_SESSION_TIMEOUT_MINUTES", "60"))

production_settings = None
production_session_factory = None
if PRODUCTION_FOUNDATION_AVAILABLE:
    try:
        production_settings = get_production_settings()
        production_session_factory = create_production_session_factory(production_settings)
        init_production_database(production_session_factory)
        if production_settings.demo_mode and not production_settings.is_production:
            seed_production_demo_tenant(production_session_factory)
    except Exception:
        PRODUCTION_FOUNDATION_AVAILABLE = False

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "auth_last_seen" not in st.session_state:
    st.session_state.auth_last_seen = None


def current_user() -> dict:
    return st.session_state.auth_user or {"username": "anonymous", "role": "viewer"}


def enterprise_context() -> TenantContext:
    user = current_user()
    fallback_role = st.session_state.get("active_tenant_role") or ("platform_admin" if user.get("role") == "admin" else "viewer")
    fallback_username = DEMO_PLATFORM_ADMIN if fallback_role == "platform_admin" else user.get("username", "viewer")
    return TenantContext(
        tenant_id=st.session_state.get("active_tenant_id", DEMO_TENANT_ID),
        username=fallback_username,
        role=fallback_role,
    )


def can_mutate() -> bool:
    return has_mutation_access(current_user())


def audit(action: str, details: dict | None = None) -> None:
    user = current_user()
    log_audit(user["username"], user["role"], action, details or {})


def render_login() -> None:
    st.markdown("## Supplier Intelligence Platform")
    st.caption("Secure internal pilot login")
    if pilot_db_info.get("requires_first_admin_setup"):
        st.warning("Production mode is enabled and no admin user exists. Create the first admin before continuing.")
        with st.form("first_admin_setup", clear_on_submit=False):
            setup_username = st.text_input("Admin username")
            setup_password = st.text_input("Admin password", type="password")
            submitted_setup = st.form_submit_button("Create first admin", type="primary")
        if submitted_setup:
            try:
                create_initial_admin(setup_username, setup_password)
                st.success("Initial admin created. Sign in to continue.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        st.stop()
    if pilot_db_info["default_password_in_use"]:
        st.warning(
            "Default admin is enabled for this pilot. Login with "
            f"`{pilot_db_info['default_admin_user']}` / `ChangeMe123!`, then change it in Admin & Audit."
        )
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        user = authenticate_user(username, password)
        if user:
            st.session_state.auth_user = user
            st.session_state.auth_last_seen = datetime.utcnow()
            log_audit(user["username"], user["role"], "auth.login", {})
            st.rerun()
        else:
            log_audit(username or "unknown", "unknown", "auth.failed_login", {})
            st.error("Invalid username or password.")
    st.stop()


if st.session_state.auth_user is None:
    render_login()

last_seen = st.session_state.auth_last_seen
if last_seen and datetime.utcnow() - last_seen > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
    expired_user = st.session_state.auth_user
    if expired_user:
        log_audit(expired_user["username"], expired_user["role"], "auth.session_timeout", {})
    st.session_state.auth_user = None
    st.session_state.auth_last_seen = None
    st.warning("Session timed out. Please sign in again.")
    render_login()

st.session_state.auth_last_seen = datetime.utcnow()


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR — must come before data load so we can pick network
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🔬 Supplier Intelligence Platform")
    st.markdown(
        "<span style='font-size:0.75rem;color:#64748b;'>"
        "Risk Assessment · Decision Intelligence</span>",
        unsafe_allow_html=True,
    )
    user = current_user()
    st.caption(f"Signed in as `{user['username']}` · `{user['role']}`")
    if PRODUCTION_FOUNDATION_AVAILABLE and production_session_factory is not None:
        try:
            with production_session_factory() as prod_session:
                tenant_repo = ProductionTenantRepository(prod_session)
                tenants = tenant_repo.list_tenants()
                tenant_options = [tenant.tenant_id for tenant in tenants] or [DEMO_TENANT_ID]
                if st.session_state.get("active_tenant_id") not in tenant_options:
                    st.session_state.active_tenant_id = tenant_options[0]
                active_tenant = st.selectbox(
                    "Active Tenant",
                    options=tenant_options,
                    index=tenant_options.index(st.session_state.active_tenant_id),
                    help="Tenant context used by production API, jobs, alerts, and Enterprise Admin views.",
                )
                st.session_state.active_tenant_id = active_tenant
                membership = tenant_repo.get_membership(active_tenant, user["username"])
                if membership:
                    st.session_state.active_tenant_role = membership.role
                elif user.get("role") == "admin":
                    st.session_state.active_tenant_role = "platform_admin"
                else:
                    st.session_state.active_tenant_role = "viewer"
                st.caption(f"Tenant role: `{st.session_state.active_tenant_role}`")
        except Exception as exc:
            st.caption(f"Tenant selector unavailable: {exc}")
    if not can_mutate():
        st.info("Viewer mode: dashboards are read-only.")
    if st.button("Sign out", use_container_width=True):
        audit("auth.logout", {})
        st.session_state.auth_user = None
        st.session_state.auth_last_seen = None
        st.rerun()
    st.divider()

    selected_network_label = st.selectbox(
        "**Active Network**",
        options=list(NETWORK_OPTIONS.keys()),
        index=0,
        help="Switch between the live Lake Cable demo and the synthetic sample network.",
    )
    selected_network_path = NETWORK_OPTIONS[selected_network_label]
    st.divider()

# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════

if st.session_state.using_uploaded_data and st.session_state.uploaded_supplier_df is not None:
    data = build_uploaded_network(st.session_state.uploaded_supplier_df)
    active_network_source = "uploaded"
else:
    data = load_network(selected_network_path)
    active_network_source = "demo"

risk_df = compute_all_risks(data)
G = build_networkx_graph(data)
centralities, resilience, spofs = get_graph_metrics(data)

# ─── Re-open sidebar to add network stats (now that data is loaded) ────────────
with st.sidebar:

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
    if st.session_state.using_uploaded_data and st.session_state.uploaded_supplier_df is not None:
        n_up = len(st.session_state.uploaded_supplier_df)
        st.success(f"✅ Live data: {n_up} suppliers")
        st.caption("Uploaded data is powering the full analytics network, not just the scorecard.")
        if can_mutate() and st.button("↩️ Reset to demo data", use_container_width=True):
            st.session_state.using_uploaded_data = False
            st.session_state.uploaded_supplier_df = None
            audit("data.reset_to_demo", {"source": "sidebar"})
            st.rerun()
    else:
        st.info("📁 Using demo data\nUpload real data in **Data Upload** tab to replace the full analytics network")

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

tab_labels = [
    "📊 Risk Dashboard",
    "🕸️ Network Analysis",
    "⚡ Scenario Engine",
    "💰 Financial Impact",
    "📋 Performance Scorecard",
    "🎯 Decision Intelligence",
    "📁 Data Upload",
    "📡 Sentinel Agent",
]
tab_labels.insert(8, "Evidence Chains")
tab_labels.extend(["Production Command Center", "Alerts & Health"])
if enterprise_context().role == "platform_admin":
    tab_labels.append("Enterprise Admin")
if is_admin(current_user()):
    tab_labels.append("🛡️ Admin & Audit")

tabs = st.tabs(tab_labels)
tab_dashboard, tab_network, tab_scenarios, tab_monte_carlo, tab_scorecard, tab_decision, tab_upload, tab_sentinel, tab_evidence = tabs[:9]
tab_command = tabs[9]
tab_alerts = tabs[10]
next_tab_index = 11
tab_enterprise_admin = None
if enterprise_context().role == "platform_admin":
    tab_enterprise_admin = tabs[next_tab_index]
    next_tab_index += 1
tab_admin = tabs[next_tab_index] if is_admin(current_user()) and len(tabs) > next_tab_index else None


# ═══════════════════════════════════════════════════════════════════
# TAB 1: RISK DASHBOARD
# ═══════════════════════════════════════════════════════════════════

with tab_dashboard:
    if active_network_source == "uploaded":
        st.caption("Live uploaded supplier network is active across all analytics tabs.")

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

    if active_network_source == "uploaded":
        st.info("This topology was generated from your uploaded supplier file. Upstream links are inferred from tier, category, country, and spend so the network analytics can still run end to end.")

    # Build network visualization
    pos = {}
    positive_tiers = sorted({node.get("tier", 0) for node in data["nodes"] if node.get("tier", 0) > 0})
    tier_x = {-2: 5, -1: 4, 0: 3}
    for offset, tier in enumerate(positive_tiers, start=1):
        tier_x[tier] = 3 - offset
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
        label = {-2: "Customers", -1: "Distribution", 0: "OEM"}
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
        if active_network_source == "uploaded":
            st.info("Scenarios below were generated automatically from the risk profile of your uploaded suppliers.")
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

        run_sim = st.button(
            "🚀 Run Cascade Simulation",
            type="primary",
            use_container_width=True,
            disabled=not can_mutate(),
        )
        if not can_mutate():
            st.caption("Viewer role can inspect results but cannot run new simulations.")

    with col_results:
        if run_sim:
            audit(
                "simulation.run_cascade",
                {"scenario": selected_name, "beta": beta, "gamma": gamma, "runs": n_runs},
            )
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
        if suppliers_with_spend.empty:
            suppliers_with_spend = risk_df.copy()
        selected_supplier = st.selectbox(
            "Select Supplier",
            suppliers_with_spend["name"].tolist(),
        )
        sup_row = suppliers_with_spend[suppliers_with_spend["name"] == selected_supplier].iloc[0]

        st.markdown(f"**{selected_supplier}** — Tier {sup_row['tier']} · {sup_row['region']}")
        st.markdown(f"Risk: **{sup_row['risk_level']}** ({sup_row['risk_prob']:.0%})")

        st.markdown("---")
        st.markdown("**Cost Parameters**")

        annual_spend_default = int(sup_row["spend"]) if sup_row["spend"] > 0 else 250000
        annual_spend = st.number_input("Annual Spend ($)", value=annual_spend_default, step=100000)
        daily_demand = st.number_input("Daily Demand (units)", value=200, step=50)
        unit_cost_default = 55.0
        if st.session_state.using_uploaded_data and st.session_state.uploaded_supplier_df is not None:
            uploaded_match = st.session_state.uploaded_supplier_df[
                st.session_state.uploaded_supplier_df["supplier_name"] == selected_supplier
            ]
            if not uploaded_match.empty and uploaded_match.iloc[0].get("unit_cost", 0) > 0:
                unit_cost_default = float(uploaded_match.iloc[0]["unit_cost"])
        unit_cost = st.number_input("Unit Cost ($)", value=unit_cost_default, step=5.0)
        profit_margin = st.number_input("Profit Margin/Unit ($)", value=22.0, step=2.0)
        holding_rate = st.slider("Holding Cost Rate (%)", 15, 40, 25) / 100

        st.markdown("---")
        st.markdown("**Disruption Parameters**")
        min_delay = st.number_input("Min Delay (days)", value=7, step=1)
        mode_delay = st.number_input("Most Likely Delay (days)", value=21, step=1)
        max_delay = st.number_input("Max Delay (days)", value=90, step=5)
        mc_iters = st.slider("MC Iterations", 1000, 10000, 5000, 500)

        run_mc = st.button(
            "💰 Run Financial Simulation",
            type="primary",
            use_container_width=True,
            disabled=not can_mutate(),
        )
        if not can_mutate():
            st.caption("Viewer role can inspect results but cannot run new financial simulations.")

    with col_output:
        if run_mc:
            audit(
                "simulation.run_financial",
                {"supplier": selected_supplier, "iterations": mc_iters, "annual_spend": annual_spend},
            )
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
            "Hidden Cost %": ((copq + delivery_cost) / direct_cost * 100) if direct_cost > 0 else 0,
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
# TAB 7: DATA UPLOAD
# ═══════════════════════════════════════════════════════════════════

with tab_upload:
    st.markdown("#### 📁 Upload Your Supplier Data")
    st.markdown(
        "Replace the demo data with your real supplier portfolio. "
        "Upload an Excel or CSV file — the system automatically maps your columns."
    )
    col_dl, col_spacer = st.columns([1, 2])
    with col_dl:
        try:
            template_bytes = generate_sample_template()
            st.download_button(
                label="⬇️ Download Excel Template",
                data=template_bytes,
                file_name="supplier_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception:
            st.info("Install openpyxl for template download: pip install openpyxl")

    st.markdown("---")
    col_upload, col_info = st.columns([1.2, 1])
    with col_upload:
        if not can_mutate():
            st.info("Viewer role is read-only. Ask an admin or analyst to upload/activate supplier data.")
        uploaded_file = st.file_uploader(
            "Upload Supplier Data",
            type=["xlsx", "xls", "csv"],
            help="Column names are auto-detected — exact match not required.",
            disabled=not can_mutate(),
        )
        if uploaded_file is not None:
            with st.spinner("Processing your file..."):
                file_bytes = uploaded_file.read()
                result = ingest_file(file_bytes, uploaded_file.name)
            if result.success:
                st.success(f"✅ Loaded **{result.row_count} suppliers** from {uploaded_file.name}")
                for warning in result.warnings:
                    st.warning(f"⚠️ {warning}")
                with st.expander("📋 Column Mapping Report", expanded=True):
                    st.markdown("**Mapped Columns:**")
                    for raw_col, canonical in result.column_mapping.items():
                        st.markdown(f"  `{raw_col}` → `{canonical}`")
                    if result.unmapped_columns:
                        st.markdown("**Excluded (unmapped):**")
                        for col in result.unmapped_columns:
                            st.markdown(f"  `{col}`")
                preview_cols = [c for c in [
                    "supplier_name", "country", "tier", "unit_cost",
                    "quality_score", "on_time_delivery_pct", "defect_rate_pct", "annual_spend"
                ] if c in result.df.columns]
                st.dataframe(
                    result.df[preview_cols].head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "unit_cost":            st.column_config.NumberColumn(format="$%.2f"),
                        "annual_spend":         st.column_config.NumberColumn(format="$%,.0f"),
                        "quality_score":        st.column_config.ProgressColumn(min_value=0, max_value=100),
                        "on_time_delivery_pct": st.column_config.NumberColumn(format="%.1f%%"),
                        "defect_rate_pct":      st.column_config.NumberColumn(format="%.2f%%"),
                    }
                )
                col_act, col_cancel = st.columns(2)
                with col_act:
                    if st.button("🚀 Activate This Dataset", type="primary", use_container_width=True):
                        st.session_state.uploaded_supplier_df = result.df
                        st.session_state.using_uploaded_data = True
                        upload_id = save_supplier_upload(
                            current_user()["username"],
                            uploaded_file.name,
                            result.df,
                        )
                        audit(
                            "data.activate_upload",
                            {
                                "upload_id": upload_id,
                                "filename": uploaded_file.name,
                                "row_count": result.row_count,
                            },
                        )
                        st.success("✅ Dataset activated!")
                        st.rerun()
                with col_cancel:
                    if st.session_state.using_uploaded_data:
                        if st.button("↩️ Revert to Demo Data", use_container_width=True):
                            st.session_state.using_uploaded_data = False
                            st.session_state.uploaded_supplier_df = None
                            audit("data.reset_to_demo", {"source": "upload_tab"})
                            st.rerun()
            else:
                audit("data.upload_failed", {"filename": uploaded_file.name, "errors": result.errors})
                for err in result.errors:
                    st.error(f"❌ {err}")

    with col_info:
        st.markdown("#### 📖 Supported Columns")
        st.markdown("Auto-detected from your headers — no exact match needed.")
        col_groups = {
            "✅ Required": {"Supplier Name": "supplier, vendor, company, manufacturer"},
            "📊 Performance": {
                "Quality Score (0–100)":  "quality_rating, audit_score, qms_score",
                "On-Time Delivery %":     "otd, delivery_rate, on_time",
                "Defect Rate %":          "defects, ppm, reject_rate, scrap_rate",
                "Lead Time (days)":       "lead_time, turnaround_days",
            },
            "💰 Financial": {
                "Unit Cost ($)":    "price, unit_price, cost_per_unit",
                "Annual Spend ($)": "spend, total_spend, purchase_value",
                "Annual Volume":    "quantity, units_per_year, order_volume",
            },
            "⚠️ Risk": {
                "Risk Score (0–100)":       "risk_rating, risk_index",
                "Financial Health (0–1)":   "credit_score, financial_stability",
                "Geopolitical Risk (0–1)":  "geo_risk, political_risk",
                "Tariff Exposure (0–1)":    "tariff_risk, trade_risk",
            },
            "ℹ️ Info": {
                "Country":        "nation, region, geography, location",
                "Tier (1/2/3)":   "supplier_tier, level",
                "Category":       "commodity, product_type, segment",
                "Certifications": "certs, certificates, compliance",
            },
        }
        for group, cols in col_groups.items():
            with st.expander(group, expanded=(group == "✅ Required")):
                for col_name, aliases in cols.items():
                    st.markdown(f"**{col_name}**")
                    st.caption(aliases)

    st.markdown("---")
    st.markdown("#### Current Dataset")
    if st.session_state.using_uploaded_data and st.session_state.uploaded_supplier_df is not None:
        df_active = st.session_state.uploaded_supplier_df
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Suppliers", len(df_active))
        if "country" in df_active.columns:
            c2.metric("Countries", df_active["country"].nunique())
        if "tier" in df_active.columns:
            c3.metric("Tiers", df_active["tier"].nunique())
        if "annual_spend" in df_active.columns:
            c4.metric("Total Spend", f"${df_active['annual_spend'].sum():,.0f}")
    else:
        st.info("📊 Using demo data. Upload your file above to activate real data.")


# ═══════════════════════════════════════════════════════════════════
# TAB 8: SENTINEL AGENT
# ═══════════════════════════════════════════════════════════════════

with tab_sentinel:
    st.markdown("#### 📡 Sentinel Agent — Supply Chain News Intelligence")
    st.markdown(
        "Monitors supply chain disruptions and maps them to your portfolio. "
        "Use **Live News + AI** for real articles classified by OpenAI or Claude, "
        "**NewsAPI Rules** for local rule-based matching, or **Demo** for no-key demos."
    )

    with st.expander("ℹ️ How the Sentinel Agent works (read if confused)", expanded=False):
        st.markdown("""
        **Why these modes?**
        
        Streamlit Cloud's network proxy **blocks external news APIs** (NewsAPI, Reuters, BBC, etc.).
        This is a Streamlit Cloud infrastructure restriction, not a bug in the code.
        
        | Mode | Requires | Works on Streamlit Cloud? | What you get |
        |------|----------|--------------------------|-------------|
        | Live News + AI | NewsAPI + OpenAI/Anthropic key | Depends on host egress | Real articles classified by AI; supplier matching stays local |
        | NewsAPI Rules | NewsAPI key | Depends on host egress | Real articles with local keyword matching |
        | AI Scenario Briefing | Anthropic API key | ✅ YES | Synthetic planning scenarios, not verified live news |
        | 🎯 Demo | Nothing | ✅ YES | 8 realistic pre-built scenarios |
        
        **Recommendation:** For a company deployment, use **Live News + AI** with secrets
        configured server-side. This gives real-time articles while keeping uploaded supplier
        rows out of external LLM prompts.
        """)

    col_cfg, col_res = st.columns([1, 1.8])

    with col_cfg:
        st.markdown("**Mode Selection**")
        sentinel_mode = st.radio(
            "Scan Mode",
            ["Live News + AI", "NewsAPI Rules", "AI Scenario Briefing", "Demo Mode"],
            help="Live News + AI fetches real articles, then OpenAI or Claude classifies public article text.",
        )
        st.markdown("---")
        anthropic_key_sentinel = ""
        openai_key_sentinel = ""
        news_api_key = ""
        llm_provider = "anthropic"
        llm_model = ""
        env_news_key = get_secret_or_env("NEWSAPI_KEY")
        env_openai_key = get_secret_or_env("OPENAI_API_KEY")
        env_anthropic_key = get_secret_or_env("ANTHROPIC_API_KEY")

        if sentinel_mode == "Live News + AI":
            st.markdown("**API Configuration**")
            llm_provider_label = st.selectbox("LLM provider", ["OpenAI", "Anthropic"])
            llm_provider = llm_provider_label.lower()
            typed_news_key = st.text_input(
                "NewsAPI key",
                type="password",
                placeholder="Using NEWSAPI_KEY from env/secrets" if env_news_key else "Get a key at newsapi.org",
            )
            news_api_key = typed_news_key or env_news_key
            if llm_provider == "openai":
                typed_openai_key = st.text_input(
                    "OpenAI API key",
                    type="password",
                    placeholder="Using OPENAI_API_KEY from env/secrets" if env_openai_key else "sk-...",
                )
                openai_key_sentinel = typed_openai_key or env_openai_key
                llm_model = st.text_input("OpenAI model override (optional)", value=get_secret_or_env("OPENAI_MODEL"))
            else:
                typed_anthropic_key = st.text_input(
                    "Anthropic API key",
                    type="password",
                    placeholder="Using ANTHROPIC_API_KEY from env/secrets" if env_anthropic_key else "sk-ant-...",
                )
                anthropic_key_sentinel = typed_anthropic_key or env_anthropic_key
                llm_model = st.text_input("Claude model override (optional)", value=get_secret_or_env("ANTHROPIC_MODEL"))
            if news_api_key and (openai_key_sentinel or anthropic_key_sentinel):
                st.success("Ready for real-time NewsAPI + AI analysis")
            else:
                st.warning("Add NewsAPI plus one LLM key, or configure them in .env / Streamlit secrets.")
            st.caption("Privacy-safe path: only public article text goes to the LLM; uploaded supplier rows stay local.")
        elif sentinel_mode == "AI Scenario Briefing":
            st.markdown("**🔑 Anthropic API Key**")
            typed_anthropic_key = st.text_input(
                "Anthropic API Key",
                type="password",
                placeholder="Using ANTHROPIC_API_KEY from env/secrets" if env_anthropic_key else "sk-ant-...",
                help="Claude generates synthetic portfolio-specific scenarios. Use Live News + AI for real articles.",
                label_visibility="collapsed",
            )
            anthropic_key_sentinel = typed_anthropic_key or env_anthropic_key
            if not anthropic_key_sentinel:
                st.warning("Enter an Anthropic API key or configure ANTHROPIC_API_KEY")
            else:
                st.success("✅ Ready to generate AI intelligence briefing")
        elif sentinel_mode == "NewsAPI Rules":
            st.markdown("**🔑 NewsAPI Key**")
            typed_news_key = st.text_input(
                "NewsAPI Key",
                type="password",
                placeholder="Using NEWSAPI_KEY from env/secrets" if env_news_key else "Get free key at newsapi.org",
                label_visibility="collapsed",
            )
            news_api_key = typed_news_key or env_news_key
            st.info("Fetches real articles and uses local rule-based matching. Add an LLM key and choose Live News + AI for better classification.")
        else:
            st.info("🎯 Demo mode: No API key needed. Shows 8 realistic supply chain scenarios tailored to your portfolio.")
        st.markdown("---")
        st.markdown("**🔍 Settings**")
        n_events = st.slider("Number of intelligence events", 5, 20, 10)
        custom_query = st.text_input(
            "Focus area (optional)",
            placeholder="e.g. tariffs, semiconductor, Mexico",
        )
        if st.session_state.get("using_uploaded_data") and st.session_state.get("uploaded_supplier_df") is not None:
            n_sup = len(st.session_state.uploaded_supplier_df)
            n_countries = st.session_state.uploaded_supplier_df["country"].nunique() if "country" in st.session_state.uploaded_supplier_df.columns else "?"
            st.success(f"✅ Analyzing {n_sup} real suppliers across {n_countries} countries")
        else:
            st.info("📊 Using demo supplier portfolio. Upload real data for better matching.")
        can_run = (
            (sentinel_mode == "Live News + AI" and bool(news_api_key) and bool(openai_key_sentinel or anthropic_key_sentinel)) or
            (sentinel_mode == "AI Scenario Briefing" and bool(anthropic_key_sentinel)) or
            (sentinel_mode == "NewsAPI Rules" and bool(news_api_key)) or
            (sentinel_mode == "Demo Mode")
        )
        can_run = can_run and can_mutate()
        run_scan = st.button("📡 Run Sentinel Scan", type="primary", use_container_width=True, disabled=not can_run)
        if not can_mutate():
            st.caption("Viewer role can inspect saved intelligence but cannot run new scans.")
        if st.session_state.get("sentinel_results"):
            st.markdown("---")
            st.markdown("**🔽 Filter Results**")
            show_severities = st.multiselect("Severity levels", ["critical", "high", "medium", "low"], default=["critical", "high", "medium"])
            show_matched_only = st.checkbox("Only supplier-matched events", False)

    with col_res:
        if run_scan:
            if st.session_state.get("using_uploaded_data") and st.session_state.get("uploaded_supplier_df") is not None:
                scan_df = st.session_state.uploaded_supplier_df.copy()
            else:
                demo_raw = get_scorecard_data()
                scan_df = demo_raw.rename(columns={"Supplier": "supplier_name", "Country": "country", "Category": "category", "Unit_Cost": "unit_cost", "Annual_Volume": "annual_volume"})
                scan_df["annual_spend"] = scan_df["unit_cost"] * scan_df["annual_volume"]
            mode_map = {
                "Live News + AI": "live_ai",
                "NewsAPI Rules": "newsapi",
                "AI Scenario Briefing": "ai",
                "Demo Mode": "demo",
            }
            chosen_mode = mode_map.get(sentinel_mode, "demo")
            with st.spinner("🔍 Running Sentinel scan..."):
                impacts, mode_used, error_msg = run_sentinel_scan(
                    news_api_key=news_api_key,
                    supplier_df=scan_df,
                    anthropic_api_key=anthropic_key_sentinel,
                    openai_api_key=openai_key_sentinel,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    max_articles=n_events,
                    custom_query=custom_query,
                    mode=chosen_mode,
                )
                st.session_state.sentinel_results = impacts
                st.session_state.sentinel_mode_used = mode_used
                st.session_state.sentinel_error = error_msg
                st.session_state.last_scan_time = datetime.utcnow()
                scan_id = save_sentinel_scan(current_user()["username"], mode_used, impacts)
                audit(
                    "sentinel.run_scan",
                    {
                        "scan_id": scan_id,
                        "mode": mode_used,
                        "event_count": len(impacts),
                        "error": error_msg,
                    },
                )
            if error_msg:
                st.warning(error_msg)
            if impacts:
                st.success(f"✅ {len(impacts)} intelligence events generated — **{mode_used}**")
            else:
                st.error("No events returned. Check your API key or try Demo Mode.")

        results = st.session_state.get("sentinel_results", [])
        mode_used_display = st.session_state.get("sentinel_mode_used", "")

        if results:
            if st.session_state.get("last_scan_time"):
                scan_time = st.session_state.last_scan_time.strftime("%Y-%m-%d %H:%M UTC")
                st.caption(f"Last scan: {scan_time} · Mode: {mode_used_display}")
            n_critical = sum(1 for r in results if str(r.severity).lower() == "critical")
            n_high = sum(1 for r in results if str(r.severity).lower() == "high")
            n_matched = sum(1 for r in results if r.affected_suppliers)
            total_exp = sum(r.estimated_exposure_usd for r in results)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Events Analyzed", len(results))
            mc2.metric("Critical / High", f"{n_critical} / {n_high}", delta=f"{n_critical} critical" if n_critical else None, delta_color="inverse" if n_critical > 0 else "off")
            mc3.metric("Supplier Matches", n_matched)
            mc4.metric("Est. Total Exposure", f"${total_exp:,.0f}" if total_exp > 0 else "–")
            if "Demo" in mode_used_display:
                st.info("🎯 **Demo mode** — these are example events to illustrate platform capabilities, not real-time news.")
            elif "Live News + AI" in mode_used_display:
                st.info("**Live News + AI** — real NewsAPI articles classified by the selected LLM; supplier matching stayed local.")
            elif "AI" in mode_used_display:
                st.info("🤖 **AI Scenario Briefing** — Claude synthesized scenario-style intelligence for planning, not verified live news.")
            st.markdown("---")
            sev_filter = locals().get("show_severities", ["critical", "high", "medium", "low"])
            match_filter = locals().get("show_matched_only", False)
            filtered = [r for r in results if str(r.severity).lower() in sev_filter and (not match_filter or r.affected_suppliers)]
            if not filtered:
                st.info("No events match current filters.")
            else:
                st.markdown(f"Showing **{len(filtered)}** of {len(results)} events")
                for i, impact in enumerate(filtered):
                    sev_icon = SEVERITY_ICONS.get(impact.severity, "⚪")
                    dis_icon = DISRUPTION_ICONS.get(impact.disruption_type, "📦")
                    title_short = impact.article.title[:80] + ("..." if len(impact.article.title) > 80 else "")
                    impact_severity = str(impact.severity).lower()
                    with st.expander(f"{sev_icon} {dis_icon} {title_short}", expanded=(i < 2 and impact_severity in ["critical", "high"])):
                        m1, m2, m3, m4 = st.columns(4)
                        m1.markdown(f"**Severity**  \n{sev_icon} {impact_severity}")
                        m2.markdown(f"**Type**  \n{dis_icon} {impact.disruption_type}")
                        src_clean = impact.article.source.replace('DEMO — ', '').replace('AI Intelligence Briefing (', '').rstrip(')')
                        m3.markdown(f"**Source**  \n{src_clean}")
                        m4.markdown(f"**Date**  \n{impact.article.published_at.strftime('%b %d, %Y')}")
                        st.markdown("---")
                        if impact.affected_suppliers:
                            names = ", ".join(impact.affected_suppliers[:5])
                            extra = f" +{len(impact.affected_suppliers)-5} more" if len(impact.affected_suppliers) > 5 else ""
                            st.markdown(f"**🏭 Affected Suppliers:** {names}{extra}")
                        else:
                            st.markdown("**🏭 Affected Suppliers:** None directly matched in portfolio")
                        if impact.affected_countries:
                            st.markdown(f"**🌍 Countries:** {', '.join(impact.affected_countries[:6])}")
                        if impact.affected_categories:
                            st.markdown(f"**📦 Categories:** {', '.join(impact.affected_categories[:4])}")
                        if impact.estimated_exposure_usd > 0:
                            st.markdown(f"**💰 Estimated Exposure:** ${impact.estimated_exposure_usd:,.0f}")
                        st.markdown(f"**📝 Analysis:** {impact.summary}")
                        if impact.recommended_actions:
                            st.markdown("**✅ Recommended Actions:**")
                            for j, action in enumerate(impact.recommended_actions[:5], 1):
                                st.markdown(f"{j}. {action}")
                        col_link, col_conf = st.columns([2, 1])
                        with col_link:
                            if impact.article.url and impact.article.url != "#":
                                st.markdown(f"[🔗 Read Full Article]({impact.article.url})")
                        with col_conf:
                            st.caption(f"Confidence: {impact.confidence} · {impact.analysis_method}")
            st.markdown("---")
            export_rows = [{
                "Title": r.article.title, "Source": r.article.source,
                "Date": r.article.published_at.strftime("%Y-%m-%d"),
                "Type": r.disruption_type, "Severity": r.severity, "Score": r.severity_score,
                "Affected Suppliers": "; ".join(r.affected_suppliers),
                "Countries": "; ".join(r.affected_countries),
                "Est. Exposure ($)": r.estimated_exposure_usd,
                "Summary": r.summary,
                "Top Action": r.recommended_actions[0] if r.recommended_actions else "",
                "Mode": r.analysis_method,
            } for r in results]
            st.download_button(
                "⬇️ Export Sentinel Report (CSV)",
                data=pd.DataFrame(export_rows).to_csv(index=False).encode(),
                file_name=f"sentinel_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                disabled=not can_mutate(),
                on_click=audit,
                args=("report.export_sentinel_csv", {"row_count": len(export_rows), "mode": mode_used_display}),
            )
            if not can_mutate():
                st.caption("Viewer role cannot export reports.")
        elif not run_scan:
            st.markdown(
                "<div style='text-align:center;padding:60px 20px;color:#475569;'>"
                "<div style='font-size:3rem;'>📡</div>"
                "<div style='font-size:1.1rem;font-weight:500;margin-top:12px;'>Sentinel Agent Ready</div>"
                "<div style='font-size:0.85rem;margin-top:10px;line-height:1.6;'>"
                "Select a mode on the left and click <b>Run Sentinel Scan</b>.<br><br>"
                "No API key? Choose <b>Demo Mode</b> to see example intelligence events."
                "</div></div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════
# ADMIN & AUDIT
# ═══════════════════════════════════════════════════════════════════

with tab_evidence:
    st.markdown("#### Supplier Risk Evidence Chain")
    st.caption(
        "Deterministic pilot view connecting weak signals to explainable risk drivers, "
        "supplier-level scores, and practical mitigation actions."
    )

    evidence_api_url = get_secret_or_env("SUPPLIER_API_BASE_URL", "")
    evidence_reports = []
    if not evidence_api_url:
        st.caption("Local demo evidence is active. Configure SUPPLIER_API_BASE_URL for tenant-backed evidence.")
        scenario_profiles = load_demo_scenario_profiles()
        scenario_by_label = {
            f"{item['supplier_name']} - {item['scenario'].replace('_', ' ').title()}": item
            for item in scenario_profiles
        }
        selected_scenario_label = st.selectbox(
            "Demo scenario",
            ["Portfolio overview", *scenario_by_label],
            key="evidence_demo_scenario",
        )
        if selected_scenario_label != "Portfolio overview":
            scenario = scenario_by_label[selected_scenario_label]
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Supplier", scenario["supplier_name"])
            sc2.metric("Scenario Risk", f"{scenario['risk_score']}/100")
            sc3.metric("Category", scenario["category"])
            st.info(scenario["story"])

    if evidence_api_url and STREAMLIT_API_CLIENT_AVAILABLE:
        try:
            active_evidence_context = enterprise_context()
            evidence_api = StreamlitApiClient.from_env(
                evidence_api_url,
                tenant_id=active_evidence_context.tenant_id,
            )
            connector_catalog = evidence_api.request("/evidence/connectors")
            connector_syncs = evidence_api.request("/evidence/connectors/syncs")
            recent_api_signals = evidence_api.request("/evidence/signals")
            evidence_runs = evidence_api.request("/evidence/runs")
            if evidence_runs:
                latest_run = evidence_api.request(f"/evidence/runs/{evidence_runs[0]['run_id']}")
                evidence_reports = list(latest_run.get("suppliers") or [])

            st.markdown("##### Connector Operations")
            co1, co2, co3 = st.columns(3)
            co1.metric("Connector Mode", str(connector_catalog.get("mode", "unknown")).title())
            co2.metric(
                "Last Sync",
                connector_syncs[0]["status"].title() if connector_syncs else "Not Run",
            )
            co3.metric("Ingested Signals", len(recent_api_signals))

            selected_connector = st.selectbox(
                "Connector",
                ["news", "filings", "hiring"],
                key="evidence_connector_name",
            )
            if st.button(
                "Run Connector Sync",
                disabled=not can_mutate(),
                key="evidence_connector_sync",
            ):
                sync_result = evidence_api.request(
                    f"/evidence/connectors/{selected_connector}/sync",
                    method="POST",
                )
                if sync_result.get("status") == "completed":
                    st.success(
                        f"{selected_connector.title()} sync completed with "
                        f"{sync_result.get('records_accepted', 0)} accepted signals."
                    )
                else:
                    st.warning(
                        f"{selected_connector.title()} sync {sync_result.get('status', 'degraded')}: "
                        f"{sync_result.get('error', 'No records accepted.')}"
                    )
                st.rerun()

            if connector_syncs:
                st.dataframe(
                    pd.DataFrame(connector_syncs[:10])[
                        [
                            "source_system",
                            "status",
                            "records_accepted",
                            "started_at",
                            "finished_at",
                            "error",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            if recent_api_signals:
                st.markdown("##### Recently Ingested Signals")
                st.dataframe(
                    pd.DataFrame(recent_api_signals[:10])[
                        [
                            "supplier_name",
                            "signal_type",
                            "driver",
                            "source",
                            "observed_at",
                            "severity",
                            "confidence",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            if st.button(
                "Run Evidence Chain",
                disabled=not can_mutate(),
                key="evidence_run_chain",
            ):
                run_result = evidence_api.request(
                    "/evidence/runs",
                    method="POST",
                    payload={"include_demo_signals": False},
                )
                evidence_reports = list(run_result.get("suppliers") or [])
                st.success(
                    f"Evidence run completed for {len(evidence_reports)} suppliers "
                    f"with scoring version {run_result.get('scoring_version', 'default-v1')}."
                )
        except Exception as exc:
            st.error(friendly_api_error(exc))

    if not evidence_api_url:
        evidence_reports = get_supplier_risk_evidence_chain_reports()
    elif not evidence_reports:
        st.info(
            "No persisted evidence run is available for the active tenant. "
            "Ingest signals, then run the evidence chain."
        )
    total_signals = sum(len(report["evidence_chain"]) for report in evidence_reports)
    high_or_worse = sum(1 for report in evidence_reports if report["risk_level"] in {"critical", "high"})
    avg_confidence = (
        sum(report["confidence"] for report in evidence_reports) / len(evidence_reports)
        if evidence_reports
        else 0
    )

    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.metric("Suppliers", len(evidence_reports))
    ec2.metric("Weak Signals", total_signals)
    ec3.metric("Critical / High", high_or_worse)
    ec4.metric("Avg Confidence", f"{avg_confidence:.0%}")

    summary_rows = [
        {
            "Supplier": report["supplier_name"],
            "Risk Score": report["risk_score"],
            "Risk Level": report["risk_level"].title(),
            "Top Driver": report["top_risk_drivers"][0]["driver"] if report["top_risk_drivers"] else "",
            "Confidence": f"{report['confidence']:.0%}",
        }
        for report in evidence_reports
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    chart_df = pd.DataFrame(
        [
            {
                "supplier_name": report["supplier_name"],
                "risk_score": report["risk_score"],
                "risk_level": report["risk_level"],
            }
            for report in evidence_reports
        ]
    )
    if not chart_df.empty:
        chart = px.bar(
            chart_df.sort_values("risk_score"),
            x="risk_score",
            y="supplier_name",
            orientation="h",
            color="risk_level",
            color_discrete_map={
                "critical": "#ef4444",
                "high": "#f59e0b",
                "medium": "#3b82f6",
                "low": "#10b981",
            },
            labels={"risk_score": "Risk score", "supplier_name": ""},
        )
        chart.update_layout(height=320, margin=dict(l=8, r=8, t=10, b=8), showlegend=True)
        st.plotly_chart(chart, use_container_width=True)

    for report in evidence_reports:
        title = f"{report['supplier_name']} - {report['risk_level'].title()} ({report['risk_score']:.1f})"
        with st.expander(title, expanded=report["risk_level"] in {"critical", "high"}):
            r1, r2, r3 = st.columns(3)
            r1.metric("Risk Score", f"{report['risk_score']:.1f}/100")
            r2.metric("Risk Level", report["risk_level"].title())
            r3.metric("Confidence", f"{report['confidence']:.0%}")

            st.markdown("**Top Risk Drivers**")
            driver_rows = [
                {
                    "Driver": driver["driver"],
                    "Contribution": driver["contribution"],
                    "Signal IDs": ", ".join(driver["signal_ids"]),
                    "Sources": ", ".join(driver["sources"]),
                }
                for driver in report["top_risk_drivers"]
            ]
            st.dataframe(pd.DataFrame(driver_rows), use_container_width=True, hide_index=True)

            st.markdown("**Evidence Chain**")
            for evidence in report["evidence_chain"]:
                st.markdown(
                    f"- **{evidence['driver']}** from `{evidence['source']}` "
                    f"(`{evidence['signal_id']}`, {evidence['signal_type']}, "
                    f"severity {evidence['severity']}, confidence {evidence['confidence']:.0%}): "
                    f"{evidence['summary']}"
                )

            st.markdown("**Recommended Actions**")
            for action in report["recommended_actions"]:
                st.markdown(f"- {action}")

    st.download_button(
        "Export Evidence Chain JSON",
        data=json.dumps(evidence_reports, indent=2).encode(),
        file_name=f"supplier_risk_evidence_chain_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
        disabled=not can_mutate(),
        on_click=audit,
        args=("report.export_evidence_chain_json", {"supplier_count": len(evidence_reports)}),
    )
    if not can_mutate():
        st.caption("Viewer role can inspect evidence chains but cannot export reports.")


with tab_command:
    st.markdown("#### Production Command Center")
    st.caption("Operational view for API, database, monitoring, security, and data freshness.")

    if not PRODUCTION_FOUNDATION_AVAILABLE or production_session_factory is None or production_settings is None:
        st.error("Production foundation modules are unavailable. Check backend dependencies and logs.")
    else:
        active_context = enterprise_context()
        status = get_production_system_status(production_settings, production_session_factory, active_context.tenant_id)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Database", "Online" if status["database"]["ok"] else "Degraded", status["database"]["driver"])
        c2.metric("API", status["api"]["status"].title())
        c3.metric("Security Mode", status["security_mode"].title(), active_context.role)
        c4.metric("Open Alerts", status["open_alerts"])

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Monitored Suppliers", status["monitored_suppliers"] or len(risk_df))
        c6.metric("Demo Mode", "On" if production_settings.demo_mode else "Off")
        c7.metric("Scheduler", "Enabled" if status["worker"]["enabled"] else "Disabled", status["worker"]["mode"])
        c8.metric("Sentinel API", "Configured" if status["sentinel"]["configured"] else "Demo/Fallback")

        c9, c10, c11, c12 = st.columns(4)
        c9.metric("Worker Mode", status["worker_mode"]["active"].upper(), status["worker_mode"]["requested"])
        c10.metric("Rate Limit", "On" if status["rate_limit"]["enabled"] else "Off", f"{status['rate_limit']['requests']}/window")
        c11.metric("Secrets", status["secrets"]["provider"].upper(), status["secrets"]["kms_provider"])
        c12.metric("Retention", "On" if status["retention"]["enabled"] else "Dry-run")

        if status["production_issues"]:
            for issue in status["production_issues"]:
                st.warning(issue)

        configured_api_url = get_secret_or_env("SUPPLIER_API_BASE_URL", "")
        if configured_api_url:
            try:
                api_client = StreamlitApiClient.from_env(configured_api_url)
                api_health = api_client.request("/health")
                st.success(f"Configured API reachable: {api_health.get('status', 'unknown')}")
            except Exception as exc:
                st.error(friendly_api_error(exc))

        st.markdown("##### Monitoring Jobs")
        col_job1, col_job2, col_job3 = st.columns(3)
        scheduler = LocalJobScheduler(production_settings, production_session_factory)
        if col_job1.button("Run Sentinel Scan", use_container_width=True, disabled=not can_mutate()):
            result = scheduler.run_job_now("sentinel_scan", active_context.tenant_id)
            st.success(f"Sentinel job {result.status}.")
            st.rerun()
        if col_job2.button("Recalculate Risk", use_container_width=True, disabled=not can_mutate()):
            result = scheduler.run_job_now("risk_recalculate", active_context.tenant_id)
            st.success(f"Risk job {result.status}.")
            st.rerun()
        if col_job3.button("Recalculate Exposure", use_container_width=True, disabled=not can_mutate()):
            result = scheduler.run_job_now("exposure_recalculate", active_context.tenant_id)
            st.success(f"Exposure job {result.status}.")
            st.rerun()
        if not can_mutate():
            st.caption("Viewer role can inspect system status but cannot run jobs.")

        st.markdown("##### Production Queue Tasks")
        tq1, tq2, tq3 = st.columns(3)
        task_runner = EnterpriseTaskRunner(production_settings, production_session_factory)
        if tq1.button("Run Retention Dry Run", use_container_width=True, disabled=not can_mutate()):
            result = task_runner.run_task("retention_cleanup_task", active_context.tenant_id)
            st.success(f"Retention task {result.status}.")
            st.rerun()
        if tq2.button("Run Audit Export Task", use_container_width=True, disabled=not can_mutate()):
            result = task_runner.run_task("audit_export_task", active_context.tenant_id)
            st.success(f"Audit export task {result.status}.")
            st.rerun()
        if tq3.button("Record Backup Metadata", use_container_width=True, disabled=not can_mutate()):
            result = task_runner.run_task("backup_metadata_task", active_context.tenant_id)
            st.success(f"Backup metadata task {result.status}.")
            st.rerun()

        st.markdown("##### Data Source Status")
        st.dataframe(
            pd.DataFrame(
                [
                    {"source": "Supplier database", "status": "online" if status["database"]["ok"] else "degraded"},
                    {"source": "FastAPI backend", "status": status["api"]["status"]},
                    {"source": "Sentinel NewsAPI", "status": "configured" if production_settings.newsapi_key else "missing key"},
                    {
                        "source": "LLM classifier",
                        "status": "configured" if (production_settings.openai_api_key or production_settings.anthropic_api_key) else "rule/demo fallback",
                    },
                    {"source": "Streamlit command center", "status": "online"},
                    {"source": "Redis / Celery", "status": "configured" if status["worker_mode"]["available"] else "local fallback"},
                    {"source": "SIEM export", "status": "configured" if status["siem"]["configured"] else "not configured"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(f"Last successful Sentinel scan: {status['worker'].get('last_successful_sentinel_scan') or 'not yet run'}")
        if status["last_failed_job"]:
            st.error(f"Last failed job: {status['last_failed_job']['job_name']} - {status['last_failed_job']['error']}")


with tab_alerts:
    st.markdown("#### Alerts & System Health")
    st.caption("Supplier risk, Sentinel, ingestion, API, and background job alerts.")
    if not PRODUCTION_FOUNDATION_AVAILABLE or production_session_factory is None:
        st.info("Production alert repository is unavailable.")
    else:
        with production_session_factory() as prod_session:
            active_context = enterprise_context()
            repo = ProductionAlertRepository(prod_session, active_context.tenant_id)
            alerts = repo.list(limit=300)
            open_alerts = [alert for alert in alerts if alert.status == "open"]

            a1, a2, a3 = st.columns(3)
            a1.metric("Open Alerts", len(open_alerts))
            a2.metric("High/Critical", sum(1 for alert in open_alerts if alert.severity in {"high", "critical"}))
            a3.metric("Total Exposure", f"${sum(alert.exposure or 0 for alert in open_alerts):,.0f}")

            if alerts:
                alert_df = pd.DataFrame([ProductionAlertRepository.to_dict(alert) for alert in alerts])
                st.dataframe(alert_df, use_container_width=True, hide_index=True)
                if can_mutate():
                    open_ids = [alert.id for alert in open_alerts]
                    if open_ids:
                        selected_alert = st.selectbox("Acknowledge alert", open_ids)
                        if st.button("Acknowledge Selected Alert", type="primary"):
                            repo.acknowledge(int(selected_alert), actor=current_user()["username"])
                            prod_session.commit()
                            audit("alerts.acknowledge", {"alert_id": int(selected_alert)})
                            st.rerun()
            else:
                st.info("No production alerts have been created yet.")


if tab_enterprise_admin is not None:
    with tab_enterprise_admin:
        st.markdown("#### Enterprise Admin")
        st.caption("Shared-SaaS tenant administration, access review, API keys, and SOC 2 readiness controls.")
        if not PRODUCTION_FOUNDATION_AVAILABLE or production_session_factory is None or production_settings is None:
            st.error("Enterprise admin is unavailable because production foundation modules did not load.")
        else:
            active_context = enterprise_context()
            with production_session_factory() as prod_session:
                tenant_repo = ProductionTenantRepository(prod_session)
                tenants = tenant_repo.list_tenants()
                current_tenant = tenant_repo.get_tenant(active_context.tenant_id)

                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Active Tenant", active_context.tenant_id)
                e2.metric("Tenant Role", active_context.role)
                e3.metric("Auth Provider", get_secret_or_env("AUTH_PROVIDER", "local").upper())
                e4.metric("Tenants", len(tenants))

                if current_tenant:
                    st.markdown("##### Tenant")
                    st.json(ProductionTenantRepository.tenant_to_dict(current_tenant))

                admin_tabs = st.tabs(
                    [
                        "Memberships",
                        "API Keys",
                        "Access Reviews",
                        "Backup & Retention",
                        "Security Status",
                        "Audit Export",
                        "Compliance",
                    ]
                )

                with admin_tabs[0]:
                    st.markdown("##### Tenant Memberships")
                    memberships = tenant_repo.list_memberships(active_context.tenant_id)
                    st.dataframe(
                        pd.DataFrame([ProductionTenantRepository.membership_to_dict(row) for row in memberships]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    with st.form("enterprise_invite_user"):
                        invite_username = st.text_input("User email / username")
                        invite_role = st.selectbox(
                            "Tenant role",
                            ["viewer", "auditor", "analyst", "risk_manager", "org_admin", "platform_admin"],
                        )
                        invite_submitted = st.form_submit_button("Invite / assign role")
                    if invite_submitted:
                        tenant_repo.create_membership(active_context.tenant_id, invite_username, invite_role)
                        prod_session.commit()
                        audit("enterprise.membership_upsert", {"tenant_id": active_context.tenant_id, "username": invite_username, "role": invite_role})
                        st.success("Membership updated.")
                        st.rerun()

                with admin_tabs[1]:
                    st.markdown("##### Tenant API Keys")
                    st.caption("For security, raw API keys are shown only once after creation.")
                    with st.form("enterprise_create_api_key"):
                        key_username = st.text_input("API key user", value=active_context.username)
                        key_role = st.selectbox("API key role", ["viewer", "auditor", "analyst", "risk_manager", "org_admin"])
                        key_label = st.text_input("Label", value="Streamlit-created key")
                        key_submitted = st.form_submit_button("Create API key")
                    if key_submitted:
                        raw_key = tenant_repo.create_api_key(active_context.tenant_id, key_username, key_role, key_label)
                        prod_session.commit()
                        audit("enterprise.api_key_created", {"tenant_id": active_context.tenant_id, "username": key_username, "role": key_role})
                        st.success(f"New API key: `{raw_key}`")

                with admin_tabs[2]:
                    st.markdown("##### Access Reviews")
                    reviews = tenant_repo.list_access_reviews(active_context.tenant_id)
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "id": row.id,
                                    "reviewer": row.reviewer,
                                    "status": row.status,
                                    "notes": row.notes,
                                    "created_at": row.created_at,
                                }
                                for row in reviews
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    if st.button("Open access review for this tenant"):
                        tenant_repo.create_access_review(active_context.tenant_id, active_context.username, "Quarterly access review opened from demo console.")
                        prod_session.commit()
                        audit("enterprise.access_review_opened", {"tenant_id": active_context.tenant_id})
                        st.rerun()

                with admin_tabs[3]:
                    st.markdown("##### Backup, Retention, and SOC 2 Stubs")
                    st.info(
                        "Configured stubs document backup/restore and retention controls. "
                        "Production should wire these to encrypted Postgres backups and tested restore drills."
                    )
                    if st.button("Record backup check"):
                        tenant_repo.create_backup_run(active_context.tenant_id, status="recorded", location="manual/demo check")
                        prod_session.commit()
                        audit("enterprise.backup_check_recorded", {"tenant_id": active_context.tenant_id})
                        st.rerun()

                with admin_tabs[4]:
                    st.markdown("##### Tenant Security Status")
                    status = get_production_system_status(production_settings, production_session_factory, active_context.tenant_id)
                    schema_status = validate_production_tenant_schema(prod_session)
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"control": "Auth provider", "status": production_settings.auth_provider},
                                {"control": "MFA policy", "status": "required" if production_settings.mfa_required else "not required"},
                                {"control": "SCIM", "status": "enabled" if production_settings.scim_enabled else "placeholder/off"},
                                {"control": "Worker mode", "status": get_worker_mode(production_settings)["active"]},
                                {"control": "Redis", "status": "configured" if production_settings.redis_url else "not configured"},
                                {"control": "Rate limit", "status": "enabled" if production_settings.rate_limit_enabled else "disabled"},
                                {"control": "Secrets provider", "status": production_settings.secrets_provider},
                                {"control": "KMS provider", "status": production_settings.kms_provider},
                                {"control": "Migration schema", "status": "ok" if schema_status["ok"] else "needs review"},
                                {"control": "Retention", "status": "enabled" if production_settings.retention_enabled else "dry-run/off"},
                                {"control": "SIEM sink", "status": production_settings.siem_sink},
                                {"control": "Open alerts", "status": status["open_alerts"]},
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    if schema_status["tables_missing_tenant_id"]:
                        st.error(f"Tables missing tenant_id: {schema_status['tables_missing_tenant_id']}")

                with admin_tabs[5]:
                    st.markdown("##### Audit Export / SIEM Readiness")
                    export_service = ProductionAuditExportService(prod_session, active_context.tenant_id)
                    export_format = st.radio("Export format", ["jsonl", "csv"], horizontal=True)
                    if st.button("Preview tenant audit export"):
                        payload = export_service.export_csv(limit=200) if export_format == "csv" else export_service.export_jsonl(limit=200)
                        st.text_area("Export payload", payload, height=240)
                    st.caption("Production WORM export should target S3 Object Lock, Azure immutable blob, or GCP retention lock.")

                with admin_tabs[6]:
                    st.markdown("##### Compliance Evidence")
                    evidence_service = ProductionEvidenceService(prod_session, active_context.tenant_id)
                    if st.button("Collect access-control evidence"):
                        st.json(evidence_service.collect_access_control_evidence())
                    if st.button("Collect operations evidence"):
                        st.json(evidence_service.collect_operational_evidence())
                    st.caption("This gathers readiness evidence; it is not a claim of SOC 2 compliance.")


if tab_admin is not None:
    with tab_admin:
        st.markdown("#### 🛡️ Admin & Audit")
        st.caption("Pilot controls for users, roles, persistence, and auditability.")

        admin_user = current_user()
        c1, c2, c3 = st.columns(3)
        c1.metric("Database", Path(pilot_db_info["db_path"]).name)
        c2.metric("Users", len(list_users()))
        c3.metric("Saved Sentinel Scans", len(list_sentinel_scans(limit=500)))

        if pilot_db_info["default_password_in_use"]:
            st.warning(
                "Default admin password is still possible unless changed. "
                "Set `SUPPLIER_APP_ADMIN_PASSWORD` before first deployment and rotate this admin password now."
            )

        tab_users, tab_audit, tab_history = st.tabs(["Users", "Audit Log", "Saved Scans"])

        with tab_users:
            st.markdown("##### Create User")
            with st.form("create_user_form"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Temporary password", type="password")
                new_role = st.selectbox("Role", ["analyst", "viewer", "admin"])
                create_submitted = st.form_submit_button("Create user", type="primary")
            if create_submitted:
                try:
                    create_user(new_username, new_password, new_role, admin_user)
                    st.success(f"Created user `{new_username}`.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

            st.markdown("##### Existing Users")
            users_df = pd.DataFrame(list_users())
            if not users_df.empty:
                users_df["is_active"] = users_df["is_active"].astype(bool)
                st.dataframe(users_df, use_container_width=True, hide_index=True)

            with st.expander("Manage Existing User"):
                usernames = [u["username"] for u in list_users()]
                if usernames:
                    target_user = st.selectbox("User", usernames)
                    active = st.checkbox("Active", value=next(u["is_active"] for u in list_users() if u["username"] == target_user) == 1)
                    if st.button("Update active status"):
                        if target_user == admin_user["username"] and not active:
                            st.error("You cannot deactivate your own active admin session.")
                        else:
                            set_user_active(target_user, active, admin_user)
                            st.success("Updated user status.")
                            st.rerun()
                    new_pw = st.text_input("New password", type="password")
                    if st.button("Reset password"):
                        try:
                            change_user_password(target_user, new_pw, admin_user)
                            st.success("Password updated.")
                        except Exception as exc:
                            st.error(str(exc))

        with tab_audit:
            audit_rows = list_audit_logs(limit=300)
            if audit_rows:
                audit_df = pd.DataFrame(audit_rows)
                audit_df["details"] = audit_df["details_json"].apply(lambda x: json.dumps(json.loads(x), indent=0))
                st.dataframe(
                    audit_df[["timestamp", "username", "role", "action", "details"]],
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Export Audit Log (CSV)",
                    data=audit_df.to_csv(index=False).encode(),
                    file_name=f"audit_log_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    on_click=audit,
                    args=("admin.export_audit_log", {"row_count": len(audit_rows)}),
                )
            else:
                st.info("No audit events yet.")

        with tab_history:
            scans = list_sentinel_scans(limit=100)
            if scans:
                scans_df = pd.DataFrame(scans)
                st.dataframe(scans_df, use_container_width=True, hide_index=True)
            else:
                st.info("No saved Sentinel scans yet.")

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
