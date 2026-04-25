"""
data_ingestion.py
=================
Real-world data ingestion for the Supplier Intelligence Platform.

Handles:
- Excel (.xlsx, .xls) and CSV file uploads
- Smart column mapping (fuzzy matching against expected schema)
- Data validation and cleaning
- Schema normalization so downstream models always get consistent data
"""

import pandas as pd
import numpy as np
import io
import re
from datetime import date
from dataclasses import dataclass, field
from typing import Optional


# ─── EXPECTED SCHEMA ──────────────────────────────────────────────
# These are the canonical column names the rest of the app expects.
# The ingestion engine maps whatever the user uploads onto these.

REQUIRED_COLUMNS = {
    "supplier_name": {
        "aliases": [
            "supplier", "supplier_name", "name", "company", "vendor",
            "vendor_name", "company_name", "manufacturer", "partner"
        ],
        "dtype": "str",
        "description": "Supplier company name",
    },
}

OPTIONAL_COLUMNS = {
    "country": {
        "aliases": ["country", "nation", "location", "country_of_origin", "origin", "region", "geography"],
        "dtype": "str",
        "description": "Supplier country",
        "default": "Unknown",
    },
    "tier": {
        "aliases": ["tier", "supplier_tier", "level", "tier_level", "supply_tier"],
        "dtype": "int",
        "description": "Supply chain tier (1, 2, 3)",
        "default": 1,
    },
    "category": {
        "aliases": [
            "category", "product_category", "commodity", "type", "product_type",
            "material", "part_type", "segment", "classification"
        ],
        "dtype": "str",
        "description": "Product/service category",
        "default": "Uncategorized",
    },
    "unit_cost": {
        "aliases": [
            "unit_cost", "cost", "price", "unit_price", "avg_cost",
            "average_cost", "cost_per_unit", "purchase_price", "buy_price"
        ],
        "dtype": "float",
        "description": "Unit purchase cost ($)",
        "default": 0.0,
    },
    "annual_spend": {
        "aliases": [
            "annual_spend", "spend", "total_spend", "yearly_spend",
            "annual_purchase", "total_cost", "purchase_value", "annual_value"
        ],
        "dtype": "float",
        "description": "Annual spend ($)",
        "default": 0.0,
    },
    "annual_volume": {
        "aliases": [
            "annual_volume", "volume", "quantity", "annual_quantity",
            "units_per_year", "order_volume", "annual_units"
        ],
        "dtype": "float",
        "description": "Annual purchase volume (units)",
        "default": 0.0,
    },
    "quality_score": {
        "aliases": [
            "quality_score", "quality", "quality_rating", "qms_score",
            "audit_score", "quality_index", "supplier_quality"
        ],
        "dtype": "float",
        "description": "Quality score (0-100)",
        "default": 70.0,
    },
    "on_time_delivery_pct": {
        "aliases": [
            "on_time_delivery_pct", "on_time_delivery", "otd", "delivery_rate",
            "on_time_rate", "delivery_performance", "schedule_adherence",
            "on_time", "otd_pct", "delivery_pct"
        ],
        "dtype": "float",
        "description": "On-time delivery % (0-100)",
        "default": 85.0,
    },
    "defect_rate_pct": {
        "aliases": [
            "defect_rate_pct", "defect_rate", "defects", "ppm",
            "reject_rate", "quality_defects", "failure_rate", "scrap_rate",
            "defect_pct", "non_conformance_rate"
        ],
        "dtype": "float",
        "description": "Defect rate % (0-100)",
        "default": 2.0,
    },
    "lead_time_days": {
        "aliases": [
            "lead_time_days", "lead_time", "delivery_days",
            "avg_lead_time", "turnaround_days", "cycle_time", "lt_days"
        ],
        "dtype": "float",
        "description": "Lead time (days)",
        "default": 30.0,
    },
    "risk_score": {
        "aliases": [
            "risk_score", "risk", "risk_rating", "risk_index",
            "risk_level_score", "supplier_risk", "overall_risk"
        ],
        "dtype": "float",
        "description": "Risk score (0-100, higher = riskier)",
        "default": 50.0,
    },
    "financial_health": {
        "aliases": [
            "financial_health", "financial_stability", "credit_score",
            "financial_rating", "fiscal_health", "creditworthiness"
        ],
        "dtype": "float",
        "description": "Financial health (0-1, higher = healthier)",
        "default": 0.6,
    },
    "geopolitical_risk": {
        "aliases": [
            "geopolitical_risk", "geo_risk", "political_risk",
            "country_risk", "political_stability_risk"
        ],
        "dtype": "float",
        "description": "Geopolitical risk (0-1, higher = riskier)",
        "default": 0.3,
    },
    "tariff_exposure": {
        "aliases": [
            "tariff_exposure", "tariff_risk", "trade_risk",
            "import_duty_risk", "customs_risk"
        ],
        "dtype": "float",
        "description": "Tariff exposure (0-1, higher = more exposed)",
        "default": 0.3,
    },
    "compliance_score": {
        "aliases": [
            "compliance_score", "regulatory_compliance", "compliance_rating",
            "regulatory_score", "regulatory_adherence_score", "adherence_score"
        ],
        "dtype": "float",
        "description": "Regulatory compliance score (0-1, higher = more compliant)",
        "default": 0.7,
    },
    "dependency_score": {
        "aliases": [
            "dependency_score", "sole_source_risk", "supplier_dependency",
            "dependency", "sole_source", "concentration_risk"
        ],
        "dtype": "float",
        "description": "Supplier dependency / sole-source risk (0-1, higher = more dependent)",
        "default": 0.3,
    },
    "sub_tier_count": {
        "aliases": [
            "sub_tier_count", "tier_2_suppliers", "sub_suppliers",
            "sub_tier_suppliers", "n_sub_tiers", "subtier_count"
        ],
        "dtype": "float",
        "description": "Number of sub-tier (Tier-2) suppliers",
        "default": 3.0,
    },
    "criticality": {
        "aliases": [
            "criticality", "business_criticality", "strategic_importance",
            "supplier_criticality", "risk_category", "risk_classification",
            "importance_level"
        ],
        "dtype": "str",
        "description": "Business criticality classification (e.g. High / Medium / Low)",
        "default": "Medium",
    },
    "certifications": {
        "aliases": [
            "certifications", "certs", "certificates", "certifications_held",
            "quality_certifications"
        ],
        "dtype": "str",
        "description": "Certifications (comma-separated)",
        "default": "",
    },
    "years_in_business": {
        "aliases": [
            "years_in_business", "years", "company_age",
            "established_years", "age", "years_operating"
        ],
        "dtype": "float",
        "description": "Years in business",
        "default": 5.0,
    },
    "contact_email": {
        "aliases": ["email", "contact_email", "supplier_email", "contact"],
        "dtype": "str",
        "description": "Contact email",
        "default": "",
    },
    "website": {
        "aliases": ["website", "url", "web", "site", "homepage"],
        "dtype": "str",
        "description": "Supplier website",
        "default": "",
    },
    "notes": {
        "aliases": ["notes", "comments", "remarks", "description", "additional_info"],
        "dtype": "str",
        "description": "Free-form notes",
        "default": "",
    },
}


@dataclass
class IngestionResult:
    """Result of a data ingestion operation."""
    success: bool
    df: Optional[pd.DataFrame] = None
    column_mapping: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    row_count: int = 0
    mapped_columns: list = field(default_factory=list)
    unmapped_columns: list = field(default_factory=list)
    missing_defaults_applied: list = field(default_factory=list)


def _normalize_col_name(col: str) -> str:
    """Lowercase, strip, replace spaces/dashes with underscores."""
    return re.sub(r'[\s\-\.\/]+', '_', col.strip().lower())


def _fuzzy_match_column(col: str, all_aliases: dict) -> Optional[str]:
    """
    Try to match a raw column name to a canonical column name.
    Returns canonical name or None if no match.
    """
    col_norm = _normalize_col_name(col)

    # 1. Exact match
    for canonical, aliases in all_aliases.items():
        if col_norm in aliases:
            return canonical

    # 2. Substring match — col contains an alias OR alias contains col
    for canonical, aliases in all_aliases.items():
        for alias in aliases:
            if alias in col_norm or col_norm in alias:
                return canonical

    # 3. Partial token match (for multi-word columns)
    col_tokens = set(col_norm.split('_'))
    for canonical, aliases in all_aliases.items():
        alias_tokens = set(' '.join(aliases).replace('_', ' ').split())
        overlap = col_tokens & alias_tokens
        if len(overlap) >= 2 or (len(col_tokens) == 1 and len(overlap) == 1 and len(list(overlap)[0]) > 3):
            return canonical

    return None


def _build_alias_map() -> dict:
    """Build a flat alias → canonical lookup from all column definitions."""
    alias_map = {}
    all_defs = {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS}
    for canonical, meta in all_defs.items():
        alias_map[canonical] = meta["aliases"]
    return alias_map


def _clean_numeric(series: pd.Series, col_name: str) -> pd.Series:
    """
    Clean a numeric column:
    - Strip currency symbols, commas, % signs
    - Convert to float
    - Replace negatives with 0 for columns that can't be negative
    """
    def parse_val(v):
        if pd.isna(v):
            return np.nan
        v = str(v)
        v = re.sub(r'[$,€£¥\s]', '', v)  # Remove currency symbols and spaces
        if v.endswith('%'):
            v = v[:-1]
        try:
            return float(v)
        except ValueError:
            return np.nan

    return series.apply(parse_val)


def _apply_pct_detection(df: pd.DataFrame, col: str) -> pd.Series:
    """
    For percentage columns (OTD, defect rate), normalize mixed formats where
    some rows are stored as decimals (0-1) and others as percentages (0-100).
    """
    series = df[col]
    non_null = series.dropna()
    if len(non_null) == 0:
        return series

    # Values between 0 and 1.5 are almost always decimal percentages and
    # should be converted row-by-row so mixed-format uploads still normalize.
    return series.apply(
        lambda value: value * 100 if pd.notna(value) and 0 <= value <= 1.5 else value
    )


def ingest_file(file_bytes: bytes, filename: str) -> IngestionResult:
    """
    Main ingestion function. Takes raw file bytes and filename.
    Returns IngestionResult with normalized DataFrame.
    """
    result = IngestionResult(success=False)

    # ── 1. Parse raw file ──────────────────────────────────────────
    try:
        ext = filename.lower().split('.')[-1]
        if ext in ('xlsx', 'xls', 'xlsm'):
            # Try first sheet, then let user pick if multiple
            raw_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
        elif ext == 'csv':
            # Try to detect separator
            sample = file_bytes[:2048].decode('utf-8', errors='ignore')
            sep = ',' if sample.count(',') >= sample.count(';') else ';'
            raw_df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
        else:
            result.errors.append(f"Unsupported file type: .{ext}. Upload .xlsx, .xls, or .csv")
            return result

        # Drop completely empty rows/columns
        raw_df.dropna(how='all', inplace=True)
        raw_df.dropna(axis=1, how='all', inplace=True)

        if len(raw_df) == 0:
            result.errors.append("File appears empty after removing blank rows.")
            return result

    except Exception as e:
        result.errors.append(f"Could not parse file: {str(e)}")
        return result

    # ── 2. Column mapping ──────────────────────────────────────────
    alias_map = _build_alias_map()
    column_mapping = {}  # raw_col → canonical_col
    used_canonical = set()

    for raw_col in raw_df.columns:
        canonical = _fuzzy_match_column(str(raw_col), alias_map)
        if canonical and canonical not in used_canonical:
            column_mapping[raw_col] = canonical
            used_canonical.add(canonical)

    result.column_mapping = column_mapping

    # Check required columns are mapped
    required_mapped = {v for v in column_mapping.values() if v in REQUIRED_COLUMNS}
    missing_required = set(REQUIRED_COLUMNS.keys()) - required_mapped
    if missing_required:
        result.errors.append(
            f"Could not find required column(s): {', '.join(missing_required)}. "
            f"Your file needs a column for supplier/company name."
        )
        return result

    # ── 3. Build normalized DataFrame ─────────────────────────────
    out_rows = []
    all_defs = {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS}

    # Reverse map: canonical → raw col
    rev_map = {v: k for k, v in column_mapping.items()}

    for _, raw_row in raw_df.iterrows():
        out_row = {}

        for canonical, meta in all_defs.items():
            if canonical in rev_map:
                raw_col = rev_map[canonical]
                val = raw_row[raw_col]
            else:
                # Use default
                val = meta.get("default", None)
                if canonical not in REQUIRED_COLUMNS:
                    result.missing_defaults_applied.append(canonical)

            # Type coercion
            try:
                if meta["dtype"] == "float":
                    val = float(val) if pd.notna(val) else meta.get("default", 0.0)
                elif meta["dtype"] == "int":
                    val = int(float(val)) if pd.notna(val) else meta.get("default", 1)
                elif meta["dtype"] == "str":
                    val = str(val).strip() if pd.notna(val) else meta.get("default", "")
            except (ValueError, TypeError):
                val = meta.get("default", None)

            out_row[canonical] = val

        out_rows.append(out_row)

    out_df = pd.DataFrame(out_rows)

    # ── 4. Post-processing fixes ───────────────────────────────────

    # Strip currency/% from numeric columns that came through as strings
    for col in ["unit_cost", "annual_spend", "annual_volume",
                "quality_score", "on_time_delivery_pct", "defect_rate_pct",
                "lead_time_days", "risk_score", "financial_health",
                "geopolitical_risk", "tariff_exposure", "years_in_business"]:
        if col in out_df.columns:
            out_df[col] = _clean_numeric(out_df[col], col)

    # Fix percentage detection (0-1 vs 0-100)
    for pct_col in ["on_time_delivery_pct", "defect_rate_pct"]:
        if pct_col in out_df.columns:
            out_df[pct_col] = _apply_pct_detection(out_df, pct_col)

    # Clamp scores to valid ranges
    out_df["quality_score"] = out_df["quality_score"].clip(0, 100)
    out_df["on_time_delivery_pct"] = out_df["on_time_delivery_pct"].clip(0, 100)
    out_df["defect_rate_pct"] = out_df["defect_rate_pct"].clip(0, 100)
    out_df["risk_score"] = out_df["risk_score"].clip(0, 100)

    # Fill NaN unit_cost from annual_spend / annual_volume where possible
    mask = (out_df["unit_cost"] == 0) & (out_df["annual_spend"] > 0) & (out_df["annual_volume"] > 0)
    out_df.loc[mask, "unit_cost"] = out_df.loc[mask, "annual_spend"] / out_df.loc[mask, "annual_volume"]

    # Drop duplicate supplier names (keep first)
    before = len(out_df)
    out_df.drop_duplicates(subset=["supplier_name"], keep='first', inplace=True)
    if len(out_df) < before:
        result.warnings.append(f"Removed {before - len(out_df)} duplicate supplier entries.")

    # Drop rows with empty supplier names
    out_df = out_df[out_df["supplier_name"].str.strip() != ""]
    out_df = out_df[out_df["supplier_name"] != "nan"]
    out_df.reset_index(drop=True, inplace=True)

    # ── 5. Unmapped columns (keep as metadata) ─────────────────────
    mapped_raw = set(column_mapping.keys())
    unmapped_raw = [c for c in raw_df.columns if c not in mapped_raw]
    result.unmapped_columns = unmapped_raw

    if unmapped_raw:
        result.warnings.append(
            f"These columns were not mapped and are excluded: {', '.join(unmapped_raw)}"
        )

    # ── 6. Finalize ────────────────────────────────────────────────
    result.success = True
    result.df = out_df
    result.row_count = len(out_df)
    result.mapped_columns = list(used_canonical)

    return result


def generate_sample_template() -> bytes:
    """
    Generate a sample Excel template users can download to see expected format.
    """
    sample_data = {
        "Supplier Name": [
            "Apex Manufacturing Mexico",
            "BrightStar Vietnam",
            "Precision Parts India",
            "Taiwan Quality Corp",
            "Midwest US Machining"
        ],
        "Country": ["Mexico", "Vietnam", "India", "Taiwan", "USA"],
        "Tier": [1, 1, 2, 1, 1],
        "Category": ["Electronics", "Plastics", "Metals", "Electronics", "Precision Parts"],
        "Unit Cost ($)": [12.50, 8.75, 9.20, 15.00, 22.00],
        "Annual Spend ($)": [625000, 700000, 552000, 450000, 440000],
        "Annual Volume (units)": [50000, 80000, 60000, 30000, 20000],
        "Quality Score (0-100)": [82, 71, 68, 94, 97],
        "On-Time Delivery (%)": [88.0, 72.0, 75.0, 95.0, 98.0],
        "Defect Rate (%)": [1.8, 4.2, 3.5, 0.5, 0.3],
        "Lead Time (days)": [21, 35, 28, 18, 10],
        "Risk Score (0-100)": [35, 55, 60, 25, 15],
        "Financial Health (0-1)": [0.75, 0.60, 0.55, 0.85, 0.90],
        "Geopolitical Risk (0-1)": [0.30, 0.45, 0.40, 0.25, 0.10],
        "Tariff Exposure (0-1)": [0.25, 0.40, 0.35, 0.20, 0.05],
        "Certifications": [
            "ISO 9001, IATF 16949",
            "ISO 9001",
            "ISO 9001",
            "ISO 9001, AS9100, ISO 14001",
            "ISO 9001, AS9100, NADCAP"
        ],
        "Years in Business": [10, 6, 8, 18, 25],
        "Contact Email": [
            "contact@apex-mx.com", "sales@brightstar.vn",
            "export@precisionind.in", "sales@twqualitycorp.tw", ""
        ],
        "Notes": ["Primary electronics supplier", "", "Secondary metals source", "", "Premium backup"]
    }
    df = pd.DataFrame(sample_data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Suppliers')
        # Auto-size columns
        ws = writer.sheets['Suppliers']
        for col_cells in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 40)
    buf.seek(0)
    return buf.read()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _sanitize_node_id(tier: int, ordinal: int) -> str:
    return f"T{tier}-{chr(64 + ordinal)}" if ordinal <= 26 else f"T{tier}-{ordinal}"


def _country_risk_hint(country: str) -> tuple[float, float, float]:
    country = (country or "").lower()

    if any(token in country for token in ["china", "drc", "congo", "russia", "ukraine"]):
        return 0.75, 0.65, 0.7
    if any(token in country for token in ["taiwan", "peru", "mexico", "vietnam", "india", "chile"]):
        return 0.45, 0.35, 0.35
    if any(token in country for token in ["usa", "united states", "canada", "germany", "japan"]):
        return 0.12, 0.18, 0.1
    return 0.30, 0.25, 0.25


def _infer_financial_health(row: pd.Series) -> float:
    financial_health = _safe_float(row.get("financial_health"), 0.0)
    if financial_health > 0:
        return _clamp(financial_health, 0.05, 0.99)

    risk_score = _safe_float(row.get("risk_score"), 50.0)
    quality = _safe_float(row.get("quality_score"), 70.0) / 100
    on_time = _safe_float(row.get("on_time_delivery_pct"), 85.0) / 100
    health = (1 - risk_score / 100) * 0.5 + quality * 0.2 + on_time * 0.3
    return _clamp(health, 0.1, 0.95)


def _infer_concentration_risk(row: pd.Series, category_counts: dict, total_spend: float) -> float:
    dependency = _safe_float(row.get("dependency_score"), -1.0)
    if dependency >= 0:
        return _clamp(dependency, 0.05, 0.99)

    category = row.get("category", "Uncategorized")
    peers = category_counts.get(category, 1)
    spend_share = 0.0
    annual_spend = _safe_float(row.get("annual_spend"), 0.0)
    if total_spend > 0:
        spend_share = annual_spend / total_spend

    base = {1: 0.85, 2: 0.60, 3: 0.42}.get(peers, 0.28)
    return _clamp(base + spend_share * 0.35, 0.1, 0.95)


def _edge_weight_from_spend(spend: float, max_spend: float) -> float:
    if max_spend <= 0:
        return 0.6
    return _clamp(0.35 + 0.63 * (spend / max_spend), 0.35, 0.98)


def _link_similarity(child: dict, parent: dict) -> float:
    score = 0.0
    if child.get("category") == parent.get("category"):
        score += 0.45
    if child.get("country") == parent.get("country"):
        score += 0.20
    if child.get("tier") == parent.get("tier", 0) + 1:
        score += 0.20
    score += min(parent.get("spend", 0.0) / max(child.get("spend", 1.0), 1.0), 2.0) * 0.05
    return score


def _generate_uploaded_scenarios(supplier_nodes: list[dict], edges: list[dict]) -> list[dict]:
    if not supplier_nodes:
        return []

    node_map = {node["id"]: node for node in supplier_nodes}
    upstream = {}
    for edge in edges:
        upstream.setdefault(edge["source"], set()).add(edge["target"])
        upstream.setdefault(edge["target"], set()).add(edge["source"])

    def scenario_nodes(primary_id: str) -> list[str]:
        primary = node_map[primary_id]
        related = [primary_id]
        for node in supplier_nodes:
            if node["id"] == primary_id:
                continue
            if node.get("category") == primary.get("category") or node.get("region") == primary.get("region"):
                related.append(node["id"])
            if len(related) >= 3:
                break
        for neighbor in upstream.get(primary_id, set()):
            if neighbor in node_map and neighbor not in related:
                related.append(neighbor)
            if len(related) >= 3:
                break
        return related[:3]

    ranked = {
        "tariff": max(supplier_nodes, key=lambda n: n.get("tariff_exposure", 0.0)),
        "geopolitical": max(supplier_nodes, key=lambda n: n.get("geopolitical_risk", 0.0)),
        "weather": max(supplier_nodes, key=lambda n: n.get("weather_risk", 0.0)),
        "concentration": max(supplier_nodes, key=lambda n: n.get("concentration_risk", 0.0)),
        "financial": min(supplier_nodes, key=lambda n: n.get("financial_health", 1.0)),
    }

    scenario_specs = [
        (
            "tariff",
            "Tariff Shock on Critical Import Lane",
            "Tariff escalation increases landed costs and customs delays for a highly exposed supplier lane.",
            (10, 24, 60, 14, 28, 0.08),
        ),
        (
            "geopolitical",
            "Geopolitical Disruption in a High-Risk Region",
            "Political instability or sanctions interrupt supplier operations and export reliability.",
            (14, 35, 90, 14, 30, 0.06),
        ),
        (
            "weather",
            "Severe Weather / Natural Disaster Event",
            "Flooding, storms, or port disruption slow production and logistics for weather-exposed suppliers.",
            (7, 18, 45, 10, 24, 0.12),
        ),
        (
            "concentration",
            "Single-Source Supplier Outage",
            "A concentrated supplier fails, creating outsized flow disruption due to limited alternatives.",
            (14, 30, 75, 10, 21, 0.18),
        ),
        (
            "financial",
            "Supplier Financial Distress",
            "A financially weak supplier triggers shipment instability, quality escapes, and recovery delays.",
            (21, 45, 120, 14, 30, 0.20),
        ),
    ]

    scenarios = []
    for idx, (metric_name, name, description, params) in enumerate(scenario_specs, start=1):
        primary = ranked[metric_name]
        scenarios.append({
            "id": f"uploaded_{metric_name}_{idx}",
            "name": name,
            "description": f"{description} Primary trigger: {primary['name']} ({primary['region']}).",
            "affected_nodes": scenario_nodes(primary["id"]),
            "min_delay": params[0],
            "mode_delay": params[1],
            "max_delay": params[2],
            "expedite_threshold": params[3],
            "stockout_threshold": params[4],
            "quality_impact_prob": params[5],
        })

    return scenarios


def dataframe_to_network_data(
    df: pd.DataFrame,
    company_name: str = "Focal Company (OEM)",
) -> dict:
    """
    Convert uploaded supplier rows into a complete network payload that
    the existing app and models can use directly.
    """
    if df is None or len(df) == 0:
        return {
            "network_name": "Uploaded Supplier Network",
            "description": "Generated network from uploaded supplier data.",
            "created": str(date.today()),
            "nodes": [],
            "edges": [],
            "scenarios": [],
        }

    work_df = df.copy()
    work_df["tier"] = work_df["tier"].fillna(1).clip(lower=1).astype(int)
    work_df["annual_spend"] = work_df["annual_spend"].fillna(0.0)
    work_df["category"] = work_df["category"].fillna("Uncategorized")
    work_df["country"] = work_df["country"].fillna("Unknown")
    work_df = work_df.sort_values(["tier", "annual_spend", "supplier_name"], ascending=[True, False, True]).reset_index(drop=True)

    total_spend = max(_safe_float(work_df["annual_spend"].sum()), 1.0)
    max_spend = _safe_float(work_df["annual_spend"].max(), 0.0)
    category_counts = work_df["category"].value_counts(dropna=False).to_dict()
    tier_counters = {}
    supplier_nodes = []

    for _, row in work_df.iterrows():
        tier = max(int(row.get("tier", 1)), 1)
        tier_counters[tier] = tier_counters.get(tier, 0) + 1
        node_id = _sanitize_node_id(tier, tier_counters[tier])

        spend = _safe_float(row.get("annual_spend"), 0.0)
        if spend <= 0:
            spend = _safe_float(row.get("unit_cost"), 0.0) * _safe_float(row.get("annual_volume"), 0.0)

        geo_hint, weather_hint, tariff_hint = _country_risk_hint(row.get("country", ""))
        node = {
            "id": node_id,
            "name": str(row.get("supplier_name", node_id)),
            "type": "supplier",
            "tier": tier,
            "region": str(row.get("country", "Unknown")),
            "country": str(row.get("country", "Unknown")),
            "category": str(row.get("category", "Uncategorized")),
            "spend": spend,
            "annual_volume": _safe_float(row.get("annual_volume"), 0.0),
            "unit_cost": _safe_float(row.get("unit_cost"), 0.0),
            "on_time_rate": _clamp(_safe_float(row.get("on_time_delivery_pct"), 85.0) / 100, 0.05, 0.999),
            "financial_health": _infer_financial_health(row),
            "geopolitical_risk": _clamp(
                _safe_float(row.get("geopolitical_risk"), geo_hint if row.get("geopolitical_risk") is not None else geo_hint),
                0.01,
                0.99,
            ),
            "weather_risk": _clamp(weather_hint + (_safe_float(row.get("lead_time_days"), 30.0) / 1200), 0.05, 0.95),
            "concentration_risk": _infer_concentration_risk(row, category_counts, total_spend),
            "tariff_exposure": _clamp(
                _safe_float(row.get("tariff_exposure"), tariff_hint if row.get("tariff_exposure") is not None else tariff_hint),
                0.01,
                0.99,
            ),
            "notes": str(row.get("notes", "")),
        }
        supplier_nodes.append(node)

    nodes = [
        {
            "id": "OEM",
            "name": company_name,
            "type": "focal",
            "tier": 0,
            "region": "US-Primary",
            "spend": 0,
            "on_time_rate": 1.0,
            "financial_health": 1.0,
            "geopolitical_risk": 0.0,
            "weather_risk": 0.0,
            "concentration_risk": 0.0,
            "tariff_exposure": 0.0,
        },
        *supplier_nodes,
        {
            "id": "DIST",
            "name": "Distribution Network",
            "type": "distributor",
            "tier": -1,
            "region": "Multi-Region",
        },
        {
            "id": "CUST",
            "name": "End Customers",
            "type": "customer",
            "tier": -2,
            "region": "Multi-Region",
        },
    ]

    edges = []

    tier_one_nodes = [node for node in supplier_nodes if node["tier"] == 1]
    if not tier_one_nodes:
        tier_one_nodes = supplier_nodes[:]

    for node in tier_one_nodes:
        edges.append({
            "source": node["id"],
            "target": "OEM",
            "weight": _edge_weight_from_spend(node.get("spend", 0.0), max_spend),
            "type": "direct-supply",
            "annual_value": node.get("spend", 0.0),
        })

    for node in supplier_nodes:
        if node["tier"] <= 1:
            continue

        candidate_parents = [parent for parent in supplier_nodes if parent["tier"] == node["tier"] - 1]
        if not candidate_parents:
            candidate_parents = [parent for parent in supplier_nodes if parent["tier"] < node["tier"]]

        if not candidate_parents:
            edges.append({
                "source": node["id"],
                "target": "OEM",
                "weight": 0.45,
                "type": "direct-supply",
                "annual_value": node.get("spend", 0.0),
            })
            continue

        ranked_parents = sorted(candidate_parents, key=lambda parent: _link_similarity(node, parent), reverse=True)
        max_links = 2 if len(ranked_parents) > 1 else 1
        for parent in ranked_parents[:max_links]:
            edges.append({
                "source": node["id"],
                "target": parent["id"],
                "weight": _clamp(0.35 + _link_similarity(node, parent), 0.25, 0.95),
                "type": "material-supply",
            })

    edges.append({"source": "OEM", "target": "DIST", "weight": 1.0, "type": "distribution"})
    edges.append({"source": "DIST", "target": "CUST", "weight": 1.0, "type": "sales"})

    return {
        "network_name": "Uploaded Supplier Network",
        "description": "Generated from uploaded supplier data. Upstream edges and scenarios are inferred to keep analytics runnable end to end.",
        "created": str(date.today()),
        "nodes": nodes,
        "edges": edges,
        "scenarios": _generate_uploaded_scenarios(supplier_nodes, edges),
    }


def dataframe_to_network_nodes(df: pd.DataFrame) -> list:
    """
    Backward-compatible helper retained for older call sites.
    """
    return dataframe_to_network_data(df)["nodes"]
