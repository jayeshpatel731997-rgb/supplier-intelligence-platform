"""Seed deterministic supplier-risk demo data for local or staging."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from src.config import Settings, get_settings
from src.database import create_session_factory, init_database
from src.models import SupplierConnectorSync, SupplierEvidenceRun, SupplierHistoricalOutcome
from src.repositories.suppliers import SupplierRepository
from src.repositories.tenants import TenantRepository
from src.services.supplier_evidence_service import SupplierEvidenceService
from src.services.supplier_risk_evidence_chain import load_demo_supplier_signals
from src.tenancy import DEMO_PLATFORM_ADMIN, DEMO_TENANT_ID


DEMO_SCENARIOS: list[dict[str, Any]] = [
    {
        "scenario": "normal_supplier",
        "supplier_id": "demo-normal-1",
        "supplier_name": "Northstar Fasteners",
        "country": "USA",
        "category": "Fasteners",
        "risk_score": 18,
        "story": "Stable supplier with normal delivery and quality signals.",
    },
    {
        "scenario": "financial_distress",
        "supplier_id": "demo-finance-1",
        "supplier_name": "Apex Castings",
        "country": "USA",
        "category": "Castings",
        "risk_score": 78,
        "story": "Financial weak signals indicate liquidity pressure.",
    },
    {
        "scenario": "logistics_disruption",
        "supplier_id": "demo-logistics-1",
        "supplier_name": "Pacific Harness",
        "country": "Vietnam",
        "category": "Wire Harness",
        "risk_score": 66,
        "story": "Port and lane disruption threatens lead times.",
    },
    {
        "scenario": "hiring_slowdown",
        "supplier_id": "demo-hiring-1",
        "supplier_name": "Metro Machining",
        "country": "Mexico",
        "category": "Machining",
        "risk_score": 58,
        "story": "Hiring slowdown could constrain capacity.",
    },
    {
        "scenario": "compliance_issue",
        "supplier_id": "demo-compliance-1",
        "supplier_name": "Riverbend Chemicals",
        "country": "USA",
        "category": "Chemicals",
        "risk_score": 63,
        "story": "Audit/compliance finding requires corrective action.",
    },
    {
        "scenario": "multi_signal_high_risk",
        "supplier_id": "demo-multi-1",
        "supplier_name": "Global Electronics Subtier",
        "country": "Taiwan",
        "category": "Electronics",
        "risk_score": 91,
        "story": "Multiple financial, logistics, and hiring signals indicate high risk.",
    },
]


def load_demo_scenarios(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path and Path(path).exists():
        with Path(path).open(encoding="utf-8") as handle:
            return list(json.load(handle))
    return list(DEMO_SCENARIOS)


def _seed_suppliers(session, tenant_id: str, scenarios: list[dict[str, Any]]) -> int:
    repo = SupplierRepository(session, tenant_id)
    for item in scenarios:
        repo.upsert_supplier(
            supplier_id=item["supplier_id"],
            name=item["supplier_name"],
            country=item["country"],
            category=item["category"],
            annual_spend=1_000_000,
            risk_score=item["risk_score"],
            source=f"demo:{item['scenario']}",
        )
    return len(scenarios)


def _scenario_signals(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signals = []
    for item in scenarios:
        if item["scenario"] == "normal_supplier":
            severity = 18
            signal_type = "operational"
            driver = "Normal operating pattern"
        elif item["scenario"] == "financial_distress":
            severity = 82
            signal_type = "financial"
            driver = "Liquidity pressure"
        elif item["scenario"] == "logistics_disruption":
            severity = 74
            signal_type = "news"
            driver = "Logistics disruption"
        elif item["scenario"] == "hiring_slowdown":
            severity = 64
            signal_type = "hiring"
            driver = "Hiring slowdown"
        elif item["scenario"] == "compliance_issue":
            severity = 70
            signal_type = "audit"
            driver = "Compliance audit issue"
        else:
            severity = 92
            signal_type = "financial"
            driver = "Multi-signal elevated risk"
        signals.append(
            {
                "supplier_id": item["supplier_id"],
                "supplier_name": item["supplier_name"],
                "signal_id": f"demo-{item['scenario']}",
                "signal_type": signal_type,
                "driver": driver,
                "source": "Demo scenario seed",
                "source_url": "",
                "observed_at": "2026-06-01",
                "severity": severity,
                "confidence": 0.82,
                "summary": item["story"],
            }
        )
    return signals


def _seed_historical_outcomes(session, tenant_id: str) -> int:
    rows = [
        SupplierHistoricalOutcome(
            tenant_id=tenant_id,
            supplier_id="demo-finance-1",
            event_type="late_delivery",
            event_date=datetime(2026, 5, 1, tzinfo=UTC),
            severity=0.8,
            notes="Escalated delivery miss after financial distress signal.",
            source="demo",
        ),
        SupplierHistoricalOutcome(
            tenant_id=tenant_id,
            supplier_id="demo-compliance-1",
            event_type="audit_finding",
            event_date=datetime(2026, 5, 10, tzinfo=UTC),
            severity=0.6,
            notes="Corrective action requested by quality team.",
            source="demo",
        ),
    ]
    count = 0
    for row in rows:
        existing = session.scalar(
            select(SupplierHistoricalOutcome).where(
                SupplierHistoricalOutcome.tenant_id == tenant_id,
                SupplierHistoricalOutcome.supplier_id == row.supplier_id,
                SupplierHistoricalOutcome.event_type == row.event_type,
                SupplierHistoricalOutcome.event_date == row.event_date,
            )
        )
        if existing is None:
            session.add(row)
            count += 1
    return count


def seed_demo_data(
    *,
    session_factory: sessionmaker | None = None,
    tenant_id: str = DEMO_TENANT_ID,
    scenario_path: str | Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    if tenant_id != DEMO_TENANT_ID:
        raise ValueError("The deterministic demo seed only supports tenant_id=demo-tenant.")
    active_settings = settings or get_settings()
    factory = session_factory or create_session_factory(active_settings)
    staging_api_key = os.getenv("SUPPLIER_DEMO_API_KEY", "").strip()
    staging_username = os.getenv("SUPPLIER_STAGING_SEED_USERNAME", "").strip()
    if active_settings.is_staging_or_production:
        if active_settings.auth_provider == "oidc" and not staging_username:
            raise RuntimeError(
                "SUPPLIER_STAGING_SEED_USERNAME must match the staging OIDC subject or verified email."
            )
        if active_settings.auth_provider == "local" and (
            not staging_api_key or staging_api_key == "demo-api-key"
        ):
            raise RuntimeError(
                "SUPPLIER_DEMO_API_KEY must be explicitly set to a non-default secret for a local-auth staging exception."
            )
    with factory() as session:
        if active_settings.is_staging_or_production:
            tenants = TenantRepository(session)
            if active_settings.auth_provider == "oidc":
                tenants.ensure_demo_oidc_membership(staging_username, role="risk_manager")
            else:
                tenants.ensure_demo_seed(
                    raw_key=staging_api_key,
                    username="demo-staging-risk-manager",
                    role="risk_manager",
                    revoke_usernames=(DEMO_PLATFORM_ADMIN,),
                )
        else:
            TenantRepository(session).ensure_demo_seed()
        scenarios = load_demo_scenarios(scenario_path)
        suppliers = _seed_suppliers(session, tenant_id, scenarios)
        evidence = SupplierEvidenceService(session, tenant_id)
        evidence.save_scoring_config(
            version="demo-staging-v1",
            description="Deterministic staging/demo supplier risk scoring configuration.",
            actor="seed_demo_data",
        )
        seed_signals = [*load_demo_supplier_signals(), *_scenario_signals(scenarios)]
        existing_seed_sync = session.scalar(
            select(SupplierConnectorSync).where(
                SupplierConnectorSync.tenant_id == tenant_id,
                SupplierConnectorSync.source_system == "demo_seed",
            )
        )
        if existing_seed_sync is None:
            evidence.import_signals(
                source_system="demo_seed",
                connector_type="local_demo",
                signals=seed_signals,
                actor="seed_demo_data",
            )
        else:
            evidence.upsert_signals("demo_seed", seed_signals)
        existing_run = session.scalar(
            select(SupplierEvidenceRun)
            .where(
                SupplierEvidenceRun.tenant_id == tenant_id,
                SupplierEvidenceRun.scoring_version == "demo-staging-v1",
            )
            .order_by(SupplierEvidenceRun.created_at.asc())
            .limit(1)
        )
        if existing_run is None:
            run = evidence.run_evidence_chain(scoring_version="demo-staging-v1", actor="seed_demo_data")
        else:
            run = evidence.get_run(existing_run.run_id) or {"actions": []}
        _seed_historical_outcomes(session, tenant_id)
        signals = evidence.list_signals(limit=10_000)
        actions = run.get("actions", [])
        session.commit()
        return {
            "tenant_id": tenant_id,
            "suppliers": suppliers,
            "weak_signals": len(signals),
            "evidence_run": run.get("run_id"),
            "actions": len(actions),
            "scoring_version": "demo-staging-v1",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed deterministic supplier risk demo data.")
    parser.add_argument("--tenant-id", default=DEMO_TENANT_ID)
    parser.add_argument("--scenario-path", default="")
    args = parser.parse_args()
    settings = get_settings()
    factory = create_session_factory(settings)
    if not settings.is_staging_or_production:
        init_database(factory)
    result = seed_demo_data(
        session_factory=factory,
        tenant_id=args.tenant_id,
        scenario_path=args.scenario_path or settings.demo_scenario_path,
        settings=settings,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
