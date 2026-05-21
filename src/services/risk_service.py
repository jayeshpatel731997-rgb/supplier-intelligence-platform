"""Risk score and exposure services."""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from models.bayesian_risk import SupplierSignals, compute_bayesian_risk
from src.models import FinancialExposureRun, ScenarioRun, SupplierRiskScore
from src.repositories.alerts import AlertRepository
from src.repositories.suppliers import SupplierRepository
from src.tenancy import DEMO_TENANT_ID


def _risk_for_supplier(row) -> dict:
    signals = SupplierSignals(
        financial_health=max(0.0, min(1.0, 1.0 - (row.risk_score / 100.0))),
        geopolitical_risk=0.55 if row.country.lower() in {"china", "russia", "ukraine"} else 0.25,
        weather_risk=0.25,
        concentration_risk=0.45 if row.annual_spend > 250_000 else 0.25,
        historical_reliability=max(0.0, min(1.0, row.on_time_delivery_pct / 100.0)),
        tariff_exposure=0.5 if row.country.lower() not in {"usa", "united states", "canada"} else 0.2,
    )
    risk = compute_bayesian_risk(signals)
    return {
        "supplier_id": row.supplier_id,
        "supplier_name": row.name,
        "risk_probability": risk.posterior_probability,
        "risk_level": risk.risk_level,
        "dominant_factor": risk.dominant_risk_factor,
        "confidence": risk.confidence,
        "drivers": list(risk.individual_likelihood_ratios.keys()),
    }


class RiskService:
    def __init__(self, session: Session, tenant_id: str = DEMO_TENANT_ID):
        self.session = session
        self.tenant_id = tenant_id

    def recalculate(self) -> list[dict]:
        scores: list[dict] = []
        alerts = AlertRepository(self.session, self.tenant_id)
        for supplier in SupplierRepository(self.session, self.tenant_id).list(limit=10_000):
            score = _risk_for_supplier(supplier)
            row = SupplierRiskScore(
                tenant_id=self.tenant_id,
                supplier_id=supplier.supplier_id,
                risk_probability=score["risk_probability"],
                risk_level=score["risk_level"],
                dominant_factor=score["dominant_factor"],
                confidence=score["confidence"],
                drivers_json=json.dumps(score["drivers"]),
            )
            self.session.add(row)
            if score["risk_level"] in {"high", "critical"}:
                alerts.create_alert(
                    alert_type="supplier_high_risk",
                    severity=score["risk_level"],
                    message=f"{supplier.name} reached {score['risk_level']} risk ({score['risk_probability']:.0%}).",
                    supplier_id=supplier.supplier_id,
                    exposure=supplier.annual_spend,
                )
            scores.append(score)
        return scores

    def latest_scores(self) -> list[dict]:
        latest = []
        for supplier in SupplierRepository(self.session, self.tenant_id).list(limit=10_000):
            row = self.session.scalar(
                select(SupplierRiskScore)
                .where(SupplierRiskScore.supplier_id == supplier.supplier_id)
                .order_by(SupplierRiskScore.created_at.desc())
            )
            if row is None:
                latest.append(_risk_for_supplier(supplier))
            else:
                latest.append(
                    {
                        "supplier_id": supplier.supplier_id,
                        "supplier_name": supplier.name,
                        "risk_probability": row.risk_probability,
                        "risk_level": row.risk_level,
                        "dominant_factor": row.dominant_factor,
                        "confidence": row.confidence,
                        "drivers": json.loads(row.drivers_json or "[]"),
                    }
                )
        return latest

    def financial_exposure(self) -> dict:
        suppliers = SupplierRepository(self.session, self.tenant_id).list(limit=10_000)
        total_spend = sum(s.annual_spend for s in suppliers)
        high_risk_exposure = sum(s.annual_spend for s in suppliers if s.risk_score >= 70)
        row = FinancialExposureRun(
            tenant_id=self.tenant_id,
            expected_loss=high_risk_exposure * 0.15,
            var95=high_risk_exposure * 0.35,
            cvar95=high_risk_exposure * 0.50,
            result_json=json.dumps({"total_spend": total_spend, "high_risk_exposure": high_risk_exposure}),
        )
        self.session.add(row)
        return {
            "total_spend": total_spend,
            "high_risk_exposure": high_risk_exposure,
            "expected_loss": row.expected_loss,
            "var95": row.var95,
            "cvar95": row.cvar95,
        }

    def run_scenario(self, scenario_name: str = "api_manual_scenario") -> dict:
        exposure = self.financial_exposure()
        row = ScenarioRun(tenant_id=self.tenant_id, scenario_name=scenario_name, status="completed", result_json=json.dumps(exposure))
        self.session.add(row)
        return {"scenario_name": scenario_name, "status": "completed", "financial_exposure": exposure}
