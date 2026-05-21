"""Supplier repository."""

from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models import Supplier, SupplierKPI
from src.tenancy import DEMO_TENANT_ID


def supplier_id_from_name(name: str) -> str:
    cleaned = " ".join(str(name or "Unknown").strip().split())
    digest = hashlib.sha1(cleaned.lower().encode("utf-8")).hexdigest()[:10]
    return f"SUP-{digest}"


class SupplierRepository:
    def __init__(self, session: Session, tenant_id: str = DEMO_TENANT_ID):
        self.session = session
        self.tenant_id = tenant_id

    def list(self, limit: int = 500) -> list[Supplier]:
        return list(
            self.session.scalars(
                select(Supplier).where(Supplier.tenant_id == self.tenant_id).order_by(Supplier.name).limit(limit)
            )
        )

    def get(self, supplier_id: str) -> Supplier | None:
        return self.session.scalar(
            select(Supplier).where(Supplier.tenant_id == self.tenant_id, Supplier.supplier_id == supplier_id)
        )

    def upsert_supplier(
        self,
        supplier_id: str | None,
        name: str,
        country: str = "Unknown",
        category: str = "Uncategorized",
        annual_spend: float = 0.0,
        unit_cost: float = 0.0,
        on_time_delivery_pct: float = 85.0,
        defect_rate_pct: float = 2.0,
        risk_score: float = 50.0,
        tier: int = 1,
        source: str = "manual",
    ) -> Supplier:
        active_id = supplier_id or supplier_id_from_name(name)
        supplier = self.get(active_id)
        if supplier is None:
            supplier = Supplier(tenant_id=self.tenant_id, supplier_id=active_id, name=name)
            self.session.add(supplier)
        supplier.tenant_id = self.tenant_id
        supplier.name = name
        supplier.country = country or "Unknown"
        supplier.category = category or "Uncategorized"
        supplier.annual_spend = float(annual_spend or 0.0)
        supplier.unit_cost = float(unit_cost or 0.0)
        supplier.on_time_delivery_pct = float(on_time_delivery_pct or 85.0)
        supplier.defect_rate_pct = float(defect_rate_pct or 2.0)
        supplier.risk_score = float(risk_score or 50.0)
        supplier.tier = int(tier or 1)
        supplier.source = source
        return supplier

    def upsert_from_dataframe(self, df: pd.DataFrame, source: str = "upload") -> list[Supplier]:
        suppliers: list[Supplier] = []
        for _, row in df.iterrows():
            name = str(row.get("supplier_name", row.get("Supplier", "Unknown"))).strip()
            suppliers.append(
                self.upsert_supplier(
                    supplier_id=supplier_id_from_name(name),
                    name=name,
                    country=str(row.get("country", row.get("Country", "Unknown"))),
                    category=str(row.get("category", row.get("Category", "Uncategorized"))),
                    annual_spend=float(row.get("annual_spend", row.get("Annual_Spend", 0.0)) or 0.0),
                    unit_cost=float(row.get("unit_cost", row.get("Unit_Cost", 0.0)) or 0.0),
                    on_time_delivery_pct=float(row.get("on_time_delivery_pct", 85.0) or 85.0),
                    defect_rate_pct=float(row.get("defect_rate_pct", 2.0) or 2.0),
                    risk_score=float(row.get("risk_score", row.get("Risk_Score", 50.0)) or 50.0),
                    tier=int(row.get("tier", 1) or 1),
                    source=source,
                )
            )
        return suppliers

    def replace_kpis(self, supplier_id: str, metrics: dict[str, float], period: str = "current") -> None:
        existing = list(
            self.session.scalars(
                select(SupplierKPI).where(SupplierKPI.tenant_id == self.tenant_id, SupplierKPI.supplier_id == supplier_id)
            )
        )
        for item in existing:
            self.session.delete(item)
        for metric_name, metric_value in metrics.items():
            self.session.add(
                SupplierKPI(
                    tenant_id=self.tenant_id,
                    supplier_id=supplier_id,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    period=period,
                )
            )

    @staticmethod
    def to_dict(supplier: Supplier) -> dict:
        return {
            "id": supplier.supplier_id,
            "supplier_id": supplier.supplier_id,
            "name": supplier.name,
            "country": supplier.country,
            "category": supplier.category,
            "tier": supplier.tier,
            "annual_spend": supplier.annual_spend,
            "unit_cost": supplier.unit_cost,
            "on_time_delivery_pct": supplier.on_time_delivery_pct,
            "defect_rate_pct": supplier.defect_rate_pct,
            "risk_score": supplier.risk_score,
            "source": supplier.source,
        }

    @staticmethod
    def many_to_dict(suppliers: Iterable[Supplier]) -> list[dict]:
        return [SupplierRepository.to_dict(supplier) for supplier in suppliers]
