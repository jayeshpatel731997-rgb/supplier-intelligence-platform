"""supplier evidence chains

Revision ID: 0002_supplier_evidence_chains
Revises: 0001_tenant_scoped_schema
Create Date: 2026-06-05
"""

from __future__ import annotations

from alembic import op

from src.models import Base


revision = "0002_supplier_evidence_chains"
down_revision = "0001_tenant_scoped_schema"
branch_labels = None
depends_on = None

EVIDENCE_TABLES = [
    "supplier_weak_signals",
    "supplier_evidence_scoring_versions",
    "supplier_evidence_runs",
    "supplier_evidence_run_suppliers",
    "supplier_evidence_actions",
    "supplier_connector_syncs",
    "supplier_historical_outcomes",
]


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind, tables=[Base.metadata.tables[name] for name in EVIDENCE_TABLES])


def downgrade() -> None:
    bind = op.get_bind()
    for name in reversed(EVIDENCE_TABLES):
        Base.metadata.tables[name].drop(bind=bind, checkfirst=True)
