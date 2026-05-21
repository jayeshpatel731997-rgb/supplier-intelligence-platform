"""Production ingestion service built on the existing ingestion parser."""

from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy.orm import Session

from data_ingestion import ingest_file
from src.models import IngestionJob
from src.repositories.audit import AuditRepository
from src.repositories.suppliers import SupplierRepository
from src.tenancy import DEMO_TENANT_ID


@dataclass(slots=True)
class IngestionServiceResult:
    success: bool
    job_id: int
    job_status: str
    row_count: int
    errors: list[str]
    warnings: list[str]
    column_mapping: dict


class IngestionService:
    def __init__(self, session: Session, tenant_id: str = DEMO_TENANT_ID):
        self.session = session
        self.tenant_id = tenant_id

    def process_upload(self, file_bytes: bytes, filename: str, username: str = "system") -> IngestionServiceResult:
        job = IngestionJob(tenant_id=self.tenant_id, filename=filename, status="running")
        self.session.add(job)
        self.session.flush()

        result = ingest_file(file_bytes, filename)
        report = {
            "errors": result.errors,
            "warnings": result.warnings,
            "column_mapping": result.column_mapping,
            "unmapped_columns": result.unmapped_columns,
            "missing_defaults_applied": result.missing_defaults_applied,
        }

        if result.success:
            SupplierRepository(self.session, self.tenant_id).upsert_from_dataframe(result.df, source="upload")
            job.status = "completed"
            job.row_count = result.row_count
            AuditRepository(self.session, self.tenant_id).log(
                "ingestion.upload_completed",
                username=username,
                role="system",
                details={"filename": filename, "row_count": result.row_count},
            )
        else:
            job.status = "failed"
            job.error = "; ".join(result.errors)
            AuditRepository(self.session, self.tenant_id).log(
                "ingestion.upload_failed",
                username=username,
                role="system",
                details={"filename": filename, "errors": result.errors},
            )

        job.report_json = json.dumps(report, default=str)
        self.session.flush()
        return IngestionServiceResult(
            success=result.success,
            job_id=job.id,
            job_status=job.status,
            row_count=result.row_count,
            errors=result.errors,
            warnings=result.warnings,
            column_mapping=result.column_mapping,
        )
