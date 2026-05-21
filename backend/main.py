"""FastAPI backend for production integrations."""

from __future__ import annotations

import uuid
from dataclasses import asdict

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.config import Settings, get_settings
from src.database import create_session_factory, init_database, seed_demo_tenant
from src.repositories.alerts import AlertRepository
from src.repositories.audit import AuditRepository
from src.repositories.jobs import JobRepository
from src.repositories.suppliers import SupplierRepository
from src.repositories.tenants import TenantRepository
from src.security.auth_providers import build_auth_provider
from src.security.rate_limit import RateLimitMiddleware
from src.security.secrets import EnvSecretProvider, require_production_secret
from src.services.audit_export import AuditExportService
from src.services.ingestion_service import IngestionService
from src.services.risk_service import RiskService
from src.services.scheduler import LocalJobScheduler
from src.services.sentinel_service import SentinelService
from src.services.system_service import system_status
from src.services.worker_queue import EnterpriseTaskRunner, get_worker_mode
from src.tenancy import DEMO_TENANT_ID, TenantContext, require_permission


settings = get_settings()
SessionFactory = create_session_factory(settings)
init_database(SessionFactory)
if settings.demo_mode and not settings.is_production:
    seed_demo_tenant(SessionFactory)

app = FastAPI(
    title="Supplier Intelligence Platform API",
    version="1.0.0",
    description="Tenant-scoped supplier risk intelligence API.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[item.strip() for item in settings.cors_allow_origins.split(",") if item.strip()] or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    RateLimitMiddleware,
    enabled=settings.rate_limit_enabled,
    limit=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)


class SentinelScanRequest(BaseModel):
    mode: str = "demo"


class ScenarioRunRequest(BaseModel):
    scenario_name: str = "api_manual_scenario"


class TenantCreateRequest(BaseModel):
    tenant_id: str
    name: str


class MembershipCreateRequest(BaseModel):
    username: str
    role: str


class ApiKeyCreateRequest(BaseModel):
    username: str
    role: str
    label: str = ""


class AccessReviewCreateRequest(BaseModel):
    reviewer: str
    notes: str = ""


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except HTTPException as exc:
        response = JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "request_id": request_id},
        )
    except Exception:
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
        )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
    return response


def get_db():
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()


def get_active_settings() -> Settings:
    return settings


def get_tenant_context(request: Request, session: Session = Depends(get_db)) -> TenantContext:
    tenant_id = request.headers.get("X-Tenant-ID", "").strip()
    api_key = request.headers.get("X-API-Key", "").strip()
    if not tenant_id or not api_key:
        _safe_auth_failure_audit(session, "missing_tenant_or_api_key", request.state.request_id)
        raise HTTPException(status_code=401, detail="X-Tenant-ID and X-API-Key are required.")
    context = TenantRepository(session).validate_api_key(tenant_id, api_key)
    if context is None:
        _safe_auth_failure_audit(session, "invalid_tenant_or_api_key", request.state.request_id)
        raise HTTPException(status_code=403, detail="Invalid tenant or API key.")
    context.request_id = request.state.request_id
    return context


def _safe_auth_failure_audit(session: Session, reason: str, request_id: str) -> None:
    try:
        AuditRepository(session, DEMO_TENANT_ID).log(
            "auth.failure",
            details={"reason": reason, "request_id": request_id},
        )
        session.commit()
    except Exception:
        session.rollback()


def require_context(permission: str):
    def dependency(context: TenantContext = Depends(get_tenant_context)) -> TenantContext:
        if not require_permission(context, permission):
            raise HTTPException(status_code=403, detail=f"Permission denied: {permission}")
        return context

    return dependency


@app.get("/live", tags=["health"])
def live():
    return {"status": "alive"}


@app.get("/ready", tags=["health"])
def ready(active_settings: Settings = Depends(get_active_settings)):
    status = system_status(active_settings, SessionFactory, DEMO_TENANT_ID)
    return {"status": "ready" if status["database"]["ok"] else "degraded", "database": status["database"]}


@app.get("/health", tags=["health"])
def health(active_settings: Settings = Depends(get_active_settings)):
    status = system_status(active_settings, SessionFactory, DEMO_TENANT_ID)
    return {"status": "ok" if status["database"]["ok"] else "degraded", "database": status["database"]}


@app.get("/worker/health", tags=["health"])
def worker_health(active_settings: Settings = Depends(get_active_settings)):
    return get_worker_mode(active_settings)


@app.get("/auth/status", tags=["auth"])
def auth_status(active_settings: Settings = Depends(get_active_settings)):
    provider = build_auth_provider(active_settings)
    return {
        "auth_provider": active_settings.auth_provider,
        "mfa_required": active_settings.mfa_required,
        "scim_enabled": active_settings.scim_enabled,
        "runtime": provider.validate_runtime(),
    }


@app.get("/security/secrets/status", tags=["health"])
def secrets_status(active_settings: Settings = Depends(get_active_settings)):
    provider = EnvSecretProvider()
    checks = [
        require_production_secret("DATABASE_URL", provider, active_settings.is_production),
        require_production_secret("OIDC_CLIENT_SECRET", provider, active_settings.is_production and active_settings.auth_provider == "oidc"),
    ]
    return {
        "provider": active_settings.secrets_provider,
        "kms_provider": active_settings.kms_provider,
        "checks": [{"name": check.name, "ok": check.ok, "message": check.message} for check in checks],
    }


@app.get("/suppliers", tags=["suppliers"])
def suppliers(session: Session = Depends(get_db), context: TenantContext = Depends(require_context("supplier.read"))):
    return SupplierRepository.many_to_dict(SupplierRepository(session, context.tenant_id).list())


@app.get("/suppliers/{supplier_id}", tags=["suppliers"])
def supplier(supplier_id: str, session: Session = Depends(get_db), context: TenantContext = Depends(require_context("supplier.read"))):
    row = SupplierRepository(session, context.tenant_id).get(supplier_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Supplier not found")
    return SupplierRepository.to_dict(row)


@app.get("/risk/scores", tags=["risk"])
def risk_scores(session: Session = Depends(get_db), context: TenantContext = Depends(require_context("risk.read"))):
    return RiskService(session, context.tenant_id).latest_scores()


@app.post("/risk/recalculate", tags=["risk"])
def risk_recalculate(session: Session = Depends(get_db), context: TenantContext = Depends(require_context("risk.run"))):
    scores = RiskService(session, context.tenant_id).recalculate()
    session.commit()
    return {"status": "completed", "scores": scores}


@app.get("/financial/exposure", tags=["risk"])
def financial_exposure(session: Session = Depends(get_db), context: TenantContext = Depends(require_context("risk.read"))):
    result = RiskService(session, context.tenant_id).financial_exposure()
    session.commit()
    return result


@app.post("/scenario/run", tags=["scenario"])
def scenario_run(payload: ScenarioRunRequest, session: Session = Depends(get_db), context: TenantContext = Depends(require_context("scenario.run"))):
    result = RiskService(session, context.tenant_id).run_scenario(payload.scenario_name)
    session.commit()
    return result


@app.post("/sentinel/scan", tags=["sentinel"])
def sentinel_scan(
    payload: SentinelScanRequest,
    session: Session = Depends(get_db),
    active_settings: Settings = Depends(get_active_settings),
    context: TenantContext = Depends(require_context("sentinel.run")),
):
    result = SentinelService(session, active_settings, context.tenant_id).scan(mode=payload.mode)
    session.commit()
    return result.to_dict()


@app.get("/sentinel/events", tags=["sentinel"])
def sentinel_events(
    session: Session = Depends(get_db),
    active_settings: Settings = Depends(get_active_settings),
    context: TenantContext = Depends(require_context("supplier.read")),
):
    return SentinelService(session, active_settings, context.tenant_id).list_events()


@app.get("/alerts", tags=["alerts"])
def alerts(session: Session = Depends(get_db), context: TenantContext = Depends(require_context("alerts.read"))):
    return [AlertRepository.to_dict(row) for row in AlertRepository(session, context.tenant_id).list()]


@app.post("/alerts/{alert_id}/acknowledge", tags=["alerts"])
def acknowledge_alert(alert_id: int, session: Session = Depends(get_db), context: TenantContext = Depends(require_context("alerts.acknowledge"))):
    row = AlertRepository(session, context.tenant_id).acknowledge(alert_id, actor=context.username)
    session.commit()
    return AlertRepository.to_dict(row)


@app.post("/ingestion/upload", tags=["ingestion"])
async def ingestion_upload(
    file: UploadFile = File(...),
    session: Session = Depends(get_db),
    context: TenantContext = Depends(require_context("ingestion.upload")),
):
    data = await file.read()
    result = IngestionService(session, context.tenant_id).process_upload(data, file.filename or "upload.csv", username=context.username)
    session.commit()
    return asdict(result)


@app.get("/ingestion/jobs", tags=["ingestion"])
def ingestion_jobs(session: Session = Depends(get_db), context: TenantContext = Depends(require_context("system.read"))):
    from src.models import IngestionJob
    from sqlalchemy import select

    rows = session.scalars(
        select(IngestionJob).where(IngestionJob.tenant_id == context.tenant_id).order_by(IngestionJob.created_at.desc()).limit(100)
    )
    return [
        {
            "id": row.id,
            "filename": row.filename,
            "status": row.status,
            "row_count": row.row_count,
            "error": row.error,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


@app.get("/audit/logs", tags=["audit"])
def audit_logs(session: Session = Depends(get_db), context: TenantContext = Depends(require_context("audit.read"))):
    return [AuditRepository.to_dict(row) for row in AuditRepository(session, context.tenant_id).list()]


@app.get("/system/status", tags=["health"])
def get_system_status(active_settings: Settings = Depends(get_active_settings), context: TenantContext = Depends(require_context("system.read"))):
    return system_status(active_settings, SessionFactory, context.tenant_id)


@app.get("/background/jobs", tags=["jobs"])
def background_jobs(session: Session = Depends(get_db), context: TenantContext = Depends(require_context("system.read"))):
    return [JobRepository.to_dict(row) for row in JobRepository(session, context.tenant_id).list()]


@app.post("/background/jobs/{job_name}/run", tags=["jobs"])
def run_background_job(
    job_name: str,
    active_settings: Settings = Depends(get_active_settings),
    context: TenantContext = Depends(require_context("jobs.run")),
):
    return LocalJobScheduler(active_settings, SessionFactory).run_job_now(job_name, context.tenant_id).to_dict()


@app.post("/worker/tasks/{task_name}/run", tags=["jobs"])
def run_worker_task(
    task_name: str,
    active_settings: Settings = Depends(get_active_settings),
    context: TenantContext = Depends(require_context("jobs.run")),
):
    return EnterpriseTaskRunner(active_settings, SessionFactory).run_task(
        task_name,
        context.tenant_id,
        correlation_id=context.request_id,
    ).to_dict()


@app.get("/audit/export/jsonl", tags=["audit"])
def export_audit_jsonl(session: Session = Depends(get_db), context: TenantContext = Depends(require_context("audit.read"))):
    return {"tenant_id": context.tenant_id, "format": "jsonl", "payload": AuditExportService(session, context.tenant_id).export_jsonl()}


@app.get("/tenants", tags=["admin"])
def tenants(session: Session = Depends(get_db), context: TenantContext = Depends(require_context("tenant.read"))):
    repo = TenantRepository(session)
    if context.role == "platform_admin":
        return [TenantRepository.tenant_to_dict(row) for row in repo.list_tenants()]
    return [TenantRepository.tenant_to_dict(repo.get_tenant(context.tenant_id))]


@app.post("/tenants", tags=["admin"])
def create_tenant(payload: TenantCreateRequest, session: Session = Depends(get_db), context: TenantContext = Depends(require_context("tenant.manage_users"))):
    if context.role != "platform_admin":
        raise HTTPException(status_code=403, detail="Only platform_admin can create tenants.")
    row = TenantRepository(session).create_tenant(payload.tenant_id, payload.name)
    session.commit()
    return TenantRepository.tenant_to_dict(row)


@app.get("/tenants/{tenant_id}/memberships", tags=["admin"])
def memberships(tenant_id: str, session: Session = Depends(get_db), context: TenantContext = Depends(require_context("tenant.manage_users"))):
    if context.tenant_id != tenant_id and context.role != "platform_admin":
        raise HTTPException(status_code=403, detail="Cannot manage another tenant.")
    return [TenantRepository.membership_to_dict(row) for row in TenantRepository(session).list_memberships(tenant_id)]


@app.post("/tenants/{tenant_id}/memberships", tags=["admin"])
def create_membership(
    tenant_id: str,
    payload: MembershipCreateRequest,
    session: Session = Depends(get_db),
    context: TenantContext = Depends(require_context("tenant.manage_users")),
):
    if context.tenant_id != tenant_id and context.role != "platform_admin":
        raise HTTPException(status_code=403, detail="Cannot manage another tenant.")
    row = TenantRepository(session).create_membership(tenant_id, payload.username, payload.role)
    session.commit()
    return TenantRepository.membership_to_dict(row)


@app.post("/tenants/{tenant_id}/api-keys", tags=["admin"])
def create_api_key(
    tenant_id: str,
    payload: ApiKeyCreateRequest,
    session: Session = Depends(get_db),
    context: TenantContext = Depends(require_context("tenant.manage_api_keys")),
):
    if context.tenant_id != tenant_id and context.role != "platform_admin":
        raise HTTPException(status_code=403, detail="Cannot manage another tenant.")
    key = TenantRepository(session).create_api_key(tenant_id, payload.username, payload.role, payload.label)
    session.commit()
    return {"tenant_id": tenant_id, "username": payload.username, "role": payload.role, "api_key": key}


@app.get("/tenants/{tenant_id}/access-reviews", tags=["admin"])
def access_reviews(tenant_id: str, session: Session = Depends(get_db), context: TenantContext = Depends(require_context("audit.read"))):
    if context.tenant_id != tenant_id and context.role != "platform_admin":
        raise HTTPException(status_code=403, detail="Cannot read another tenant.")
    return [
        {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "reviewer": row.reviewer,
            "status": row.status,
            "notes": row.notes,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in TenantRepository(session).list_access_reviews(tenant_id)
    ]


@app.post("/tenants/{tenant_id}/access-reviews", tags=["admin"])
def create_access_review(
    tenant_id: str,
    payload: AccessReviewCreateRequest,
    session: Session = Depends(get_db),
    context: TenantContext = Depends(require_context("audit.read")),
):
    if context.tenant_id != tenant_id and context.role != "platform_admin":
        raise HTTPException(status_code=403, detail="Cannot create review for another tenant.")
    row = TenantRepository(session).create_access_review(tenant_id, payload.reviewer, payload.notes)
    session.commit()
    return {"id": row.id, "tenant_id": row.tenant_id, "reviewer": row.reviewer, "status": row.status, "notes": row.notes}


def app_import_smoke() -> str:
    return "ok"
