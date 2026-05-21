"""Background worker process for scheduled monitoring jobs."""

from __future__ import annotations

from apscheduler.schedulers.blocking import BlockingScheduler

from src.config import get_settings
from src.database import create_session_factory, init_database
from src.observability.logging import get_logger
from src.services.scheduler import LocalJobScheduler
from src.services.worker_queue import EnterpriseTaskRunner, build_celery_app, get_worker_mode, register_celery_tasks


logger = get_logger(__name__)

settings = get_settings()
SessionFactory = create_session_factory(settings)
init_database(SessionFactory)
celery_app = register_celery_tasks(build_celery_app(settings), settings, SessionFactory)


def main() -> None:
    session_factory = SessionFactory
    if settings.worker_mode == "celery" and celery_app is not None:
        logger.info("celery worker app configured; run with celery -A backend.worker.celery_app worker")
        return
    runner = LocalJobScheduler(settings, session_factory)
    task_runner = EnterpriseTaskRunner(settings, session_factory)

    if not settings.scheduler_enabled:
        logger.info("scheduler disabled by configuration")
        return

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(lambda: task_runner.run_task_for_all_tenants("sentinel_scan_task"), "interval", minutes=settings.sentinel_interval_minutes, id="sentinel_scan")
    scheduler.add_job(lambda: task_runner.run_task_for_all_tenants("risk_recalculation_task"), "interval", minutes=settings.risk_interval_minutes, id="risk_recalculate")
    scheduler.add_job(lambda: task_runner.run_task_for_all_tenants("exposure_recalculation_task"), "interval", minutes=settings.risk_interval_minutes, id="exposure_recalculate")
    scheduler.add_job(lambda: runner.run_all_tenants("retention_cleanup"), "interval", minutes=1440, id="retention_cleanup")
    logger.info("background worker started", extra={"worker_mode": get_worker_mode(settings)})
    scheduler.start()


if __name__ == "__main__":
    main()
