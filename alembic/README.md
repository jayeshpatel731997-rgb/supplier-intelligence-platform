# Alembic Migrations

This scaffold is ready for PostgreSQL-backed SaaS deployments.

Local/demo mode still uses `Base.metadata.create_all()` so Streamlit and tests keep working without a migration step.

Generate a migration after model changes:

```bash
alembic revision --autogenerate -m "tenant scoped schema"
alembic upgrade head
```
