#!/usr/bin/env sh
set -e

python scripts/migrate.py
exec uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-10000}"
