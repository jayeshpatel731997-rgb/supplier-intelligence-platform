"""Verify a SQLite backup can be opened and queried."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a SQLite backup file.")
    parser.add_argument("backup_path")
    args = parser.parse_args()
    path = Path(args.backup_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA quick_check")
        table_count = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    print({"backup": str(path), "table_count": table_count, "quick_check": "ok"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
