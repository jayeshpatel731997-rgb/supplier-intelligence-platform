"""Export a SQLite database file to a timestamped backup path."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings


def _sqlite_path(database_url: str) -> Path:
    if not database_url.startswith("sqlite:///"):
        raise ValueError("SQLite export only supports sqlite:/// URLs.")
    return Path(database_url.replace("sqlite:///", "", 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local SQLite backup copy.")
    parser.add_argument("--output-dir", default="backups", help="Backup destination directory.")
    args = parser.parse_args()
    source = _sqlite_path(get_settings().database_url)
    if not source.exists():
        raise FileNotFoundError(source)
    destination_dir = Path(args.output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = destination_dir / f"{source.stem}-{timestamp}{source.suffix}"
    shutil.copy2(source, destination)
    print({"source": str(source), "backup": str(destination)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
