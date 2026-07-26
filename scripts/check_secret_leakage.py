"""Scan repository files for high-confidence secret shapes without printing values."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 2_000_000
SKIP_SUFFIXES = {".db", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pyc", ".xlsx", ".xls"}
PLACEHOLDER_MARKERS = {
    "",
    "...",
    "<secret>",
    "<token>",
    "<password>",
    "<api-key>",
    "<tenant-api-key>",
    "<oidc-token>",
    "replace_me",
}
PATTERNS = [
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{32,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "credential-bearing database URL",
        re.compile(r"\bpostgres(?:ql)?(?:\+psycopg)?://([^:\s/@]+):([^@\s/]+)@", re.IGNORECASE),
    ),
]


def repository_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def _is_placeholder(value: str) -> bool:
    clean = value.strip().strip("\"'")
    lowered = clean.lower()
    return (
        lowered in PLACEHOLDER_MARKERS
        or clean.startswith("<")
        or clean.endswith("...")
        or len(clean) < 16
        or any(
            marker in lowered
            for marker in (
                "replace",
                "change_me",
                "example",
                "placeholder",
                "password",
                "secret",
                "test",
            )
        )
        or any(character in clean for character in ("$", "{", "}"))
    )


def scan_file(path: Path) -> list[tuple[int, str]]:
    if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return []
    findings: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines, start=1):
        for category, pattern in PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            if category == "credential-bearing database URL" and _is_placeholder(match.group(2)):
                continue
            findings.append((line_number, category))
    return findings


def main() -> int:
    findings: list[tuple[Path, int, str]] = []
    for path in repository_files():
        for line_number, category in scan_file(path):
            findings.append((path, line_number, category))
    if findings:
        for path, line_number, category in findings:
            print(f"{path.relative_to(ROOT)}:{line_number}: possible {category}")
        return 1
    print("Secret leakage scan passed: no high-confidence secret shapes found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
