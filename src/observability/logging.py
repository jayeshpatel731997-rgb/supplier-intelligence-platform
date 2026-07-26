"""Structured logging setup without leaking secrets."""

from __future__ import annotations

import logging
import re
import sys
from urllib.parse import urlsplit, urlunsplit


_URL_CREDENTIAL_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)([^/\s:@]+):([^@\s/]+)@")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|client[_-]?secret)\s*=\s*([^,\s]+)"
)
_URL_RE = re.compile(r"https?://[^\s,)\]}]+", re.IGNORECASE)
_RELATIVE_ERROR_URL_RE = re.compile(r"(?i)(\burl:\s*)/[^\s,)\]}]+")


def _redact_url(match: re.Match[str]) -> str:
    try:
        parsed = urlsplit(match.group(0))
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return "https://***"
    if port:
        host = f"{host}:{port}"
    return urlunsplit((parsed.scheme, host, "", "", ""))


def redact_secret_text(value: object) -> str:
    text = str(value)
    text = _URL_CREDENTIAL_RE.sub(r"\1***:***@", text)
    text = _SECRET_ASSIGNMENT_RE.sub(r"\1=***", text)
    text = _URL_RE.sub(_redact_url, text)
    return _RELATIVE_ERROR_URL_RE.sub(r"\1/***", text)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=False,
    )


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
