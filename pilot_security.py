"""Pilot-grade authentication, persistence, and audit logging.

This module is intentionally small and local-first. It gives the Streamlit app
enough controls for an internal company pilot without pretending to be a full
enterprise IAM/database layer.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


DB_PATH = Path(os.getenv("SUPPLIER_APP_DB_PATH", "data/pilot_app.db"))
VALID_ROLES = {"admin", "analyst", "viewer"}
MUTATING_ROLES = {"admin", "analyst"}
MAX_FAILED_ATTEMPTS = int(os.getenv("SUPPLIER_APP_MAX_FAILED_ATTEMPTS", "5"))
LOCKOUT_MINUTES = int(os.getenv("SUPPLIER_APP_LOCKOUT_MINUTES", "15"))


def security_mode() -> str:
    return os.getenv("SUPPLIER_SECURITY_MODE", os.getenv("SECURITY_MODE", "local")).strip().lower()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@contextmanager
def _connect(db_path: Path | None = None):
    active_path = db_path or DB_PATH
    active_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(active_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 250_000)
    return f"pbkdf2_sha256${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, salt_b64, digest_b64 = stored_hash.split("$", 2)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 250_000)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def validate_password_strength(password: str) -> None:
    value = password or ""
    if len(value) < 10:
        raise ValueError("Password must be at least 10 characters.")
    if not re.search(r"[A-Z]", value) or not re.search(r"[a-z]", value) or not re.search(r"\d", value):
        raise ValueError("Password must include uppercase, lowercase, and numeric characters.")


def _ensure_user_columns(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "failed_login_count" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN failed_login_count INTEGER NOT NULL DEFAULT 0")
    if "locked_until" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN locked_until TEXT")


def init_pilot_database() -> dict[str, Any]:
    """Create tables and seed a first admin if the DB is empty."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'analyst', 'viewer')),
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                failed_login_count INTEGER NOT NULL DEFAULT 0,
                locked_until TEXT,
                last_login_at TEXT
            );

            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                action TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS supplier_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                username TEXT NOT NULL,
                filename TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                data_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sentinel_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                username TEXT NOT NULL,
                mode TEXT NOT NULL,
                event_count INTEGER NOT NULL,
                total_exposure_usd REAL NOT NULL,
                results_json TEXT NOT NULL
            );
            """
        )
        _ensure_user_columns(conn)
        count = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
        if count == 0:
            mode = security_mode()
            username = os.getenv("SUPPLIER_APP_ADMIN_USER", "admin")
            password_set = "SUPPLIER_APP_ADMIN_PASSWORD" in os.environ
            password = os.getenv("SUPPLIER_APP_ADMIN_PASSWORD", "ChangeMe123!")
            if mode == "production" and not password_set:
                return {
                    "db_path": str(DB_PATH),
                    "default_admin_user": username,
                    "default_password_in_use": False,
                    "requires_first_admin_setup": True,
                    "security_mode": mode,
                }
            validate_password_strength(password)
            conn.execute(
                """
                INSERT INTO users (username, password_hash, role, is_active, created_at)
                VALUES (?, ?, 'admin', 1, ?)
                """,
                (username, hash_password(password), utc_now_iso()),
            )
            conn.execute(
                """
                INSERT INTO audit_logs (timestamp, username, role, action, details_json)
                VALUES (?, ?, 'admin', 'system.seed_admin', ?)
                """,
                (
                    utc_now_iso(),
                    username,
                    json.dumps({"default_password_used": "SUPPLIER_APP_ADMIN_PASSWORD" not in os.environ}),
                ),
            )

    return {
        "db_path": str(DB_PATH),
        "default_admin_user": os.getenv("SUPPLIER_APP_ADMIN_USER", "admin"),
        "default_password_in_use": security_mode() != "production" and "SUPPLIER_APP_ADMIN_PASSWORD" not in os.environ,
        "requires_first_admin_setup": False,
        "security_mode": security_mode(),
    }


def create_initial_admin(username: str, password: str) -> None:
    validate_password_strength(password)
    username = (username or "").strip()
    if not username:
        raise ValueError("Username is required.")
    with _connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
        if count:
            raise ValueError("Initial admin can only be created before any users exist.")
        conn.execute(
            """
            INSERT INTO users (username, password_hash, role, is_active, created_at)
            VALUES (?, ?, 'admin', 1, ?)
            """,
            (username, hash_password(password), utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO audit_logs (timestamp, username, role, action, details_json)
            VALUES (?, ?, 'admin', 'system.create_initial_admin', '{}')
            """,
            (utc_now_iso(), username),
        )


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    username = (username or "").strip()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT username, password_hash, role, is_active, failed_login_count, locked_until
            FROM users WHERE username = ?
            """,
            (username,),
        ).fetchone()
        if not row or not row["is_active"]:
            return None
        now = utc_now_iso()
        if row["locked_until"] and row["locked_until"] > now:
            return None
        if not verify_password(password, row["password_hash"]):
            failed = int(row["failed_login_count"] or 0) + 1
            locked_until = None
            if failed >= MAX_FAILED_ATTEMPTS:
                from datetime import timedelta

                locked_until = (datetime.now(UTC) + timedelta(minutes=LOCKOUT_MINUTES)).isoformat(timespec="seconds")
            conn.execute(
                "UPDATE users SET failed_login_count = ?, locked_until = ? WHERE username = ?",
                (failed, locked_until, username),
            )
            return None
        conn.execute(
            "UPDATE users SET last_login_at = ?, failed_login_count = 0, locked_until = NULL WHERE username = ?",
            (utc_now_iso(), username),
        )
        return {"username": row["username"], "role": row["role"]}


def log_audit(username: str, role: str, action: str, details: dict[str, Any] | None = None) -> None:
    safe_details = details or {}
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO audit_logs (timestamp, username, role, action, details_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (utc_now_iso(), username or "anonymous", role or "unknown", action, json.dumps(safe_details, default=str)),
        )


def list_audit_logs(limit: int = 250) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, username, role, action, details_json
            FROM audit_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_users() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT username, role, is_active, created_at, last_login_at
            FROM users
            ORDER BY username
            """
        ).fetchall()
    return [dict(row) for row in rows]


def create_user(username: str, password: str, role: str, actor: dict[str, str]) -> None:
    username = (username or "").strip()
    role = (role or "").strip().lower()
    if not username:
        raise ValueError("Username is required.")
    if role not in VALID_ROLES:
        raise ValueError("Role must be admin, analyst, or viewer.")
    validate_password_strength(password)

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO users (username, password_hash, role, is_active, created_at)
            VALUES (?, ?, ?, 1, ?)
            """,
            (username, hash_password(password), role, utc_now_iso()),
        )
    log_audit(actor["username"], actor["role"], "admin.create_user", {"created_username": username, "role": role})


def set_user_active(username: str, is_active: bool, actor: dict[str, str]) -> None:
    with _connect() as conn:
        conn.execute("UPDATE users SET is_active = ? WHERE username = ?", (1 if is_active else 0, username))
    log_audit(actor["username"], actor["role"], "admin.set_user_active", {"target": username, "is_active": is_active})


def change_user_password(username: str, new_password: str, actor: dict[str, str]) -> None:
    validate_password_strength(new_password)
    with _connect() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE username = ?", (hash_password(new_password), username))
    log_audit(actor["username"], actor["role"], "admin.change_password", {"target": username})


def save_supplier_upload(username: str, filename: str, df: pd.DataFrame) -> int:
    payload = df.to_json(orient="records", date_format="iso")
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO supplier_uploads (timestamp, username, filename, row_count, data_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (utc_now_iso(), username, filename, len(df), payload),
        )
        upload_id = int(cur.lastrowid)
    return upload_id


def save_sentinel_scan(username: str, mode: str, impacts: list[Any]) -> int:
    rows = []
    for impact in impacts:
        rows.append(
            {
                "title": impact.article.title,
                "source": impact.article.source,
                "url": impact.article.url,
                "severity": impact.severity,
                "severity_score": impact.severity_score,
                "disruption_type": impact.disruption_type,
                "affected_suppliers": impact.affected_suppliers,
                "affected_countries": impact.affected_countries,
                "affected_categories": impact.affected_categories,
                "estimated_exposure_usd": impact.estimated_exposure_usd,
                "summary": impact.summary,
                "recommended_actions": impact.recommended_actions,
                "confidence": impact.confidence,
                "analysis_method": impact.analysis_method,
            }
        )
    total_exposure = sum(float(row["estimated_exposure_usd"] or 0) for row in rows)
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO sentinel_scans (timestamp, username, mode, event_count, total_exposure_usd, results_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (utc_now_iso(), username, mode, len(rows), total_exposure, json.dumps(rows, default=str)),
        )
        scan_id = int(cur.lastrowid)
    return scan_id


def list_sentinel_scans(limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, username, mode, event_count, total_exposure_usd
            FROM sentinel_scans
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def has_mutation_access(user: dict[str, str]) -> bool:
    return user.get("role") in MUTATING_ROLES


def is_admin(user: dict[str, str]) -> bool:
    return user.get("role") == "admin"
