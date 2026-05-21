"""Production-oriented auth primitives used by API tests and future services."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models import User


class PasswordPolicyError(ValueError):
    pass


def validate_password_strength(password: str) -> None:
    value = password or ""
    if len(value) < 12:
        raise PasswordPolicyError("Password must be at least 12 characters.")
    if not re.search(r"[A-Z]", value):
        raise PasswordPolicyError("Password must include an uppercase letter.")
    if not re.search(r"[a-z]", value):
        raise PasswordPolicyError("Password must include a lowercase letter.")
    if not re.search(r"\d", value):
        raise PasswordPolicyError("Password must include a number.")
    if not re.search(r"[^A-Za-z0-9]", value):
        raise PasswordPolicyError("Password must include a symbol.")


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return f"pbkdf2_sha256${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, salt_b64, digest_b64 = stored_hash.split("$", 2)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            base64.b64decode(salt_b64),
            310_000,
        )
        return hmac.compare_digest(actual, base64.b64decode(digest_b64))
    except Exception:
        return False


class AuthService:
    VALID_ROLES = {"viewer", "analyst", "admin", "platform_admin", "org_admin", "risk_manager", "auditor"}

    def __init__(self, session: Session, max_failed_attempts: int = 5, lockout_minutes: int = 15):
        self.session = session
        self.max_failed_attempts = max_failed_attempts
        self.lockout_minutes = lockout_minutes

    def create_user(self, username: str, password: str, role: str = "viewer") -> User:
        validate_password_strength(password)
        role = role.lower()
        if role not in self.VALID_ROLES:
            raise ValueError("Role must be a valid local or tenant role.")
        user = User(username=username.strip(), password_hash=hash_password(password), role=role)
        self.session.add(user)
        self.session.flush()
        return user

    def get_user(self, username: str) -> User | None:
        return self.session.scalar(select(User).where(User.username == username.strip()))

    def authenticate(self, username: str, password: str) -> User | None:
        user = self.get_user(username)
        now = datetime.now(UTC)
        if user is None or not user.is_active:
            return None
        if user.locked_until and user.locked_until > now:
            return None
        if verify_password(password, user.password_hash):
            user.failed_login_count = 0
            user.locked_until = None
            user.last_login_at = now
            return user
        user.failed_login_count += 1
        if user.failed_login_count >= self.max_failed_attempts:
            user.locked_until = now + timedelta(minutes=self.lockout_minutes)
        return None

    def user_has_role(self, username: str, required_role: str) -> bool:
        user = self.get_user(username)
        if user is None or not user.is_active:
            return False
        levels = {
            "viewer": 1,
            "auditor": 1,
            "analyst": 2,
            "risk_manager": 3,
            "admin": 4,
            "org_admin": 4,
            "platform_admin": 5,
        }
        return levels.get(user.role, 0) >= levels.get(required_role, 0)
