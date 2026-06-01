"""Upload safety checks, scanner hooks, and tenant-scoped storage adapters."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.config import Settings


SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,254}$")
SAFE_TENANT_RE = re.compile(r"[^A-Za-z0-9._-]+")


class UploadSafetyError(ValueError):
    """Raised when an upload violates safety policy."""


@dataclass(slots=True)
class UploadScanResult:
    ok: bool
    scanner: str
    message: str = ""


@dataclass(slots=True)
class StoredUpload:
    provider: str
    key: str
    bucket: str = ""


def _csv_items(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def validate_upload_filename(filename: str) -> str:
    safe_name = (filename or "").strip()
    if (
        not safe_name
        or "/" in safe_name
        or "\\" in safe_name
        or safe_name in {".", ".."}
        or not SAFE_FILENAME_RE.fullmatch(safe_name)
    ):
        raise UploadSafetyError("Unsafe upload filename.")
    return safe_name


def validate_upload_type(filename: str, content_type: str, settings: Settings) -> str:
    safe_name = validate_upload_filename(filename)
    extension = Path(safe_name).suffix.lower()
    allowed_extensions = _csv_items(settings.allowed_upload_extensions)
    if extension not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise UploadSafetyError(f"Unsupported upload file type. Allowed extensions: {allowed}")

    allowed_mime_types = _csv_items(settings.allowed_upload_mime_types)
    detected_mime = (content_type or "").split(";", 1)[0].strip().lower()
    if detected_mime in {"application/octet-stream", "binary/octet-stream"}:
        detected_mime = ""
    if detected_mime and allowed_mime_types and detected_mime not in allowed_mime_types:
        allowed = ", ".join(sorted(allowed_mime_types))
        raise UploadSafetyError(f"Unsupported upload content type. Allowed MIME types: {allowed}")
    return safe_name


def scan_upload(data: bytes, settings: Settings) -> UploadScanResult:
    provider = settings.upload_scanner_provider
    if provider == "none":
        if settings.is_production and settings.upload_scanner_required:
            raise UploadSafetyError("Upload scanner is required but not configured.")
        return UploadScanResult(ok=True, scanner="none", message="Scanning not configured.")
    if not settings.upload_scanner_endpoint_url:
        raise UploadSafetyError("Upload scanner endpoint is not configured.")

    # Placeholder integration point for a future ICAP/ClamAV/vendor scanner.
    # Keep file contents in memory only; do not log or emit payload bytes.
    _ = data
    return UploadScanResult(ok=True, scanner=provider, message="Scanner integration stub accepted upload.")


def tenant_upload_key(tenant_id: str, filename: str, key_prefix: str = "uploads") -> str:
    tenant = SAFE_TENANT_RE.sub("-", tenant_id.strip()).strip("-") or "tenant"
    prefix = SAFE_TENANT_RE.sub("-", key_prefix.strip()).strip("-/")
    name = validate_upload_filename(filename)
    object_name = f"{uuid.uuid4().hex}-{name}"
    if prefix:
        return f"{tenant}/{prefix}/{object_name}"
    return f"{tenant}/{object_name}"


class LocalUploadStorage:
    provider = "local"

    def __init__(self, root: str) -> None:
        self.root = Path(root)

    def save(self, tenant_id: str, filename: str, data: bytes, settings: Settings) -> StoredUpload:
        key = tenant_upload_key(tenant_id, filename, settings.upload_storage_key_prefix)
        root = self.root.resolve()
        target = (root / key).resolve()
        if root not in target.parents:
            raise UploadSafetyError("Unsafe tenant-scoped upload path.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return StoredUpload(provider=self.provider, key=key)


class S3UploadStorage:
    provider = "s3"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def save(self, tenant_id: str, filename: str, data: bytes, settings: Settings) -> StoredUpload:
        key = tenant_upload_key(tenant_id, filename, settings.upload_storage_key_prefix)
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for S3-compatible upload storage.") from exc

        client = boto3.client(
            "s3",
            endpoint_url=settings.upload_storage_endpoint_url or None,
            region_name=settings.upload_storage_region or None,
            aws_access_key_id=settings.upload_storage_access_key_id or None,
            aws_secret_access_key=settings.upload_storage_secret_access_key or None,
        )
        client.put_object(Bucket=settings.upload_storage_bucket, Key=key, Body=data)
        return StoredUpload(provider=self.provider, key=key, bucket=settings.upload_storage_bucket)


def build_upload_storage(settings: Settings):
    if settings.upload_storage_provider == "local":
        return LocalUploadStorage(settings.upload_storage_path)
    if settings.upload_storage_provider == "s3":
        return S3UploadStorage(settings)
    raise UploadSafetyError(f"Unsupported upload storage provider: {settings.upload_storage_provider}.")
