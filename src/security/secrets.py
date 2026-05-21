"""Secrets manager and KMS abstractions with local-safe defaults."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SecretValue:
    name: str
    value: str

    def __str__(self) -> str:
        return "***" if self.value else ""

    def __repr__(self) -> str:
        return f"SecretValue(name={self.name!r}, value='***')"


@dataclass(frozen=True, slots=True)
class SecretCheck:
    ok: bool
    name: str
    message: str


class SecretProvider(Protocol):
    def get_secret(self, name: str, default: str = "") -> SecretValue:
        ...


class EnvSecretProvider:
    def get_secret(self, name: str, default: str = "") -> SecretValue:
        return SecretValue(name=name, value=os.getenv(name, default))


class StreamlitSecretsProvider:
    def get_secret(self, name: str, default: str = "") -> SecretValue:
        try:
            import streamlit as st

            value = st.secrets.get(name, default)
        except Exception:
            value = default
        return SecretValue(name=name, value=str(value or ""))


class AwsSecretsManagerProvider(EnvSecretProvider):
    """Placeholder adapter. Wire boto3 retrieval here in AWS deployments."""


class AzureKeyVaultProvider(EnvSecretProvider):
    """Placeholder adapter. Wire azure-keyvault-secrets here in Azure deployments."""


class GcpSecretManagerProvider(EnvSecretProvider):
    """Placeholder adapter. Wire google-cloud-secret-manager here in GCP deployments."""


class KMSProvider(Protocol):
    def encrypt(self, plaintext: str) -> str:
        ...

    def decrypt(self, ciphertext: str) -> str:
        ...


class LocalDevKMSProvider:
    """Reversible local obfuscation for development only, not production cryptography."""

    def __init__(self, master_key: str = "local-dev-key"):
        self.master_key = master_key or "local-dev-key"

    def encrypt(self, plaintext: str) -> str:
        key = self.master_key.encode("utf-8")
        data = plaintext.encode("utf-8")
        mixed = bytes(byte ^ key[index % len(key)] for index, byte in enumerate(data))
        return base64.urlsafe_b64encode(mixed).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        key = self.master_key.encode("utf-8")
        data = base64.urlsafe_b64decode(ciphertext.encode("ascii"))
        plain = bytes(byte ^ key[index % len(key)] for index, byte in enumerate(data))
        return plain.decode("utf-8")


class AwsKMSProvider(LocalDevKMSProvider):
    """Placeholder adapter. Use AWS KMS Encrypt/Decrypt in production."""


class AzureKeyVaultKMSProvider(LocalDevKMSProvider):
    """Placeholder adapter. Use Azure Key Vault cryptography clients in production."""


class GcpKMSProvider(LocalDevKMSProvider):
    """Placeholder adapter. Use Cloud KMS encrypt/decrypt in production."""


def require_production_secret(name: str, provider: SecretProvider, production: bool) -> SecretCheck:
    value = provider.get_secret(name)
    if production and not value.value:
        return SecretCheck(ok=False, name=name, message=f"{name} is required in production.")
    return SecretCheck(ok=True, name=name, message="configured" if value.value else "not configured")
