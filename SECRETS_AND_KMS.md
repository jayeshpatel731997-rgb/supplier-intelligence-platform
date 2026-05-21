# Secrets and KMS

The code includes cloud-ready abstractions without requiring cloud services locally.

## Secret providers

- `EnvSecretProvider`
- `StreamlitSecretsProvider`
- `AwsSecretsManagerProvider` placeholder
- `AzureKeyVaultProvider` placeholder
- `GcpSecretManagerProvider` placeholder

## KMS providers

- `LocalDevKMSProvider` for reversible local development only
- AWS/Azure/GCP placeholders for production envelope encryption

Never log `SecretValue.value`; its string and repr forms are redacted.

## Rotation

Rotate API keys through the tenant admin/API key flow. Rotate cloud secrets in the external secrets manager, then restart services or reload configuration.
