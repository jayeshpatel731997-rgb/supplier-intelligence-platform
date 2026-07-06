# Upload Malware And Content Scanning Requirements

## Current State

The platform already bounds uploads by filename, extension, MIME type, size, and
tenant-scoped storage key. Local/demo mode stores files under the configured
local upload path. Staging/production readiness requires S3-compatible object
storage configuration before `/ready` can pass.

This update adds a deterministic `eicar-test` / `staging-safe` scanner adapter.
It rejects the harmless EICAR-style scanner test signature so tests can prove
that a scanner finding blocks storage/ingestion. It is not real malware
inspection.

## Real Staging Requirement

Managed staging should use:

- S3-compatible object storage with tenant-scoped keys.
- A scanner service such as ClamAV daemon, ICAP, or a managed vendor scanner.
- Quarantine behavior for scanner findings and scanner errors.
- A clear reject/allow policy:
  - reject known malware or disallowed content;
  - fail closed when scanning is required but unavailable;
  - allow only clean files to proceed to ingestion;
  - never log uploaded file contents or scanner secrets.
- Audit records that capture tenant, actor, filename, storage key, scanner
  provider, result, request ID, and timestamp without storing secret values.

## Validation Checklist

1. Set `SUPPLIER_UPLOAD_STORAGE_PROVIDER=s3`.
2. Configure bucket, endpoint, region, access key, secret key, and key prefix.
3. Set `SUPPLIER_UPLOAD_SCANNER_REQUIRED=true`.
4. Set `SUPPLIER_UPLOAD_SCANNER_PROVIDER=<clamav|icap|vendor>`.
5. Set `SUPPLIER_UPLOAD_SCANNER_ENDPOINT_URL=<scanner-endpoint>`.
6. Upload a clean file and verify ingestion succeeds.
7. Upload the scanner test file and verify rejection/quarantine.
8. Stop the scanner and verify uploads fail closed.
9. Review audit logs and confirm no file contents or secrets are emitted.

## Evidence Status

Current repository evidence proves application decision plumbing only. It does
not prove a real scanner service, object storage bucket policy, quarantine
workflow, or managed staging scanner SLA.
