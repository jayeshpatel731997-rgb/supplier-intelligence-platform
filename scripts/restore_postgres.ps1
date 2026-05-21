param(
    [Parameter(Mandatory=$true)][string]$BackupPath,
    [string]$DatabaseUrl = $env:DATABASE_URL
)

if (-not $DatabaseUrl) {
    Write-Error "DATABASE_URL is required."
    exit 1
}
if (-not (Test-Path $BackupPath)) {
    Write-Error "Backup file not found: $BackupPath"
    exit 1
}

pg_restore --clean --if-exists --dbname $DatabaseUrl $BackupPath
exit $LASTEXITCODE
