param(
    [string]$DatabaseUrl = $env:DATABASE_URL,
    [string]$OutputDir = "backups"
)

if (-not $DatabaseUrl) {
    Write-Error "DATABASE_URL is required."
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$stamp = Get-Date -Format "yyyyMMddTHHmmssZ"
$path = Join-Path $OutputDir "postgres-$stamp.dump"
pg_dump $DatabaseUrl -Fc -f $path
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output @{ backup = $path; status = "created" }
