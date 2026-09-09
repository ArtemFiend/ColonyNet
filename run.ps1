$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Environment missing. Run: py -3.10 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e ."
}
Push-Location $RepoRoot
try {
    & $Python -m colonynet @args
    if ($LASTEXITCODE -ne 0) { throw "ColonyNet exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
