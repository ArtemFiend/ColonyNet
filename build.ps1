$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Create .venv and install the project first." }
Push-Location $RepoRoot
try {
    & $Python -m PyInstaller packaging\ColonyNet.spec --noconfirm --clean
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
