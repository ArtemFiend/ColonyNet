$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$AppPython = Join-Path $ProjectRoot ".venv-app\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $AppPython)) {
    $AppPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $AppPython)) {
    throw "Python environment is missing. Run: py -3.10 -m venv .venv; then install requirements-app.txt."
}
Push-Location $ProjectRoot
try {
    & $AppPython -m full_pipline.colony_pipeline_app @args
    if ($LASTEXITCODE -ne 0) { throw "ColonyNet exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
