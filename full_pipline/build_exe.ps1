param(
    [switch]$NoClean
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$Python = Join-Path $ProjectRoot ".venv-app\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}

if (-not (Test-Path $Python)) {
    $Python = "python"
}

Push-Location $ProjectRoot
try {
    $Args = @("-m", "PyInstaller", "full_pipline\ColonyNetPipeline.spec", "--noconfirm")
    if (-not $NoClean) {
        $Args += "--clean"
    }
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
