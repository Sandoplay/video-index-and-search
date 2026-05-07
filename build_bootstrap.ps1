$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Developer .venv not found. Create it first, then run: .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python -c "import PyInstaller" *> $null
$PyInstallerAvailable = $LASTEXITCODE -eq 0
$ErrorActionPreference = $PreviousErrorActionPreference

if (-not $PyInstallerAvailable) {
    & $Python -m pip install pyinstaller
}

$PythonBase = & $Python -c "import sys; print(sys.base_prefix)"
$env:TCL_LIBRARY = Join-Path $PythonBase "tcl\tcl8.6"
$env:TK_LIBRARY = Join-Path $PythonBase "tcl\tk8.6"

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --console `
    --name bootstrap `
    --distpath (Join-Path $RepoRoot "dist") `
    --workpath (Join-Path $RepoRoot "build") `
    --specpath $RepoRoot `
    (Join-Path $RepoRoot "bootstrap_launcher.py")

$ExePath = Join-Path $RepoRoot "dist\bootstrap.exe"
if (-not (Test-Path $ExePath)) {
    throw "Build finished, but dist\bootstrap.exe was not created."
}

Write-Host "Created $ExePath"
