$ErrorActionPreference = "Stop"

$Repo       = "https://github.com/kaissaradi/RGCViewer.git"
$InstallDir = if ($env:ENCORE_HOME) { $env:ENCORE_HOME } else { Join-Path $env:USERPROFILE ".encore" }
$BinDir     = Join-Path $env:USERPROFILE ".local\bin"
$MinPython  = [version]"3.10"

function Info  { param($msg) Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn  { param($msg) Write-Host "WARN: $msg" -ForegroundColor Yellow }
function Fail  { param($msg) Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# --- locate python ---------------------------------------------------------
$PythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver -and [version]$ver -ge $MinPython) {
            $PythonCmd = $cmd
            break
        }
    } catch {}
}
if (-not $PythonCmd) { Fail "Python >= $MinPython is required but not found. Install it from python.org." }

$PythonVer = & $PythonCmd --version
Info "Using $PythonVer"

# --- clone or update --------------------------------------------------------
if (Test-Path (Join-Path $InstallDir ".git")) {
    Info "Updating existing installation in $InstallDir"
    git -C $InstallDir pull --ff-only
    if ($LASTEXITCODE -ne 0) { Fail "git pull failed in $InstallDir." }
} else {
    Info "Cloning Encore into $InstallDir"
    git clone $Repo $InstallDir
    if ($LASTEXITCODE -ne 0) { Fail "git clone failed. Is git installed and the network reachable?" }
}

# --- virtual environment ----------------------------------------------------
$Venv = Join-Path $InstallDir ".venv"
if (-not (Test-Path $Venv)) {
    Info "Creating virtual environment"
    & $PythonCmd -m venv $Venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create a virtual environment at $Venv." }
}

$VenvPython = Join-Path $Venv "Scripts\python.exe"

Info "Installing dependencies (this may take a few minutes on first run)"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Fail "Failed to upgrade pip." }

& $VenvPython -m pip install -e $InstallDir
if ($LASTEXITCODE -ne 0) { Fail "Dependency installation failed. Scroll up for the pip error." }

# --- verify Qt bindings -----------------------------------------------------
# qtpy reports a generic QtBindingsNotFoundError when the binding is missing OR
# when its DLLs fail to load, so import PyQt6 directly for a useful error. Go
# through qt_bootstrap, the same way the app does, or a conda Qt on PATH will
# fail this check even though Encore itself would start fine.
Push-Location $InstallDir
$QtCheck = & $VenvPython -c "import qt_bootstrap as b; b.prefer_bundled_qt(); import PyQt6.QtCore" 2>&1
$QtOk = ($LASTEXITCODE -eq 0)
Pop-Location

if (-not $QtOk) {
    Write-Host ""
    Write-Host ($QtCheck | Out-String)
    Fail @"
PyQt6 could not be imported, so Encore will not start.

  * 'DLL load failed' -> another Qt6 on your system is shadowing PyQt6's own.
        Try launching Encore from a plain PowerShell window rather than an
        Anaconda one, or install the Microsoft Visual C++ Redistributable:
        https://aka.ms/vs/17/release/vc_redist.x64.exe
  * 'No module named PyQt6' -> your Python version may have no PyQt6 wheel.
        $PythonVer is in use; Python 3.10-3.13 are known to work.
"@
}

# --- create launcher shim ---------------------------------------------------
if (-not (Test-Path $BinDir)) { New-Item -ItemType Directory -Path $BinDir -Force | Out-Null }

$ShimPath = Join-Path $BinDir "encore.cmd"
$VenvExe  = Join-Path $Venv "Scripts\encore.exe"

Set-Content -Path $ShimPath -Value "@echo off`r`n`"$VenvExe`" %*" -Encoding ASCII

$Ps1Path = Join-Path $BinDir "encore.ps1"
Set-Content -Path $Ps1Path -Value "& `"$VenvExe`" @args" -Encoding UTF8

# --- PATH advice -------------------------------------------------------------
$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($UserPath -notlike "*$BinDir*") {
    Info "Adding $BinDir to your user PATH"
    [Environment]::SetEnvironmentVariable("PATH", "$BinDir;$UserPath", "User")
    $env:PATH = "$BinDir;$env:PATH"
    Warn "Restart your terminal for PATH changes to take effect."
}

Info "Encore installed! Run it with:"
Write-Host ""
Write-Host "    encore"
Write-Host ""
Write-Host "  Options:"
Write-Host "    encore --debug"
Write-Host "    encore --kilosort-dir C:\path\to\run"
Write-Host "    encore --dat-file C:\path\to\raw.dat"
Write-Host ""
