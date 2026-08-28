$ErrorActionPreference = "Stop"

$Repo       = "https://github.com/kaissaradi/RGCViewer.git"
$InstallDir = if ($env:RGCVIEWER_HOME) { $env:RGCVIEWER_HOME } else { Join-Path $env:USERPROFILE ".rgcviewer" }
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
} else {
    Info "Cloning RGCViewer into $InstallDir"
    git clone $Repo $InstallDir
}

# --- virtual environment ----------------------------------------------------
$Venv = Join-Path $InstallDir ".venv"
if (-not (Test-Path $Venv)) {
    Info "Creating virtual environment"
    & $PythonCmd -m venv $Venv
}

$VenvPython = Join-Path $Venv "Scripts\python.exe"

Info "Installing dependencies (this may take a few minutes on first run)"
& $VenvPython -m pip install --upgrade pip -q
& $VenvPython -m pip install -e $InstallDir -q

# --- create launcher shim ---------------------------------------------------
if (-not (Test-Path $BinDir)) { New-Item -ItemType Directory -Path $BinDir -Force | Out-Null }

$ShimPath = Join-Path $BinDir "rgcviewer.cmd"
$VenvExe  = Join-Path $Venv "Scripts\rgcviewer.exe"

Set-Content -Path $ShimPath -Value "@echo off`r`n`"$VenvExe`" %*" -Encoding ASCII

# Also create a ps1 wrapper for PowerShell users
$Ps1Path = Join-Path $BinDir "rgcviewer.ps1"
Set-Content -Path $Ps1Path -Value "& `"$VenvExe`" @args" -Encoding UTF8

# --- PATH advice -------------------------------------------------------------
$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($UserPath -notlike "*$BinDir*") {
    Info "Adding $BinDir to your user PATH"
    [Environment]::SetEnvironmentVariable("PATH", "$BinDir;$UserPath", "User")
    $env:PATH = "$BinDir;$env:PATH"
    Warn "Restart your terminal for PATH changes to take effect."
}

Info "RGCViewer installed! Run it with:"
Write-Host ""
Write-Host "    rgcviewer"
Write-Host ""
Write-Host "  Options:"
Write-Host "    rgcviewer --debug"
Write-Host "    rgcviewer --kilosort-dir C:\path\to\run"
Write-Host "    rgcviewer --dat-file C:\path\to\raw.dat"
Write-Host ""
