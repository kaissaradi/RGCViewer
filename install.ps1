$ErrorActionPreference = "Stop"

$Repo       = "https://github.com/kaissaradi/RGCViewer.git"
$InstallDir = if ($env:ENCORE_HOME) { $env:ENCORE_HOME } else { Join-Path $env:USERPROFILE ".encore" }
$BinDir     = Join-Path $env:USERPROFILE ".local\bin"
$MinPython  = [version]"3.10"

function Info  { param($msg) Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn  { param($msg) Write-Host "WARN: $msg" -ForegroundColor Yellow }
function Fail  { param($msg) Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# --- pick an environment manager --------------------------------------------
# A plain venv built on top of an Anaconda python is broken on Windows: the venv
# is never activated through conda, so conda's *base* Library\bin stays ahead of
# it on PATH and Qt6Core.dll binds to base's older MSVC runtime, failing with
# "DLL load failed ... The specified procedure could not be found." Use a conda
# environment whenever conda is available so activation happens properly.
$CondaCmd = (Get-Command conda -ErrorAction SilentlyContinue)
$UseConda = [bool]$CondaCmd

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
if ($UseConda) {
    Info "Using conda at $($CondaCmd.Source)"
    $PythonVer = "python 3.12 (conda)"
} else {
    if (-not $PythonCmd) { Fail "Python >= $MinPython is required but not found. Install it from python.org." }
    $PythonVer = & $PythonCmd --version
    Info "Using $PythonVer"
}

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

# --- environment ------------------------------------------------------------
$Venv = Join-Path $InstallDir ".venv"

if ($UseConda) {
    # A previous run may have left a plain venv here; conda create refuses to
    # write into a non-empty directory, and that venv is exactly what we are
    # replacing, so clear it out.
    if ((Test-Path $Venv) -and -not (Test-Path (Join-Path $Venv "conda-meta"))) {
        Warn "Replacing the old virtual environment in $Venv with a conda environment"
        Remove-Item -Recurse -Force $Venv
    }
    if (-not (Test-Path (Join-Path $Venv "conda-meta"))) {
        Info "Creating conda environment in $Venv"
        & conda create -y -p $Venv "python=3.12"
        if ($LASTEXITCODE -ne 0) { Fail "Could not create a conda environment at $Venv." }
    }
    # Always go through "conda run", which activates the environment first. That
    # is the whole point: activation puts this env's Library\bin ahead of conda
    # base's on PATH, so Qt6Core.dll binds to a matching MSVC runtime.
    $Run = @("conda", "run", "--no-capture-output", "-p", $Venv, "python")
} else {
    if (-not (Test-Path $Venv)) {
        Info "Creating virtual environment"
        & $PythonCmd -m venv $Venv
        if ($LASTEXITCODE -ne 0) { Fail "Could not create a virtual environment at $Venv." }
    }
    $Run = @((Join-Path $Venv "Scripts\python.exe"))
}

$RunExe  = $Run[0]
$RunArgs = @($Run | Select-Object -Skip 1)
function RunPy { & $RunExe @($RunArgs + $args) }

Info "Installing dependencies (this may take a few minutes on first run)"
RunPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Fail "Failed to upgrade pip." }

RunPy -m pip install -e $InstallDir
if ($LASTEXITCODE -ne 0) { Fail "Dependency installation failed. Scroll up for the pip error." }

# --- verify Qt bindings -----------------------------------------------------
# qtpy reports a generic QtBindingsNotFoundError when the binding is missing OR
# when its DLLs fail to load, so import PyQt6 directly for a useful error. Go
# through qt_bootstrap, the same way the app does, or a conda Qt on PATH will
# fail this check even though Encore itself would start fine.
Push-Location $InstallDir
# stderr from a native command becomes a terminating error under
# $ErrorActionPreference = "Stop", so relax it while we capture the output.
$PrevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$QtCheck = (RunPy -c "import qt_bootstrap as b; b.prefer_bundled_qt(); import PyQt6.QtCore" 2>&1 | Out-String)
$QtOk = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $PrevEAP
Pop-Location

if (-not $QtOk) {
    Write-Host ""
    Write-Host ($QtCheck | Out-String)
    Fail @"
PyQt6 could not be imported, so Encore will not start.

  * 'DLL load failed' -> Qt is loading a mismatched DLL from elsewhere on your
        PATH. This is usually an Anaconda base environment. Installing conda's
        own Python here normally avoids it; if you are seeing this anyway, run
        the diagnostics and send the output to whoever maintains Encore:
            conda run -p "$Venv" python -c "import qt_bootstrap as b; b.explain_qt_import_failure()"
  * 'No module named PyQt6' -> your Python version may have no PyQt6 wheel.
        $PythonVer is in use; Python 3.10-3.13 are known to work.
"@
}

# --- create launcher shim ---------------------------------------------------
if (-not (Test-Path $BinDir)) { New-Item -ItemType Directory -Path $BinDir -Force | Out-Null }

$ShimPath = Join-Path $BinDir "encore.cmd"
$Ps1Path  = Join-Path $BinDir "encore.ps1"
$VenvExe  = Join-Path $Venv "Scripts\encore.exe"

if ($UseConda) {
    # Launch through "conda run" as well -- calling the exe directly would skip
    # activation and reintroduce the DLL mismatch the conda env exists to avoid.
    Set-Content -Path $ShimPath -Value "@echo off`r`nconda run --no-capture-output -p `"$Venv`" encore %*" -Encoding ASCII
    Set-Content -Path $Ps1Path  -Value "& conda run --no-capture-output -p `"$Venv`" encore @args" -Encoding UTF8
} else {
    Set-Content -Path $ShimPath -Value "@echo off`r`n`"$VenvExe`" %*" -Encoding ASCII
    Set-Content -Path $Ps1Path  -Value "& `"$VenvExe`" @args" -Encoding UTF8
}

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
