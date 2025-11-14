# Requires -Version 5.1
<#
.SYNOPSIS
  Stops, cleans (deletes venv, node_modules), and reinstalls all
  dependencies for the WildLens backend and frontend.

USAGE:
  .\rebuild-all.ps1
  .\rebuild-all.ps1 -SkipBackend (rebuild Frontend only)
#>

# =============================================================================
# Configuration (Must match deploy-all.ps1)
# =============================================================================

param(
    [string]$BackendRelPath = "..\Wildlens-Web\Backend",
    [string]$FrontendRelPath = "..\Wildlens-Web\Frontend",
    [string]$VenvName = ".venv",
    [switch]$SkipBackend = $false ## bypass Backend
)

Set-StrictMode -Version Latest
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $ScriptDir

# =============================================================================
# Helper Functions
# =============================================================================

function Write-Info([string]$m){ Write-Host $m -ForegroundColor Cyan }
function Write-Warn([string]$m){ Write-Host $m -ForegroundColor Yellow }
function Write-Err([string]$m){ Write-Host $m -ForegroundColor Red }
function Write-Success([string]$m){ Write-Host $m -ForegroundColor Green }

# =============================================================================
# Main Functions
# =============================================================================

function Stop-Services {
    Write-Info "=== 1. Stopping running services ==="
    $deployScript = Join-Path $ScriptDir "deploy-all.ps1"

    if (-not (Test-Path $deployScript)) {
        Write-Err "ERROR: 'deploy-all.ps1' not found at $deployScript."
        Write-Err "Cannot stop services. Please stop them manually before re-running this script."
        throw "deploy-all.ps1 not found"
    }

    try {
        # Call the deploy script (which you just fixed)
        & $deployScript -Command stop
    } catch {
        Write-Warn "Could not stop services completely. (They might not have been running)."
        Write-Warn "Error: $_"
    }

    # Give processes time to fully close
    Write-Host "Waiting 2 seconds for processes to shut down..."
    Start-Sleep -Seconds 2
}

function Rebuild-Backend {
    Write-Info "=== 2. Rebuilding Backend ==="
    $backendPath = (Resolve-Path (Join-Path $ScriptDir $BackendRelPath) -ErrorAction SilentlyContinue).Path
    if (-not $backendPath) {
        Write-Err "Backend directory not found at '$BackendRelPath'. Skipping."
        return
    }
    Write-Host "Found Backend at: $backendPath"

    $ReqFile = Join-Path $backendPath "requirements.txt"
    if (-not (Test-Path $ReqFile)) {
        Write-Err "'requirements.txt' not found in $backendPath. Skipping Backend setup."
        return
    }

    $VenvPath = Join-Path $backendPath $VenvName

    # --- Clean up Backend ---
    if (Test-Path $VenvPath) {
        Write-Warn "Deleting old virtual environment: $VenvPath"
        try {
            Remove-Item -Path $VenvPath -Recurse -Force -ErrorAction Stop
            Write-Success "Old venv deleted."
        } catch {
            Write-Err "ERROR: Could not delete $VenvPath. Check permissions or close code editors."
            Write-Err "Error: $_"
            throw "Could not clean Backend"
        }
    }

    # --- Reinstall Backend ---
    Write-Host "Creating new virtual environment (python -m venv $VenvName)..."
    try {
        Push-Location $backendPath
        & python -m venv $VenvName

        if ($LASTEXITCODE -ne 0) {
            throw "python -m venv failed with exit code $LASTEXITCODE"
        }

        Pop-Location
        Write-Success "Virtual environment created."
    } catch {
        Pop-Location
        Write-Err "ERROR: Failed to create venv. Ensure 'python' is in your PATH."
        Write-Err "Error: $_"
        return
    }

    Write-Host "Installing dependencies from requirements.txt..."
    $PipExe = Join-Path $VenvPath "Scripts" "pip.exe"
    if (-not (Test-Path $PipExe)) {
        Write-Err "ERROR: pip.exe not found at $PipExe. Installation failed."
        return
    }

    try {
        & $PipExe install -r $ReqFile

        if ($LASTEXITCODE -ne 0) {
            throw "'pip install' failed with exit code $LASTEXITCODE"
        }

        Write-Success "Backend dependencies installed successfully."
    } catch {
        Write-Err "ERROR: 'pip install' failed."
        Write-Err "Error: $_"
    }
}

function Rebuild-Frontend {
    Write-Info "=== 3. Rebuilding Frontend ==="
    $frontendPath = (Resolve-Path (Join-Path $ScriptDir $FrontendRelPath) -ErrorAction SilentlyContinue).Path
    if (-not $frontendPath) {
        Write-Err "Frontend directory not found at '$FrontendRelPath'. Skipping."
        return
    }
    Write-Host "Found Frontend at: $frontendPath"

    $PkgFile = Join-Path $frontendPath "package.json"
    if (-not (Test-Path $PkgFile)) {
        Write-Err "'package.json' not found in $frontendPath. Skipping Frontend setup."
        return
    }

    # --- Clean up Frontend ---
    $NodeModules = Join-Path $frontendPath "node_modules"
    $LockFile = Join-Path $frontendPath "package-lock.json"
    $NextCache = Join-Path $frontendPath ".next"

    foreach ($item in $NodeModules, $LockFile, $NextCache) {
        if (Test-Path $item) {
            Write-Warn "Deleting: $item"
            try {
                Remove-Item -Path $item -Recurse -Force -ErrorAction Stop
            } catch {
                Write-Err "ERROR: Could not delete $item. Check permissions or close code editors."
                Write-Err "Error: $_"
            }
        }
    }
    Write-Success "Cleaned up old Frontend files."

    # --- Reinstall Frontend ---
    Write-Host "Running 'npm install' (this may take a few minutes)..."
    try {
        Push-Location $frontendPath
        & npm install # Removed the incorrect "-ErrorAction Stop"

        if ($LASTEXITCODE -ne 0) {
            throw "'npm install' failed with exit code $LASTEXITCODE"
        }

        Pop-Location
        Write-Success "Frontend dependencies installed successfully."
    } catch {
        Pop-Location
        Write-Err "ERROR: 'npm install' failed. Check the npm log for details."
        Write-Err "Error: $_"
    }
}

# =============================================================================
# Execution
# =============================================================================

try {
    Stop-Services

    ## check if not bypass SkipBackend
    if (-not $SkipBackend) {
        Rebuild-Backend
    }
    else {
        Write-Warn "=== SKIPPING Backend Rebuild (-SkipBackend) ==="
    }

    Rebuild-Frontend

    Write-Host ""
    Write-Success "======================================================="
    Write-Success " REBUILD COMPLETE!"
    Write-Success " You can now start the project using:"
    Write-Info "   .\deploy-all.ps1 -Command start"
    Write-Success "======================================================="

} catch {
    Write-Err "Script stopped abruptly due to a critical error."
}

Pop-Location