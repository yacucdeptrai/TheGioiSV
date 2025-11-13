# Requires -Version 5.1
<#
.SYNOPSIS
  Start/stop/status/restart WildLens backend(frontend) on Windows.

USAGE:
  .\deploy-all.ps1 -Command start
  .\deploy-all.ps1 -Command stop
  .\deploy-all.ps1 -Command restart
  .\deploy-all.ps1 -Command status

This script:
- Starts backend: python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
- Starts frontend: sets NEXT_PUBLIC_API_URL then runs npm run dev
- Uses Start-Process -PassThru so child processes continue after script exits
#>

param(
    [ValidateSet("start","stop","restart","status")]
    [string]$Command = "start",

    [string]$BackendHost = "127.0.0.1",
    [int]$BackendPort = 8000,

    # Adjust these defaults to match your repository layout
    [string]$BackendRelPath = "..\Wildlens-Web\Backend",
    [string]$FrontendRelPath = "..\Wildlens-Web\Frontend",

    [switch]$OpenBrowser
)

Set-StrictMode -Version Latest
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $ScriptDir

# Logs & pid dir
$LogsDir = Join-Path $ScriptDir "logs"
If (!(Test-Path $LogsDir)) { New-Item -Path $LogsDir -ItemType Directory | Out-Null }

$backendPidFile  = Join-Path $LogsDir "backend.pid"
$frontendPidFile = Join-Path $LogsDir "frontend.pid"

$backendOutLog  = Join-Path $LogsDir "backend-out.log"
$backendErrLog  = Join-Path $LogsDir "backend-err.log"
$frontendOutLog = Join-Path $LogsDir "frontend-out.log"
$frontendErrLog = Join-Path $LogsDir "frontend-err.log"

function Write-Warn([string]$m){ Write-Host $m -ForegroundColor Yellow }
function Write-Err([string]$m){ Write-Host $m -ForegroundColor Red }

function Is-ProcessRunning($pid){
    try {
        $p = Get-Process -Id $pid -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Start-Backend {
    Write-Host "=== Starting Backend (uvicorn) ===" -ForegroundColor Cyan

    $backendPath = Resolve-Path (Join-Path $ScriptDir $BackendRelPath) -ErrorAction SilentlyContinue
    if (-not $backendPath) {
        Write-Warn "Backend directory not found at '$BackendRelPath'. Adjust \$BackendRelPath in the script."
        return $null
    }
    $backendPath = $backendPath.Path

    $args = "-m uvicorn main:app --host $BackendHost --port $BackendPort --reload"
    $pythonExe = "python"  # change to full path if you need specific interpreter

    Write-Host "Backend working dir: $backendPath"
    Write-Host "Command: $pythonExe $args"

    try {
        $backendProc = Start-Process -FilePath $pythonExe -ArgumentList $args -WorkingDirectory $backendPath -PassThru -NoNewWindow
        Start-Sleep -Milliseconds 300
        $backendProc.Id | Out-File -FilePath $backendPidFile -Encoding ascii
        Write-Host "Backend started (PID $($backendProc.Id))"
        return $backendProc
    } catch {
        Write-Err "Failed to start backend: $($_.Exception.Message)"
        return $null
    }
}

function Start-Frontend {
    Write-Host "=== Starting Frontend (next dev) ===" -ForegroundColor Cyan

    $frontendPath = Resolve-Path (Join-Path $ScriptDir $FrontendRelPath) -ErrorAction SilentlyContinue
    if (-not $frontendPath) {
        Write-Warn "Frontend directory not found at '$FrontendRelPath'. Adjust \$FrontendRelPath in the script."
        return $null
    }
    $frontendPath = $frontendPath.Path

    $pkg = Join-Path $frontendPath "package.json"
    if (-not (Test-Path $pkg)) {
        Write-Err "No package.json found in frontend directory: $frontendPath"
        return $null
    }

    # If node_modules/.bin/next missing, try npm install
    $nextBin = Join-Path $frontendPath "node_modules\.bin\next.cmd"
    if (-not (Test-Path $nextBin)) {
        Write-Warn "Next.js binary not found. Running 'npm install' (this may take a while)..."
        try {
            Push-Location $frontendPath
            & npm install
            Pop-Location
        } catch {
            Pop-Location
            Write-Err "npm install failed. Ensure Node & npm are installed and working."
            return $null
        }
    }

    # Build command that sets env var for the cmd session and runs npm run dev
    $apiUrl = "http://$BackendHost`:$BackendPort"
    Write-Host "Setting NEXT_PUBLIC_API_URL=$apiUrl for frontend process"

    # Use cmd.exe /c "set \"NEXT_PUBLIC_API_URL=...\" && npm run dev"
    $cmd = "cmd.exe"
    $cmdArgs = "/c set `"NEXT_PUBLIC_API_URL=$apiUrl`" && npm run dev"

    Write-Host "Frontend working dir: $frontendPath"
    Write-Host "Command: $cmd $cmdArgs"

    try {
        $frontendProc = Start-Process -FilePath $cmd -ArgumentList $cmdArgs -WorkingDirectory $frontendPath -PassThru -NoNewWindow
        Start-Sleep -Milliseconds 300
        $frontendProc.Id | Out-File -FilePath $frontendPidFile -Encoding ascii
        Write-Host "Frontend started (PID $($frontendProc.Id))"
        return $frontendProc
    } catch {
        Write-Err "Failed to start frontend: $($_.Exception.Message)"
        return $null
    }
}

# PASTE THIS INTO deploy-all.ps1, REPLACING THE OLD FUNCTIONS

function Stop-ByPidFile($pidFile, [string]$name){
    if (Test-Path $pidFile) {
        $processId = (Get-Content $pidFile -ErrorAction SilentlyContinue) -as [int]
        if ($processId -and (Is-ProcessRunning $processId)) {
            try {
                Write-Host "Stopping $name (PID $processId)..."
                Stop-Process -Id $processId -Force -ErrorAction Stop
                Remove-Item $pidFile -ErrorAction SilentlyContinue
                Write-Host "$name stopped."
            } catch {
                Write-Warn "Could not stop $name (PID $processId): $_"
            }
        } else {
            Write-Warn "$name pid file exists but process not running. Removing pid file."
            Remove-Item $pidFile -ErrorAction SilentlyContinue
        }
    } else {
        Write-Warn "No pid file found for $name."
    }
}

function Show-Status {
    Write-Host "=== Status ===" -ForegroundColor Cyan
    if (Test-Path $backendPidFile) {
        $processId = (Get-Content $backendPidFile) -as [int]
        if ($processId -and (Is-ProcessRunning $processId)) {
            Write-Host "Backend: running (PID $processId)"
        } else {
            Write-Host "Backend: pid file present but process not running."
        }
    } else {
        Write-Host "Backend: not started"
    }

    if (Test-Path $frontendPidFile) {
        $processId = (Get-Content $frontendPidFile) -as [int]
        if ($processId -and (Is-ProcessRunning $processId)) {
            Write-Host "Frontend: running (PID $processId)"
        } else {
            Write-Host "Frontend: pid file present but process not running."
        }
    } else {
        Write-Host "Frontend: not started"
    }
}

# Main command dispatcher
switch ($Command) {
    "start" {
        # Start backend first
        $b = Start-Backend
        Start-Sleep -Seconds 1

        # Then frontend
        $f = Start-Frontend
        Start-Sleep -Seconds 1

        $backendUrl = "http://$BackendHost`:$BackendPort"
        $frontendUrl = "http://localhost:3000"

        Write-Host ""
        Write-Host "=============================================================" -ForegroundColor DarkCyan
        Write-Host " WildLens is running (or starting)."
        if ($b) { Write-Host " Backend : $backendUrl   (PID $($b.Id))" }
        else     { Write-Host " Backend : failed to start." }
        if ($f) { Write-Host " Frontend: $frontendUrl  (PID $($f.Id))" }
        else    { Write-Host " Frontend: failed to start." }
        Write-Host " Logs    : $LogsDir"
        Write-Host " Stop    : .\deploy-all.ps1 -Command stop" -ForegroundColor Yellow
        Write-Host " Restart : .\deploy-all.ps1 -Command restart" -ForegroundColor Yellow
        Write-Host " Status  : .\deploy-all.ps1 -Command status" -ForegroundColor Yellow
        Write-Host "=============================================================" -ForegroundColor DarkCyan

        if ($OpenBrowser) {
            try { Start-Process $frontendUrl } catch { Write-Warn "Could not open browser automatically." }
        }
    }

    "stop" {
        Write-Host "Stopping services..."
        Stop-ByPidFile $frontendPidFile "Frontend"
        Stop-ByPidFile $backendPidFile  "Backend"
    }

    "restart" {
        & $MyInvocation.MyCommand.Path -Command stop
        Start-Sleep -Seconds 1
        & $MyInvocation.MyCommand.Path -Command start
    }

    "status" {
        Show-Status
    }
}

# Return to original dir
Pop-Location
