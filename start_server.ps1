# NovelToComic - One-Click Startup Script
# Usage: .\start_server.ps1
#
# Steps:
#   1. Kills ALL Ollama processes (server + tray app)
#   2. Starts ollama serve so OLLAMA_MODELS is correctly inherited
#   3. Verifies llama3 is visible
#   4. Checks all HuggingFace models on D drive
#   5. Frees port 8000 if a stale process is holding it
#   6. Launches the FastAPI server

$OLLAMA_MODEL_PATH = "D:\AI_Models\Ollama"
$LLM_MODEL        = "llama3"
$VENV_PYTHON      = ".\venv\Scripts\python.exe"
$UVICORN          = ".\venv\Scripts\uvicorn"

# --- Set env vars for all child processes spawned from this shell ---
$env:OLLAMA_MODELS         = $OLLAMA_MODEL_PATH
$env:HF_HOME               = "D:\AI_Models\HuggingFace"
$env:HF_HUB_CACHE          = "D:\AI_Models\HuggingFace"
$env:HUGGINGFACE_HUB_CACHE = "D:\AI_Models\HuggingFace"

# Persist at User level so future shells inherit it automatically
[System.Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $OLLAMA_MODEL_PATH, "User")

Write-Host ""
Write-Host "=== NovelToComic Startup ===" -ForegroundColor Cyan
Write-Host "OLLAMA_MODELS = $env:OLLAMA_MODELS"
Write-Host "HF_HOME       = $env:HF_HOME"
Write-Host ""

# -----------------------------------------------------------------------
# Step 1: Kill ALL Ollama processes (server + tray app)
# -----------------------------------------------------------------------
Write-Host "[1/5] Stopping all Ollama processes..." -ForegroundColor Yellow
$killed = 0
Get-Process | Where-Object { $_.Name -like "*ollama*" } | ForEach-Object {
    Write-Host "      Killing: $($_.Name) (PID $($_.Id))"
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    $killed++
}
if ($killed -eq 0) {
    Write-Host "      No Ollama processes were running."
}
Start-Sleep -Seconds 3

# -----------------------------------------------------------------------
# Step 2: Start ollama serve — truly detached via Start-Process
# -----------------------------------------------------------------------
Write-Host "[2/5] Starting ollama serve with OLLAMA_MODELS=$OLLAMA_MODEL_PATH ..." -ForegroundColor Yellow

# $env:OLLAMA_MODELS is already set above; Start-Process inherits it.
# -WindowStyle Hidden uses Windows CreateProcess which creates a fully
# independent process that survives the parent PowerShell session.
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden

# Probe until port 11434 responds (max 30 s)
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:11434/" -Method HEAD -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop | Out-Null
        Write-Host "      Ollama ready after $($i+1)s." -ForegroundColor Green
        $ready = $true
        break
    } catch { }
}
if (-not $ready) {
    Write-Host "  ERROR: Ollama did not start in 30s." -ForegroundColor Red
    exit 1
}



# -----------------------------------------------------------------------
# Step 3: Verify llama3 is visible
# -----------------------------------------------------------------------
Write-Host "[3/5] Checking LLM model ($LLM_MODEL)..." -ForegroundColor Yellow
$modelList = (ollama list 2>&1) | Out-String
Write-Host $modelList.Trim()

if ($modelList -notmatch $LLM_MODEL) {
    Write-Host "  '$LLM_MODEL' not found - attempting pull from local cache..." -ForegroundColor Red
    ollama pull $LLM_MODEL
    Start-Sleep -Seconds 2
    $modelList = (ollama list 2>&1) | Out-String
    if ($modelList -notmatch $LLM_MODEL) {
        Write-Host "  ERROR: Could not load $LLM_MODEL. Check D:\AI_Models\Ollama" -ForegroundColor Red
        Write-Host "  Run: ollama pull llama3" -ForegroundColor Gray
        exit 1
    }
}
Write-Host "  llama3 OK" -ForegroundColor Green

# -----------------------------------------------------------------------
# Step 4: Check HuggingFace models on D drive
# -----------------------------------------------------------------------
Write-Host ""
Write-Host "[4/5] Checking HuggingFace models on D drive..." -ForegroundColor Yellow
& $VENV_PYTHON check_models.py

# -----------------------------------------------------------------------
# Step 5: Free port 8000 if a stale process is holding it
# -----------------------------------------------------------------------
Write-Host ""
Write-Host "[5/5] Checking if port 8000 is already in use..." -ForegroundColor Yellow

$netstatOutput = netstat -ano | Select-String "127.0.0.1:8000\s"
if ($netstatOutput) {
    $portPid = ($netstatOutput | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -First 1).Trim()
    if ($portPid -match '^\d+$') {
        Write-Host "      Port 8000 held by PID $portPid - killing it..." -ForegroundColor Red
        Stop-Process -Id ([int]$portPid) -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        Write-Host "      Port 8000 freed." -ForegroundColor Green
    }
} else {
    Write-Host "      Port 8000 is free." -ForegroundColor Green
}

# -----------------------------------------------------------------------
# Launch FastAPI server
# -----------------------------------------------------------------------
Write-Host ""
Write-Host ">>> Starting FastAPI server at http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "    Press Ctrl+C to stop BOTH the server and Ollama." -ForegroundColor Gray
Write-Host ""

try {
    & $UVICORN api.main:app --host 127.0.0.1 --port 8000
} finally {
    Write-Host ""
    Write-Host ">>> Stopping NovelToComic and cleaning up..." -ForegroundColor Yellow

    # Kill Ollama when the app stops so it does not stay in the background
    Get-Process | Where-Object { $_.Name -like "*ollama*" } | ForEach-Object {
        Write-Host "      Stopping: $($_.Name) (PID $($_.Id))"
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }

    Write-Host ">>> Cleanup complete. Goodbye!" -ForegroundColor Green
}
