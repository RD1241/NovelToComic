# NovelToComic - One-Click Startup Script
# Usage: .\start_server.ps1
#
# This script:
#   1. Kills ALL Ollama processes (server + tray app)
#   2. Starts ollama serve directly so OLLAMA_MODELS is correctly inherited
#   3. Verifies llama3 is visible
#   4. Checks all HuggingFace models on D drive
#   5. Launches the FastAPI server

$OLLAMA_MODEL_PATH = "D:\AI_Models\Ollama"
$LLM_MODEL        = "llama3"
$VENV_PYTHON      = ".\venv\Scripts\python.exe"
$UVICORN          = ".\venv\Scripts\uvicorn"

# --- Set env vars for all child processes spawned from this shell ---
$env:OLLAMA_MODELS         = $OLLAMA_MODEL_PATH
$env:HF_HOME               = "D:\AI_Models\HuggingFace"
$env:HF_HUB_CACHE          = "D:\AI_Models\HuggingFace"
$env:HUGGINGFACE_HUB_CACHE = "D:\AI_Models\HuggingFace"

# Also persist at User level so future shells inherit it automatically
[System.Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $OLLAMA_MODEL_PATH, "User")

Write-Host ""
Write-Host "=== NovelToComic Startup ===" -ForegroundColor Cyan
Write-Host "OLLAMA_MODELS = $env:OLLAMA_MODELS"
Write-Host "HF_HOME       = $env:HF_HOME"
Write-Host ""

# --- Step 1: Kill ALL Ollama processes (server + tray app) ---
# The tray app ("ollama app") spawns ollama without OLLAMA_MODELS so we kill both.
Write-Host "[1/4] Stopping all Ollama processes..." -ForegroundColor Yellow
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

# --- Step 2: Start ollama serve directly from this shell ---
# This inherits OLLAMA_MODELS = D:\AI_Models\Ollama correctly.
Write-Host "[2/4] Starting ollama serve with OLLAMA_MODELS=$OLLAMA_MODEL_PATH ..." -ForegroundColor Yellow
Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow -PassThru | Out-Null
Start-Sleep -Seconds 6

# --- Step 3: Verify llama3 is visible ---
# BUG FIX: Use Out-String to get a single string so -match works as a boolean.
# If you use an array, PowerShell -notmatch returns filtered elements, not a bool.
Write-Host "[3/4] Checking LLM model ($LLM_MODEL)..." -ForegroundColor Yellow
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

# --- Step 4: Check HuggingFace models on D drive ---
Write-Host ""
Write-Host "[4/4] Checking HuggingFace models on D drive..." -ForegroundColor Yellow
& $VENV_PYTHON check_models.py

# --- Launch FastAPI server ---
Write-Host ""
Write-Host ">>> Starting FastAPI server at http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "    Press Ctrl+C to stop BOTH the server and Ollama." -ForegroundColor Gray
Write-Host ""

try {
    & $UVICORN api.main:app --host 127.0.0.1 --port 8000
} finally {
    Write-Host ""
    Write-Host ">>> Stopping NovelToComic and cleaning up..." -ForegroundColor Yellow
    
    # Kill Ollama when the app stops so it doesn't stay in the background
    Get-Process | Where-Object { $_.Name -like "*ollama*" } | ForEach-Object {
        Write-Host "      Stopping: $($_.Name) (PID $($_.Id))"
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
    
    Write-Host ">>> Cleanup complete. Goodbye!" -ForegroundColor Green
}
