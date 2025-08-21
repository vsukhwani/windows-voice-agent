# Windows Voice Agent Server Stop Script
Write-Host "Stopping Windows Voice Agent Server..." -ForegroundColor Yellow

# Find and stop Python processes running the bot
$processes = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -eq "python" }
if ($processes) {
    Write-Host "Found $($processes.Count) Python process(es)" -ForegroundColor Cyan
    $processes | Stop-Process -Force
    Write-Host "Server stopped successfully!" -ForegroundColor Green
} else {
    Write-Host "No Python processes found running the server" -ForegroundColor Red
}

# Check if port is still in use
$portCheck = netstat -an | findstr :7860
if ($portCheck) {
    Write-Host "Warning: Port 7860 is still in use" -ForegroundColor Red
} else {
    Write-Host "Port 7860 is now free" -ForegroundColor Green
}
