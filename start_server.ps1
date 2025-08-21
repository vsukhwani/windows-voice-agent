# Windows Voice Agent Server Startup Script
Write-Host "Starting Windows Voice Agent Server..." -ForegroundColor Green
Write-Host "Server will run on http://localhost:7860" -ForegroundColor Yellow
Write-Host "To stop the server, use: Get-Process python | Stop-Process" -ForegroundColor Red

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

# Start server in background
Start-Process -FilePath "python" -ArgumentList "server\bot_windows_simple.py", "--host", "0.0.0.0", "--port", "7860" -WindowStyle Hidden

Write-Host "Server started successfully!" -ForegroundColor Green
Write-Host "Check if it's running with: netstat -an | findstr :7860" -ForegroundColor Cyan
