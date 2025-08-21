@echo off
echo ========================================
echo      Local Voice Agent Startup
echo ========================================
echo.
echo Starting Server and Client...
echo Server will be available at: http://localhost:7860
echo Client will be available at: http://localhost:3000
echo.
echo Press Ctrl+C in either window to stop
echo ========================================
echo.

cd /d "%~dp0"

REM Start server in background
echo Starting Server...
start "Voice Agent Server" cmd /c "venv\Scripts\python.exe server\bot_windows_simple.py --host 0.0.0.0 --port 7860 & pause"

REM Wait 10 seconds for server to start
echo Waiting for server to initialize...
timeout /t 10 /nobreak >nul

REM Start client in background  
echo Starting Client...
start "Voice Agent Client" cmd /c "cd client && npm run dev"

echo.
echo ========================================
echo Both server and client have been started!
echo.
echo Server: http://localhost:7860
echo Client: http://localhost:3000
echo.
echo Check the opened windows for status.
echo You can now use the voice agent at: http://localhost:3000
echo ========================================
pause
