@echo off
echo ========================================
echo      Hindi Voice Agent Startup
echo ========================================
echo.
echo हिंदी वॉयस एजेंट शुरू हो रहा है...
echo (Hindi Voice Agent Starting...)
echo.
echo Server will be available at: http://localhost:7861
echo Client will be available at: http://localhost:3000
echo.
echo Press Ctrl+C in either window to stop
echo ========================================
echo.

cd /d "%~dp0"

REM Start Hindi server in background
echo Starting Hindi Server...
start "Hindi Voice Agent Server" cmd /c "venv\Scripts\python.exe server\bot_hindi.py --host 0.0.0.0 --port 7861 & pause"

REM Wait 10 seconds for server to start
echo Waiting for server to initialize...
timeout /t 10 /nobreak >nul

REM Start client in background  
echo Starting Client...
start "Voice Agent Client" cmd /c "cd client && npm run dev"

echo.
echo ========================================
echo Hindi Voice Agent started!
echo हिंदी वॉयस एजेंट शुरू हो गया!
echo.
echo Server: http://localhost:7861
echo Client: http://localhost:3000
echo.
echo Test Hindi support: http://localhost:7861/api/test-hindi
echo.
echo Check the opened windows for status.
echo You can now use the Hindi voice agent at: http://localhost:3000
echo अब आप हिंदी में बात कर सकते हैं!
echo ========================================
pause
