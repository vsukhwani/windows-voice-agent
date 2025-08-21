@echo off
echo ========================================
echo      Stopping Voice Agent
echo ========================================
echo.

REM Stop any running Node.js processes (client)
echo Stopping client processes...
taskkill /F /IM node.exe 2>nul
if %errorlevel% == 0 (
    echo ✓ Client stopped
) else (
    echo - No client processes found
)

REM Stop any Python processes running the server
echo Stopping server processes...
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| findstr "bot_windows_simple"') do (
    taskkill /F /PID %%i 2>nul
    if %errorlevel% == 0 echo ✓ Server stopped
)

echo.
echo ========================================
echo Voice Agent stopped!
echo ========================================
pause
