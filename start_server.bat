@echo off
echo Starting Windows Voice Agent Server...
echo Server will run on http://localhost:7860
echo Press Ctrl+C to stop the server

cd /d "%~dp0"
call venv\Scripts\activate.bat
python server\bot_windows_simple.py --host 0.0.0.0 --port 7860

pause
