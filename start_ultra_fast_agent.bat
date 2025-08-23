@echo off
echo ========================================
echo    🚀 ULTRA LOW-LATENCY VOICE AGENT 🚀
echo ========================================
echo.
echo This version includes aggressive optimizations:
echo - Ultra-fast VAD (50ms speech detection)
echo - Tiny Whisper model for speed
echo - Micro-chunk streaming TTS (4 words)
echo - Speed-optimized LLM settings
echo - Performance monitoring
echo.
echo Expected latencies:
echo - Speech detection: 50-100ms
echo - Transcription: 100-300ms  
echo - LLM response: 200-800ms
echo - TTS start: 50-150ms
echo - TOTAL PERCEIVED: 400-1200ms
echo.
echo Server: http://localhost:7860
echo Client: http://localhost:3000
echo Performance API: http://localhost:7860/api/performance
echo ========================================
echo.

cd /d "%~dp0"

REM Check if Ollama is running
echo 🔍 Checking Ollama status...
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ❌ Ollama is not running! Please start Ollama first.
    echo    Run: ollama serve
    pause
    exit /b 1
)
echo ✅ Ollama is running

REM Start ultra-fast server
echo 🚀 Starting ultra-fast voice agent server...
start "Ultra-Fast Voice Agent" cmd /c "venv\Scripts\python.exe server\bot_ultra_fast.py --host 0.0.0.0 --port 7860 & pause"

REM Wait for server startup
echo ⏳ Waiting for server to initialize...
timeout /t 3 /nobreak >nul

REM Test server
echo 🧪 Testing server response...
curl -s http://localhost:7860/api/performance >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Server may still be starting...
    timeout /t 2 /nobreak >nul
)

REM Start client
echo 🎨 Starting client interface...
start "Ultra-Fast Client" cmd /c "cd client && npm run dev"

echo.
echo ========================================
echo 🎯 Ultra low-latency voice agent ready!
echo.
echo 📊 Performance dashboard:
echo    http://localhost:7860/api/performance
echo 🎙️ Voice interface:
echo    http://localhost:3000
echo 🔧 Server API:
echo    http://localhost:7860
echo.
echo 💡 Tips for best performance:
echo - Speak clearly and pause briefly between sentences
echo - Use a good microphone for clean audio input
echo - Ensure stable internet connection
echo - Keep other CPU-intensive apps closed
echo ========================================
pause
