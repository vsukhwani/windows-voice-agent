@echo off
echo ========================================
echo      Git Repository Setup Helper
echo ========================================
echo.
echo This script will help you push your changes to GitHub
echo.
echo Prerequisites:
echo 1. Create a new repository on GitHub at: https://github.com/vsukhwani
echo 2. Name it something like: windows-voice-agent
echo 3. DO NOT initialize with README
echo.
echo After creating the repository, you'll get a URL like:
echo https://github.com/vsukhwani/windows-voice-agent.git
echo.
pause
echo.
echo ========================================
echo Setting up git repository...
echo ========================================

REM Add all changes
echo Adding all files to git...
git add .

REM Show status
echo.
echo Current git status:
git status

echo.
echo Ready to commit? 
pause

REM Commit changes
echo.
echo Committing changes...
git commit -m "feat: Add Windows support with Ollama, Whisper, and Kokoro TTS

- Add Windows-specific bot configurations (bot_windows_simple.py, bot_windows.py)
- Integrate Ollama for local LLM (Gemma3 12B)
- Add ONNX-based Kokoro TTS service
- Add Whisper STT with optimized settings
- Create automated startup/shutdown scripts
- Add comprehensive troubleshooting tools (microphone_test.html, test_voice_input.py)
- Update documentation with detailed Windows setup guide
- Add NLTK data management utilities
- Optimize VAD settings for better voice detection
- Add environment configuration for Windows paths"

echo.
echo ========================================
echo Now you need to:
echo 1. Update the remote URL to your GitHub repository
echo 2. Push the changes
echo ========================================
echo.
echo Example commands (replace with your actual repository URL):
echo git remote set-url origin https://github.com/vsukhwani/YOUR-REPO-NAME.git
echo git push -u origin main
echo.
pause
