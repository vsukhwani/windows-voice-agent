# Windows Setup Guide - Updated Configuration

This guide helps you set up and run the voice agent on Windows using **Ollama** for LLM, **faster-whisper** for STT, and **kokoro-onnx** for TTS.

## 🚀 Quick Start (Recommended)

### One-Click Setup
```bash
# Start everything automatically
start_voice_agent.bat

# To stop everything  
stop_voice_agent.bat
```

This will start both server and client, and open the browser automatically.

## 📋 Prerequisites

- **Python 3.10+**  
- **Node.js 18+**
- **Ollama** - [Download from ollama.ai](https://ollama.ai/)
- Windows 10/11

## 🔧 Detailed Setup

### 1. Install Ollama and Download Models

```bash
# Install Ollama from https://ollama.ai/ 

# Pull the Gemma 3 12B model
ollama pull gemma3:12b

# Verify Ollama is running
ollama serve
```

### 2. Project Setup

```bash
# Clone the repository
git clone <this-repo>
cd macos-local-voice-agents

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r server\requirements_windows.txt

# Install client dependencies
cd client
npm install
cd ..
```

### 3. Model Configuration

The project is pre-configured to use Kokoro models from:
```
C:\Users\[YourUsername]\kokoro\kokoro-onnx\examples\
├── kokoro-v1.0.onnx     # Main TTS model (~310MB)
└── voices-v1.0.bin      # Voice data (~27MB)
```

The `.env` file is already configured with the correct paths.
pip install -r server/requirements_windows.txt
```

### 4. Run the Voice Agent

#### Option 1: Automated (Recommended)
```bash
# Start both server and client
start_voice_agent.bat

# Stop everything  
stop_voice_agent.bat
```

#### Option 2: Manual
```bash
# Terminal 1: Server
venv\Scripts\activate
python server\bot_windows_simple.py --host 0.0.0.0 --port 7860

# Terminal 2: Client (in new terminal)
cd client
npm run dev
```

### 5. Access the Application

1. Open browser to `http://localhost:3000`
2. Allow microphone permissions when prompted
3. You should hear "Hello, I'm Pipecat!" 
4. Click microphone button and start talking!

## 🎛️ Configuration

### Current Setup
- **LLM**: Ollama with Gemma3 12B (`http://127.0.0.1:11434/v1`)
- **STT**: Whisper BASE model (faster processing)
- **TTS**: Kokoro ONNX with `af_heart` voice
- **VAD**: Optimized sensitivity settings

### Environment File (`.env`)
```env
# Kokoro TTS Paths  
KOKORO_MODEL_PATH="C:/Users/[Username]/kokoro/kokoro-onnx/examples/kokoro-v1.0.onnx"
KOKORO_VOICES_PATH="C:/Users/[Username]/kokoro/kokoro-onnx/examples/voices-v1.0.bin"

# Whisper Settings
WHISPER_DEVICE="cpu"           # Use "cuda" if you have NVIDIA GPU
WHISPER_COMPUTE_TYPE="int8"    # Use "float16" for GPU

# NLTK Configuration
NLTK_DATA="C:/Users/[Username]/AppData/Roaming/nltk_data"
```

### Available Voices
Edit `server/bot_windows_simple.py` to change the voice:

```python
tts = KokoroOnnxTTSService(
    model_path=model_path,
    voices_path=voices_path,
    voice="af_bella",  # Change this line
    sample_rate=24000,
)
```

**Available options:**
- `af_heart` (default) - American Female (warm)
- `af_bella` - American Female (bright)  
- `af_nicole` - American Female (neutral)
- `am_adam` - American Male
- `bf_emma` - British Female
- `bm_george` - British Male

## 🐛 Troubleshooting

### Microphone Issues
1. **Test microphone**: Open `microphone_test.html` in your browser
2. **Check permissions**: Ensure browser allows microphone access
3. **Volume check**: Speak loudly and clearly into microphone
4. **Wait for processing**: Allow 2-3 seconds after speaking

### Common Problems

**"Connection error" / "No audio frame received"**
```bash
# Check if Ollama is running
ollama serve

# Test Ollama connection
curl http://localhost:11434/api/tags

# Visit server test endpoint
# Open: http://localhost:7860/api/test-ollama
```

**"NLTK data not found"**
```bash
# Download required NLTK data
python download_nltk_data.py
```

**"Kokoro model not found"**
- Verify model files exist at the paths in `.env`
- Check file permissions
- Ensure paths use forward slashes: `/` not `\`

**"WhisperSTT initialization failed"**
```bash
# First run downloads Whisper models (~1.5GB)
# This is automatic but may take time
# Check internet connection if it fails
```

### Debug Tools

**Server diagnostics:**
```bash
# Test voice input processing
python test_voice_input.py

# Check server logs for detailed errors  
python server\bot_windows_simple.py --host 0.0.0.0 --port 7860
```

**Ollama verification:**
```bash
# List available models
ollama list

# Test specific model
ollama run gemma3:12b "Hello, how are you?"
```

**Browser tools:**
- **Microphone test**: `microphone_test.html`
- **Developer console**: F12 in browser for client errors
- **Network tab**: Check WebRTC connection status

## ⚡ Performance Tips

### CPU-Only Setup (Default)
```env
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```
- Expected latency: 1-3 seconds
- Memory usage: ~2-3GB RAM
- Works on all Windows systems

### GPU Acceleration (Optional)  
```env
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```
- Requires NVIDIA GPU with CUDA
- Expected latency: 0.5-1 seconds  
- Memory usage: ~1-2GB VRAM + 2GB RAM

### Model Size Options
In `server/bot_windows_simple.py`, change Whisper model:

```python
stt = WhisperSTTService(
    model=Model.TINY,     # Fastest, least accurate
    # model=Model.BASE,   # Balanced (current default)  
    # model=Model.SMALL,  # Better accuracy, slower
    # model=Model.MEDIUM, # High accuracy, much slower
)
```

## 📁 Files Overview

**Main Components:**
- `server/bot_windows_simple.py` - **Primary Windows voice agent (Ollama)**
- `server/bot_windows.py` - Alternative Windows agent (LM Studio)
- `server/kokoro_tts_onnx.py` - ONNX-based TTS service
- `client/` - Next.js web interface

**Setup & Utilities:**
- `start_voice_agent.bat` - **One-click startup script**
- `stop_voice_agent.bat` - Stop all processes  
- `microphone_test.html` - Browser microphone testing
- `test_voice_input.py` - Voice processing diagnostics
- `download_nltk_data.py` - NLTK setup utility

**Configuration:**
- `.env` - **Main configuration file**
- `server/requirements_windows.txt` - Python dependencies

## 🚀 Next Steps

Once your voice agent is working:

1. **Customize the personality**: Edit the `SYSTEM_INSTRUCTION` in the bot file
2. **Add tool calling**: Integrate with APIs or local functions  
3. **Try different voices**: Change the Kokoro voice setting
4. **Optimize performance**: Adjust Whisper model size and compute settings
5. **Extend functionality**: Add custom processing steps or integrations

## 📞 Support

If you encounter issues:

1. **Check server logs** for detailed error messages
2. **Test components individually** using the debug tools  
3. **Verify prerequisites** (Ollama running, models downloaded)
4. **Try the microphone test page** to isolate audio issues
5. **Check the main README.md** for additional troubleshooting

The voice agent architecture is highly modular - you can swap out any component (STT, LLM, TTS) or add custom processing steps as needed.
