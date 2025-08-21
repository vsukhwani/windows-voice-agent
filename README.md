# Local Voice Agents with Pipecat

![screenshot](assets/debug-console-screenshot.png)

Pipecat is an open-source, vendor-neutral framework for building real-time voice (and video) AI applications.

This repository contains an example of a voice agent running with **all local models** on both **macOS** and **Windows**. On an M-series Mac, you can achieve voice-to-voice latency of <800 ms with relatively strong models.

## 🌟 Multi-Platform Support

### macOS Version ([server/bot.py](server/bot.py))
- **STT**: MLX Whisper
- **LLM**: Gemma3 12B (via LM Studio)
- **TTS**: Kokoro TTS (MLX-based)
- **VAD**: Silero VAD + smart-turn v2

### Windows Version ([server/bot_windows_simple.py](server/bot_windows_simple.py))
- **STT**: Whisper (faster-whisper backend)
- **LLM**: Gemma3 12B (via Ollama)
- **TTS**: Kokoro TTS (ONNX-based)
- **VAD**: Silero VAD

## 🚀 Quick Start

### For Windows Users
```bash
# 1. Start everything automatically
start_voice_agent.bat

# 2. Open browser to http://localhost:3000
# 3. Allow microphone permissions and start talking!

# To stop everything:
stop_voice_agent.bat
```

### For macOS Users  
```bash
# 1. Start server
cd server/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python bot.py

# 2. Start client (in new terminal)
cd client/
npm install
npm run dev
```

## 📋 Prerequisites

### Windows
- **Python 3.10+** 
- **Node.js 18+**
- **Ollama** (for local LLM) - [Download here](https://ollama.ai/)
- **Kokoro Model Files** (automatically configured in this repo)

### macOS
- **Python 3.10+**
- **Node.js 18+** 
- **LM Studio** (for local LLM) - [Download here](https://lmstudio.ai/)

## 🛠️ Detailed Setup

### Windows Setup (Recommended Path)

1. **Install Ollama and Pull Model**
   ```bash
   # Install Ollama from https://ollama.ai/
   ollama pull gemma3:12b
   ```

2. **Clone and Setup Project**
   ```bash
   git clone <this-repo>
   cd macos-local-voice-agents
   
   # Create virtual environment
   python -m venv venv
   venv\Scripts\activate
   
   # Install dependencies  
   pip install -r server\requirements_windows.txt
   cd client
   npm install
   cd ..
   ```

3. **Configure Kokoro TTS**
   - Model files are pre-configured at: `C:/Users/[Username]/kokoro/kokoro-onnx/examples/`
   - The `.env` file is already set up with correct paths

4. **Start the Voice Agent**
   ```bash
   # Option 1: Use batch files (easiest)
   start_voice_agent.bat
   
   # Option 2: Manual start
   # Terminal 1: Server
   venv\Scripts\activate
   python server\bot_windows_simple.py --host 0.0.0.0 --port 7860
   
   # Terminal 2: Client  
   cd client
   npm run dev
   ```

### macOS Setup

See original instructions - use LM Studio for the LLM and MLX-based components.

## 🎤 Usage

1. **Open Browser**: Navigate to `http://localhost:3000`
2. **Allow Microphone**: Grant microphone permissions when prompted
3. **Start Talking**: Click the microphone button and speak
4. **First Response**: You should hear "Hello, I'm Pipecat!" 
5. **Have Conversations**: The AI will respond to your voice in real-time

## 🔧 Configuration

### Environment Variables (`.env`)
```env
# Kokoro TTS Model Paths (Windows)
KOKORO_MODEL_PATH="C:/Users/[Username]/kokoro/kokoro-onnx/examples/kokoro-v1.0.onnx"
KOKORO_VOICES_PATH="C:/Users/[Username]/kokoro/kokoro-onnx/examples/voices-v1.0.bin"

# Whisper Settings
WHISPER_DEVICE="cpu"           # or "cuda" for GPU
WHISPER_COMPUTE_TYPE="int8"    # or "float16" for GPU

# NLTK Data Path
NLTK_DATA="C:/Users/[Username]/AppData/Roaming/nltk_data"
```

### Available Voices
Change the voice in your bot file:
- `af_heart` (default) - American Female
- `af_bella` - American Female  
- `am_adam` - American Male
- `bf_emma` - British Female
- `bm_george` - British Male

## 🐛 Troubleshooting

### Audio Issues
1. **Test Microphone**: Open `microphone_test.html` in browser
2. **Browser Permissions**: Ensure microphone access is allowed
3. **Speak Clearly**: Talk directly into microphone with good volume
4. **Wait for Processing**: Allow 2-3 seconds after speaking

### Common Problems
- **"No audio frame received"**: Check microphone permissions
- **NLTK errors**: Run `python download_nltk_data.py`
- **Connection errors**: Ensure Ollama is running (`ollama serve`)
- **Model not found**: Verify paths in `.env` file

### Debug Tools
- **Server Logs**: Check terminal output for detailed debugging
- **Ollama Test**: Visit `http://localhost:7860/api/test-ollama`
- **Microphone Test**: Open `microphone_test.html` 
- **Debug Script**: Run `python test_voice_input.py`

## 📁 Project Structure
```

```
├── server/
│   ├── bot_windows_simple.py    # Windows voice agent (Ollama + ONNX)
│   ├── bot_windows.py           # Windows voice agent (LM Studio + ONNX)  
│   ├── bot.py                   # macOS voice agent (LM Studio + MLX)
│   ├── kokoro_tts_onnx.py      # ONNX-based Kokoro TTS service
│   ├── requirements_windows.txt # Windows Python dependencies
│   └── requirements.txt         # macOS Python dependencies
├── client/                      # Next.js React client
├── .env                         # Environment configuration
├── start_voice_agent.bat        # Windows: Start both server & client
├── stop_voice_agent.bat         # Windows: Stop all processes
├── microphone_test.html         # Browser-based microphone test
├── test_voice_input.py          # Voice input debugging script
└── download_nltk_data.py        # NLTK data setup script
```

## 🔗 Architecture

The bot and web client communicate using a **low-latency, local, serverless WebRTC connection**. For more information on serverless WebRTC, see the Pipecat [SmallWebRTCTransport docs](https://docs.pipecat.ai/server/services/transport/small-webrtc) and this [article](https://www.daily.co/blog/you-dont-need-a-webrtc-server-for-your-voice-agents/).

You can easily:
- **Swap models**: Change STT, LLM, or TTS services
- **Add tool calling**: Integrate MCP servers or custom functions
- **Customize pipeline**: Add parallel processing or custom steps
- **Configure interrupts**: Modify how the agent handles user interruptions
- **Change transport**: Switch to WebSocket or other Pipecat transports

## 📚 Resources

- **[Pipecat Documentation](https://docs.pipecat.ai/)** - Complete framework docs
- **[Voice AI Guide](https://voiceaiandvoiceagents.com/)** - Deep dive into voice AI architecture
- **[Voice UI Kit](https://github.com/pipecat-ai/voice-ui-kit)** - React components for voice interfaces
- **[Kokoro TTS](https://github.com/Blaizzy/mlx-audio)** - Local, high-quality TTS

## ⚡ Performance

- **macOS (M-series)**: <800ms voice-to-voice latency
- **Windows (CPU)**: ~1-2s voice-to-voice latency  
- **Windows (GPU)**: ~800ms-1s voice-to-voice latency
- **Memory Usage**: 2-4GB RAM for all models
- **Storage**: ~2GB for all model files

## 🤝 Contributing

Contributions are welcome! This repo demonstrates local voice AI capabilities and can be extended with:
- Additional TTS voices
- More efficient model configurations  
- UI/UX improvements
- Additional platform support
- Integration examples

## 📄 License

This project follows the original repository's license terms.