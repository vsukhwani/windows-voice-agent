# हिंदी वॉयस एजेंट - Hindi Voice Agent

![Hindi Support](https://img.shields.io/badge/Language-Hindi-orange.svg)
![Voice AI](https://img.shields.io/badge/Voice-AI-blue.svg)
![Local](https://img.shields.io/badge/100%25-Local-green.svg)

**Windows पर पूर्ण हिंदी समर्थन के साथ स्थानीय वॉयस एजेंट**  
**Local voice agent with full Hindi language support on Windows**

## 🇮🇳 विशेषताएं (Features)

- **🎤 Hindi Speech Recognition** - Whisper SMALL model with Hindi language support
- **🧠 Hindi Conversation** - Gemma3 12B LLM with excellent Hindi capabilities  
- **🔊 Text-to-Speech** - Kokoro TTS attempting Hindi pronunciation
- **⚡ Real-time Response** - <2 seconds voice-to-voice latency
- **🏠 100% Local** - No external APIs, complete privacy
- **🖥️ Windows Optimized** - Full Windows 10/11 support

## 🚀 त्वरित शुरुआत (Quick Start)

### एक-क्लिक सेटअप (One-Click Setup)
```bash
# Start Hindi voice agent
start_hindi_voice_agent.bat
```

Browser में जाएं: `http://localhost:3000`  
हिंदी में बोलना शुरू करें! (Start speaking in Hindi!)

## 📋 आवश्यकताएं (Prerequisites)

- **Python 3.10+**
- **Node.js 18+**  
- **Ollama** with Gemma3 12B model
- **Kokoro TTS Models** (pre-configured)
- Windows 10/11

## 🔧 विस्तृत सेटअप (Detailed Setup)

### 1. Ollama और मॉडल इंस्टॉल करें (Install Ollama and Models)

```bash
# Ollama download from https://ollama.ai/
# Install Hindi-capable model
ollama pull gemma3:12b

# Verify Ollama is running
ollama serve
```

### 2. प्रोजेक्ट सेटअप (Project Setup)

```bash
# Clone repository
git clone https://github.com/vsukhwani/windows-voice-agent.git
cd windows-voice-agent

# Switch to Hindi branch
git checkout hindi-support

# Setup virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r server\requirements_windows.txt
cd client && npm install && cd ..
```

### 3. हिंदी वॉयस एजेंट चलाएं (Run Hindi Voice Agent)

#### विकल्प 1: स्वचालित (Automatic)
```bash
start_hindi_voice_agent.bat
```

#### विकल्प 2: मैन्युअल (Manual)
```bash
# Terminal 1: Hindi Server  
venv\Scripts\activate
python server\bot_hindi.py --host 0.0.0.0 --port 7861

# Terminal 2: Client
cd client && npm run dev
```

### 4. परीक्षण (Testing)

```bash
# Run comprehensive Hindi tests
python test_hindi_voice_agent.py

# Test server endpoints
# http://localhost:7861/api/test-hindi
# http://localhost:7861/api/test-ollama
```

## 🎛️ कॉन्फ़िगरेशन (Configuration)

### भाषा सेटिंग्स (Language Settings)
```env
# .env file
WHISPER_LANGUAGE="hi"
WHISPER_MODEL="small" 
SYSTEM_LANGUAGE="hindi"
```

### Hindi Bot Configuration
```python
# server/bot_hindi.py
SYSTEM_INSTRUCTION_HINDI = """
आप Pipecat हैं, एक दोस्ताना और सहायक चैटबॉट।
बातचीत की शुरुआत "नमस्ते, मैं Pipecat हूँ!" कहकर करें।
"""

# Whisper STT for Hindi
stt = WhisperSTTService(
    model=Model.SMALL,  # Better Hindi support
    language="hi",      # Hindi language code
    initial_prompt="नमस्ते, आप कैसे हैं? यह एक हिंदी बातचीत है।"
)
```

## 🎯 कैसे उपयोग करें (How to Use)

1. **Browser खोलें**: `http://localhost:3000`
2. **Microphone Permission दें**: Allow microphone access
3. **हिंदी में बोलें**: "नमस्ते, आप कैसे हैं?"
4. **जवाब सुनें**: AI will respond in Hindi
5. **बातचीत जारी रखें**: Continue conversation naturally

### उदाहरण बातचीत (Example Conversation)
```
👤 उपयोगकर्ता: "नमस्ते, आप कौन हैं?"
🤖 Pipecat: "नमस्ते! मैं Pipecat हूँ। मैं आपकी सहायता के लिए यहाँ हूँ।"

👤 उपयोगकर्ता: "मौसम कैसा है?"  
🤖 Pipecat: "मुझे मौसम की वर्तमान जानकारी नहीं है, लेकिन मैं अन्य विषयों पर चर्चा कर सकता हूँ।"
```

## 🐛 समस्या निवारण (Troubleshooting)

### सामान्य समस्याएं (Common Issues)

**🎤 माइक्रोफोन काम नहीं कर रहा**
```bash
# Test microphone
# Open: microphone_test.html in browser
# Check browser permissions
```

**🔗 कनेक्शन एरर**
```bash
# Check Ollama is running
ollama serve
curl http://localhost:11434/api/tags

# Restart Hindi server
start_hindi_voice_agent.bat
```

**🧠 हिंदी समझ नहीं रहा**
```bash
# Verify Hindi model is downloaded
python test_hindi_voice_agent.py

# Check Whisper language setting in .env
WHISPER_LANGUAGE="hi"
```

**📚 NLTK डेटा एरर**
```bash
python download_nltk_data.py
```

### डिबग टूल्स (Debug Tools)

```bash
# Comprehensive Hindi testing
python test_hindi_voice_agent.py

# Server status check
curl http://localhost:7861/api/test-hindi

# Component testing
python test_voice_input.py
```

## ⚡ प्रदर्शन (Performance)

### अपेक्षित विलंबता (Expected Latency)
- **CPU Only**: 2-3 seconds voice-to-voice
- **GPU (CUDA)**: 1-2 seconds voice-to-voice
- **Memory Usage**: ~3-4GB RAM for Hindi models

### अनुकूलन (Optimization)
```env
# For faster processing (less accurate)
WHISPER_MODEL="base"

# For better accuracy (slower)  
WHISPER_MODEL="medium"

# GPU acceleration (if available)
WHISPER_DEVICE="cuda"
WHISPER_COMPUTE_TYPE="float16"
```

## 🎵 वॉयस विकल्प (Voice Options)

Kokoro TTS में उपलब्ध आवाजें:
```python
# server/bot_hindi.py में बदलें
voice="af_heart"    # American Female (clearest for Hindi)
voice="af_bella"    # American Female (bright)  
voice="am_adam"     # American Male
voice="bf_emma"     # British Female
```

**नोट**: Kokoro मूल रूप से अंग्रेजी के लिए है, हिंदी उच्चारण में एक्सेंट हो सकता है।

## 📊 भाषा समर्थन स्थिति (Language Support Status)

| Component | Hindi Support | Quality | Notes |
|-----------|---------------|---------|-------|
| **Whisper STT** | ✅ Full | Excellent | SMALL model recommended |
| **Gemma3 LLM** | ✅ Full | Excellent | Natural Hindi conversation |
| **Kokoro TTS** | ⚠️ Limited | Fair | English accent on Hindi |
| **UI/Interface** | ✅ Full | Good | Bilingual documentation |

## 🔮 भविष्य के सुधार (Future Improvements)

- **🎯 Native Hindi TTS** - Better Hindi pronunciation
- **🗣️ More Voice Options** - Different Hindi accents
- **🎨 Hindi UI** - Complete Hindi interface
- **📱 Mobile Support** - Android/iOS apps
- **🎪 Regional Languages** - Support for more Indian languages

## 🤝 योगदान (Contributing)

हिंदी समर्थन बेहतर बनाने के लिए योगदान करें:

1. **Fork** करें और **hindi-support** branch पर काम करें
2. नई सुविधाएं जोड़ें या बग फिक्स करें  
3. **Pull Request** भेजें
4. **Issues** रिपोर्ट करें

### विशेष योगदान क्षेत्र:
- Hindi TTS voice models
- Regional language support
- UI/UX improvements
- Performance optimization

## 📄 लाइसेंस (License)

यह प्रोजेक्ट मूल रिपॉजिटरी के लाइसेंस का पालन करता है।

## 📞 सहायता (Support)

समस्या का समाधान:
1. **test_hindi_voice_agent.py** चलाएं
2. **README** और troubleshooting section देखें  
3. **GitHub Issues** में रिपोर्ट करें
4. **Discussions** में सवाल पूछें

---

## 🎉 हिंदी वॉयस AI का आनंद लें!
## 🎉 Enjoy Hindi Voice AI!

**स्थानीय, निजी, और शक्तिशाली - आपका अपना हिंदी वॉयस असिस्टेंट**  
**Local, Private, and Powerful - Your own Hindi Voice Assistant**
