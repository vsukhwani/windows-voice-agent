# Ultra Low-Latency Voice Agent Optimizations

This document details all the performance optimizations implemented to achieve ultra-low latency in voice conversations.

## 🎯 Latency Breakdown & Optimizations

### 1. Speech Detection (Target: 50-100ms)
**Optimizations Applied:**
- Ultra-aggressive VAD settings:
  - `confidence=0.3` (very sensitive)
  - `start_secs=0.05` (50ms detection time)
  - `stop_secs=0.3` (300ms stop detection)
  - `min_volume=0.3` (lower threshold)

### 2. Speech-to-Text (Target: 100-300ms)
**Optimizations Applied:**
- Whisper TINY model (smallest, fastest)
- Optimized parameters:
  - `condition_on_previous_text=False` (no context lookback)
  - `beam_size=1` (single beam search)
  - `best_of=1` (no alternative generation)
  - `no_speech_threshold=0.4` (faster silence detection)

### 3. LLM Response (Target: 200-800ms)
**Optimizations Applied:**
- Gemma3:270M model (smallest viable model)
- Reduced `max_tokens=150` (shorter responses)
- `temperature=0.7` (balanced speed/quality)
- Optimized system prompt (very brief instructions)
- Ultra-fast aggregation (`timeout=0.01`)

### 4. Text-to-Speech (Target: 50-150ms to start)
**Optimizations Applied:**
- Micro-chunk streaming (4 words per chunk)
- Ultra-minimal delays (10ms between chunks)
- Smart sentence boundary detection
- Parallel processing of chunks
- Memory pre-allocation for frames

### 5. Pipeline & System Optimizations
**Optimizations Applied:**
- Disabled metrics collection
- Disabled usage tracking  
- Optimized frame copying
- Performance timing throughout
- Memory-efficient operations

## 📊 Expected Performance Metrics

| Component | Optimized Latency | Previous Latency | Improvement |
|-----------|-------------------|------------------|-------------|
| Speech Detection | 50-100ms | 200-500ms | 2-5x faster |
| STT Processing | 100-300ms | 300-800ms | 2-3x faster |
| LLM Response | 200-800ms | 500-2000ms | 2-3x faster |
| TTS Start | 50-150ms | 200-600ms | 3-4x faster |
| **Total Perceived** | **400-1200ms** | **1200-3900ms** | **3x faster** |

## 🔧 System Requirements for Optimal Performance

### Recommended Hardware:
- **CPU**: 8+ cores, 3.0GHz+ (for Whisper & LLM)
- **RAM**: 16GB+ (for model loading)
- **Storage**: SSD (for faster model loading)
- **Network**: Stable broadband (for WebRTC)

### Recommended Software Settings:
- **Ollama**: Run with `--num-gpu 0` for CPU-only consistency
- **Windows**: Set process priority to "High" for voice agent
- **Microphone**: Use high-quality USB mic for clean audio
- **Browser**: Chrome/Edge for best WebRTC performance

## 🚀 Usage Instructions

### Starting the Ultra-Fast Agent:
```bash
# Option 1: Use the optimized batch file
start_ultra_fast_agent.bat

# Option 2: Manual startup
venv\Scripts\activate
python server\bot_ultra_fast.py --host 0.0.0.0 --port 7860
```

### Performance Monitoring:
- Real-time performance logs in console
- Performance API: `http://localhost:7860/api/performance`
- Timing data logged for each request component

### Best Practices for Users:
1. **Speak clearly** with good microphone positioning
2. **Pause briefly** between sentences for clean segmentation
3. **Keep responses short** for fastest turnaround
4. **Use stable internet** for consistent WebRTC performance
5. **Close other apps** that might compete for CPU resources

## ⚡ Advanced Optimizations (Future)

### Potential Further Improvements:
1. **GPU Acceleration**: CUDA/OpenCL for Whisper and TTS
2. **Model Quantization**: INT4 quantization for even smaller models
3. **Streaming LLM**: Real-time token streaming from Ollama
4. **WebAssembly**: Client-side processing for some components
5. **Edge Deployment**: Local model serving with optimized inference
6. **Custom Models**: Fine-tuned smaller models for specific use cases

### Experimental Features:
- Voice activity prediction (anticipate speech)
- Predictive text generation (start generating before STT completes)
- Audio fingerprinting for ultra-fast speaker detection
- Context-aware response caching

## 🔍 Troubleshooting Performance Issues

### If latency is higher than expected:

1. **Check Ollama Performance**:
   ```bash
   curl http://localhost:11434/api/generate -d '{"model":"gemma3:270m","prompt":"test","stream":false}'
   ```

2. **Monitor CPU Usage**: Voice agent should use 30-60% CPU during conversations

3. **Check Network**: Ensure stable connection for WebRTC

4. **Verify Audio Settings**: Test microphone quality and positioning

5. **Review Logs**: Check for performance timing in console output

### Performance Debugging:
- Enable detailed timing logs
- Monitor memory usage patterns
- Check for audio quality issues
- Verify Whisper model loading time
- Test TTS synthesis speed

This ultra-optimized version should provide a significantly more responsive voice conversation experience!
