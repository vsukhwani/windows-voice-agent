#!/usr/bin/env python3
"""
Test script to check Hindi language support across all components
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Add local pipecat to Python path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server", "pipecat", "src"))

load_dotenv(override=True)

from loguru import logger

async def test_whisper_hindi():
    """Test Whisper STT with Hindi language support"""
    try:
        logger.info("Testing Whisper Hindi support...")
        from pipecat.services.whisper.stt import WhisperSTTService, Model
        
        # Test with Hindi language
        stt = WhisperSTTService(
            model=Model.SMALL,  # Better model for non-English languages
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            language="hi",  # Hindi language code
            no_speech_threshold=0.6,
            initial_prompt="नमस्ते, आप कैसे हैं? मैं आपसे बात करना चाहता हूं।",  # Hindi prompt
        )
        
        logger.info("✅ Whisper Hindi initialization successful!")
        logger.info("Language: Hindi (hi)")
        logger.info("Model: SMALL (better for non-English)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Whisper Hindi test failed: {e}")
        return False

async def test_ollama_hindi():
    """Test Ollama LLM with Hindi support"""
    try:
        logger.info("Testing Ollama Hindi support...")
        import httpx
        
        # Test Ollama with Hindi prompt
        url = "http://127.0.0.1:11434/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        
        data = {
            "model": "gemma3:12b",
            "messages": [
                {
                    "role": "system", 
                    "content": "आप एक सहायक AI हैं। कृपया हिंदी में उत्तर दें।"
                },
                {
                    "role": "user", 
                    "content": "नमस्ते, आप कैसे हैं?"
                }
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            
        if response.status_code == 200:
            result = response.json()
            hindi_response = result['choices'][0]['message']['content']
            logger.info("✅ Ollama Hindi test successful!")
            logger.info(f"Hindi Response: {hindi_response}")
            return True
        else:
            logger.error(f"❌ Ollama request failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ollama Hindi test failed: {e}")
        return False

async def test_kokoro_hindi():
    """Test Kokoro TTS with Hindi text"""
    try:
        logger.info("Testing Kokoro Hindi support...")
        
        # Note: Kokoro TTS may have limited Hindi support
        # It's primarily trained on English voices
        logger.warning("⚠️  Kokoro TTS is primarily designed for English")
        logger.warning("    Hindi text may not sound natural")
        logger.warning("    Consider using alternative TTS for Hindi")
        
        # Test basic initialization
        from kokoro_tts_onnx import KokoroOnnxTTSService
        
        model_path = os.getenv("KOKORO_MODEL_PATH", "")
        voices_path = os.getenv("KOKORO_VOICES_PATH", "")
        
        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            logger.error("❌ Kokoro model files not found")
            return False
            
        tts = KokoroOnnxTTSService(
            model_path=model_path,
            voices_path=voices_path,
            voice="af_heart",
            sample_rate=24000,
        )
        
        logger.info("✅ Kokoro TTS initialized (limited Hindi support)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Kokoro Hindi test failed: {e}")
        return False

def get_language_recommendations():
    """Provide recommendations for Hindi language support"""
    logger.info("🌐 Hindi Language Support Analysis:")
    logger.info("")
    
    logger.info("📝 WHISPER STT:")
    logger.info("   ✅ Full Hindi support (language='hi')")
    logger.info("   ✅ Trained on Hindi audio data")
    logger.info("   ✅ Recommended: Use SMALL or MEDIUM model")
    logger.info("")
    
    logger.info("🧠 OLLAMA LLM:")
    logger.info("   ✅ Gemma3 has good Hindi support")
    logger.info("   ✅ Can understand and respond in Hindi")
    logger.info("   ✅ Supports mixed Hindi-English conversations")
    logger.info("")
    
    logger.info("🔊 KOKORO TTS:")
    logger.info("   ⚠️  Limited Hindi support (English voices only)")
    logger.info("   ⚠️  May pronounce Hindi text with English accent")
    logger.info("   💡 Alternative: Consider Azure TTS or Google TTS for Hindi")
    logger.info("")
    
    logger.info("🎯 RECOMMENDATIONS:")
    logger.info("   1. Use Whisper SMALL/MEDIUM model for better Hindi recognition")
    logger.info("   2. Set system prompt to respond in Hindi")
    logger.info("   3. Consider alternative TTS for natural Hindi speech")
    logger.info("   4. Test with mixed Hindi-English conversations")

async def main():
    """Run all Hindi language tests"""
    logger.info("🇮🇳 Testing Hindi Language Support")
    logger.info("=" * 50)
    
    # Component tests
    whisper_ok = await test_whisper_hindi()
    ollama_ok = await test_ollama_hindi()
    kokoro_ok = await test_kokoro_hindi()
    
    logger.info("=" * 50)
    
    # Results summary
    if whisper_ok and ollama_ok:
        logger.info("✅ Hindi voice agent is possible!")
        logger.info("   - STT: Whisper supports Hindi")
        logger.info("   - LLM: Gemma3 supports Hindi") 
        logger.info("   - TTS: Limited support (consider alternatives)")
    else:
        logger.warning("⚠️  Some components need configuration")
    
    logger.info("")
    get_language_recommendations()

if __name__ == "__main__":
    asyncio.run(main())
