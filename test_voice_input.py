#!/usr/bin/env python3
"""
Test script to verify Whisper STT functionality
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Add local pipecat to Python path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server", "pipecat", "src"))

load_dotenv(override=True)

from pipecat.services.whisper.stt import WhisperSTTService, Model
from loguru import logger

async def test_whisper():
    """Test Whisper STT service initialization"""
    try:
        logger.info("Testing Whisper STT Service...")
        
        # Test with BASE model (smaller, faster)
        stt = WhisperSTTService(
            model=Model.BASE,
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            language="en",
            no_speech_threshold=0.6,
            initial_prompt="Hello, how are you today? This is a conversation.",
        )
        
        logger.info("✅ Whisper STT initialized successfully!")
        logger.info(f"Model: {Model.BASE}")
        logger.info(f"Device: {os.getenv('WHISPER_DEVICE', 'cpu')}")
        logger.info(f"Compute Type: {os.getenv('WHISPER_COMPUTE_TYPE', 'int8')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Whisper STT failed to initialize: {e}")
        return False

async def test_audio_input():
    """Test basic audio input detection"""
    try:
        import sounddevice as sd
        import numpy as np
        
        logger.info("Testing audio input...")
        
        # List available audio devices
        devices = sd.query_devices()
        logger.info("Available audio devices:")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                logger.info(f"  {i}: {device['name']} (inputs: {device['max_input_channels']})")
        
        # Test recording a short sample
        logger.info("Recording 2 seconds of audio (speak now)...")
        duration = 2  # seconds
        sample_rate = 16000  # Hz
        
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()  # Wait until recording is finished
        
        # Check if we captured any audio
        max_amplitude = np.max(np.abs(audio_data))
        logger.info(f"Max audio amplitude: {max_amplitude:.4f}")
        
        if max_amplitude > 0.001:
            logger.info("✅ Audio input detected successfully!")
            return True
        else:
            logger.warning("⚠️  Very low or no audio input detected")
            return False
            
    except ImportError:
        logger.warning("sounddevice not available, skipping audio test")
        return True
    except Exception as e:
        logger.error(f"❌ Audio input test failed: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("🎤 Voice Agent Audio Tests")
    logger.info("=" * 40)
    
    # Test 1: Whisper STT
    whisper_ok = await test_whisper()
    
    # Test 2: Audio Input
    audio_ok = await test_audio_input()
    
    logger.info("=" * 40)
    if whisper_ok and audio_ok:
        logger.info("✅ All tests passed! Voice input should work.")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above.")
        
    logger.info("🎤 Test complete")

if __name__ == "__main__":
    asyncio.run(main())
