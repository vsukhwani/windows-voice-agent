#!/usr/bin/env python3
"""
Comprehensive Hindi Voice Agent Testing Script
हिंदी वॉयस एजेंट परीक्षण स्क्रिप्ट
"""

import os
import sys
import asyncio
import httpx
from dotenv import load_dotenv

# Add local pipecat to Python path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server", "pipecat", "src"))

load_dotenv(override=True)

from pipecat.services.whisper.stt import WhisperSTTService, Model
from loguru import logger

async def test_hindi_components():
    """Test all Hindi voice agent components"""
    print("🇮🇳 हिंदी वॉयस एजेंट परीक्षण शुरू हो रहा है...")
    print("🇮🇳 Hindi Voice Agent Testing Starting...")
    print("=" * 60)
    
    results = {
        "whisper_hindi": False,
        "ollama_hindi": False,
        "kokoro_models": False,
        "nltk_data": False
    }
    
    # Test 1: Hindi Whisper STT
    print("\n📝 Testing Hindi Whisper STT...")
    try:
        stt = WhisperSTTService(
            model=Model.SMALL,  # Better for Hindi
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            language="hi",  # Hindi language
            no_speech_threshold=0.6,
            initial_prompt="नमस्ते, आप कैसे हैं? यह एक हिंदी बातचीत है।",
        )
        print("✅ Hindi Whisper STT initialized successfully!")
        print(f"   Model: {Model.SMALL}")
        print(f"   Language: Hindi (hi)")
        print(f"   Device: {os.getenv('WHISPER_DEVICE', 'cpu')}")
        print(f"   Initial Prompt: नमस्ते, आप कैसे हैं? यह एक हिंदी बातचीत है।")
        results["whisper_hindi"] = True
    except Exception as e:
        print(f"❌ Hindi Whisper STT failed: {e}")
    
    # Test 2: Ollama Hindi Support
    print("\n🤖 Testing Ollama Hindi LLM Support...")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:  # Increased timeout
            # Test basic connectivity
            response = await client.get("http://127.0.0.1:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                print(f"✅ Ollama connected. Available models: {model_names}")
                
                # Test Hindi response
                if "llama3.2:latest" in model_names:
                    print("🧠 Testing Hindi conversation with Llama 3.2...")
                    hindi_request = {
                        "model": "llama3.2:latest",
                        "messages": [
                            {
                                "role": "user", 
                                "content": "आप कौन हैं? कृपया हिंदी में छोटा उत्तर दें।"
                            }
                        ],
                        "stream": False
                    }
                    
                    chat_response = await client.post(
                        "http://127.0.0.1:11434/v1/chat/completions",
                        json=hindi_request,
                        timeout=120.0  # Increased timeout for chat
                    )
                    
                    if chat_response.status_code == 200:
                        chat_data = chat_response.json()
                        hindi_reply = chat_data["choices"][0]["message"]["content"]
                        print(f"✅ Hindi Response: {hindi_reply}")
                        results["ollama_hindi"] = True
                    else:
                        print(f"❌ Hindi chat failed: {chat_response.status_code}")
                elif "gemma3:12b" in model_names:
                    print("⚠️  Using Gemma3 12B (backup option)")
                    print("💡 For better Hindi support, consider using: ollama pull llama3.2:latest")
                else:
                    print("⚠️  Neither llama3.2:latest nor gemma3:12b found.")
                    print("💡 Install recommended: ollama pull llama3.2:latest")
            else:
                print(f"❌ Ollama not accessible: {response.status_code}")
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        import traceback
        print("Full error:")
        traceback.print_exc()
        print("💡 Make sure Ollama is running: ollama serve")
    
    # Test 3: Kokoro Model Files
    print("\n🎵 Testing Kokoro TTS Model Files...")
    try:
        model_path = os.getenv("KOKORO_MODEL_PATH", "")
        voices_path = os.getenv("KOKORO_VOICES_PATH", "")
        
        if os.path.exists(model_path) and os.path.exists(voices_path):
            model_size = os.path.getsize(model_path) / (1024*1024)  # MB
            voices_size = os.path.getsize(voices_path) / (1024*1024)  # MB
            print(f"✅ Kokoro model files found!")
            print(f"   Model: {model_path} ({model_size:.1f}MB)")
            print(f"   Voices: {voices_path} ({voices_size:.1f}MB)")
            print("ℹ️  Note: Kokoro will attempt Hindi pronunciation with English voices")
            results["kokoro_models"] = True
        else:
            print(f"❌ Kokoro model files not found:")
            print(f"   Model: {model_path} (exists: {os.path.exists(model_path)})")
            print(f"   Voices: {voices_path} (exists: {os.path.exists(voices_path)})")
    except Exception as e:
        print(f"❌ Kokoro test failed: {e}")
    
    # Test 4: NLTK Data
    print("\n📚 Testing NLTK Data...")
    try:
        import nltk
        nltk_data_path = os.getenv("NLTK_DATA", "")
        if nltk_data_path:
            nltk.data.path.append(nltk_data_path)
        
        # Test required NLTK data
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
            print("✅ NLTK averaged_perceptron_tagger_eng found")
            results["nltk_data"] = True
        except LookupError:
            print("⚠️  NLTK data missing. Run: python download_nltk_data.py")
    except Exception as e:
        print(f"❌ NLTK test failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Hindi Voice Agent Test Summary:")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n🎯 Overall Score: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 सभी टेस्ट पास हो गए! आपका हिंदी वॉयस एजेंट तैयार है!")
        print("🎉 All tests passed! Your Hindi voice agent is ready!")
        print("\n🚀 Start Hindi voice agent with: start_hindi_voice_agent.bat")
    else:
        print("\n🔧 कुछ समस्याएं हैं। कृपया ऊपर दी गई जानकारी देखें।")
        print("🔧 Some issues found. Please check the information above.")
    
    return results

async def test_hindi_server():
    """Test if Hindi server is running"""
    print("\n🌐 Testing Hindi Server Connection...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:7861/api/test-hindi")
            if response.status_code == 200:
                data = response.json()
                print("✅ Hindi server is running!")
                print(f"   Status: {data.get('status')}")
                print(f"   Test Phrase: {data.get('test_phrase')}")
                print(f"   Components: {data.get('components')}")
                return True
            else:
                print(f"⚠️  Hindi server returned: {response.status_code}")
                return False
    except Exception as e:
        print(f"ℹ️  Hindi server not running: {e}")
        print("💡 Start with: start_hindi_voice_agent.bat")
        return False

async def main():
    """Main test function"""
    print("🇮🇳 हिंदी वॉयस एजेंट व्यापक परीक्षण")
    print("🇮🇳 Hindi Voice Agent Comprehensive Testing")
    print("=" * 60)
    
    # Test components
    component_results = await test_hindi_components()
    
    # Test server if it's running
    server_running = await test_hindi_server()
    
    print("\n" + "=" * 60)
    print("🎯 Final Recommendations:")
    print("=" * 60)
    
    if all(component_results.values()):
        print("1. ✅ All components are ready for Hindi support")
        if not server_running:
            print("2. 🚀 Run: start_hindi_voice_agent.bat")
            print("3. 🌐 Visit: http://localhost:3000")
            print("4. 🎤 बोलिए हिंदी में! (Speak in Hindi!)")
    else:
        print("1. 🔧 Fix the failing components above")
        if not component_results["ollama_hindi"]:
            print("2. 📥 Install Llama 3.2: ollama pull llama3.2:latest")
            print("3. 🔄 Backup option: ollama pull gemma3:12b")
        if not component_results["nltk_data"]:
            print("4. 📚 Run: python download_nltk_data.py")
    
    print("\n🎉 हिंदी वॉयस एजेंट परीक्षण पूरा!")
    print("🎉 Hindi Voice Agent Testing Complete!")

if __name__ == "__main__":
    asyncio.run(main())
