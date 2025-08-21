#!/usr/bin/env python3
"""
Debug test script to check all components of the voice agent
"""

import asyncio
import httpx
import json
import os
from pathlib import Path

async def test_ollama():
    """Test Ollama connection and model availability"""
    print("🔍 Testing Ollama connection...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:11434/api/tags", timeout=10.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                print(f"✅ Ollama accessible. Available models: {model_names}")
                
                # Check if gemma3:12b is available
                if "gemma3:12b" in model_names:
                    print("✅ gemma3:12b model found")
                else:
                    print("❌ gemma3:12b model not found")
                return True
            else:
                print(f"❌ Ollama returned status code: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False

async def test_server():
    """Test server connection"""
    print("\n🔍 Testing server connection...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:7860/api/test-ollama", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Server accessible: {data}")
                return True
            else:
                print(f"❌ Server returned status code: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False

def test_model_files():
    """Test if Kokoro model files exist"""
    print("\n🔍 Testing model files...")
    
    model_path = Path("server/models/kokoro/kokoro-v1.0.onnx")
    voices_path = Path("server/models/kokoro/voices-v1.0.bin")
    
    if model_path.exists():
        print(f"✅ Kokoro model found: {model_path}")
        print(f"   Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
    else:
        print(f"❌ Kokoro model not found: {model_path}")
    
    if voices_path.exists():
        print(f"✅ Voices file found: {voices_path}")
        print(f"   Size: {voices_path.stat().st_size / (1024*1024):.1f} MB")
    else:
        print(f"❌ Voices file not found: {voices_path}")
    
    return model_path.exists() and voices_path.exists()

def test_environment():
    """Test environment variables"""
    print("\n🔍 Testing environment variables...")
    
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
        with open(env_file, 'r') as f:
            content = f.read()
            if "KOKORO_MODEL_PATH" in content:
                print("✅ KOKORO_MODEL_PATH found in .env")
            else:
                print("❌ KOKORO_MODEL_PATH not found in .env")
    else:
        print("❌ .env file not found")
    
    # Check actual environment variables
    model_path = os.getenv("KOKORO_MODEL_PATH", "")
    voices_path = os.getenv("KOKORO_VOICES_PATH", "")
    
    if model_path:
        print(f"✅ KOKORO_MODEL_PATH: {model_path}")
    else:
        print("❌ KOKORO_MODEL_PATH not set")
    
    if voices_path:
        print(f"✅ KOKORO_VOICES_PATH: {voices_path}")
    else:
        print("❌ KOKORO_VOICES_PATH not set")

async def test_llm_generation():
    """Test LLM generation directly"""
    print("\n🔍 Testing LLM generation...")
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "model": "gemma3:12b",
                "messages": [
                    {"role": "user", "content": "Say hello in one sentence."}
                ],
                "stream": False
            }
            
            response = await client.post(
                "http://127.0.0.1:11434/v1/chat/completions",
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"✅ LLM generation successful: {content}")
                return True
            else:
                print(f"❌ LLM generation failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
    except Exception as e:
        print(f"❌ Error testing LLM generation: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting debug tests...\n")
    
    # Test environment
    test_environment()
    
    # Test model files
    models_ok = test_model_files()
    
    # Test Ollama
    ollama_ok = await test_ollama()
    
    # Test server
    server_ok = await test_server()
    
    # Test LLM generation
    llm_ok = await test_llm_generation()
    
    print("\n" + "="*50)
    print("📊 SUMMARY:")
    print(f"   Environment: {'✅' if models_ok else '❌'}")
    print(f"   Model Files: {'✅' if models_ok else '❌'}")
    print(f"   Ollama: {'✅' if ollama_ok else '❌'}")
    print(f"   Server: {'✅' if server_ok else '❌'}")
    print(f"   LLM Generation: {'✅' if llm_ok else '❌'}")
    print("="*50)
    
    if all([models_ok, ollama_ok, server_ok, llm_ok]):
        print("\n🎉 All tests passed! The voice agent should work.")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")

if __name__ == "__main__":
    asyncio.run(main())
