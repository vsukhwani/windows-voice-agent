#!/usr/bin/env python3
"""
Setup script for Windows-compatible voice agent.
Downloads Kokoro model files and creates necessary directories.
"""

import os
import sys
import urllib.request
from pathlib import Path

def download_file(url: str, filepath: Path, description: str):
    """Download a file with progress indication."""
    print(f"Downloading {description}...")
    print(f"URL: {url}")
    print(f"Destination: {filepath}")
    
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ Downloaded {description}")
    except Exception as e:
        print(f"✗ Failed to download {description}: {e}")
        return False
    return True

def main():
    # Create models directory
    models_dir = Path("server/models/kokoro")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("Setting up Windows-compatible voice agent...")
    print(f"Models directory: {models_dir.absolute()}")
    
    # Model files to download
    model_files = {
        "kokoro-v1.0.onnx": {
            "url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
            "description": "Kokoro ONNX model (310MB)"
        },
        "voices-v1.0.bin": {
            "url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin", 
            "description": "Kokoro voices file"
        }
    }
    
    # Check if files already exist
    missing_files = []
    for filename, info in model_files.items():
        filepath = models_dir / filename
        if not filepath.exists():
            missing_files.append((filename, info))
        else:
            print(f"✓ {filename} already exists")
    
    if not missing_files:
        print("\nAll model files are already downloaded!")
        print("\nNext steps:")
        print("1. Copy env_windows.example to .env and update paths:")
        print(f"   KOKORO_MODEL_PATH={models_dir / 'kokoro-v1.0.onnx'}")
        print(f"   KOKORO_VOICES_PATH={models_dir / 'voices-v1.0.bin'}")
        print("2. Install dependencies: pip install -r server/requirements_windows.txt")
        print("3. Run the server: python server/bot_windows.py")
        return
    
    # Download missing files
    print(f"\nDownloading {len(missing_files)} missing files...")
    success_count = 0
    
    for filename, info in missing_files:
        filepath = models_dir / filename
        if download_file(info["url"], filepath, info["description"]):
            success_count += 1
    
    print(f"\nDownloaded {success_count}/{len(missing_files)} files successfully.")
    
    if success_count == len(missing_files):
        print("\n✓ Setup complete!")
        print("\nNext steps:")
        print("1. Copy env_windows.example to .env and update paths:")
        print(f"   KOKORO_MODEL_PATH={models_dir / 'kokoro-v1.0.onnx'}")
        print(f"   KOKORO_VOICES_PATH={models_dir / 'voices-v1.0.bin'}")
        print("2. Install dependencies: pip install -r server/requirements_windows.txt")
        print("3. Run the server: python server/bot_windows.py")
    else:
        print("\n✗ Some files failed to download. Please download them manually:")
        for filename, info in missing_files:
            print(f"   {filename}: {info['url']}")

if __name__ == "__main__":
    main()
