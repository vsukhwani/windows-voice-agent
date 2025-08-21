#!/usr/bin/env python3
"""
Script to download required NLTK data for Kokoro TTS
"""

import nltk
import sys

def download_nltk_data():
    """Download required NLTK data"""
    try:
        print("Downloading NLTK data...")
        
        # Download the specific resource that was missing
        nltk.download('averaged_perceptron_tagger_eng', quiet=False)
        
        # Download other commonly needed resources for TTS
        nltk.download('punkt', quiet=False)
        nltk.download('averaged_perceptron_tagger', quiet=False)
        nltk.download('punkt_tab', quiet=False)
        
        print("✅ All NLTK data downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

if __name__ == "__main__":
    success = download_nltk_data()
    sys.exit(0 if success else 1)
