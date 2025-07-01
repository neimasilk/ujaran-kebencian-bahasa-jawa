#!/usr/bin/env python3
"""Debug script to test Settings class instantiation."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config.settings import Settings

def main():
    print("Testing Settings class...")
    
    try:
        settings = Settings()
        print(f"Settings type: {type(settings)}")
        print(f"Settings dict: {settings.__dict__ if hasattr(settings, '__dict__') else 'No __dict__'}")
        
        # Test accessing deepseek_base_url
        print(f"deepseek_base_url: {settings.deepseek_base_url}")
        print(f"deepseek_api_key: {settings.deepseek_api_key}")
        
        # Check if it's actually a dict
        if isinstance(settings, dict):
            print("ERROR: Settings is a dict!")
        else:
            print("SUCCESS: Settings is not a dict")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()