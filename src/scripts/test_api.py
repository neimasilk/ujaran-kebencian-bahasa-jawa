#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.deepseek_client import DeepSeekAPIClient

def test_deepseek_api():
    """Test DeepSeek API connection"""
    print("🔍 Testing DeepSeek API Connection...")
    print("-" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check if API key exists
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not found in environment")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Test connection with a simple labeling request
        client = DeepSeekAPIClient()
        print("🔄 Testing API with sample text...")
        
        # Test with a simple Javanese text
        test_text = "Sugeng enjing, piye kabare?"
        result = client.label_single_text(test_text)
        
        if result and not result.error:
            print("✅ API Connection: SUCCESS")
            print(f"📊 Test result: Label={result.label_id}, Confidence={result.confidence:.2f}")
            print(f"⏱️ Response time: {result.response_time:.2f}s")
            print("🎉 DeepSeek API is working properly!")
            return True
        else:
            print("❌ API Connection: FAILED")
            if result and result.error:
                print(f"Error: {result.error}")
            return False
            
    except Exception as e:
        print(f"❌ API Connection Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_deepseek_api()
    print("-" * 50)
    if success:
        print("✅ Ready to proceed with Google Drive setup!")
    else:
        print("❌ Please check your DeepSeek API key configuration")