#!/usr/bin/env python3
import os
from openai import OpenAI
from google import genai

# Keys from environment variables
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ZAI_KEY = os.getenv("ZAI_API_KEY")

def check_deepseek():
    print("\n🤖 Checking DeepSeek...")
    if not DEEPSEEK_KEY:
        print("   ❌ No API Key")
        return False
    try:
        client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Tes satu kata Jawa."}],
            max_tokens=10
        )
        print(f"   ✅ Success: {response.choices[0].message.content.strip()}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_gemini_new_sdk():
    print("\n🌟 Checking Gemini (New SDK: gemini-2.5-flash)...")
    if not GEMINI_KEY:
        print("   ❌ No API Key")
        return False
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Tes satu kata Jawa."
        )
        print(f"   ✅ Success: {response.text.strip()}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_zai_openai_style():
    print("\n🔮 Checking Z.AI (via OpenAI SDK)...")
    if not ZAI_KEY:
        print("   ❌ No API Key")
        return False
    try:
        # Using OpenAI client for Z.AI
        client = OpenAI(api_key=ZAI_KEY, base_url="https://api.z.ai/api/paas/v4")
        response = client.chat.completions.create(
            model="glm-4.6",
            messages=[{"role": "user", "content": "Tes satu kata Jawa."}],
            max_tokens=10
        )
        print(f"   ✅ Success: {response.choices[0].message.content.strip()}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("--- API Health Check v3 ---")
    ds_ok = check_deepseek()
    gem_ok = check_gemini_new_sdk()
    zai_ok = check_zai_openai_style()
    
    print("\n--- Summary ---")
    print(f"DeepSeek: {'✅' if ds_ok else '❌'}")
    print(f"Gemini:   {'✅' if gem_ok else '❌'}")
    print(f"Z.AI:     {'✅' if zai_ok else '❌'}")
