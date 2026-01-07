import os
import google.generativeai as genai
from dotenv import dotenv_values

def list_available_models():
    env_vars = dotenv_values(".env")
    # Coba key pertama saja untuk testing
    key = env_vars.get("GEMINI_API_KEY_1", "").strip()
    
    if not key:
        print("❌ Key 1 tidak ditemukan di .env")
        return

    print(f"🔍 Mencoba list model dengan Key 1 ({key[:10]}...)")
    try:
        genai.configure(api_key=key)
        models = genai.list_models()
        print("\n✅ Model yang tersedia untuk Key ini:")
        for m in models:
            print(f" - {m.name}")
    except Exception as e:
        print(f"❌ Gagal mengambil list model: {e}")

if __name__ == "__main__":
    list_available_models()

