import os
import google.generativeai as genai
from dotenv import dotenv_values

def check_gemini_keys():
    env_vars = dotenv_values(".env")
    MODEL_NAME = 'gemini-2.0-flash'
    
    print(f"🔬 Testing with model: {MODEL_NAME}")
    print("-" * 50)

    for i in range(1, 6):
        key_name = f"GEMINI_API_KEY_{i}"
        key_value = env_vars.get(key_name)
        
        if not key_value:
            print(f"❌ {key_name}: Not found in .env")
            continue
            
        key_value = key_value.strip()
        print(f"🔑 Key {i} ({key_value[:10]}...):", end=" ", flush=True)
        
        print("Mencoba koneksi...", end=" ", flush=True)
        try:
            genai.configure(api_key=key_value)
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content("Hi", request_options={"timeout": 10})
            
            if response and response.text:
                print("✅ VALID & WORKING")
            else:
                print("⚠️ EMPTY RESPONSE")
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "429" in error_str:
                print("⏳ QUOTA EXCEEDED (But key is valid)")
            elif "invalid" in error_str:
                print("❌ INVALID KEY")
            else:
                print(f"❌ ERROR: {str(e)[:50]}...")

if __name__ == "__main__":
    check_gemini_keys()
