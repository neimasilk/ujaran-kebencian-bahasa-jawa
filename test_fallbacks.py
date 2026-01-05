import os
import google.generativeai as genai
from dotenv import dotenv_values

def check_model(model_name, key):
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hi", request_options={"timeout": 5})
        return True if response.text else False
    except:
        return False

env_vars = dotenv_values(".env")
key = env_vars.get("GEMINI_API_KEY_1", "").strip()

models_to_test = ['gemini-2.0-flash-lite', 'gemini-flash-lite-latest', 'gemini-1.5-flash-8b']

print(f"Testing fallback models for Key 1...")
for m in models_to_test:
    print(f"Testing {m}: ", end="")
    if check_model(m, key):
        print("✅ WORKING")
    else:
        print("❌ FAILED/LIMIT")
