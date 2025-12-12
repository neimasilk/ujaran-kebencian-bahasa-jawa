#!/usr/bin/env python3
"""
Script to generate synthetic Javanese data using DeepSeek API.
Focus: Informal, Slang, Hate Speech patterns (for robustness), and Code-Switching.
"""

import os
import time
from openai import OpenAI
import csv

# CONFIGURATION
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"  # Check DeepSeek docs for exact base URL
OUTPUT_FILE = "data/corpus/synthetic_javanese.txt"

prompts = [
    "Buatkan 20 kalimat bahasa Jawa Ngoko kasar yang biasa dipakai saat marah di sosial media. Tidak perlu terjemahan.",
    "Buatkan 20 komentar netizen julid dalam bahasa Jawa campur bahasa Indonesia (code-mixing).",
    "Buatkan 20 kalimat bahasa Jawa Timuran yang menggunakan kata 'Jancok', 'Asu', 'Gatel' dalam konteks kemarahan.",
    "Buatkan 20 percakapan pendek bahasa Jawa tentang politik yang memanas.",
    "Buatkan 20 kalimat bahasa Jawa yang menyindir seseorang secara sarkas (satire)."
]

def generate_data():
    if not API_KEY:
        print("❌ Error: Please set DEEPSEEK_API_KEY environment variable.")
        print("Example: set DEEPSEEK_API_KEY=sk-your-key (Windows CMD)")
        print("Or: $env:DEEPSEEK_API_KEY='sk-your-key' (PowerShell)")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    print(f"🚀 Starting generation with DeepSeek...")
    print(f"💾 Output will be appended to {OUTPUT_FILE}")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    total_generated = 0
    
    # Run multiple iterations to get volume
    iterations = 5  # Adjust as needed
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for i in range(iterations):
            print(f"\n🔄 Iteration {i+1}/{iterations}")
            
            for p_idx, prompt in enumerate(prompts):
                try:
                    print(f"   Asking: '{prompt[:30]}...'")
                    response = client.chat.completions.create(
                        model="deepseek-chat", # or deepseek-coder
                        messages=[
                            {"role": "system", "content": "You are a native Javanese speaker from East Java. You speak natural, informal, and sometimes rude Javanese (Ngoko). Output ONLY the Javanese sentences, one per line. No numbering, no translations."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=1.2, # High creativity for variance
                        max_tokens=1000
                    )
                    
                    content = response.choices[0].message.content
                    lines = [l.strip() for l in content.split('\n') if l.strip() and not l[0].isdigit()]
                    
                    for line in lines:
                        # Clean potential numbering like "1. " or "- "
                        clean_line = line.lstrip("1234567890.- ")
                        f.write(clean_line + "\n")
                    
                    count = len(lines)
                    total_generated += count
                    print(f"   ✅ Received {count} sentences.")
                    
                    # Be nice to the API
                    time.sleep(2) 
                    
                except Exception as e:
                    print(f"   ❌ API Error: {e}")
                    time.sleep(5)

    print(f"\n✨ Generation Complete! Total {total_generated} synthetic sentences saved.")

if __name__ == "__main__":
    print("--- DeepSeek Javanese Data Generator ---")
    generate_data()
