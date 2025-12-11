#!/usr/bin/env python3
"""
Massive Synthetic Data Generator with API Key Rotation.
DeepSeek + Multi-Key Gemini (Smart Fallback).
"""

import os
import time
import random
from google import genai
from google.genai import types
from openai import OpenAI

# ================= CONFIGURATION =================
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# List of Gemini Keys to rotate
GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY"), # Key 1
    "AIzaSyB6nyh6qpbx7VXuOxwLZvnOabFxClStC2k", # Key 2
    "AIzaSyBD4ja-P06uiGYJlOz3syNR9WT3p0nfSsE"  # Key 3
]

OUTPUT_FILE = "data/corpus/massive_synthetic_javanese.txt"
ITERATIONS = 50 

# Remove None/Empty keys
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

# ================= PROMPTS =================
# Mixed prompts for maximum diversity
all_prompts = [
    # Ngoko & Dialects
    "Buatkan 20 kalimat percakapan sehari-hari dalam Bahasa Jawa Ngoko Lugu antara teman akrab. Topik bebas. Output hanya kalimat Jawa.",
    "Buatkan 20 kalimat Bahasa Jawa dialek Jawa Timur (Suroboyoan/Malang) yang ekspresif. Gunakan kata 'rek', 'cak', 'cuk'. Output hanya kalimat Jawa.",
    "Buatkan 20 kalimat curhat sedih atau galau dalam Bahasa Jawa Ngoko. Output hanya kalimat Jawa.",
    "Buatkan 20 komentar netizen di media sosial yang 'julid' atau pedas dalam Bahasa Jawa Ngoko. Output hanya kalimat Jawa.",
    "Buatkan 20 kalimat Bahasa Jawa dialek Ngapak (Banyumas) tentang kehidupan sehari-hari. Output hanya kalimat Jawa.",
    
    # Code-Switching & Context
    "Buatkan 20 kalimat 'Code-Switching' (Campur Kode) gabungan Jawa, Indonesia, Inggris ala anak muda jaman now. Output hanya kalimat.",
    "Buatkan 20 kalimat yang mengandung istilah budaya Jawa (slametan, weton, pamali, kualat) dalam konteks modern. Output hanya kalimat Jawa.",
    "Buatkan 20 kalimat opini tentang teknologi atau berita terkini dalam Bahasa Jawa campur Bahasa Indonesia. Output hanya kalimat.",
    
    # Krama & Formal
    "Buatkan 20 kalimat dialog Krama Alus (Sangat Halus) yang sopan untuk berbicara dengan orang tua. Output hanya kalimat Jawa.",
    "Buatkan 20 kalimat Bahasa Jawa Krama Inggil yang digunakan dalam pidato atau sambutan resmi. Output hanya kalimat Jawa.",
    "Buatkan 20 kalimat instruksi atau tutorial melakukan sesuatu dalam Bahasa Jawa Krama. Output hanya kalimat Jawa."
]

# ================= ENGINE SETUP =================
def setup_deepseek():
    if not DEEPSEEK_KEY: return None
    return OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE_URL)

# Global index for Gemini key rotation
current_gemini_key_index = 0

def get_gemini_client():
    global current_gemini_key_index
    if not GEMINI_KEYS: return None
    
    # Get current key
    key = GEMINI_KEYS[current_gemini_key_index]
    return genai.Client(api_key=key)

def rotate_gemini_key():
    global current_gemini_key_index
    if len(GEMINI_KEYS) <= 1:
        print("   ⚠️ No other Gemini keys to rotate to.")
        return False
    
    # Rotate index
    current_gemini_key_index = (current_gemini_key_index + 1) % len(GEMINI_KEYS)
    print(f"   🔄 Rotating to Gemini Key #{current_gemini_key_index + 1}")
    return True

# ================= GENERATORS =================
def generate_deepseek(client, prompt):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a native Javanese speaker. Generate ONLY Javanese text sentences. One sentence per line. No translations. No numbering/bullets."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.2,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"   ❌ DeepSeek Error: {e}")
        return None

def generate_gemini(prompt):
    # Retry logic with key rotation
    max_retries = len(GEMINI_KEYS) # Try each key once per prompt
    
    for _ in range(max_retries):
        client = get_gemini_client()
        if not client: return None
        
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="You are a Javanese expert. Generate ONLY Javanese text sentences. One sentence per line. No translations. No numbering. \n\n" + prompt
            )
            return response.text
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "exhausted" in error_str.lower():
                print(f"   ⚠️ Gemini Quota Exceeded on Key #{current_gemini_key_index + 1}. Switching...")
                if not rotate_gemini_key():
                    return None # Stop if we can't rotate
            else:
                print(f"   ❌ Gemini Error: {e}")
                return None
                
    return None

# ================= MAIN LOOP =================
def main():
    print(f"🚀 Starting Multi-Key Generation with {len(GEMINI_KEYS)} Gemini Keys + DeepSeek...")
    
    ds_client = setup_deepseek()
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    total_count = 0
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        
        for i in range(ITERATIONS):
            print(f"\n🔄 --- Batch {i+1}/{ITERATIONS} ---")
            
            # 1. DeepSeek (Always Primary)
            if ds_client:
                prompt = random.choice(all_prompts)
                print(f"   🤖 DeepSeek: '{prompt[:40]}...'")
                content = generate_deepseek(ds_client, prompt)
                
                if content:
                    lines = [l.strip().lstrip("1234567890.-• ") for l in content.split('\n') if l.strip()]
                    valid = [l for l in lines if len(l) > 10]
                    for line in valid: f.write(line + "\n")
                    print(f"      ✅ Saved {len(valid)} lines.")
                    total_count += len(valid)
                time.sleep(1)

            # 2. Gemini (With Rotation)
            prompt = random.choice(all_prompts)
            print(f"   🌟 Gemini:   '{prompt[:40]}...'")
            content = generate_gemini(prompt)
            
            if content:
                lines = [l.strip().lstrip("1234567890.-• ") for l in content.split('\n') if l.strip()]
                valid = [l for l in lines if len(l) > 10]
                for line in valid: f.write(line + "\n")
                print(f"      ✅ Saved {len(valid)} lines.")
                total_count += len(valid)
            else:
                print("      ⚠️ Gemini skipped this turn (All keys exhausted or error).")
            
            # Safe delay to protect even the fresh keys
            time.sleep(3) 
                
    print(f"\n✨ Generation Complete. Added {total_count} sentences.")
    print(f"💾 File: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
