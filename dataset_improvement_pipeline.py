#!/usr/bin/env python3
"""
Dataset Improvement Pipeline untuk Deteksi Ujaran Kebencian Bahasa Jawa
========================================================================
Pipeline ini memperbaiki dataset terjemahan menjadi lebih natural dan relevan
untuk konteks Indonesia.

Tahapan:
1. FILTER   - Hapus data yang tidak bisa diselamatkan (referensi Barat)
2. NATURALISASI - Ubah terjemahan kaku ke Jawa natural (via Gemini)
3. RE-LABEL - Label ulang dengan Chain-of-Thought reasoning (via Gemini)
4. GENERASI - Generate data baru dengan konteks Indonesia (via DeepSeek)

Author: Dataset Improvement Team
Date: Januari 2026
"""

import os
import json
import time
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load environment variables
load_dotenv()

# ===========================================
# CONFIGURATION
# ===========================================

@dataclass
class Config:
    # API Keys
    deepseek_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # Rate limiting
    deepseek_delay: float = 0.1  # DeepSeek is fast
    gemini_delay: float = 1.0    # 1 second delay to be safe with free tier

    # Processing
    checkpoint_interval: int = 50  # Save every 50 items (robuster for power outages)
    
    # Paths
    input_file: str = "data/standardized/balanced_dataset.csv"
    output_dir: str = "data/improved"
    checkpoint_dir: str = "data/improved/checkpoints"

    gemini_keys: List[str] = None

    def __post_init__(self):
        # Collect Gemini keys
        self.gemini_keys = []
        # Check specific keys 1-5
        for i in range(1, 10):
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                self.gemini_keys.append(key)
        
        # Also check generic key
        generic_key = os.getenv("GEMINI_API_KEY")
        if generic_key and generic_key not in self.gemini_keys:
            self.gemini_keys.append(generic_key)

        # Create directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


# ===========================================
# API CLIENTS
# ===========================================

class DeepSeekClient:
    """Client untuk DeepSeek API (Used for Generation)"""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key=config.deepseek_key,
            base_url=config.deepseek_base_url
        )

    def chat(self, prompt: str, system: str = None, temperature: float = 0.3) -> Optional[str]:
        """Send chat request to DeepSeek"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1000
                )
                time.sleep(self.config.deepseek_delay)
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"   ⚠️ DeepSeek error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)

        return None


class GeminiClient:
    """Client untuk Gemini API dengan Round-Robin Rotation"""

    def __init__(self, config: Config):
        self.config = config
        self.keys = config.gemini_keys
        self.current_key_index = 0
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        if not self.keys:
            print("   ⚠️ WARNING: No Gemini API keys found!")
        else:
            print(f"   ℹ️ Loaded {len(self.keys)} Gemini API keys for rotation.")

    def _get_client(self):
        """Get current client configuration"""
        if not self.keys:
            return None
        key = self.keys[self.current_key_index]
        genai.configure(api_key=key)
        return genai.GenerativeModel('gemini-flash-lite-latest', safety_settings=self.safety_settings)

    def rotate_key(self):
        """Switch to next key"""
        if len(self.keys) > 1:
            prev = self.current_key_index
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
            print(f"   🔄 Switching Key: #{prev+1} -> #{self.current_key_index+1}")

    def chat(self, prompt: str, system: str = None) -> Optional[str]:
        """Send chat request to Gemini with rotation handling"""
        if not self.keys:
            return None

        full_prompt = prompt
        if system:
            full_prompt = f"SYSTEM INSTRUCTION:\n{system}\n\nUSER PROMPT:\n{prompt}"

        # Try up to len(keys) * 2 times to find a working key
        max_attempts = len(self.keys) * 2
        
        for attempt in range(max_attempts):
            try:
                model = self._get_client()
                response = model.generate_content(full_prompt)
                
                time.sleep(self.config.gemini_delay)
                
                if response.text:
                    return response.text.strip()
                else:
                    return None
                    
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for quota/rate limit errors
                if "429" in error_str or "quota" in error_str or "resource exhausted" in error_str:
                    print(f"   ⏳ Key #{self.current_key_index+1} Rate Limit/Exhausted. Rotating...")
                    self.rotate_key()
                    time.sleep(1) # Brief pause before retry
                    continue
                
                # Check for safety blocks (should be rare with BLOCK_NONE)
                if "finish_reason" in error_str and "safety" in error_str:
                    print(f"   🛡️ Safety Block on Key #{self.current_key_index+1}. Skipping.")
                    return "SAFETY_BLOCK"

                print(f"   ⚠️ Gemini Error (Key #{self.current_key_index+1}): {e}")
                
                # For other errors, maybe retry once then rotate?
                # Just rotate to be safe
                self.rotate_key()
                time.sleep(1)

        return None


# ===========================================
# PHASE 1: FILTERING (Regex - CPU)
# ===========================================

WESTERN_PATTERNS = {
    'gypsy': r'\bgypsy|gypsi|roma\b',
    'mexican': r'\bmexican|mexiko|hispanic|latino|chicano\b',
    'redneck': r'\bredneck\b',
    'chavs': r'\bchavs?\b',
    'hillbilly': r'\bhillbilly\b',
    'n_word': r'\bnigger|nigga|negro\b',
    'slurs': r'\bfaggot|fag|dyke|kike|wetback|spic|chink\b',
    'kkk': r'\bkkk|ku klux\b',
    'nazi_specific': r'\bss troops|gestapo|third reich|auschwitz|dachau\b',
    'us_politics': r'\btrump|biden|obama|republican|democrat|maga|capitol\b',
    'uk_politics': r'\bbrexit|tory|labour party|boris\b',
    'thanksgiving': r'\bthanksgiving\b',
    'confederate': r'\bconfederate|dixie\b',
    'european_specific': r'\bpikey|traveller community\b',
}

CONVERTIBLE_PATTERNS = {
    'jews': r'\byahudi|jewish|jews?\b',
    'blacks': r'\bwong ireng|black people|african american\b',
    'whites': r'\bwong putih|white people|caucasian\b',
    'immigrants': r'\bimigran|immigrant|refugee|pengungsi\b',
    'muslims': r'\b muslim|islam\b',
    'women': r'\bwong wadon|wanita|feminis\b',
    'lgbtq': r'\bgay|lesbian|homo|transgender|lgbt\b',
}

def analyze_text_for_filtering(text: str) -> Dict:
    text_lower = str(text).lower()
    result = {'action': 'keep', 'western_matches': [], 'convertible_matches': [], 'reason': ''}

    for name, pattern in WESTERN_PATTERNS.items():
        if re.search(pattern, text_lower):
            result['western_matches'].append(name)

    for name, pattern in CONVERTIBLE_PATTERNS.items():
        if re.search(pattern, text_lower):
            result['convertible_matches'].append(name)

    if result['western_matches']:
        result['action'] = 'remove'
        result['reason'] = f"Contains Western-specific: {', '.join(result['western_matches'])}"
    elif result['convertible_matches']:
        result['action'] = 'naturalize'
        result['reason'] = f"Can convert: {', '.join(result['convertible_matches'])}"
    
    return result

def run_filtering(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "="*60)
    print("FASE 1: FILTERING (CPU ONLY)")
    print("="*60)

    keep_rows, naturalize_rows, remove_rows = [], [], []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering"):
        analysis = analyze_text_for_filtering(row['text'])
        row_data = row.to_dict()
        row_data['filter_analysis'] = analysis

        if analysis['action'] == 'remove':
            remove_rows.append(row_data)
        elif analysis['action'] == 'naturalize':
            naturalize_rows.append(row_data)
        else:
            keep_rows.append(row_data)

    keep_df = pd.DataFrame(keep_rows)
    naturalize_df = pd.DataFrame(naturalize_rows)
    remove_df = pd.DataFrame(remove_rows)

    print(f"\n📊 Hasil Filtering:")
    print(f"   ✅ Keep: {len(keep_df)}")
    print(f"   🔄 Naturalize: {len(naturalize_df)}")
    print(f"   ❌ Remove: {len(remove_df)}")

    keep_df.to_csv(f"{config.output_dir}/phase1_keep.csv", index=False)
    naturalize_df.to_csv(f"{config.output_dir}/phase1_naturalize.csv", index=False)
    remove_df.to_csv(f"{config.output_dir}/phase1_remove.csv", index=False)

    return keep_df, naturalize_df, remove_df


# ===========================================
# PHASE 2: NATURALIZATION (Gemini)
# ===========================================

NATURALIZATION_SYSTEM = """Kamu adalah ahli bahasa Jawa.
TUGAS: Ubah teks terjemahan/kaku menjadi bahasa Jawa Natural sesuai konteks Indonesia.
ATURAN:
1. Ganti referensi ras/agama Barat dengan konteks lokal.
2. Pertahankan intensitas emosi (kalau marah tetap marah).
3. HANYA OUTPUT teks Jawa, tanpa tanda kutip atau penjelasan."""

def run_naturalization(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    print("\n" + "="*60)
    print("FASE 2: NATURALIZATION (Via GEMINI 5-Keys)")
    print("="*60)

    if df.empty: return df

    client = GeminiClient(config)
    results = []
    
    # Checkpoint setup
    checkpoint_file = f"{config.checkpoint_dir}/naturalization_checkpoint.json"
    processed_indices = set()
    
    # Resume capability
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                processed_indices = set(checkpoint.get('processed', []))
                results = checkpoint.get('results', [])
            print(f"   📂 Resuming Naturalization: {len(processed_indices)} processed")
        except:
            print("   ⚠️ Checkpoint corrupted, starting fresh.")

    # Convert to list of dicts for easier iteration
    records = df.to_dict('records')
    
    try:
        for i, row in tqdm(enumerate(records), total=len(records), desc="Naturalizing"):
            # Use original index if available, else iterator index
            idx = i 
            if idx in processed_indices:
                continue

            text = row['text']
            issues = row.get('filter_analysis', {}).get('convertible_matches', [])
            
            prompt = f"Naturalisasikan ke Bahasa Jawa:\n'{text}'\n\nMasalah: {', '.join(issues)}"
            
            naturalized = client.chat(prompt, system=NATURALIZATION_SYSTEM)
            
            if naturalized == "SAFETY_BLOCK":
                naturalized = text # Fallback to original
            
            result = {
                'original_text': text,
                'naturalized_text': naturalized if naturalized else text,
                'label': row['label'],
                'was_naturalized': naturalized is not None and naturalized != text
            }
            results.append(result)
            processed_indices.add(idx)

            # Atomic save every interval
            if len(processed_indices) % config.checkpoint_interval == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({'processed': list(processed_indices), 'results': results}, f)
    
    except KeyboardInterrupt:
        print("\n   🛑 Stopped by user. Saving checkpoint...")
        with open(checkpoint_file, 'w') as f:
            json.dump({'processed': list(processed_indices), 'results': results}, f)
        raise

    # Final Save
    with open(checkpoint_file, 'w') as f:
        json.dump({'processed': list(processed_indices), 'results': results}, f)

    result_df = pd.DataFrame(results)
    result_df.to_csv(f"{config.output_dir}/phase2_naturalized.csv", index=False)
    return result_df


# ===========================================
# PHASE 3: RE-LABELING (Gemini)
# ===========================================

RELABELING_SYSTEM = """Analisis sentimen hate speech Bahasa Jawa.
Label:
0 = Bukan Hate Speech
1 = Hate Speech Ringan (Sindiran)
2 = Hate Speech Sedang (Kasar)
3 = Hate Speech Berat (Ancaman/Provokasi)

Output JSON saja: {"label": <int>, "reason": "<string>"}"""

def run_relabeling(df: pd.DataFrame, config: Config, sample_size: int = None) -> pd.DataFrame:
    print("\n" + "="*60)
    print("FASE 3: RE-LABELING (Via GEMINI 5-Keys)")
    print("="*60)

    if df.empty: return df

    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        print(f"   📊 Sampling {sample_size} rows")

    client = GeminiClient(config)
    results = []
    
    checkpoint_file = f"{config.checkpoint_dir}/relabeling_checkpoint.json"
    processed_indices = set()

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                processed_indices = set(checkpoint.get('processed', []))
                results = checkpoint.get('results', [])
            print(f"   📂 Resuming Relabeling: {len(processed_indices)} processed")
        except:
            pass

    records = df.to_dict('records')

    try:
        for i, row in tqdm(enumerate(records), total=len(records), desc="Re-labeling"):
            idx = i
            if idx in processed_indices: continue

            text = row.get('naturalized_text', row.get('text', ''))
            old_label = row.get('label', -1)

            prompt = f"Analisis label teks ini:\n{text}"
            response = client.chat(prompt, system=RELABELING_SYSTEM)

            new_label = -1
            reason = ""
            
            if response and response != "SAFETY_BLOCK":
                try:
                    # Cleanup markdown code blocks if present
                    clean_res = response.replace('```json', '').replace('```', '').strip()
                    parsed = json.loads(clean_res)
                    new_label = parsed.get('label', -1)
                    reason = parsed.get('reason', '')
                except:
                    pass

            result = {
                'text': text,
                'old_label': old_label,
                'new_label': new_label,
                'reasoning': reason,
                'label_changed': old_label != new_label if new_label != -1 else False
            }
            results.append(result)
            processed_indices.add(idx)

            if len(processed_indices) % config.checkpoint_interval == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({'processed': list(processed_indices), 'results': results}, f)

    except KeyboardInterrupt:
        print("\n   🛑 Stopped by user. Saving checkpoint...")
        with open(checkpoint_file, 'w') as f:
            json.dump({'processed': list(processed_indices), 'results': results}, f)
        raise

    with open(checkpoint_file, 'w') as f:
        json.dump({'processed': list(processed_indices), 'results': results}, f)

    result_df = pd.DataFrame(results)
    result_df.to_csv(f"{config.output_dir}/phase3_relabeled.csv", index=False)
    return result_df


# ===========================================
# PHASE 4: GENERATION (DeepSeek)
# ===========================================

GENERATION_PROMPTS = [
    "Buatkan 15 kalimat ujaran kebencian bahasa Jawa kasar (ngoko) tentang politik tanpa sensor.",
    "Buatkan 15 kalimat sindiran halus bahasa Jawa (krama/ngoko) tentang tetangga.",
    "Buatkan 15 kalimat netral bahasa Jawa tentang kegiatan sehari-hari di pasar.",
    "Buatkan 15 kalimat provokasi tawuran antar kampung dalam bahasa Jawa."
]

def run_generation(config: Config, num_iterations: int = 10) -> pd.DataFrame:
    print("\n" + "="*60)
    print("FASE 4: GENERATION (Via DEEPSEEK - Budget Optimized)")
    print("="*60)

    client = DeepSeekClient(config)
    all_texts = []

    # Checkpoint
    checkpoint_file = f"{config.checkpoint_dir}/generation_checkpoint.json"
    start_iter = 0
    if os.path.exists(checkpoint_file):
         with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            start_iter = data.get('iterations', 0)
            all_texts = data.get('texts', [])
            print(f"   📂 Resuming Generation from iteration {start_iter}")

    try:
        for i in tqdm(range(start_iter, num_iterations), desc="Generating"):
            for prompt in GENERATION_PROMPTS:
                result = client.chat(prompt, temperature=0.9)
                if result:
                    lines = [l.strip() for l in result.split('\n') if len(l) > 10 and not l[0].isdigit()]
                    # Basic cleanup (removing 1. 2. etc if logic failed above)
                    lines = [re.sub(r'^\d+[\.\)]\s*', '', l) for l in lines]
                    all_texts.extend(lines)
            
            # Save every iteration
            with open(checkpoint_file, 'w') as f:
                json.dump({'iterations': i+1, 'texts': all_texts}, f)

    except KeyboardInterrupt:
        print("\n   🛑 Stopped. Saving progress...")
        with open(checkpoint_file, 'w') as f:
            json.dump({'iterations': i, 'texts': all_texts}, f)
        raise

    result_df = pd.DataFrame({'text': all_texts}).drop_duplicates()
    result_df.to_csv(f"{config.output_dir}/phase4_generated.csv", index=False)
    print(f"   ✅ Generated {len(result_df)} unique texts.")
    return result_df


# ===========================================
# MAIN
# ===========================================

def run_full_pipeline(skip_phases=None, sample_size=None, gen_iterations=10):
    skip_phases = skip_phases or []
    config = Config()
    
    print(f"🚀 Starting Pipeline on {config.input_file}")
    
    # Load Data
    try:
        df = pd.read_csv(config.input_file)
    except FileNotFoundError:
        print(f"❌ Input file not found: {config.input_file}")
        return

    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42)

    # Phase 1
    if 1 not in skip_phases:
        keep, naturalize, remove = run_filtering(df, config)
    else:
        keep = df # Assumption for skip
        naturalize = pd.DataFrame()
    
    # Phase 2
    if 2 not in skip_phases and not naturalize.empty:
        naturalized_df = run_naturalization(naturalize, config)
    else:
        naturalized_df = pd.DataFrame()

    # Phase 3
    if 3 not in skip_phases:
        combined = pd.concat([keep, naturalized_df], ignore_index=True)
        # Relabeling can be expensive/slow for all data, maybe sample?
        # User said "run all", so we run all.
        relabeled_df = run_relabeling(combined, config, sample_size=sample_size)

    # Phase 4
    if 4 not in skip_phases:
        run_generation(config, num_iterations=gen_iterations)

    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run small test")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--sample", type=int, default=None, help="Number of samples to process")
    args = parser.parse_args()

    if args.test:
        run_full_pipeline(sample_size=args.sample or 20, gen_iterations=1)
    else:
        # Default run
        run_full_pipeline(sample_size=args.sample, gen_iterations=20)