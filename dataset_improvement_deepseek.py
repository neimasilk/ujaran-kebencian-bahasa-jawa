#!/usr/bin/env python3
"""
Dataset Improvement Pipeline - DEEPSEEK ONLY VERSION
=====================================================
Pipeline ini memperbaiki dataset terjemahan menjadi lebih natural dan relevan
untuk konteks Indonesia menggunakan HANYA DeepSeek API.

Tahapan:
1. FILTER   - Hapus data yang tidak bisa diselamatkan (referensi Barat)
2. NATURALISASI - Ubah terjemahan kaku ke Jawa natural (via DeepSeek)
3. RE-LABEL - Label ulang dengan Chain-of-Thought reasoning (via DeepSeek)
4. GENERASI - Generate data baru dengan konteks Indonesia (via DeepSeek)

Author: Dataset Improvement Team
Date: Januari 2026
Modified: DeepSeek-only version
"""

import os
import json
import time
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

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
    deepseek_delay: float = 0.5  # Delay between requests

    # Processing
    checkpoint_interval: int = 50  # Save every 50 items
    batch_size: int = 10  # Process in small batches

    # Paths
    input_file: str = "data/standardized/balanced_dataset.csv"
    output_dir: str = "data/improved"
    checkpoint_dir: str = "data/improved/checkpoints"

    def __post_init__(self):
        # Create directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


# ===========================================
# API CLIENT
# ===========================================

class DeepSeekClient:
    """Client untuk DeepSeek API (Used for ALL phases)"""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key=config.deepseek_key,
            base_url=config.deepseek_base_url
        )

    def chat(self, prompt: str, system: str = None, temperature: float = 0.3,
             max_tokens: int = 1000, response_format: str = "text") -> Optional[str]:
        """Send chat request to DeepSeek"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(3):
            try:
                kwargs = {
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

                # Try to use structured output if available
                if response_format == "json":
                    # Try to request JSON format
                    try:
                        kwargs["response_format"] = {"type": "json_object"}
                    except:
                        pass

                response = self.client.chat.completions.create(**kwargs)
                time.sleep(self.config.deepseek_delay)
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"   ! DeepSeek error (attempt {attempt+1}): {str(e)[:50]}")
                time.sleep(2 ** attempt)

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
    print("PHASE 1: FILTERING (CPU ONLY)")
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

    print(f"\n[*] Filtering Results:")
    print(f"   [OK] Keep: {len(keep_df)}")
    print(f"   [~] Naturalize: {len(naturalize_df)}")
    print(f"   [X] Remove: {len(remove_df)}")

    keep_df.to_csv(f"{config.output_dir}/phase1_keep.csv", index=False)
    naturalize_df.to_csv(f"{config.output_dir}/phase1_naturalize.csv", index=False)
    remove_df.to_csv(f"{config.output_dir}/phase1_remove.csv", index=False)

    return keep_df, naturalize_df, remove_df


# ===========================================
# PHASE 2: NATURALIZATION (DeepSeek)
# ===========================================

NATURALIZATION_SYSTEM = """You are a Javanese language expert.
TASK: Convert stiff/translated Javanese text to NATURAL Javanese appropriate for Indonesian context.
RULES:
1. Replace Western religious/racial references with Indonesian context (e.g., 'Jews' -> 'Cina', 'Mexican' -> 'pendatang')
2. Keep the emotion intensity intact (if angry, stay angry)
3. Use appropriate Javanese register (ngoko for informal, krama for formal)
4. OUTPUT ONLY the Javanese text, no quotes or explanation."""

def run_naturalization(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    print("\n" + "="*60)
    print("PHASE 2: NATURALIZATION (Via DEEPSEEK)")
    print("="*60)

    if df.empty:
        print("[!] No data to naturalize")
        return df

    client = DeepSeekClient(config)
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
            print(f"[*] Resuming Naturalization: {len(processed_indices)} processed")
        except:
            print("[!] Checkpoint corrupted, starting fresh.")

    # Convert to list of dicts
    records = df.to_dict('records')

    try:
        for i, row in tqdm(enumerate(records), total=len(records), desc="Naturalizing"):
            idx = i
            if idx in processed_indices:
                continue

            text = row['text']
            issues = row.get('filter_analysis', {}).get('convertible_matches', [])

            # Build prompt
            prompt = f"Naturalisasi teks Bahasa Jawa ini ke konteks Indonesia:\n\n'{text}'"
            if issues:
                prompt += f"\n\nMasalah yang perlu diperbaiki: {', '.join(issues)}"

            naturalized = client.chat(prompt, system=NATURALIZATION_SYSTEM, temperature=0.4)

            result = {
                'original_text': text,
                'naturalized_text': naturalized if naturalized else text,
                'original_label': row.get('label', ''),
                'was_naturalized': naturalized is not None and naturalized != text,
                'issues_fixed': ', '.join(issues)
            }
            results.append(result)
            processed_indices.add(idx)

            # Atomic save every interval
            if len(processed_indices) % config.checkpoint_interval == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({'processed': list(processed_indices), 'results': results}, f)
                print(f"[*] Checkpoint saved: {len(processed_indices)} processed")

    except KeyboardInterrupt:
        print("\n[!] Stopped by user. Saving checkpoint...")
        with open(checkpoint_file, 'w') as f:
            json.dump({'processed': list(processed_indices), 'results': results}, f)
        raise

    # Final Save
    with open(checkpoint_file, 'w') as f:
        json.dump({'processed': list(processed_indices), 'results': results}, f)

    result_df = pd.DataFrame(results)
    result_df.to_csv(f"{config.output_dir}/phase2_naturalized.csv", index=False)

    print(f"\n[*] Naturalization complete:")
    print(f"   Total processed: {len(result_df)}")
    print(f"   Successfully naturalized: {result_df['was_naturalized'].sum()}")

    return result_df


# ===========================================
# PHASE 3: RE-LABELING (DeepSeek)
# ===========================================

RELABELING_SYSTEM = """You are a hate speech detection expert for Javanese language.
Analyze the text and classify it into ONE of these categories:
0 = Not Hate Speech (Bukan Ujaran Kebencian) - Neutral or positive content
1 = Mild Hate Speech (Ujaran Kebencian - Ringan) - Subtle sarcasm, light insults
2 = Moderate Hate Speech (Ujaran Kebencian - Sedang) - Direct insults, harsh language
3 = Severe Hate Speech (Ujaran Kebencian - Berat) - Threats, incitement to violence, dehumanization

Output ONLY a JSON object: {"label": <int>, "confidence": <float 0-1>, "reason": "<brief explanation>"}"""

def run_relabeling(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    print("\n" + "="*60)
    print("PHASE 3: RE-LABELING (Via DEEPSEEK)")
    print("="*60)

    if df.empty:
        print("[!] No data to relabel")
        return df

    client = DeepSeekClient(config)
    results = []

    # Combine keep and naturalized data
    all_data = []

    # First, get phase1_keep data
    keep_file = f"{config.output_dir}/phase1_keep.csv"
    if os.path.exists(keep_file):
        keep_df = pd.read_csv(keep_file)
        for _, row in keep_df.iterrows():
            all_data.append({
                'text': row['text'],
                'original_label': row.get('final_label', row.get('label', '')),
                'source': 'keep'
            })

    # Then add naturalized data
    for _, row in df.iterrows():
        text = row.get('naturalized_text', row.get('text', ''))
        all_data.append({
            'text': text,
            'original_label': row.get('original_label', ''),
            'source': 'naturalized'
        })

    print(f"[*] Total data to relabel: {len(all_data)}")

    checkpoint_file = f"{config.checkpoint_dir}/relabeling_checkpoint.json"
    processed_indices = set()

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                processed_indices = set(checkpoint.get('processed', []))
                results = checkpoint.get('results', [])
            print(f"[*] Resuming Relabeling: {len(processed_indices)} processed")
        except:
            pass

    try:
        for i, item in tqdm(enumerate(all_data), total=len(all_data), desc="Re-labeling"):
            idx = i
            if idx in processed_indices:
                continue

            text = item['text']
            old_label = item['original_label']

            prompt = f"Analisis label untuk teks Bahasa Jawa ini:\n\n{text}"

            response = client.chat(prompt, system=RELABELING_SYSTEM, temperature=0.2)

            new_label = -1
            confidence = 0.0
            reason = ""

            if response:
                try:
                    # Cleanup markdown code blocks
                    clean_res = response.replace('```json', '').replace('```', '').strip()
                    parsed = json.loads(clean_res)
                    new_label = parsed.get('label', -1)
                    confidence = parsed.get('confidence', 0.0)
                    reason = parsed.get('reason', '')
                except json.JSONDecodeError:
                    # Fallback: try to extract label with regex
                    match = re.search(r'"label"\s*:\s*(\d+)', response)
                    if match:
                        new_label = int(match.group(1))

            result = {
                'text': text,
                'old_label': old_label,
                'new_label': new_label,
                'confidence': confidence,
                'reasoning': reason,
                'label_changed': str(old_label) != str(new_label) if new_label != -1 else False,
                'source': item['source']
            }
            results.append(result)
            processed_indices.add(idx)

            if len(processed_indices) % config.checkpoint_interval == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({'processed': list(processed_indices), 'results': results}, f)
                print(f"[*] Checkpoint saved: {len(processed_indices)} processed")

    except KeyboardInterrupt:
        print("\n[!] Stopped by user. Saving checkpoint...")
        with open(checkpoint_file, 'w') as f:
            json.dump({'processed': list(processed_indices), 'results': results}, f)
        raise

    with open(checkpoint_file, 'w') as f:
        json.dump({'processed': list(processed_indices), 'results': results}, f)

    result_df = pd.DataFrame(results)
    result_df.to_csv(f"{config.output_dir}/phase3_relabeled.csv", index=False)

    print(f"\n[*] Re-labeling complete:")
    print(f"   Total processed: {len(result_df)}")
    print(f"   Labels changed: {result_df['label_changed'].sum()}")

    # Show label distribution
    if 'new_label' in result_df.columns:
        print(f"\n[*] New Label Distribution:")
        for label in sorted(result_df['new_label'].unique()):
            count = (result_df['new_label'] == label).sum()
            print(f"   Label {label}: {count}")

    return result_df


# ===========================================
# PHASE 4: GENERATION (DeepSeek)
# ===========================================

# Categories for generation
GENERATION_CATEGORIES = [
    {
        'category': 'politics',
        'label': 2,
        'prompt': """Buatkan 10 contoh ujaran kebencian bahasa Jawa (ngoko kasar) tentang POLITIK INDONESIA.
Gunakan tema: korupsi, kebijakan pemerintah, atau ketidakpuasan politik.
OUTPUT HANYA kalimat-kalimat, tanpa penomoran atau penjelasan."""
    },
    {
        'category': 'neighbors',
        'label': 1,
        'prompt': """Buatkan 10 contoh sindiran halus bahasa Jawa (campuran ngoko dan krama) tentang TETANGGA.
Gunakan tema: kebisingan, pinjam barang tidak dikembalikan, atau ikut campur urusan.
OUTPUT HANYA kalimat-kalimat, tanpa penomoran atau penjelasan."""
    },
    {
        'category': 'neutral',
        'label': 0,
        'prompt': """Buatkan 10 contoh kalimat NETRAL bahasa Jawa tentang kegiatan sehari-hari.
Gunakan tema: belanja di pasar, ngobrol dengan teman, atau aktivitas rumah.
OUTPUT HANYA kalimat-kalimat, tanpa penomoran atau penjelasan."""
    },
    {
        'category': 'severe',
        'label': 3,
        'prompt': """Buatkan 10 contoh ujaran kebencian BERAT bahasa Jawa dengan konteks SOSIAL INDONESIA.
Gunakan tema: konflik antar golongan, provokasi, atau ancaman.
OUTPUT HANYA kalimat-kalimat, tanpa penomoran atau penjelasan."""
    },
    {
        'category': 'regional',
        'label': 2,
        'prompt': """Buatkan 10 contoh ujaran kebencian bahasa Jawa tentang suku/daerah lain di Indonesia.
Gunakan konteks yang realistis di Indonesia (bukan negara lain).
OUTPUT HANYA kalimat-kalimat, tanpa penomoran atau penjelasan."""
    }
]

def run_generation(config: Config, num_iterations: int = 10) -> pd.DataFrame:
    print("\n" + "="*60)
    print("PHASE 4: GENERATION (Via DEEPSEEK)")
    print("="*60)

    client = DeepSeekClient(config)
    all_results = []

    checkpoint_file = f"{config.checkpoint_dir}/generation_checkpoint.json"
    start_iter = 0

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                start_iter = data.get('iteration', 0)
                all_results = data.get('results', [])
            print(f"[*] Resuming Generation from iteration {start_iter}")
            print(f"[*] Already generated: {len(all_results)} texts")
        except:
            pass

    try:
        for i in tqdm(range(start_iter, num_iterations), desc="Generating"):
            for cat in GENERATION_CATEGORIES:
                prompt = cat['prompt']

                result = client.chat(prompt, temperature=0.9, max_tokens=1500)

                if result:
                    # Parse results - split by lines and clean
                    lines = result.split('\n')
                    for line in lines:
                        line = line.strip()
                        # Remove numbering (1., 2., -, etc.)
                        line = re.sub(r'^[\d\-\*\.]+\s*', '', line)
                        # Remove quotes
                        line = line.strip('"\'')
                        # Keep only meaningful lines
                        if len(line) > 15 and not line.startswith('Buatkan'):
                            all_results.append({
                                'text': line,
                                'category': cat['category'],
                                'label': cat['label']
                            })

            # Save checkpoint every iteration
            with open(checkpoint_file, 'w') as f:
                json.dump({'iteration': i + 1, 'results': all_results}, f)

    except KeyboardInterrupt:
        print("\n[!] Stopped by user. Saving progress...")
        with open(checkpoint_file, 'w') as f:
            json.dump({'iteration': i, 'results': all_results}, f)
        raise

    result_df = pd.DataFrame(all_results).drop_duplicates(subset=['text'])
    result_df.to_csv(f"{config.output_dir}/phase4_generated.csv", index=False)

    print(f"\n[*] Generation complete:")
    print(f"   Total unique texts: {len(result_df)}")

    # Show distribution
    for cat in GENERATION_CATEGORIES:
        count = (result_df['category'] == cat['category']).sum()
        print(f"   {cat['category']} (label {cat['label']}): {count}")

    return result_df


# ===========================================
# MAIN
# ===========================================

def run_full_pipeline(skip_phases=None, sample_size=None, gen_iterations=10):
    skip_phases = skip_phases or []
    config = Config()

    print(f"\n{'='*60}")
    print(f"DATASET IMPROVEMENT PIPELINE - DEEPSEEK ONLY")
    print(f"{'='*60}")
    print(f"[*] Input: {config.input_file}")
    print(f"[*] Output: {config.output_dir}")
    print(f"[*] DeepSeek API: {'Configured' if config.deepseek_key else 'NOT CONFIGURED!'}")

    if not config.deepseek_key:
        print("\n[ERROR] DEEPSEEK_API_KEY not found in .env!")
        return

    # Load Data
    try:
        df = pd.read_csv(config.input_file)
        print(f"[*] Loaded {len(df)} samples")
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {config.input_file}")
        return

    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42)
        print(f"[*] Sampling to {len(df)} for testing")

    # Phase 1: Filtering
    if 1 not in skip_phases:
        keep, naturalize, remove = run_filtering(df, config)
    else:
        print("\n[!] Skipping Phase 1")
        keep = pd.read_csv(f"{config.output_dir}/phase1_keep.csv")
        naturalize = pd.read_csv(f"{config.output_dir}/phase1_naturalize.csv")

    # Phase 2: Naturalization
    if 2 not in skip_phases and not naturalize.empty:
        naturalized_df = run_naturalization(naturalize, config)
    else:
        print("\n[!] Skipping Phase 2")
        naturalized_df = pd.DataFrame()

    # Phase 3: Re-labeling
    if 3 not in skip_phases:
        relabeled_df = run_relabeling(naturalized_df if not naturalized_df.empty else pd.DataFrame(), config)
    else:
        print("\n[!] Skipping Phase 3")

    # Phase 4: Generation
    if 4 not in skip_phases:
        generated_df = run_generation(config, num_iterations=gen_iterations)
    else:
        print("\n[!] Skipping Phase 4")

    print("\n" + "="*60)
    print("PIPELINE COMPLETED")
    print("="*60)
    print(f"\n[*] Output files:")
    print(f"   - {config.output_dir}/phase1_keep.csv")
    print(f"   - {config.output_dir}/phase1_naturalize.csv")
    print(f"   - {config.output_dir}/phase1_remove.csv")
    print(f"   - {config.output_dir}/phase2_naturalized.csv")
    print(f"   - {config.output_dir}/phase3_relabeled.csv")
    print(f"   - {config.output_dir}/phase4_generated.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset Improvement Pipeline - DeepSeek Only")
    parser.add_argument("--test", action="store_true", help="Run small test")
    parser.add_argument("--skip", type=int, nargs='+', default=[], help="Skip phases (e.g., --skip 1 2)")
    parser.add_argument("--sample", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--gen-iter", type=int, default=10, help="Generation iterations")
    args = parser.parse_args()

    if args.test:
        run_full_pipeline(skip_phases=args.skip, sample_size=args.sample or 50, gen_iterations=1)
    else:
        run_full_pipeline(skip_phases=args.skip, sample_size=args.sample, gen_iterations=args.gen_iter)
