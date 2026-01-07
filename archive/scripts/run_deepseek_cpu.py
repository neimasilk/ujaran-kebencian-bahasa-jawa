"""
EXPERIMENT 11: DeepSeek Re-labeling untuk Komputer Biasa (CPU-ONLY)
===================================================================

Gunakan DeepSeek API untuk re-label uncertain samples.
Bisa jalan di komputer biasa (tanpa GPU).

Usage:
    python run_deepseek_cpu.py --max-samples 100
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import requests
from tqdm import tqdm
import time
import argparse


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DeepSeekRelabelConfig:
    """Configuration untuk DeepSeek Re-labeling"""

    # Data paths
    phase3_path: str = "data/improved/phase3_relabeled.csv"
    phase4_path: str = "data/improved/phase4_generated.csv"

    # Output
    output_dir: str = "results/experiment_11_deepseek"
    relabeled_output: str = "data/improved/phase5_deepseek_relabeled.csv"

    # Sampling
    max_samples_to_relabel: int = 500
    min_confidence_for_accept: float = 0.7

    # Other
    seed: int = 42


# =============================================================================
# DEEPSEEK API CLIENT
# =============================================================================

class DeepSeekClient:
    """DeepSeek API client."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    def classify(self, text: str, retry: int = 3) -> Tuple[Optional[int], float, str]:
        """Classify text menggunakan DeepSeek API."""
        prompt = f"""Classify the following Javanese/Indonesian text into ONE of these hate speech severity levels:

0 = Neutral (not hate speech)
1 = Light Hate (mild insults, sarcasm, not severe)
2 = Moderate Hate (clear insults, moderate severity)
3 = Severe Hate (extreme hate speech, threats, dehumanization)

Text: "{text}"

Analyze the text carefully. First provide your reasoning, then output FINAL: X where X is the number (0-3).

Format your response as:
REASONING: [your analysis]
FINAL: [0, 1, 2, or 3]"""

        for attempt in range(retry):
            try:
                response = requests.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                        "temperature": 0.1
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    return self._parse_response(content)
                elif response.status_code == 429:
                    print(f"Rate limit hit, waiting... (attempt {attempt + 1}/{retry})")
                    time.sleep(2 ** attempt)
                else:
                    print(f"API Error: {response.status_code} - {response.text[:200]}")

            except Exception as e:
                print(f"Error calling DeepSeek API: {e}")
                if attempt < retry - 1:
                    time.sleep(1)

        return None, 0.0, f"Failed after {retry} attempts"

    def _parse_response(self, content: str) -> Tuple[int, float, str]:
        """Parse DeepSeek response"""
        import re

        reasoning = ""
        final_label = None

        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.upper().startswith("FINAL:"):
                try:
                    final_label = int(line.split(":")[1].strip())
                except:
                    pass

        if final_label is None:
            numbers = re.findall(r'\b[0-3]\b', content)
            if numbers:
                final_label = int(numbers[-1])

        if final_label is None:
            final_label = 0

        confidence = 0.9 if final_label is not None else 0.0
        reasoning = reasoning or content[:200]

        return final_label, confidence, reasoning

    def classify_batch(self, texts: List[str], delay: float = 0.3) -> List[Tuple[Optional[int], float, str]]:
        """Classify multiple texts with rate limiting"""
        results = []

        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(texts)}")

            result = self.classify(text)
            results.append(result)
            time.sleep(delay)

        return results


# =============================================================================
# UNCERTAINTY SAMPLING
# =============================================================================

def find_uncertain_samples(df: pd.DataFrame, max_samples: int = 500) -> List[Dict]:
    """Find potentially uncertain samples menggunakan heuristics."""
    uncertain = []

    light_indicators = ['bodoh', 'goblok', 'tolol', 'salah', 'jelek', 'sampah', 'sialan']
    neutral_indicators = ['tidak', 'bukan', 'hanya', 'saja', 'kalau', 'jika', 'memang']

    for idx, row in df.iterrows():
        text = str(row['text']).lower()
        label = row['label']

        if label == 3:  # Skip severe (usually clear)
            continue

        has_light = any(word in text for word in light_indicators)
        has_neutral = any(word in text for word in neutral_indicators)
        is_short = len(text.split()) < 10

        if (has_light and has_neutral) or (is_short and has_light):
            uncertain.append({
                'index': idx,
                'text': row['text'],
                'original_label': int(label)
            })

        if len(uncertain) >= max_samples:
            break

    return uncertain


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(max_samples: int = 100, api_key: str = None):
    """Run experiment"""

    if not api_key:
        # Load .env for API key
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

        api_key = os.getenv("DEEPSEEK_API_KEY", "")

    if not api_key:
        print("ERROR: No DeepSeek API key found!")
        print("Please set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")
        return None

    print(f"Using DeepSeek API key: {api_key[:10]}...{api_key[-4:]}")

    config = DeepSeekRelabelConfig(max_samples_to_relabel=max_samples)

    print("=" * 60)
    print("EXPERIMENT 11: DeepSeek Re-labeling (CPU-ONLY)")
    print("=" * 60)
    print(f"Max samples: {config.max_samples_to_relabel}")

    # Initialize client
    client = DeepSeekClient(api_key=api_key)

    # Load data
    print("\nLoading data...")
    phase3_df = pd.read_csv(config.phase3_path)
    phase3_df = phase3_df[['text', 'new_label']].copy()
    phase3_df = phase3_df.rename(columns={'new_label': 'label'})

    phase4_df = pd.read_csv(config.phase4_path)
    phase4_df = phase4_df[['text', 'label']].copy()

    df = pd.concat([phase3_df, phase4_df], ignore_index=True)
    df = df.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    print(f"Total samples: {len(df)}")

    # Find uncertain samples
    print("\nFinding uncertain samples...")
    uncertain_samples = find_uncertain_samples(df, config.max_samples_to_relabel)
    print(f"Found {len(uncertain_samples)} uncertain samples")

    # Test API first
    print("\nTesting DeepSeek API...")
    test_result = client.classify("Testing connection")
    if test_result[0] is None:
        print("ERROR: DeepSeek API test failed!")
        print(f"Error: {test_result[2]}")
        return None
    print("API test passed!")

    # Re-label
    print(f"\nRe-labeling {len(uncertain_samples)} samples with DeepSeek API...")
    texts_to_relabel = [s['text'] for s in uncertain_samples]
    deepseek_results = client.classify_batch(texts_to_relabel, delay=0.5)

    # Process results
    relabeled_samples = []
    accepted_count = 0
    failed_count = 0

    for sample, (deepseek_label, deepseek_confidence, deepseek_reasoning) in zip(uncertain_samples, deepseek_results):
        accept = (
            deepseek_label is not None and
            deepseek_confidence >= config.min_confidence_for_accept
        )

        relabeled_sample = {
            'text': sample['text'],
            'original_label': sample['original_label'],
            'llm_label': deepseek_label if accept else sample['original_label'],
            'llm_confidence': deepseek_confidence,
            'llm_reasoning': deepseek_reasoning,
            'accepted': accept
        }
        relabeled_samples.append(relabeled_sample)

        if accept:
            accepted_count += 1
        if deepseek_label is None:
            failed_count += 1

    # Save
    print(f"\nAccepted {accepted_count}/{len(deepseek_results)} labels")
    print(f"Failed {failed_count} classifications")

    new_df = df.copy()
    for i, sample in enumerate(relabeled_samples):
        if sample['accepted']:
            # Get original index from uncertain_samples
            idx = uncertain_samples[i]['index']
            new_df.at[idx, 'label'] = sample['llm_label']

    output_path = config.relabeled_output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    results_path = os.path.join(config.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'total_samples': len(df),
            'uncertain_samples': len(uncertain_samples),
            'relabeled_samples': accepted_count,
            'failed_samples': failed_count,
            'details': relabeled_samples[:50]
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {len(df)}")
    print(f"Uncertain: {len(uncertain_samples)}")
    print(f"Re-labeled: {accepted_count}")
    print(f"Output: {output_path}")

    return {
        'total_samples': len(df),
        'uncertain_samples': len(uncertain_samples),
        'relabeled_samples': accepted_count,
        'output_path': output_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek Re-labeling (CPU-ONLY)")
    parser.add_argument('--max-samples', type=int, default=100,
                       help='Maximum samples to re-label')
    parser.add_argument('--api-key', type=str, default=None,
                       help='DeepSeek API key (optional, will read from .env)')

    args = parser.parse_args()
    run_experiment(args.max_samples, args.api_key)
