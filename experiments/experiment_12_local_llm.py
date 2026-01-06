"""
EXPERIMENT 12: Local LLM Re-labeling (GPU-Required)
===================================================

Gunakan local LLM (Llama/Mistral/Qwen) untuk re-labeling.
GRATIS - tidak perlu API key!

Requirements:
- GPU: RTX 4080 16GB (perfect!)
- Install: pip install llama-cpp-python or transformers

Models yang direkomendasikan (download otomatis):
1. Qwen 2.5 7B Instruct (4-bit quantized) - ~5GB VRAM
2. Llama 3.1 8B Instruct (4-bit quantized) - ~6GB VRAM
3. Mistral 7B Instruct v0.3 (4-bit quantized) - ~5GB VRAM
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import torch


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LocalLLMConfig:
    """Configuration untuk Local LLM Re-labeling"""

    # Data paths
    phase3_path: str = "data/improved/phase3_relabeled.csv"
    phase4_path: str = "data/improved/phase4_generated.csv"

    # Output
    output_dir: str = "results/experiment_12_local_llm"
    relabeled_output: str = "data/improved/phase6_local_llm_relabeled.csv"

    # Model selection
    model_type: str = "qwen"  # Options: 'qwen', 'llama', 'mistral', 'gemmas'

    # Sampling
    max_samples_to_relabel: int = 500
    confidence_threshold: float = 0.6

    # Generation
    max_new_tokens: int = 100
    temperature: float = 0.1  # Low temp for consistent labeling

    # Other
    seed: int = 42


# =============================================================================
# LOCAL LLM MODELS
# =============================================================================

class LocalLLMClassifier:
    """
    Local LLM classifier untuk hate speech detection.

    Supports multiple models via HuggingFace Transformers.
    """

    def __init__(self, model_type: str = "qwen", device: str = "cuda"):
        self.model_type = model_type
        self.device = device

        print(f"Loading model: {model_type}")
        print("This may take a few minutes on first run (downloading model)...")

        # Model configs
        self.models = {
            "qwen": {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "attn_implementation": "flash_attention_2"
            },
            "llama": {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "attn_implementation": None
            },
            "mistral": {
                "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                "attn_implementation": "flash_attention_2"
            },
            "gemmas": {
                "model_id": "google/gemma-2-9b-it",
                "attn_implementation": None
            }
        }

        if model_type not in self.models:
            raise ValueError(f"Unknown model: {model_type}. Choose from: {list(self.models.keys())}")

        self.model_config = self.models[model_type]
        self._load_model()

    def _load_model(self):
        """Load model dan tokenizer"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            # Quantization config untuk save VRAM
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["model_id"],
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            print("Loading model (4-bit quantization)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config["model_id"],
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation=self.model_config.get("attn_implementation")
            )
            self.model.eval()

            print(f"Model loaded successfully on {self.device}!")

        except ImportError as e:
            print("ERROR: Missing dependencies.")
            print("Install with: pip install transformers bitsandbytes accelerate")
            raise

    def classify(self, text: str) -> Tuple[int, float, str]:
        """
        Classify text using local LLM.

        Args:
            text: Input text

        Returns:
            (label, confidence, reasoning)
        """
        # Build prompt
        prompt = self._build_prompt(text)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Parse response
        return self._parse_response(response, text)

    def _build_prompt(self, text: str) -> str:
        """Build prompt untuk model"""

        # Class definitions
        class_desc = """
0 = Neutral (bukan ujaran kebencian)
1 = Light Hate (hate speech ringan, sarkasme, tidak parah)
2 = Moderate Hate (hate speech sedang, jelas menghina)
3 = Severe Hate (hate speech berat, ancaman, dehumanisasi)
"""

        if self.model_type == "qwen":
            return f"""<|im_start|>system Kamu adalah classifier hate speech bahasa Jawa/Indonesia.<|im_end|>
<|im_start|>user Classify teks berikut ke salah satu kategori:
{class_desc}
Teks: "{text}"

Jelaskan reasoning singkat, lalu output "FINAL: X" dimana X adalah angka 0-3.<|im_end|>
<|im_start|>assistant"""

        elif self.model_type == "llama":
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Kamu adalah classifier hate speech bahasa Jawa/Indonesia.<|eot_id|><|start_header_id|>user<|end_header_id|>

Classify teks berikut ke salah satu kategori:
{class_desc}

Teks: "{text}"

Jelaskan reasoning singkat, lalu output "FINAL: X" dimana X adalah angka 0-3.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        elif self.model_type == "mistral":
            return f"""[INST] Classify teks berikut ke salah satu kategori:
{class_desc}

Teks: "{text}"

Jelaskan reasoning singkat, lalu output "FINAL: X" dimana X adalah angka 0-3. [/INST]"""

        elif self.model_type == "gemmas":
            return f"""<start_of_turn>user
Classify teks berikut ke salah satu kategori:
{class_desc}

Teks: "{text}"

Jelaskan reasoning singkat, lalu output "FINAL: X" dimana X adalah angka 0-3.<end_of_turn>
<start_of_turn>model"""

        else:
            return f"""Classify teks berikut ke salah satu kategori:
{class_desc}

Teks: "{text}"

Jelaskan reasoning singkat, lalu output "FINAL: X" dimana X adalah angka 0-3."""

    def _parse_response(self, response: str, original_text: str) -> Tuple[int, float, str]:
        """Parse model response"""
        response = response.strip()

        # Try to find FINAL: X pattern
        import re
        final_match = re.search(r'FINAL[:\s]+(\d)', response, re.IGNORECASE)

        if final_match:
            label = int(final_match.group(1))
            confidence = 0.9
            reasoning = response
        else:
            # Try to find any number 0-3
            numbers = re.findall(r'\b[0-3]\b', response)
            if numbers:
                label = int(numbers[-1])
                confidence = 0.7
                reasoning = response
            else:
                # Default
                label = 0
                confidence = 0.0
                reasoning = f"Could not parse. Original: {response}"

        return label, confidence, reasoning

    def classify_batch(self, texts: List[str]) -> List[Tuple[int, float, str]]:
        """Classify multiple texts"""
        results = []

        for text in tqdm(texts, desc=f"Classifying ({self.model_type})"):
            result = self.classify(text)
            results.append(result)

        return results


# =============================================================================
# UNCERTAINTY SAMPLING
# =============================================================================

def find_uncertain_samples(df: pd.DataFrame, max_samples: int = 500) -> List[Dict]:
    """Find uncertain samples menggunakan heuristics"""

    uncertain = []

    # Words yang indicate certain classes
    light_indicators = ['bodoh', 'goblok', 'tolol', 'salah', 'jelek', 'sampah']
    neutral_indicators = ['tidak', 'bukan', 'hanya', 'saja', 'kalau', 'jika']

    for idx, row in df.iterrows():
        text = str(row['text']).lower()
        label = row['label']

        # Skip severe labels (usually clear)
        if label == 3:
            continue

        # Check for mixed signals
        has_light = any(word in text for word in light_indicators)
        has_neutral = any(word in text for word in neutral_indicators)

        # Short texts or mixed signals = uncertain
        is_short = len(text.split()) < 10
        is_mixed = has_light and has_neutral

        if is_short or is_mixed:
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

def main():
    """Main function"""

    config = LocalLLMConfig(
        model_type="qwen",  # Options: qwen, llama, mistral, gemmas
        max_samples_to_relabel=500
    )

    print("=" * 60)
    print("EXPERIMENT 12: Local LLM Re-labeling")
    print("=" * 60)
    print(f"Model: {config.model_type}")
    print(f"Max samples: {config.max_samples_to_relabel}")

    # Initialize classifier
    classifier = LocalLLMClassifier(
        model_type=config.model_type,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load data
    print("\nLoading data...")
    phase3_df = pd.read_csv(config.phase3_path)
    phase3_df = phase3_df[['text', 'new_label']].copy()
    phase3_df = phase3_df.rename(columns={'new_label': 'label'})

    phase4_df = pd.read_csv(config.phase4_path)
    phase4_df = phase4_df[['text', 'label']].copy()

    df = pd.concat([phase3_df, phase4_df], ignore_index=True)
    print(f"Total samples: {len(df)}")

    # Find uncertain samples
    print("\nFinding uncertain samples...")
    uncertain_samples = find_uncertain_samples(df, config.max_samples_to_relabel)
    print(f"Found {len(uncertain_samples)} uncertain samples")

    # Classify with local LLM
    print("\nClassifying with local LLM...")
    texts_to_relabel = [s['text'] for s in uncertain_samples]

    llm_results = classifier.classify_batch(texts_to_relabel)

    # Process results
    relabeled_samples = []
    accepted_count = 0

    for sample, (llm_label, llm_confidence, llm_reasoning) in zip(uncertain_samples, llm_results):
        accept = llm_confidence >= config.confidence_threshold and llm_label != sample['original_label']

        relabeled_sample = {
            'text': sample['text'],
            'original_label': sample['original_label'],
            'llm_label': llm_label if accept else sample['original_label'],
            'llm_confidence': llm_confidence,
            'llm_reasoning': llm_reasoning,
            'accepted': accept
        }

        relabeled_samples.append(relabeled_sample)

        if accept:
            accepted_count += 1

    # Save re-labeled data
    print(f"\nAccepted {accepted_count}/{len(llm_results)} LLM labels")

    new_df = df.copy()
    for sample in relabeled_samples:
        if sample['accepted']:
            idx = sample['index']
            new_df.at[idx, 'label'] = sample['llm_label']

    output_path = config.relabeled_output
    new_df.to_csv(output_path, index=False)
    print(f"Saved re-labeled data to: {output_path}")

    # Save details
    os.makedirs(config.output_dir, exist_ok=True)
    with open(f"{config.output_dir}/details.json", 'w') as f:
        json.dump({
            'total_samples': len(df),
            'uncertain_samples': len(uncertain_samples),
            'relabeled_samples': accepted_count,
            'model': config.model_type
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Uncertain samples: {len(uncertain_samples)}")
    print(f"Re-labeled: {accepted_count}")
    print(f"\nOutput: {output_path}")

    print("\nNext step: Train with re-labeled data")
    print("Command: python experiments/experiment_11_train_relabel.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 12: Local LLM Re-labeling")
    parser.add_argument('--model', type=str, default='qwen',
                       choices=['qwen', 'llama', 'mistral', 'gemmas'],
                       help='Model to use')
    parser.add_argument('--max-samples', type=int, default=500,
                       help='Maximum samples to re-label')

    args = parser.parse_args()

    # Update config
    config = LocalLLMConfig(
        model_type=args.model,
        max_samples_to_relabel=args.max_samples
    )

    # Run
    main()
