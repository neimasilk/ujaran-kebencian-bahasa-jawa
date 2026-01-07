"""
EXPERIMENT 11: LLM-as-Judge untuk Re-labeling Uncertain Samples
================================================================

Task ini bisa JALAN DI KOMPUTER BIASA (tanpa GPU) karena menggunakan API.

Konsep:
1. Identify uncertain samples (model confidence < threshold)
2. Gunakan LLM (Claude/GPT) untuk re-label
3. Add confident predictions to training set
4. Re-train model dengan improved labels

Expected: +0.5-1% F1-Macro improvement
Cost: ~$5-20 untuk API calls
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

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LLMRelabelConfig:
    """Configuration untuk LLM Re-labeling"""

    # Model paths (hanya untuk get predictions)
    model_path: str = "models/experiment_6a_focal_loss/checkpoint-1503"
    tokenizer_name: str = "indobenchmark/indobert-base-p1"

    # Data paths
    phase3_path: str = "data/improved/phase3_relabeled.csv"
    phase4_path: str = "data/improved/phase4_generated.csv"

    # Output
    output_dir: str = "results/experiment_11_llm_relabeled"
    relabeled_output: str = "data/improved/phase5_llm_relabeled.csv"

    # Uncertainty threshold
    confidence_threshold: float = 0.6  # Samples below this are "uncertain"

    # LLM API Configuration
    llm_provider: str = "anthropic"  # or "openai"
    api_key: Optional[str] = None  # Will read from environment variable

    # Sampling
    max_samples_to_relabel: int = 500  # Max samples to re-label (cost control)
    min_confidence_for_accept: float = 0.8  # Only accept LLM labels if confident

    # Other
    max_length: int = 128
    batch_size: int = 32
    seed: int = 42

    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


# =============================================================================
# LLM JUDGE
# =============================================================================

class LLMJudge:
    """
    LLM-as-Judge untuk hate speech classification.

    Menggunakan Claude atau GPT-4 untuk classify text dengan reasoning.
    """

    def __init__(self, provider: str = "anthropic", api_key: Optional[str] = None):
        self.provider = provider
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY"
        }
        self.api_key = api_key or os.getenv(env_map.get(provider, "API_KEY"))

        if not self.api_key:
            raise ValueError(f"API key not found. Set {env_map.get(provider, 'API_KEY')} environment variable.")

    def classify(self, text: str) -> Tuple[int, float, str]:
        """
        Classify text using LLM.

        Args:
            text: Input text

        Returns:
            (label, confidence, reasoning)
        """
        if self.provider == "anthropic":
            return self._classify_anthropic(text)
        elif self.provider == "openai":
            return self._classify_openai(text)
        elif self.provider == "deepseek":
            return self._classify_deepseek(text)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _classify_anthropic(self, text: str) -> Tuple[int, float, str]:
        """Classify using Claude API."""
        prompt = f"""Classify the following Javanese/Indonesian text into ONE of these hate speech severity levels:

0 = Neutral (not hate speech)
1 = Light Hate (mild insults, sarcasm, not severe)
2 = Moderate Hate (clear insults, moderate severity)
3 = Severe Hate (extreme hate speech, threats, dehumanization)

Text: "{text}"

First, provide your reasoning. Then output FINAL: X where X is the number (0-3).

Format your response as:
REASONING: [your reasoning]
FINAL: [0, 1, 2, or 3]"""

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        data = {
            "model": "claude-3-haiku-20240307",  # Cheapest option
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            content = result["content"][0]["text"]

            # Parse response
            return self._parse_llm_response(content)

        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return None, 0.0, f"Error: {str(e)}"

    def _classify_openai(self, text: str) -> Tuple[int, float, str]:
        """Classify using OpenAI API."""
        import openai
        openai.api_key = self.api_key

        prompt = f"""Classify the following Javanese/Indonesian text into ONE of these hate speech severity levels:

0 = Neutral (not hate speech)
1 = Light Hate (mild insults, sarcasm, not severe)
2 = Moderate Hate (clear insults, moderate severity)
3 = Severe Hate (extreme hate speech, threats, dehumanization)

Text: "{text}"

First, provide your reasoning. Then output FINAL: X where X is the number (0-3).

Format your response as:
REASONING: [your reasoning]
FINAL: [0, 1, 2, or 3]"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",  # Cheapest GPT-4 option
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )

            content = response.choices[0].message.content
            return self._parse_llm_response(content)

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None, 0.0, f"Error: {str(e)}"

    def _classify_deepseek(self, text: str) -> Tuple[int, float, str]:
        """Classify using DeepSeek API (OPENAI-COMPATIBLE)."""
        import openai

        # Configure for DeepSeek API
        openai.api_key = self.api_key
        openai.api_base = "https://api.deepseek.com"

        prompt = f"""Classify the following Javanese/Indonesian text into ONE of these hate speech severity levels:

0 = Neutral (not hate speech)
1 = Light Hate (mild insults, sarcasm, not severe)
2 = Moderate Hate (clear insults, moderate severity)
3 = Severe Hate (extreme hate speech, threats, dehumanization)

Text: "{text}"

First, provide your reasoning. Then output FINAL: X where X is the number (0-3).

Format your response as:
REASONING: [your reasoning]
FINAL: [0, 1, 2, or 3]"""

        try:
            response = openai.chat.completions.create(
                model="deepseek-chat",  # DeepSeek's model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )

            content = response.choices[0].message.content
            return self._parse_llm_response(content)

        except Exception as e:
            print(f"Error calling DeepSeek API: {e}")
            return None, 0.0, f"Error: {str(e)}"

    def _parse_llm_response(self, content: str) -> Tuple[int, float, str]:
        """Parse LLM response to extract label, confidence, and reasoning."""
        lines = content.strip().split('\n')

        reasoning = ""
        final_label = None

        for line in lines:
            line = line.strip()
            if line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.upper().startswith("FINAL:"):
                try:
                    final_label = int(line.split(":")[1].strip())
                except:
                    pass

        # Extract reasoning if not explicitly marked
        if not reasoning and final_label is None:
            # Try to find any number 0-3
            import re
            numbers = re.findall(r'\b[0-3]\b', content)
            if numbers:
                final_label = int(numbers[-1])

        # Default confidence based on whether we got a clear answer
        confidence = 0.9 if final_label is not None else 0.0

        if final_label is None:
            final_label = 0  # Default to neutral

        return final_label, confidence, reasoning or content

    def classify_batch(self, texts: List[str], delay: float = 0.5) -> List[Tuple[int, float, str]]:
        """
        Classify multiple texts with rate limiting.

        Args:
            texts: List of input texts
            delay: Delay between API calls (seconds)

        Returns:
            List of (label, confidence, reasoning) tuples
        """
        results = []

        for text in tqdm(texts, desc="LLM Classification"):
            result = self.classify(text)
            results.append(result)
            time.sleep(delay)  # Rate limiting

        return results


# =============================================================================
# UNCERTAINTY SAMPLING (CPU-ONLY VERSION)
# =============================================================================

class UncertaintySampler:
    """
    Identify uncertain samples using pre-computed probabilities.

    Since we're on CPU, we'll load pre-computed probabilities if available,
    or use a simple heuristic based on text characteristics.
    """

    def __init__(self, config: LLMRelabelConfig):
        self.config = config
        self.uncertain_samples = []

    def find_uncertain_samples_from_probs(
        self,
        texts: List[str],
        probs: np.ndarray,
        labels: np.ndarray
    ) -> List[Dict]:
        """
        Find uncertain samples from pre-computed probabilities.

        Args:
            texts: List of texts
            probs: (n_samples, n_classes) probability array
            labels: (n_samples,) true labels

        Returns:
            List of uncertain sample dicts
        """
        uncertain = []

        for i, (text, prob, label) in enumerate(zip(texts, probs, labels)):
            max_prob = prob.max()
            entropy = -np.sum(prob * np.log(prob + 1e-10))

            # Sample is uncertain if:
            # 1. Max probability is below threshold
            # 2. High entropy (confused between classes)
            if max_prob < self.config.confidence_threshold or entropy > 1.0:
                uncertain.append({
                    'index': i,
                    'text': text,
                    'original_label': int(label),
                    'max_prob': float(max_prob),
                    'entropy': float(entropy),
                    'probs': prob.tolist()
                })

        # Sort by uncertainty (lower max_prob = more uncertain)
        uncertain.sort(key=lambda x: x['max_prob'])

        return uncertain[:self.config.max_samples_to_relabel]

    def find_uncertain_samples_heuristic(
        self,
        df: pd.DataFrame
    ) -> List[Dict]:
        """
        Find potentially uncertain samples using heuristics (no model needed).

        This is for CPU-only when model predictions aren't available.

        Heuristics:
        1. Short texts (less context)
        2. Texts with mixed signals (both positive and negative words)
        3. Borderline cases (sarcasm, subtle hate)
        """
        uncertain = []

        # Words that might indicate light/ambiguous hate
        light_indicators = ['bodoh', 'goblok', 'tolol', 'salah', 'jelek']
        moderate_indicators = ['anjing', 'babi', 'tai', 'jancok']
        neutral_indicators = ['tidak', 'bukan', 'hanya', 'saja']

        for idx, row in df.iterrows():
            text = str(row['text']).lower()
            label = row['label']

            # Skip severe labels (usually clear)
            if label == 3:
                continue

            # Check for mixed signals
            has_light = any(word in text for word in light_indicators)
            has_neutral = any(word in text for word in neutral_indicators)
            has_moderate = any(word in text for word in moderate_indicators)

            # Short texts are more uncertain
            is_short = len(text.split()) < 10

            # Mixed signals = uncertain
            if (has_light and has_neutral) or (is_short and has_light):
                uncertain.append({
                    'index': idx,
                    'text': row['text'],
                    'original_label': int(label),
                    'max_prob': 0.5,  # Unknown
                    'entropy': 1.5,   # High
                    'probs': [0.25, 0.25, 0.25, 0.25]
                })

        return uncertain[:self.config.max_samples_to_relabel]


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

class LLMRelabelExperiment:
    """Main experiment untuk LLM re-labeling"""

    def __init__(self, config: LLMRelabelConfig):
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Initialize LLM Judge
        self.judge = LLMJudge(provider=config.llm_provider)

        # Initialize Uncertainty Sampler
        self.sampler = UncertaintySampler(config)

    def load_data(self) -> pd.DataFrame:
        """Load data"""
        print("Loading data...")

        phase3_df = pd.read_csv(self.config.phase3_path)
        phase3_df = phase3_df[['text', 'new_label']].copy()
        phase3_df = phase3_df.rename(columns={'new_label': 'label'})

        phase4_df = pd.read_csv(self.config.phase4_path)
        phase4_df = phase4_df[['text', 'label']].copy()

        combined_df = pd.concat([phase3_df, phase4_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)

        print(f"Total samples: {len(combined_df)}")

        return combined_df

    def run(self) -> Dict:
        """Run LLM re-labeling experiment"""

        print("=" * 60)
        print("EXPERIMENT 11: LLM-as-Judge Re-labeling")
        print("=" * 60)
        print(f"Provider: {self.config.llm_provider}")
        print(f"Max samples to re-label: {self.config.max_samples_to_relabel}")

        # Load data
        df = self.load_data()

        # Find uncertain samples (using heuristic for CPU-only)
        print("\nFinding uncertain samples...")
        uncertain_samples = self.sampler.find_uncertain_samples_heuristic(df)
        print(f"Found {len(uncertain_samples)} uncertain samples")

        # Re-label with LLM
        print("\nRe-labeling with LLM...")
        texts_to_relabel = [s['text'] for s in uncertain_samples]

        llm_results = self.judge.classify_batch(
            texts_to_relabel,
            delay=0.5  # Rate limiting
        )

        # Process results
        relabeled_samples = []
        accepted_count = 0

        for sample, (llm_label, llm_confidence, llm_reasoning) in zip(uncertain_samples, llm_results):
            # Check if we should accept the LLM label
            accept = (
                llm_label is not None and
                llm_confidence >= self.config.min_confidence_for_accept
            )

            relabeled_sample = {
                'text': sample['text'],
                'original_label': sample['original_label'],
                'llm_label': llm_label if accept else sample['original_label'],
                'llm_confidence': llm_confidence,
                'llm_reasoning': llm_reasoning,
                'max_prob': sample['max_prob'],
                'accepted': accept
            }

            relabeled_samples.append(relabeled_sample)

            if accept:
                accepted_count += 1

        # Save re-labeled data
        print(f"\nAccepted {accepted_count}/{len(llm_results)} LLM labels")

        # Create new dataframe with re-labeled samples
        new_df = df.copy()

        for sample in relabeled_samples:
            if sample['accepted']:
                idx = sample['index']
                new_df.at[idx, 'label'] = sample['llm_label']

        # Save
        output_path = self.config.relabeled_output
        new_df.to_csv(output_path, index=False)
        print(f"Saved re-labeled data to: {output_path}")

        # Save detailed results
        results_path = os.path.join(self.config.output_dir, 'relabeled_details.json')
        with open(results_path, 'w') as f:
            json.dump({
                'total_samples': len(df),
                'uncertain_samples': len(uncertain_samples),
                'relabeled_samples': accepted_count,
                'details': relabeled_samples
            }, f, indent=2)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(df)}")
        print(f"Uncertain samples: {len(uncertain_samples)}")
        print(f"Re-labeled: {accepted_count}")
        print(f"New data saved to: {output_path}")
        print("\nNext step: Train model with re-labeled data")
        print("Command: python experiments/experiment_11_train_relabel.py")

        return {
            'total_samples': len(df),
            'uncertain_samples': len(uncertain_samples),
            'relabeled_samples': accepted_count,
            'output_path': output_path
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 11: LLM Re-labeling (CPU-ONLY)")
    parser.add_argument('--provider', type=str, default='deepseek',
                       choices=['anthropic', 'openai', 'deepseek'],
                       help='LLM provider (anthropic, openai, or deepseek)')
    parser.add_argument('--max-samples', type=int, default=500,
                       help='Maximum samples to re-label')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Confidence threshold for uncertain samples')

    args = parser.parse_args()

    # Check for API key
    api_key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY"
    }
    api_key_env = api_key_map[args.provider]

    if not os.getenv(api_key_env):
        print(f"ERROR: {api_key_env} environment variable not set!")
        print(f"\nTo set API key:")
        print(f"  set {api_key_env}=your_key_here  (Windows)")
        print(f"  export {api_key_env}=your_key_here  (Linux/Mac)")
        exit(1)

    config = LLMRelabelConfig(
        llm_provider=args.provider,
        max_samples_to_relabel=args.max_samples,
        confidence_threshold=args.threshold
    )

    experiment = LLMRelabelExperiment(config)
    results = experiment.run()
