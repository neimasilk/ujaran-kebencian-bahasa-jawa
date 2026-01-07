"""
EXPERIMENT 18: Hard Negative Mining
====================================
Find samples model consistently gets wrong, then re-label with human expertise.
This targets SYSTEMATIC ERRORS, not random noise.
"""
import os
import sys

os.environ['TRANSFORMERS_NO_PROGRESS_BAR'] = '1'

import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass

import json
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


@dataclass
class Config:
    # Best model to analyze
    model_path: str = "models/experiment_6a_focal_loss/checkpoint-2505"
    data_path: str = "data/improved/phase3_phase4_combined.csv"
    output_dir: str = "results/experiment_18_hard_negatives"

    # Hard negative thresholds
    confidence_threshold: float = 0.6  # Below this = uncertain
    max_samples_per_class: int = 100   # Limit for manual labeling

    seed: int = 42


class HardNegativeMiner:
    """Find samples that model consistently misclassifies"""

    def __init__(self, model_path, tokenizer_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=4
        ).to(self.device)
        self.model.eval()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities"""
        encodings = self.tokenizer(
            texts,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.model(**encodings.to(self.device))
            probs = F.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()

    def find_hard_negatives(
        self,
        df: pd.DataFrame,
        confidence_threshold: float = 0.6
    ) -> pd.DataFrame:
        """Find samples with low confidence on correct class"""

        texts = df['text'].tolist()
        true_labels = df['label'].tolist()

        print(f"Computing predictions for {len(texts)} samples...")
        probs = self.predict_proba(texts)

        # Get confidence for true label
        true_label_confidence = np.array([probs[i][label] for i, label in enumerate(true_labels)])

        # Get predicted labels
        pred_labels = np.argmax(probs, axis=-1)

        # Find hard negatives: low confidence on true label
        hard_mask = true_label_confidence < confidence_threshold
        misclassified_mask = pred_labels != true_labels

        # Combine: uncertain OR misclassified
        problem_mask = hard_mask | misclassified_mask

        hard_negatives = df[problem_mask].copy()
        hard_negatives['true_confidence'] = true_label_confidence[problem_mask]
        hard_negatives['predicted_label'] = pred_labels[problem_mask]
        hard_negatives['max_confidence'] = np.max(probs[problem_mask], axis=1)

        # Sort by true confidence (lowest first = hardest)
        hard_negatives = hard_negatives.sort_values('true_confidence')

        print(f"Found {len(hard_negatives)} hard negatives ({len(hard_negatives)/len(df)*100:.1f}%)")

        return hard_negatives.reset_index(drop=True)

    def analyze_by_class(self, hard_negatives: pd.DataFrame) -> Dict:
        """Analyze hard negatives by true class"""

        analysis = {}
        for true_label in range(4):
            class_hard = hard_negatives[hard_negatives['label'] == true_label]
            class_names = ['Neutral', 'Light', 'Moderate', 'Severe']

            # Most common misprediction
            if len(class_hard) > 0:
                most_common_wrong = class_hard['predicted_label'].mode()[0] if len(class_hard) > 0 else None
            else:
                most_common_wrong = None

            analysis[true_label] = {
                'class_name': class_names[true_label],
                'count': len(class_hard),
                'avg_confidence': class_hard['true_confidence'].mean() if len(class_hard) > 0 else 0,
                'most_common_wrong': int(most_common_wrong) if most_common_wrong is not None else None,
            }

        return analysis


def main():
    from dataclasses import dataclass

    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("="*50)
    print("EXPERIMENT 18: Hard Negative Mining")
    print("="*50)

    # Load data
    df = pd.read_csv(cfg.data_path)
    print(f"Data: {len(df)} samples")

    # Get val set for analysis
    _, test_df = train_test_split(
        df, test_size=0.1,
        random_state=cfg.seed,
        stratify=df['label']
    )

    # Initialize miner
    miner = HardNegativeMiner(cfg.model_path)

    # Find hard negatives
    hard_negatives = miner.find_hard_negatives(
        test_df,
        confidence_threshold=cfg.confidence_threshold
    )

    # Analyze by class
    analysis = miner.analyze_by_class(hard_negatives)

    print("\n" + "="*50)
    print("HARD NEGATIVE ANALYSIS")
    print("="*50)

    class_names = ['Neutral', 'Light', 'Moderate', 'Severe']
    for label, info in analysis.items():
        print(f"\nClass {label} ({info['class_name']}):")
        print(f"  Hard samples: {info['count']}")
        print(f"  Avg confidence: {info['avg_confidence']:.3f}")
        if info['most_common_wrong'] is not None:
            print(f"  Often confused with: {class_names[info['most_common_wrong']]}")

    # Export for manual labeling
    output_file = f"{cfg.output_dir}/hard_negatives_for_labeling.csv"

    # Prepare for labeling (select most uncertain per class)
    to_label = []
    for label in range(4):
        class_hard = hard_negatives[hard_negatives['label'] == label]
        n_samples = min(len(class_hard), cfg.max_samples_per_class)
        if n_samples > 0:
            to_label.append(class_hard.head(n_samples))

    if to_label:
        to_label_df = pd.concat(to_label, ignore_index=True)

        # Keep only relevant columns for labeling
        export_cols = ['text', 'label', 'true_confidence', 'predicted_label', 'max_confidence']
        to_label_df = to_label_df[export_cols]

        to_label_df.to_csv(output_file, index=False)
        print(f"\nExported {len(to_label_df)} samples for manual labeling")
        print(f"File: {output_file}")

        # Show examples
        print("\n" + "="*50)
        print("SAMPLE HARDEST CASES (per class)")
        print("="*50)

        for label in range(4):
            class_hard = hard_negatives[hard_negatives['label'] == label]
            if len(class_hard) > 0:
                print(f"\nClass {label} ({class_names[label]}) - Top 3 hardest:")
                for i, row in class_hard.head(3).iterrows():
                    print(f"  - \"{row['text'][:60]}...\"")
                    print(f"    True conf: {row['true_confidence']:.3f}, Pred: {row['predicted_label']}")

    # Save analysis
    results = {
        'experiment': 'Experiment 18 - Hard Negative Mining',
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(test_df),
        'hard_negatives_found': len(hard_negatives),
        'hard_negative_rate': float(len(hard_negatives) / len(test_df)),
        'per_class_analysis': {
            class_names[k]: {
                'count': int(v['count']),
                'avg_confidence': float(v['avg_confidence']),
                'most_confused_with': class_names[v['most_common_wrong']] if v['most_common_wrong'] is not None else 'N/A'
            }
            for k, v in analysis.items()
        }
    }

    with open(f'{cfg.output_dir}/analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAnalysis saved to: {cfg.output_dir}/analysis.json")

    return results


if __name__ == "__main__":
    main()
