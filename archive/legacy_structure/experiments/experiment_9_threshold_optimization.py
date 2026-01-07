"""
EXPERIMENT 9: Quick Win - Threshold Optimization per-Class
==========================================================

Optimasi decision threshold per-class untuk meningkatkan F1-Macro.

Konsep:
- Default: argmax (threshold implicit 0.5 untuk semua kelas)
- Better: Find optimal threshold per class pada validation set
- Apply optimized thresholds ke test set

Expected: +0.3-0.8% F1-Macro improvement
Target: Tembus 82% untuk workshop submission
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ThresholdConfig:
    """Configuration untuk Threshold Optimization"""

    # Model paths
    model_path: str = "models/experiment_6a_focal_loss/checkpoint-2505"
    tokenizer_name: str = "indobenchmark/indobert-base-p1"

    # Data paths
    phase3_path: str = "data/improved/phase3_relabeled.csv"
    phase4_path: str = "data/improved/phase4_generated.csv"

    # Output
    output_dir: str = "results/experiment_9_threshold_opt"

    # Threshold search range
    threshold_range: Tuple[float, float] = (0.2, 0.8)
    threshold_steps: int = 61  # 0.01 increments

    # Other
    max_length: int = 128
    batch_size: int = 32
    seed: int = 42

    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


# =============================================================================
# DATASET
# =============================================================================

class TextDataset(Dataset):
    """Simple dataset untuk text classification"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# =============================================================================
# THRESHOLD OPTIMIZER
# =============================================================================

class ThresholdOptimizer:
    """
    Optimasi decision threshold per-class untuk multi-class classification.

    Untuk multi-class, kita gunakan approach:
    1. Get probabilities dari model untuk setiap sample
    2. Untuk setiap class, cari threshold optimal di validation set
    3. Apply thresholds ke test set

    Strategy:
    - Strategy 1: Threshold per class (treat each class as binary one-vs-rest)
    - Strategy 2: Global threshold untuk semua kelas
    """

    def __init__(self, num_classes: int = 4, threshold_range: Tuple[float, float] = (0.2, 0.8),
                 threshold_steps: int = 61):
        self.num_classes = num_classes
        self.threshold_range = threshold_range
        self.threshold_steps = threshold_steps
        self.optimal_thresholds = None
        self.best_f1 = 0.0

    def find_optimal_thresholds_binary(
        self,
        val_probs: np.ndarray,
        val_labels: np.ndarray,
        threshold_range: Tuple[float, float] = (0.2, 0.8),
        n_steps: int = 61
    ) -> Dict[int, float]:
        """
        Find optimal threshold per class using binary approach.

        Untuk setiap class c, kita treat sebagai binary classification:
        - Positive: class == c
        - Negative: class != c

        Args:
            val_probs: (n_samples, n_classes) probabilities
            val_labels: (n_samples,) true labels
            threshold_range: (min, max) threshold range to search
            n_steps: number of steps to search

        Returns:
            Dict mapping class_id -> optimal_threshold
        """
        optimal_thresholds = {}

        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)

        for class_id in range(self.num_classes):
            best_threshold = 0.5
            best_f1 = 0.0

            # Create binary labels
            binary_labels = (val_labels == class_id).astype(int)
            class_probs = val_probs[:, class_id]

            # Search for best threshold
            for threshold in thresholds:
                # Apply threshold
                preds = (class_probs >= threshold).astype(int)

                # Calculate F1
                if preds.sum() == 0:  # No positive predictions
                    continue
                f1 = f1_score(binary_labels, preds, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            optimal_thresholds[class_id] = best_threshold

        return optimal_thresholds

    def predict_with_thresholds(
        self,
        probs: np.ndarray,
        thresholds: Dict[int, float]
    ) -> np.ndarray:
        """
        Predict using per-class thresholds.

        Untuk setiap sample, pilih class dengan:
        adjusted_score = prob / threshold (higher is better)

        Atau alternative: pilih class dengan prob > threshold,
        jika multiple, pilih yang tertinggi.

        Args:
            probs: (n_samples, n_classes) probabilities
            thresholds: Dict mapping class_id -> threshold

        Returns:
            predictions: (n_samples,) predicted class IDs
        """
        n_samples = probs.shape[0]
        predictions = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            # Calculate adjusted scores
            adjusted_scores = np.zeros(self.num_classes)
            for class_id in range(self.num_classes):
                threshold = thresholds.get(class_id, 0.5)
                # Adjust score by threshold (lower threshold = easier to predict)
                adjusted_scores[class_id] = probs[i, class_id] / threshold

            # Predict class with highest adjusted score
            predictions[i] = np.argmax(adjusted_scores)

        return predictions

    def optimize_and_evaluate(
        self,
        val_probs: np.ndarray,
        val_labels: np.ndarray,
        test_probs: np.ndarray,
        test_labels: np.ndarray
    ) -> Dict:
        """
        Optimize thresholds on validation set and evaluate on test set.

        Args:
            val_probs: Validation probabilities
            val_labels: Validation labels
            test_probs: Test probabilities
            test_labels: Test labels

        Returns:
            Dict with results
        """
        # Find optimal thresholds
        print("Finding optimal thresholds per class...")
        self.optimal_thresholds = self.find_optimal_thresholds_binary(
            val_probs, val_labels, self.threshold_range, self.threshold_steps
        )

        print("\nOptimal Thresholds:")
        for class_id, threshold in self.optimal_thresholds.items():
            class_name = ['Neutral', 'Light', 'Moderate', 'Severe'][class_id]
            print(f"  {class_name} (Class {class_id}): {threshold:.3f}")

        # Evaluate on validation set
        val_pred_baseline = np.argmax(val_probs, axis=1)
        val_pred_optimized = self.predict_with_thresholds(val_probs, self.optimal_thresholds)

        val_f1_baseline = f1_score(val_labels, val_pred_baseline, average='macro')
        val_f1_optimized = f1_score(val_labels, val_pred_optimized, average='macro')

        print(f"\nValidation Set:")
        print(f"  Baseline F1: {val_f1_baseline:.4f}")
        print(f"  Optimized F1: {val_f1_optimized:.4f}")
        print(f"  Improvement: {(val_f1_optimized - val_f1_baseline) * 100:+.2f}%")

        # Evaluate on test set
        test_pred_baseline = np.argmax(test_probs, axis=1)
        test_pred_optimized = self.predict_with_thresholds(test_probs, self.optimal_thresholds)

        test_f1_baseline = f1_score(test_labels, test_pred_baseline, average='macro')
        test_f1_optimized = f1_score(test_labels, test_pred_optimized, average='macro')

        print(f"\nTest Set:")
        print(f"  Baseline F1: {test_f1_baseline:.4f}")
        print(f"  Optimized F1: {test_f1_optimized:.4f}")
        print(f"  Improvement: {(test_f1_optimized - test_f1_baseline) * 100:+.2f}%")

        # Per-class results
        print(f"\nPer-Class Results (Test Set):")
        class_names = ['Neutral', 'Light', 'Moderate', 'Severe']
        for class_id in range(4):
            class_f1_baseline = f1_score(
                test_labels, test_pred_baseline, labels=[class_id], average='macro', zero_division=0
            )
            class_f1_optimized = f1_score(
                test_labels, test_pred_optimized, labels=[class_id], average='macro', zero_division=0
            )
            print(f"  {class_names[class_id]}: {class_f1_baseline:.4f} -> {class_f1_optimized:.4f} ({(class_f1_optimized - class_f1_baseline) * 100:+.2f}%)")

        return {
            'optimal_thresholds': self.optimal_thresholds,
            'val_f1_baseline': val_f1_baseline,
            'val_f1_optimized': val_f1_optimized,
            'test_f1_baseline': test_f1_baseline,
            'test_f1_optimized': test_f1_optimized,
            'test_improvement': (test_f1_optimized - test_f1_baseline) * 100,
            'test_pred_baseline': test_pred_baseline.tolist(),
            'test_pred_optimized': test_pred_optimized.tolist()
        }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

class ThresholdExperiment:
    """Main experiment untuk threshold optimization"""

    def __init__(self, config: ThresholdConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Setup output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_path)
        self.model.to(self.device)
        self.model.eval()

        # Optimizer
        self.optimizer = ThresholdOptimizer(
            num_classes=4,
            threshold_range=self.config.threshold_range,
            threshold_steps=self.config.threshold_steps
        )

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load dan split data"""
        print("Loading data...")

        # Load Phase 3
        phase3_df = pd.read_csv(self.config.phase3_path)
        phase3_df = phase3_df[['text', 'new_label']].copy()
        phase3_df = phase3_df.rename(columns={'new_label': 'label'})

        # Load Phase 4
        phase4_df = pd.read_csv(self.config.phase4_path)
        phase4_df = phase4_df[['text', 'label']].copy()

        # Combine
        combined_df = pd.concat([phase3_df, phase4_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)

        print(f"Total samples: {len(combined_df)}")

        # Split: train/val/test = 80/10/10
        train_df, temp_df = train_test_split(
            combined_df,
            test_size=(self.config.val_ratio + self.config.test_ratio),
            random_state=self.config.seed,
            stratify=combined_df['label']
        )

        val_ratio_adjusted = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adjusted),
            random_state=self.config.seed,
            stratify=temp_df['label']
        )

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def get_predictions(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions and probabilities"""
        texts = df['text'].tolist()
        labels = df['label'].tolist()

        dataset = TextDataset(texts, labels, self.tokenizer, self.config.max_length)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Getting predictions"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label = batch['label']

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)

                all_probs.append(probs.cpu().numpy())
                all_labels.extend(label.numpy())

        return np.vstack(all_probs), np.array(all_labels)

    def run(self) -> Dict:
        """Run experiment"""
        print("=" * 60)
        print("EXPERIMENT 9: Threshold Optimization per-Class")
        print("=" * 60)

        # Load data
        train_df, val_df, test_df = self.load_data()

        # Get predictions
        print("\nGetting validation predictions...")
        val_probs, val_labels = self.get_predictions(val_df)

        print("\nGetting test predictions...")
        test_probs, test_labels = self.get_predictions(test_df)

        # Baseline score (argmax)
        test_pred_baseline = np.argmax(test_probs, axis=1)
        baseline_f1 = f1_score(test_labels, test_pred_baseline, average='macro')
        print(f"\nBaseline (Argmax) F1-Macro: {baseline_f1:.4f} ({baseline_f1 * 100:.2f}%)")

        # Optimize thresholds
        results = self.optimizer.optimize_and_evaluate(
            val_probs, val_labels,
            test_probs, test_labels
        )

        # Save results
        results_save = {
            'baseline_f1_macro': float(baseline_f1),
            'optimized_f1_macro': float(results['test_f1_optimized']),
            'improvement_pct': float(results['test_improvement']),
            'optimal_thresholds': {str(k): float(v) for k, v in results['optimal_thresholds'].items()},
            'per_class_baseline': {},
            'per_class_optimized': {}
        }

        # Per-class breakdown
        class_names = ['Neutral', 'Light', 'Moderate', 'Severe']
        for class_id in range(4):
            class_f1_baseline = f1_score(
                test_labels, results['test_pred_baseline'], labels=[class_id], average='macro', zero_division=0
            )
            class_f1_optimized = f1_score(
                test_labels, results['test_pred_optimized'], labels=[class_id], average='macro', zero_division=0
            )
            results_save['per_class_baseline'][class_names[class_id]] = float(class_f1_baseline)
            results_save['per_class_optimized'][class_names[class_id]] = float(class_f1_optimized)

        with open(os.path.join(self.config.output_dir, 'results.json'), 'w') as f:
            json.dump(results_save, f, indent=2)

        print(f"\nResults saved to: {self.config.output_dir}/results.json")

        # Final summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Baseline F1-Macro:  {baseline_f1 * 100:.2f}%")
        print(f"Optimized F1-Macro: {results['test_f1_optimized'] * 100:.2f}%")
        print(f"Improvement:        {results['test_improvement']:+.2f}%")

        if results['test_f1_optimized'] >= 0.82:
            print("\nTARGET ACHIEVED! Ready for workshop submission!")
        else:
            gap = 0.82 - results['test_f1_optimized']
            print(f"\nGap to 82% target: {gap * 100:.2f}%")

        return results_save


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 9: Threshold Optimization")
    parser.add_argument('--model-path', type=str, default='models/experiment_6a_focal_loss/checkpoint-2505',
                       help='Path to trained model')
    parser.add_argument('--threshold-min', type=float, default=0.2,
                       help='Minimum threshold to search')
    parser.add_argument('--threshold-max', type=float, default=0.8,
                       help='Maximum threshold to search')
    parser.add_argument('--steps', type=int, default=61,
                       help='Number of threshold steps')

    args = parser.parse_args()

    config = ThresholdConfig(
        model_path=args.model_path,
        threshold_range=(args.threshold_min, args.threshold_max),
        threshold_steps=args.steps
    )

    experiment = ThresholdExperiment(config)
    results = experiment.run()
