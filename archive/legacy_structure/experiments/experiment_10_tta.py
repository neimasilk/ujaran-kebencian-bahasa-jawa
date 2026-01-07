"""
EXPERIMENT 10: Quick Win - Test-Time Augmentation (TTA)
========================================================

Test-Time Augmentation creates multiple augmented versions of each input
and averages the predictions to improve accuracy.

Augmentations:
1. Synonym replacement (using WordNet or simple rules)
2. Random word deletion
3. Random word swap
4. Noising (typos, character repetition)

Expected: +0.5-1% F1-Macro improvement
Target: Tembus 82% untuk workshop submission
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
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
# TEXT AUGMENTATION
# =============================================================================

class TextAugmenter:
    """
    Text augmentation untuk Test-Time Augmentation.

    Augmentations:
    1. Random deletion - delete random words
    2. Random swap - swap two random words
    3. Character noise - simulate typos
    4. (Optional) Synonym replacement - using IndoBERT tokenizer
    """

    def __init__(self, tokenizer=None, seed: int = 42):
        self.tokenizer = tokenizer
        self.rng = np.random.RandomState(seed)

        # Indonesian stopwords untuk deletion
        self.stopwords = set([
            'yang', 'dan', 'di', 'ke', 'dari', 'pada', 'untuk', 'dengan',
            'adalah', 'itu', 'ini', 'aku', 'kamu', 'dia', 'mereka',
            'juga', 'lagi', 'sudah', 'belum', 'akan', 'bisa', 'dapat'
        ])

        # Common Javanese words to avoid deleting
        self.javanese_keep = set([
            'bodoh', 'goblok', 'tolol', 'babi', 'anjing', 'memek',
            'kontol', 'jancok', 'asu', 'tai', 'pantek', 'ngentot'
        ])

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words from text.

        Args:
            text: Input text
            p: Probability of deleting each word (for stopwords only)
        """
        words = text.split()
        if len(words) <= 4:  # Keep short texts intact
            return text

        new_words = []
        for word in words:
            # Only delete stopwords with probability p
            if word.lower() in self.stopwords:
                if self.rng.random() > p:
                    new_words.append(word)
            elif word.lower() in self.javanese_keep:
                # Always keep hate words
                new_words.append(word)
            else:
                # Non-stopwords: lower probability of deletion
                if self.rng.random() > p * 0.3:
                    new_words.append(word)

        return ' '.join(new_words) if new_words else text

    def random_swap(self, text: str, n: int = 1) -> str:
        """
        Randomly swap two words in text.

        Args:
            text: Input text
            n: Number of swaps
        """
        words = text.split()
        if len(words) <= 2:
            return text

        for _ in range(n):
            idx1, idx2 = self.rng.randint(0, len(words), size=2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return ' '.join(words)

    def character_noise(self, text: str, p: float = 0.05) -> str:
        """
        Add character-level noise to simulate typos.

        Args:
            text: Input text
            p: Probability of noise per character
        """
        noisy_chars = []
        for char in text:
            if self.rng.random() < p:
                # Random character duplication
                if self.rng.random() < 0.5 and char.isalpha():
                    noisy_chars.append(char * 2)
                else:
                    noisy_chars.append(char)
            else:
                noisy_chars.append(char)
        return ''.join(noisy_chars)

    def lowercase_variant(self, text: str) -> str:
        """Convert to lowercase (different from original)."""
        return text.lower()

    def augment(self, text: str, n_augmentations: int = 3) -> List[str]:
        """
        Generate multiple augmented versions of text.

        Args:
            text: Input text
            n_augmentations: Number of augmented versions to generate

        Returns:
            List of augmented texts (includes original)
        """
        augmented = [text]  # Always include original

        # Generate different augmentations
        for i in range(n_augmentations - 1):
            aug_text = text

            # Apply random augmentation based on index
            aug_type = i % 4

            if aug_type == 0:
                aug_text = self.random_deletion(text, p=0.1)
            elif aug_type == 1:
                aug_text = self.random_swap(text, n=1)
            elif aug_type == 2:
                aug_text = self.character_noise(text, p=0.03)
            elif aug_type == 3:
                aug_text = self.lowercase_variant(text)

            augmented.append(aug_text)

        return augmented


# =============================================================================
# TTA PREDICTOR
# =============================================================================

class TTAPredictor:
    """
    Test-Time Augmentation predictor.

    Creates multiple augmented versions of input and averages predictions.
    """

    def __init__(self, model, tokenizer, device, max_length: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.augmenter = TextAugmenter(tokenizer)

    def predict_single(self, text: str, n_augmentations: int = 5) -> np.ndarray:
        """
        Predict with TTA for a single text.

        Args:
            text: Input text
            n_augmentations: Number of augmentations (including original)

        Returns:
            Averaged probabilities (num_classes,)
        """
        # Generate augmented versions
        augmented_texts = self.augmenter.augment(text, n_augmentations)

        # Get predictions for all versions
        all_probs = []
        for aug_text in augmented_texts:
            probs = self._predict_single_text(aug_text)
            all_probs.append(probs)

        # Average predictions
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs

    def _predict_single_text(self, text: str) -> np.ndarray:
        """Get probabilities for a single text."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()[0]

    def predict_batch(self, texts: List[str], n_augmentations: int = 5) -> np.ndarray:
        """
        Predict with TTA for a batch of texts.

        Args:
            texts: List of input texts
            n_augmentations: Number of augmentations per text

        Returns:
            Averaged probabilities (n_texts, num_classes)
        """
        all_probs = []

        for text in tqdm(texts, desc="TTA Prediction"):
            probs = self.predict_single(text, n_augmentations)
            all_probs.append(probs)

        return np.array(all_probs)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TTAConfig:
    """Configuration untuk TTA Experiment"""

    # Model paths
    model_path: str = "models/experiment_6a_focal_loss/checkpoint-1503"
    tokenizer_name: str = "indobenchmark/indobert-base-p1"

    # Data paths
    phase3_path: str = "data/improved/phase3_relabeled.csv"
    phase4_path: str = "data/improved/phase4_generated.csv"

    # Output
    output_dir: str = "results/experiment_10_tta"

    # TTA parameters
    n_augmentations: int = 5  # Number of augmented versions

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
# MAIN EXPERIMENT
# =============================================================================

class TTAExperiment:
    """Main experiment untuk Test-Time Augmentation"""

    def __init__(self, config: TTAConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Setup output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Load tokenizer and model
        print(f"Loading model from: {self.config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_path)
        self.model.to(self.device)
        self.model.eval()

        # TTA Predictor
        self.tta_predictor = TTAPredictor(
            self.model, self.tokenizer, self.device, self.config.max_length
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

    def get_baseline_predictions(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Get baseline predictions (no augmentation)"""
        texts = df['text'].tolist()
        labels = df['label'].tolist()

        dataset = TextDataset(texts, labels, self.tokenizer, self.config.max_length)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Baseline predictions"):
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
        print("EXPERIMENT 10: Test-Time Augmentation (TTA)")
        print("=" * 60)
        print(f"Number of augmentations: {self.config.n_augmentations}")

        # Load data
        train_df, val_df, test_df = self.load_data()

        # Get baseline predictions
        print("\nGetting baseline predictions...")
        test_probs_baseline, test_labels = self.get_baseline_predictions(test_df)

        test_pred_baseline = np.argmax(test_probs_baseline, axis=1)
        baseline_f1 = f1_score(test_labels, test_pred_baseline, average='macro')
        print(f"Baseline F1-Macro: {baseline_f1:.4f} ({baseline_f1 * 100:.2f}%)")

        # Get TTA predictions
        print(f"\nGetting TTA predictions ({self.config.n_augmentations} augmentations)...")
        test_texts = test_df['text'].tolist()
        test_probs_tta = self.tta_predictor.predict_batch(test_texts, self.config.n_augmentations)

        test_pred_tta = np.argmax(test_probs_tta, axis=1)
        tta_f1 = f1_score(test_labels, test_pred_tta, average='macro')
        print(f"TTA F1-Macro: {tta_f1:.4f} ({tta_f1 * 100:.2f}%)")

        improvement = (tta_f1 - baseline_f1) * 100
        print(f"Improvement: {improvement:+.2f}%")

        # Per-class results
        class_names = ['Neutral', 'Light', 'Moderate', 'Severe']
        print(f"\nPer-Class Results:")
        for class_id in range(4):
            class_f1_baseline = f1_score(
                test_labels, test_pred_baseline, labels=[class_id], average='macro', zero_division=0
            )
            class_f1_tta = f1_score(
                test_labels, test_pred_tta, labels=[class_id], average='macro', zero_division=0
            )
            print(f"  {class_names[class_id]}: {class_f1_baseline:.4f} -> {class_f1_tta:.4f} ({(class_f1_tta - class_f1_baseline) * 100:+.2f}%)")

        # Save results
        results = {
            'baseline_f1_macro': float(baseline_f1),
            'tta_f1_macro': float(tta_f1),
            'improvement_pct': float(improvement),
            'n_augmentations': self.config.n_augmentations,
            'per_class_baseline': {},
            'per_class_tta': {}
        }

        for class_id in range(4):
            class_f1_baseline = f1_score(
                test_labels, test_pred_baseline, labels=[class_id], average='macro', zero_division=0
            )
            class_f1_tta = f1_score(
                test_labels, test_pred_tta, labels=[class_id], average='macro', zero_division=0
            )
            results['per_class_baseline'][class_names[class_id]] = float(class_f1_baseline)
            results['per_class_tta'][class_names[class_id]] = float(class_f1_tta)

        with open(os.path.join(self.config.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {self.config.output_dir}/results.json")

        # Final summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Baseline F1-Macro:  {baseline_f1 * 100:.2f}%")
        print(f"TTA F1-Macro:       {tta_f1 * 100:.2f}%")
        print(f"Improvement:        {improvement:+.2f}%")

        if tta_f1 >= 0.82:
            print("\n*** TARGET ACHIEVED! Ready for workshop submission! ***")
        else:
            gap = 0.82 - tta_f1
            print(f"\nGap to 82% target: {gap * 100:.2f}%")

        return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 10: Test-Time Augmentation")
    parser.add_argument('--model-path', type=str, default='models/experiment_6a_focal_loss/checkpoint-1503',
                       help='Path to trained model')
    parser.add_argument('--n-aug', type=int, default=5,
                       help='Number of augmentations per sample')

    args = parser.parse_args()

    config = TTAConfig(
        model_path=args.model_path,
        n_augmentations=args.n_aug
    )

    experiment = TTAExperiment(config)
    results = experiment.run()
