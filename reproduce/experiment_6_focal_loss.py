"""
EXPERIMENT 6A: Quick Wins - Focal Loss + Label Smoothing
==========================================================

Quick wins untuk meningkatkan akurasi:
1. Focal Loss - focus pada hard examples
2. Label Smoothing - handle label noise
3. Combined approach

Target: 82-83% F1-Macro
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from transformers.trainer_utils import EvalPrediction
from datasets import Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# FOCAL LOSS IMPLEMENTATION
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss untuk addressing class imbalance.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    dimana:
    - p_t: probability dari true class
    - α_t: class weight
    - γ: focusing parameter (default=2.0)
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 reduction: str = 'mean', class_weights: Optional[torch.Tensor] = None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: logits (batch_size, num_classes)
            targets: labels (batch_size,)
        """
        # Convert logits to probabilities
        p = F.softmax(inputs, dim=1)

        # Get probability of true class
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze()

        # Calculate focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma

        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Apply class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            ce_loss = ce_loss * weights

        # Combine: focal loss = focal_term * ce_loss
        focal_loss = focal_term * ce_loss

        # Apply alpha weighting
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing untuk handling label noise.

    Smoothed label: (1 - ε) * y + ε / K

    dimana:
    - ε: smoothing parameter (default=0.1)
    - K: number of classes
    """

    def __init__(self, epsilon: float = 0.1, num_classes: int = 4):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: logits (batch_size, num_classes)
            targets: labels (batch_size,)
        """
        # Log softmax for numerical stability
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create one-hot encoded targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Apply label smoothing
        smooth_targets = (1 - self.epsilon) * targets_one_hot + \
                         self.epsilon / self.num_classes

        # Calculate KL divergence (equivalent to cross entropy with smoothed labels)
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        return loss.mean()


class FocalLabelSmoothingLoss(nn.Module):
    """
    Combined Focal Loss + Label Smoothing.

    Ini memberikan benefits dari kedua approaches:
    - Focal: focus pada hard examples
    - Label Smoothing: handle label noise & prevent overconfidence
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 epsilon: float = 0.1, num_classes: int = 4,
                 class_weights: Optional[torch.Tensor] = None):
        super(FocalLabelSmoothingLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.class_weights = class_weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: logits (batch_size, num_classes)
            targets: labels (batch_size,)
        """
        # Log softmax
        log_probs = F.log_softmax(inputs, dim=-1)

        # Get probability of true class
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze()

        # Create smoothed one-hot targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        smooth_targets = (1 - self.epsilon) * targets_one_hot + \
                         self.epsilon / self.num_classes

        # Calculate focal term
        focal_term = (1 - p_t) ** self.gamma

        # Calculate loss with smoothed labels
        loss = -smooth_targets * log_probs
        loss = loss.sum(dim=-1)  # Sum over classes

        # Apply focal term
        focal_loss = focal_term * loss

        # Apply class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            focal_loss = focal_loss * weights

        # Apply alpha weighting
        focal_loss = self.alpha * focal_loss

        return focal_loss.mean()


# =============================================================================
# CUSTOM TRAINER WITH CUSTOM LOSS
# =============================================================================

class CustomLossTrainer(Trainer):
    """
    Custom Trainer yang menggunakan Focal Loss dan/atau Label Smoothing.
    """

    def __init__(self, *args, loss_type: str = "focal",
                 alpha: float = 1.0, gamma: float = 2.0,
                 epsilon: float = 0.1, class_weights: Optional[torch.Tensor] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.class_weights = class_weights

        # Setup loss function
        if loss_type == "focal":
            self.loss_fn = FocalLoss(
                alpha=alpha,
                gamma=gamma,
                class_weights=class_weights
            )
        elif loss_type == "label_smoothing":
            self.loss_fn = LabelSmoothingLoss(
                epsilon=epsilon,
                num_classes=self.model.config.num_labels
            )
        elif loss_type == "combined":
            self.loss_fn = FocalLabelSmoothingLoss(
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                num_classes=self.model.config.num_labels,
                class_weights=class_weights
            )
        else:
            self.loss_fn = None  # Use default CrossEntropyLoss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss untuk menggunakan custom loss.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.loss_fn is not None:
            # Move class weights to same device as logits
            class_weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
            if hasattr(self.loss_fn, 'class_weights'):
                self.loss_fn.class_weights = class_weights

            loss = self.loss_fn(logits, labels)
        else:
            # Use default cross-entropy loss
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration untuk Experiment 6A"""

    # Model configurations
    model_name: str = "indobenchmark/indobert-base-p1"

    # Data paths
    phase3_path: str = "data/improved/phase3_relabeled.csv"
    phase4_path: str = "data/improved/phase4_generated.csv"

    # Output paths
    output_dir: str = "results/experiment_6a_focal_loss"
    model_output_dir: str = "models/experiment_6a_focal_loss"

    # Training hyperparameters
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 5
    weight_decay: float = 0.01

    # Loss function configuration
    loss_type: str = "combined"  # Options: "focal", "label_smoothing", "combined"
    alpha: float = 1.0
    gamma: float = 2.0
    epsilon: float = 0.1

    # Class weights (optional)
    use_class_weights: bool = False

    # Random seed
    seed: int = 42

    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


# =============================================================================
# DATASET PREPARATION
# =============================================================================

class ImprovedDatasetPreparer:
    """Persiapan dataset improved untuk training"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.combined_df = None

    def load_data(self):
        """Load dataset improved dari Phase 3 dan Phase 4"""
        logger.info("Loading improved dataset...")

        # Load Phase 3 (re-labeled)
        phase3_df = pd.read_csv(self.config.phase3_path)
        logger.info(f"Phase 3 loaded: {len(phase3_df)} records")

        # Load Phase 4 (generated)
        phase4_df = pd.read_csv(self.config.phase4_path)
        logger.info(f"Phase 4 loaded: {len(phase4_df)} records")

        # Prepare Phase 3: use 'text' and 'new_label'
        phase3_train = phase3_df[['text', 'new_label']].copy()
        phase3_train = phase3_train.rename(columns={'new_label': 'label'})

        # Prepare Phase 4: use 'text' and 'label'
        phase4_train = phase4_df[['text', 'label']].copy()

        # Combine
        self.combined_df = pd.concat([phase3_train, phase4_train], ignore_index=True)

        # Shuffle
        self.combined_df = self.combined_df.sample(
            frac=1, random_state=self.config.seed
        ).reset_index(drop=True)

        logger.info(f"Combined dataset: {len(self.combined_df)} records")
        logger.info(f"\nLabel distribution:\n{self.combined_df['label'].value_counts()}")

        return self.combined_df

    def split_data(self):
        """Split data menjadi train/val/test"""
        logger.info("Splitting data...")

        # First split: train + temp (val + test)
        train_df, temp_df = train_test_split(
            self.combined_df,
            test_size=(self.config.val_ratio + self.config.test_ratio),
            random_state=self.config.seed,
            stratify=self.combined_df['label']
        )

        # Second split: val and test from temp
        val_ratio_adjusted = self.config.val_ratio / (
            self.config.val_ratio + self.config.test_ratio
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adjusted),
            random_state=self.config.seed,
            stratify=temp_df['label']
        )

        logger.info(f"Train: {len(train_df)} records")
        logger.info(f"Val: {len(val_df)} records")
        logger.info(f"Test: {len(test_df)} records")

        return train_df, val_df, test_df

    def compute_class_weights(self, train_df):
        """Compute class weights untuk imbalanced data"""
        if not self.config.use_class_weights:
            return None

        label_counts = train_df['label'].value_counts().sort_index()
        total_samples = len(train_df)
        num_classes = len(label_counts)

        # Inverse frequency weighting
        class_weights = total_samples / (num_classes * label_counts.values)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        logger.info(f"Class weights: {class_weights.tolist()}")

        return class_weights


# =============================================================================
# TRAINER CLASS
# =============================================================================

class JavaneseHateSpeechTrainer:
    """Trainer untuk eksperimen focal loss"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.class_weights = None

        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.model_output_dir).mkdir(parents=True, exist_ok=True)

    def setup_model(self):
        """Setup model dan tokenizer"""
        logger.info(f"Setting up model: {self.config.model_name}")
        logger.info(f"Loss type: {self.config.loss_type}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=4  # 4 classes: 0=Neutral, 1=Light, 2=Moderate, 3=Severe
        )

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        logger.info(f"Model loaded on: {device}")

    def tokenize_data(self, df: pd.DataFrame):
        """Tokenize dataframe"""
        texts = df['text'].tolist()
        labels = df['label'].tolist()

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors=None
        )

        # Add labels
        encodings['labels'] = labels

        return Dataset.from_dict(encodings)

    def setup_trainer(self, train_dataset, val_dataset, class_weights=None):
        """Setup trainer dengan custom loss"""
        self.class_weights = class_weights

        def compute_metrics(p: EvalPrediction):
            """Compute metrics untuk evaluation"""
            preds = np.argmax(p.predictions, axis=1)
            labels = p.label_ids

            return {
                'accuracy': accuracy_score(labels, preds),
                'f1_macro': f1_score(labels, preds, average='macro'),
                'f1_micro': f1_score(labels, preds, average='micro'),
                'precision_macro': precision_score(labels, preds, average='macro'),
                'recall_macro': recall_score(labels, preds, average='macro'),
            }

        training_args = TrainingArguments(
            output_dir=self.config.model_output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            seed=self.config.seed,
            logging_dir=os.path.join(self.config.model_output_dir, "logs"),
            logging_steps=50,
            report_to=["none"],
            save_total_limit=2,
        )

        self.trainer = CustomLossTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=compute_metrics,
            loss_type=self.config.loss_type,
            alpha=self.config.alpha,
            gamma=self.config.gamma,
            epsilon=self.config.epsilon,
            class_weights=class_weights,
        )

        logger.info("Trainer setup complete")

    def train(self):
        """Train model"""
        logger.info("Starting training...")
        logger.info(f"Loss function: {self.config.loss_type}")
        logger.info(f"Alpha: {self.config.alpha}, Gamma: {self.config.gamma}, Epsilon: {self.config.epsilon}")

        train_result = self.trainer.train()

        logger.info("Training complete")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")

        return train_result

    def evaluate(self, test_dataset):
        """Evaluate model pada test set"""
        logger.info("Evaluating on test set...")

        predictions = self.trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_micro': f1_score(labels, preds, average='micro'),
            'f1_weighted': f1_score(labels, preds, average='weighted'),
            'precision_macro': precision_score(labels, preds, average='macro'),
            'recall_macro': recall_score(labels, preds, average='macro'),
        }

        # Per-class metrics
        report = classification_report(
            labels, preds,
            target_names=['Neutral', 'Light', 'Moderate', 'Severe'],
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(labels, preds)

        logger.info("\n=== Test Results ===")
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")

        logger.info("\n=== Per-Class F1 ===")
        for class_name in ['Neutral', 'Light', 'Moderate', 'Severe']:
            logger.info(f"{class_name}: {report[class_name]['f1-score']:.4f}")

        return metrics, report, cm

    def save_results(self, metrics, report, cm, train_df):
        """Save hasil eksperimen"""
        results = {
            'experiment_name': f'Experiment 6A: {self.config.loss_type.upper()} Loss',
            'model': self.config.model_name,
            'dataset_size': len(self.combined_df) if hasattr(self, 'combined_df') else 'N/A',
            'train_size': len(train_df),
            'config': {
                'max_length': self.config.max_length,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'num_epochs': self.config.num_epochs,
                'seed': self.config.seed,
                'loss_type': self.config.loss_type,
                'alpha': self.config.alpha,
                'gamma': self.config.gamma,
                'epsilon': self.config.epsilon,
            },
            'test_metrics': metrics,
            'per_class_f1': {
                'neutral': report['Neutral']['f1-score'],
                'light': report['Light']['f1-score'],
                'moderate': report['Moderate']['f1-score'],
                'severe': report['Severe']['f1-score'],
            },
            'confusion_matrix': cm.tolist(),
        }

        # Save results
        output_path = os.path.join(self.config.output_dir, 'results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

        # Print summary
        print("\n" + "="*60)
        print(f"EXPERIMENT 6A: {self.config.loss_type.upper()} LOSS - RESULTS")
        print("="*60)
        print(f"Model: {self.config.model_name}")
        print(f"Dataset Size: {results.get('dataset_size', 'N/A')}")
        print(f"\nTest Metrics:")
        print(f"  Accuracy:    {metrics['accuracy']:.2%}")
        print(f"  F1-Macro:    {metrics['f1_macro']:.2%}")
        print(f"  F1-Weighted: {metrics['f1_weighted']:.2%}")
        print(f"\nPer-Class F1:")
        print(f"  Neutral:  {report['Neutral']['f1-score']:.2%}")
        print(f"  Light:    {report['Light']['f1-score']:.2%}")
        print(f"  Moderate: {report['Moderate']['f1-score']:.2%}")
        print(f"  Severe:   {report['Severe']['f1-score']:.2%}")
        print("="*60)

        return results


def run_experiment(loss_type: str = "combined", config: ExperimentConfig = None):
    """
    Run Experiment 6A dengan specific loss type.

    Args:
        loss_type: "focal", "label_smoothing", atau "combined"
    """
    if config is None:
        config = ExperimentConfig()

    # Override loss type
    config.loss_type = loss_type

    logger.info("="*60)
    logger.info(f"EXPERIMENT 6A: {loss_type.upper()} LOSS")
    logger.info("="*60)

    # Step 1: Prepare dataset
    preparer = ImprovedDatasetPreparer(config)
    combined_df = preparer.load_data()
    train_df, val_df, test_df = preparer.split_data()
    class_weights = preparer.compute_class_weights(train_df)

    # Step 2: Setup trainer
    trainer_instance = JavaneseHateSpeechTrainer(config)
    trainer_instance.combined_df = combined_df
    trainer_instance.setup_model()

    # Step 3: Tokenize data
    logger.info("Tokenizing data...")
    train_dataset = trainer_instance.tokenize_data(train_df)
    val_dataset = trainer_instance.tokenize_data(val_df)
    test_dataset = trainer_instance.tokenize_data(test_df)

    # Step 4: Setup trainer
    trainer_instance.setup_trainer(train_dataset, val_dataset, class_weights)

    # Step 5: Train
    train_result = trainer_instance.train()

    # Step 6: Evaluate
    metrics, report, cm = trainer_instance.evaluate(test_dataset)

    # Step 7: Save results
    results = trainer_instance.save_results(metrics, report, cm, train_df)

    logger.info(f"Experiment 6A ({loss_type}) complete!")
    return results, trainer_instance


if __name__ == "__main__":
    # Run experiments dengan berbagai loss types
    loss_types = ["focal", "label_smoothing", "combined"]

    all_results = {}

    for loss_type in loss_types:
        print(f"\n\n{'='*60}")
        print(f"Running {loss_type.upper()} experiment...")
        print(f"{'='*60}\n")

        results, _ = run_experiment(loss_type)
        all_results[loss_type] = results

    # Summary comparison
    print("\n\n" + "="*60)
    print("EXPERIMENT 6A: COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Loss Type':<20} {'F1-Macro':<12} {'Accuracy':<12}")
    print("-"*60)

    baseline_f1 = 0.7919  # Experiment 5 baseline

    for loss_type, results in all_results.items():
        f1 = results['test_metrics']['f1_macro']
        acc = results['test_metrics']['accuracy']
        improvement = (f1 - baseline_f1) * 100
        print(f"{loss_type:<20} {f1:>10.2%} {acc:>10.2%} ({improvement:+.2f}%)")

    print("-"*60)
    print(f"{'Baseline (CE)':<20} {baseline_f1:>10.2%}")
    print("="*60)
