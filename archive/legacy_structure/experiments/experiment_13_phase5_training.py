"""
EXPERIMENT 13: Training dengan Phase 5 Dataset (DeepSeek Re-labeled)
=====================================================================

GPU-REQUIRED EXPERIMENT

Menggunakan dataset Phase 5 (10,019 samples) yang sudah di-relabel oleh DeepSeek.
Ini adalah dataset kualitas terbaik yang tersedia saat ini.

Expected: 82-85% F1-Macro
Target: Lebih baik dari baseline 81.38%
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from transformers.trainer_utils import EvalPrediction
from datasets import Dataset
import torch.nn.functional as F

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/experiment_13_phase5_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration untuk Experiment 13"""

    # Model selection - bisa pilih salah satu
    model_choices = {
        "indobert_base": "indobenchmark/indobert-base-p1",
        "indobert_large": "indobenchmark/indobert-large-p1",
        "custom_bert_v3": "models/custom_javanese_bert_v3/final_model",
        "xlm_roberta": "FacebookAI/xlm-roberta-base",
        "mbert": "google-bert/bert-base-multilingual-cased",
    }

    # Default model (bisa diubah via command line)
    model_name: str = "custom_bert_v3"  # Default: use custom BERT v3
    model_path: Optional[str] = None  # Will be set from model_choices

    # Data paths
    data_path: str = "data/improved/phase5_deepseek_relabeled.csv"

    # Output paths
    output_dir: str = "results/experiment_13_phase5_training"
    model_output_dir: str = "models/experiment_13_phase5_custom_bert"

    # Training hyperparameters
    max_length: int = 256  # Increased from 128 for better context
    batch_size: int = 16  # Adjust based on GPU memory
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    num_epochs: int = 10  # More epochs for better convergence
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Advanced techniques
    label_smoothing: float = 0.1  # From best exp 6A result
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0

    # Early stopping
    early_stopping_patience: int = 3
    load_best_model_at_end: bool = True

    # Class weights (computed from data)
    use_class_weights: bool = True

    # Seed
    seed: int = 42

    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    def __post_init__(self):
        if self.model_path is None:
            self.model_path = self.model_choices.get(self.model_name, self.model_choices["indobert_base"])


class FocalLoss:
    """Focal Loss untuk handling class imbalance"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, num_classes: int = 4):
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def __call__(self, logits, labels, class_weights=None):
        """Compute focal loss"""
        ce_loss = F.cross_entropy(logits, labels, reduction='none', weight=class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class AdvancedTrainer(Trainer):
    """Trainer dengan label smoothing dan focal loss"""

    def __init__(self, *args, label_smoothing: float = 0.1, use_focal_loss: bool = False,
                 class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.class_weights = class_weights

        if use_focal_loss:
            self.focal_loss = FocalLoss(num_classes=4)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss dengan label smoothing dan/atau focal loss"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Move class weights to correct device
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None

        if self.use_focal_loss:
            # Use focal loss
            loss = self.focal_loss(logits, labels, weights)
        else:
            # Label smoothing loss
            log_probs = F.log_softmax(logits, dim=-1)
            num_classes = logits.shape[-1]

            targets_one_hot = F.one_hot(labels, num_classes=num_classes).float()
            smooth_targets = (1 - self.label_smoothing) * targets_one_hot + \
                            self.label_smoothing / num_classes

            # Apply class weights to the loss
            if weights is not None:
                smooth_targets = smooth_targets * weights.unsqueeze(0)

            loss = -(smooth_targets * log_probs).sum(dim=-1).mean()

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred: EvalPrediction) -> Dict:
    """Compute comprehensive metrics"""
    logits, labels = eval_pred
    probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)

    # Basic metrics
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
    recall_macro = recall_score(labels, preds, average='macro', zero_division=0)

    # Per-class F1
    class_names = ['Neutral', 'Light', 'Moderate', 'Severe']
    per_class_f1 = {}
    for i, name in enumerate(class_names):
        per_class_f1[f'f1_{name.lower()}'] = f1_score(
            labels, preds, labels=[i], average='macro', zero_division=0
        )

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        **per_class_f1
    }


def prepare_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    """Prepare dataset untuk training"""
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })

    return dataset


def compute_class_weights(labels: np.ndarray, num_classes: int = 4) -> torch.Tensor:
    """Compute class weights untuk handling imbalance"""
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)

    # Inverse frequency weighting
    class_weights = total_samples / (num_classes * class_counts.astype(float))

    # Normalize
    class_weights = class_weights / class_weights.sum() * num_classes

    return torch.tensor(class_weights, dtype=torch.float32)


def main():
    """Main training function"""

    config = ExperimentConfig()

    logger.info("=" * 70)
    logger.info("EXPERIMENT 13: Training dengan Phase 5 Dataset")
    logger.info("=" * 70)
    logger.info(f"Model: {config.model_name} ({config.model_path})")
    logger.info(f"Dataset: {config.data_path}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.model_output_dir, exist_ok=True)

    # Load data
    logger.info("\n" + "=" * 70)
    logger.info("1. LOADING DATA")
    logger.info("=" * 70)

    df = pd.read_csv(config.data_path)
    logger.info(f"Total samples: {len(df)}")

    # Check columns
    if 'label' not in df.columns:
        logger.error("Column 'label' not found in dataset!")
        logger.info(f"Available columns: {list(df.columns)}")
        return

    # Label distribution
    label_counts = df['label'].value_counts().sort_index()
    logger.info("\nLabel distribution:")
    class_names = ['Neutral', 'Light', 'Moderate', 'Severe']
    for i, count in enumerate(label_counts.items()):
        logger.info(f"  {class_names[i]} ({i}): {count[1]} ({count[1]/len(df)*100:.1f}%)")

    # Compute class weights
    class_weights = None
    if config.use_class_weights:
        class_weights = compute_class_weights(df['label'].values)
        logger.info(f"\nClass weights: {class_weights.tolist()}")

    # Split data
    logger.info("\n" + "=" * 70)
    logger.info("2. SPLITTING DATA")
    logger.info("=" * 70)

    train_df, temp_df = train_test_split(
        df, test_size=(config.val_ratio + config.test_ratio),
        random_state=config.seed, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=config.test_ratio / (config.val_ratio + config.test_ratio),
        random_state=config.seed,
        stratify=temp_df['label']
    )

    logger.info(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # Load tokenizer dan model
    logger.info("\n" + "=" * 70)
    logger.info("3. LOADING MODEL")
    logger.info("=" * 70)

    logger.info(f"Loading tokenizer from: {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    logger.info(f"Loading model from: {config.model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_path,
        num_labels=4
    )

    # Model size
    model_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {model_params:,}")

    # Prepare datasets
    logger.info("\n" + "=" * 70)
    logger.info("4. PREPARING DATASETS")
    logger.info("=" * 70)

    train_dataset = prepare_dataset(train_df, tokenizer, config.max_length)
    val_dataset = prepare_dataset(val_df, tokenizer, config.max_length)
    test_dataset = prepare_dataset(test_df, tokenizer, config.max_length)

    logger.info(f"Max length: {config.max_length}")
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Training arguments
    logger.info("\n" + "=" * 70)
    logger.info("5. TRAINING CONFIGURATION")
    logger.info("=" * 70)

    training_args = TrainingArguments(
        output_dir=config.model_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=25,
        save_total_limit=3,
        seed=config.seed,
        fp16=True,  # Mixed precision training
        gradient_checkpointing=False,
        report_to="none",
    )

    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Label smoothing: {config.label_smoothing}")
    logger.info(f"Class weights: {config.use_class_weights}")
    logger.info(f"Mixed precision (FP16): True")

    # Trainer dengan early stopping
    logger.info("\n" + "=" * 70)
    logger.info("6. INITIALIZING TRAINER")
    logger.info("=" * 70)

    trainer = AdvancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        label_smoothing=config.label_smoothing,
        use_focal_loss=False,  # Use label smoothing instead
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    )

    # Train
    logger.info("\n" + "=" * 70)
    logger.info("7. STARTING TRAINING")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    train_result = trainer.train()

    logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
    logger.info(f"Best checkpoint: {trainer.state.best_model_checkpoint}")

    # Evaluate on test set
    logger.info("\n" + "=" * 70)
    logger.info("8. EVALUATING ON TEST SET")
    logger.info("=" * 70)

    test_results = trainer.evaluate(test_dataset)
    logger.info(f"\nTest Results:")
    for key, value in test_results.items():
        if 'eval_' in key:
            logger.info(f"  {key.replace('eval_', '').replace('_', ' ').title()}: {value:.4f}")

    # Detailed predictions
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    logger.info("\nConfusion Matrix:")
    logger.info("              Predicted")
    logger.info("       N   L   M   H")
    for i, row in enumerate(cm):
        logger.info(f"Actual {i}  {row}")

    # Classification Report
    logger.info("\nPer-Class Performance:")
    logger.info(classification_report(labels, preds, target_names=class_names, zero_division=0))

    # Save results
    logger.info("\n" + "=" * 70)
    logger.info("9. SAVING RESULTS")
    logger.info("=" * 70)

    results = {
        'experiment_name': 'Experiment 13: Phase 5 DeepSeek Re-labeled Training',
        'model_name': config.model_name,
        'model_path': config.model_path,
        'dataset': 'phase5_deepseek_relabeled.csv',
        'timestamp': datetime.now().isoformat(),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'training_metrics': {
            'train_runtime': train_result.metrics['train_runtime'],
            'train_samples_per_second': train_result.metrics['train_samples_per_second'],
            'total_steps': trainer.state.global_step,
        },
        'test_metrics': {
            'accuracy': float(test_results.get('eval_accuracy', 0)),
            'f1_macro': float(test_results.get('eval_f1_macro', 0)),
            'f1_micro': float(test_results.get('eval_f1_micro', 0)),
            'precision_macro': float(test_results.get('eval_precision_macro', 0)),
            'recall_macro': float(test_results.get('eval_recall_macro', 0)),
        },
        'per_class_f1': {
            name: float(f1_score(labels, preds, labels=[i], average='macro', zero_division=0))
            for i, name in enumerate(class_names)
        },
        'confusion_matrix': cm.tolist(),
        'config': {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'max_length': config.max_length,
            'label_smoothing': config.label_smoothing,
            'use_class_weights': config.use_class_weights,
            'class_weights': class_weights.tolist() if class_weights is not None else None,
            'seed': config.seed,
        }
    }

    results_path = f"{config.output_dir}/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    # Compare with baseline
    baseline_f1 = 0.8138
    current_f1 = results['test_metrics']['f1_macro']
    improvement = (current_f1 - baseline_f1) * 100

    logger.info("\n" + "=" * 70)
    logger.info("10. SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Baseline (Exp 6A):        {baseline_f1 * 100:.2f}%")
    logger.info(f"Current (Phase 5):        {current_f1 * 100:.2f}%")
    logger.info(f"Improvement:              {improvement:+.2f}%")

    # Target checks
    if current_f1 >= 0.85:
        logger.info("\n*** TARGET 85% ACHIEVED! ***")
    elif current_f1 >= 0.82:
        logger.info("\n*** TARGET 82% ACHIEVED! ***")
    elif current_f1 >= baseline_f1:
        logger.info("\n*** IMPROVEMENT FROM BASELINE! ***")
    else:
        logger.info("\n*** BELOW BASELINE - NEEDS IMPROVEMENT ***")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 13: Phase 5 Training")
    parser.add_argument('--model', type=str, default='custom_bert_v3',
                       choices=['indobert_base', 'indobert_large', 'custom_bert_v3', 'xlm_roberta', 'mbert'],
                       help='Model to use')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Max sequence length')

    args = parser.parse_args()

    # Override config with command line args
    ExperimentConfig.model_name = args.model
    ExperimentConfig.num_epochs = args.epochs
    ExperimentConfig.batch_size = args.batch_size
    ExperimentConfig.learning_rate = args.lr
    ExperimentConfig.max_length = args.max_length

    main()
