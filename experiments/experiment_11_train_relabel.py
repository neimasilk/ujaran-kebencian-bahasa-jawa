"""
EXPERIMENT 11B: Train dengan LLM-Re-labeled Dataset
====================================================

Training script untuk digunakan SETELAH LLM re-labeling selesai.

Ini harus jalan di komputer GPU.
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
from sklearn.metrics import f1_score, accuracy_score, classification_report

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Configuration untuk training dengan LLM-re-labeled data"""

    # Model
    model_name: str = "indobenchmark/indobert-base-p1"

    # Data paths
    phase5_path: str = "data/improved/phase5_llm_relabeled.csv"  # LLM re-labeled

    # Output paths
    output_dir: str = "results/experiment_11_train_relabel"
    model_output_dir: str = "models/experiment_11_train_relabel"

    # Training hyperparameters
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0

    # Label smoothing (best from experiment 6A)
    label_smoothing: float = 0.1

    # Seed
    seed: int = 42

    # Split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


class LabelSmoothingTrainer(Trainer):
    """Trainer dengan label smoothing"""

    def __init__(self, *args, label_smoothing: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss dengan label smoothing"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Label smoothing loss
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        num_classes = logits.shape[-1]

        # Create one-hot targets
        targets_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

        # Apply label smoothing
        smooth_targets = (1 - self.label_smoothing) * targets_one_hot + \
                        self.label_smoothing / num_classes

        # Calculate loss
        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred: EvalPrediction) -> Dict:
    """Compute metrics untuk evaluation"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
        'f1_micro': f1_score(labels, preds, average='micro'),
    }


def prepare_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    """Prepare dataset untuk training"""
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # Tokenize
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })

    return dataset


def main():
    """Main training function"""

    config = TrainConfig()

    logger.info("=" * 60)
    logger.info("EXPERIMENT 11B: Train dengan LLM-Re-labeled Data")
    logger.info("=" * 60)

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load data
    logger.info(f"Loading data from: {config.phase5_path}")
    df = pd.read_csv(config.phase5_path)
    logger.info(f"Total samples: {len(df)}")

    # Split data
    train_df, temp_df = train_test_split(
        df,
        test_size=(config.val_ratio + config.test_ratio),
        random_state=config.seed,
        stratify=df['label']
    )

    val_ratio_adjusted = config.val_ratio / (config.val_ratio + config.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        random_state=config.seed,
        stratify=temp_df['label']
    )

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Load tokenizer dan model
    logger.info(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=4
    )

    # Prepare datasets
    train_dataset = prepare_dataset(train_df, tokenizer, config.max_length)
    val_dataset = prepare_dataset(val_df, tokenizer, config.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.model_output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=50,
        save_total_limit=2,
        seed=config.seed,
    )

    # Trainer
    trainer = LabelSmoothingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        label_smoothing=config.label_smoothing
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_dataset = prepare_dataset(test_df, tokenizer, config.max_length)

    # Get predictions
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    # Calculate metrics
    test_f1_macro = f1_score(labels, preds, average='macro')
    test_accuracy = accuracy_score(labels, preds)

    logger.info(f"Test F1-Macro: {test_f1_macro:.4f} ({test_f1_macro * 100:.2f}%)")
    logger.info(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Per-class results
    class_names = ['Neutral', 'Light', 'Moderate', 'Severe']
    logger.info("\nPer-Class F1 Scores:")
    for i, name in enumerate(class_names):
        class_f1 = f1_score(labels, preds, labels=[i], average='macro', zero_division=0)
        logger.info(f"  {name}: {class_f1:.4f}")

    # Save results
    results = {
        'experiment_name': 'Experiment 11B: LLM Re-labeled Training',
        'model': config.model_name,
        'dataset': config.phase5_path,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'test_metrics': {
            'accuracy': float(test_accuracy),
            'f1_macro': float(test_f1_macro),
        },
        'per_class_f1': {
            name: float(f1_score(labels, preds, labels=[i], average='macro', zero_division=0))
            for i, name in enumerate(class_names)
        },
        'config': {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'label_smoothing': config.label_smoothing,
        }
    }

    os.makedirs(config.output_dir, exist_ok=True)
    with open(f"{config.output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {config.output_dir}/results.json")

    # Compare with baseline
    baseline_f1 = 0.8124  # From Experiment 6A
    improvement = (test_f1_macro - baseline_f1) * 100

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Baseline F1-Macro (Exp 6A):  {baseline_f1 * 100:.2f}%")
    logger.info(f"LLM Re-labeled F1-Macro:     {test_f1_macro * 100:.2f}%")
    logger.info(f"Improvement:                 {improvement:+.2f}%")

    if test_f1_macro >= 0.82:
        logger.info("\n*** TARGET 82% ACHIEVED! ***")

    return results


if __name__ == "__main__":
    main()
