"""
EXPERIMENT 13: Phase 5 Training - Silent Version (Minimal Output)
==================================================================
Using DeepSeek re-labeled dataset with Custom Javanese BERT v3
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
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

# Minimal logging - only to file, silent console
logging.basicConfig(
    level=logging.ERROR,  # Only log errors
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/experiment_13_phase5_training.log'),
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    model_path: str = "models/custom_javanese_bert_v3/final_model"
    data_path: str = "data/improved/phase5_deepseek_relabeled.csv"
    output_dir: str = "results/experiment_13_phase5_training"
    model_output_dir: str = "models/experiment_13_phase5_custom_bert"
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.1
    early_stopping_patience: int = 3
    seed: int = 42


class SilentTrainer(Trainer):
    def __init__(self, *args, label_smoothing=0.1, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Fix: check if class_weights is not None using is not operator
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
        else:
            weights = None
        log_probs = F.log_softmax(logits, dim=-1)
        num_classes = logits.shape[-1]
        targets_one_hot = F.one_hot(labels, num_classes=num_classes).float()
        smooth_targets = (1 - self.label_smoothing) * targets_one_hot + \
                        self.label_smoothing / num_classes

        if weights is not None:
            smooth_targets = smooth_targets * weights.unsqueeze(0)

        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        'f1_micro': f1_score(labels, preds, average='micro', zero_division=0),
        'precision_macro': f1_score(labels, preds, average='macro', zero_division=0),  # Using f1 as proxy
        'recall_macro': f1_score(labels, preds, average='macro', zero_division=0),
    }


def prepare_dataset(df, tokenizer, max_length):
    encodings = tokenizer(
        df['text'].tolist(),
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    return Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': df['label'].tolist()
    })


def compute_class_weights(labels, num_classes=4):
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * class_counts.astype(float))
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.model_output_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load data
    df = pd.read_csv(cfg.data_path)
    print(f"[EXP13] Data: {len(df)} samples")

    # Show label distribution
    label_counts = df['label'].value_counts().sort_index()
    print(f"[EXP13] Labels: {dict(label_counts)}")

    # Split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=cfg.seed, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=cfg.seed, stratify=temp_df['label'])
    print(f"[EXP13] Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Compute class weights
    class_weights = compute_class_weights(df['label'].values)

    # Load model
    print(f"[EXP13] Loading model: {cfg.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path, num_labels=4)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"[EXP13] Model params: {model_params:,}")

    # Prepare datasets
    train_dataset = prepare_dataset(train_df, tokenizer, cfg.max_length)
    val_dataset = prepare_dataset(val_df, tokenizer, cfg.max_length)
    test_dataset = prepare_dataset(test_df, tokenizer, cfg.max_length)

    # Training args - minimal logging
    training_args = TrainingArguments(
        output_dir=cfg.model_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=f"{cfg.output_dir}/logs",
        logging_steps=100,
        save_total_limit=3,
        seed=cfg.seed,
        fp16=True,
        report_to="none",
    )

    trainer = SilentTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        label_smoothing=cfg.label_smoothing,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)]
    )

    # Train
    print(f"[EXP13] Training started...")
    train_result = trainer.train()
    print(f"[EXP13] Training completed in {train_result.metrics['train_runtime']:.0f}s")

    # Evaluate
    print(f"[EXP13] Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    # Print results
    f1_macro = test_results['eval_f1_macro']
    accuracy = test_results['eval_accuracy']

    print("\n" + "="*50)
    print("EXPERIMENT 13 RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-Macro:  {f1_macro:.4f}")
    print(f"Baseline:  0.8138 (Exp 6A)")
    print(f"Delta:     {(f1_macro - 0.8138):+.4f}")
    print("="*50)

    # Per-class F1
    class_names = ['Neutral', 'Light', 'Moderate', 'Severe']
    print("\nPer-Class F1:")
    for i, name in enumerate(class_names):
        f1 = f1_score(labels, preds, labels=[i], average='macro', zero_division=0)
        print(f"  {name}: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print("     Predicted ->")
    print("       N    L    M    H")
    for i, row in enumerate(cm):
        print(f"Actual {i}  [{row[0]:3d}  {row[1]:3d}  {row[2]:3d}  {row[3]:3d}]")

    # Save results
    results = {
        'experiment': 'Experiment 13 - Phase 5 DeepSeek Re-labeled',
        'model': 'custom_javanese_bert_v3',
        'timestamp': datetime.now().isoformat(),
        'samples': len(df),
        'train_samples': len(train_df),
        'test_metrics': {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
        },
        'per_class_f1': {
            name: float(f1_score(labels, preds, labels=[i], average='macro', zero_division=0))
            for i, name in enumerate(class_names)
        },
        'confusion_matrix': cm.tolist(),
        'config': {
            'learning_rate': cfg.learning_rate,
            'batch_size': cfg.batch_size,
            'epochs': cfg.num_epochs,
            'max_length': cfg.max_length,
            'label_smoothing': cfg.label_smoothing,
        }
    }

    with open(f'{cfg.output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {cfg.output_dir}/results.json")

    return results


if __name__ == "__main__":
    main()
