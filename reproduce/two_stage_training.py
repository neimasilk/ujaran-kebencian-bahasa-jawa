#!/usr/bin/env python3
"""
NEW EXPERIMENT: Two-Stage Training (Curriculum Learning)

Addresses the augmentation bias by training in two stages:
  Stage 1: Fine-tune on synthetic data (learn patterns)
  Stage 2: Further fine-tune on manual data (adapt to real distribution)

This tests whether curriculum learning can mitigate the 53.89% vs 99.41% gap.
If successful, it shows a practical solution to augmentation bias.

Input:  data/cleaned/final_dataset_with_source.csv
Output: results/two_stage_results.json
"""

import os
import sys
import json
import time

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "final_dataset_with_source.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "two_stage_results.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "two_stage")

LABEL_NAMES = [
    "Bukan Ujaran Kebencian",
    "Ujaran Kebencian - Ringan",
    "Ujaran Kebencian - Sedang",
    "Ujaran Kebencian - Berat",
]

NUM_LABELS = 4
SEED = 42
MAX_LEN = 128


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def get_detailed_metrics(y_true, y_pred):
    f1_macro = f1_score(y_true, y_pred, average="macro") * 100
    f1_weighted = f1_score(y_true, y_pred, average="weighted") * 100
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True, zero_division=0)

    per_class = {}
    for name in LABEL_NAMES:
        per_class[name] = {
            "precision": round(report[name]["precision"] * 100, 2),
            "recall": round(report[name]["recall"] * 100, 2),
            "f1": round(report[name]["f1-score"] * 100, 2),
            "support": int(report[name]["support"]),
        }

    return {
        "f1_macro": round(f1_macro, 2),
        "f1_weighted": round(f1_weighted, 2),
        "accuracy": round(accuracy, 2),
        "precision_macro": round(precision, 2),
        "recall_macro": round(recall, 2),
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def evaluate_model(trainer, dataset, y_true):
    output = trainer.predict(dataset)
    y_pred = np.argmax(output.predictions, axis=-1)
    return get_detailed_metrics(np.array(y_true), y_pred)


def main():
    print("=" * 60)
    print("TWO-STAGE TRAINING (CURRICULUM LEARNING)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data with source tracking
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    sources = df["source"].tolist()

    # Same split as main experiments
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        texts, labels, sources, test_size=0.2, random_state=SEED, stratify=labels
    )
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    # Separate training data by source
    train_manual_idx = [i for i, s in enumerate(s_train) if s == "manual"]
    train_synth_idx = [i for i, s in enumerate(s_train) if s == "synthetic"]
    X_train_manual = [X_train[i] for i in train_manual_idx]
    y_train_manual = [y_train[i] for i in train_manual_idx]
    X_train_synth = [X_train[i] for i in train_synth_idx]
    y_train_synth = [y_train[i] for i in train_synth_idx]

    # Manual-only test set
    test_manual_idx = [i for i, s in enumerate(s_test) if s == "manual"]
    X_test_manual = [X_test[i] for i in test_manual_idx]
    y_test_manual = [y_test[i] for i in test_manual_idx]

    print(f"\nData splits:")
    print(f"  Synthetic train: {len(X_train_synth)}")
    print(f"  Manual train: {len(X_train_manual)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Manual test: {len(X_test_manual)}")
    print(f"  Full test: {len(X_test)}")

    results = {}

    # ==========================================
    # Test with IndoBERT (more stable on small data)
    # ==========================================
    for model_key, pretrained, batch_size in [
        ("indobert", "indobenchmark/indobert-base-p1", 16),
        ("xlmr_large", "xlm-roberta-large", 8),
    ]:
        print(f"\n{'='*60}")
        print(f"TWO-STAGE: {model_key}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(pretrained)

        # --- Stage 1: Train on synthetic data ---
        print(f"\n--- Stage 1: Training on {len(X_train_synth)} synthetic samples ---")

        stage1_dir = os.path.join(MODELS_DIR, f"{model_key}_stage1_synthetic")
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained, num_labels=NUM_LABELS
        )

        train_synth_dataset = TextDataset(X_train_synth, y_train_synth, tokenizer, MAX_LEN)
        val_dataset = TextDataset(X_val, y_val, tokenizer, MAX_LEN)

        stage1_args = TrainingArguments(
            output_dir=stage1_dir,
            num_train_epochs=3,  # Fewer epochs — just learn patterns
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=1,
            seed=SEED,
            logging_steps=50,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer1 = Trainer(
            model=model,
            args=stage1_args,
            train_dataset=train_synth_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        start = time.time()
        trainer1.train()
        stage1_time = time.time() - start

        # Save stage 1 checkpoint
        stage1_best = os.path.join(stage1_dir, "best")
        trainer1.save_model(stage1_best)
        tokenizer.save_pretrained(stage1_best)

        # Evaluate stage 1 on manual test
        test_manual_dataset = TextDataset(X_test_manual, y_test_manual, tokenizer, MAX_LEN)
        test_full_dataset = TextDataset(X_test, y_test, tokenizer, MAX_LEN)

        stage1_manual = evaluate_model(trainer1, test_manual_dataset, y_test_manual)
        stage1_full = evaluate_model(trainer1, test_full_dataset, y_test)
        print(f"  Stage 1 -> Manual F1: {stage1_manual['f1_macro']:.2f}%")
        print(f"  Stage 1 -> Full F1:   {stage1_full['f1_macro']:.2f}%")

        del trainer1
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # --- Stage 2: Fine-tune on manual data ---
        print(f"\n--- Stage 2: Fine-tuning on {len(X_train_manual)} manual samples ---")

        stage2_dir = os.path.join(MODELS_DIR, f"{model_key}_stage2_manual")
        model2 = AutoModelForSequenceClassification.from_pretrained(
            stage1_best, num_labels=NUM_LABELS
        )

        train_manual_dataset = TextDataset(X_train_manual, y_train_manual, tokenizer, MAX_LEN)

        # Use lower LR for fine-tuning stage
        stage2_args = TrainingArguments(
            output_dir=stage2_dir,
            num_train_epochs=5,  # More epochs on smaller data
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=32,
            learning_rate=5e-6,  # Lower LR for fine-tuning
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=1,
            seed=SEED,
            logging_steps=50,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        # Use manual-only val for stage 2
        val_manual_idx_v = [i for i, s in enumerate(s_val) if s == "manual"]
        X_val_m = [X_val[i] for i in val_manual_idx_v]
        y_val_m = [y_val[i] for i in val_manual_idx_v]
        val_manual_dataset = TextDataset(X_val_m, y_val_m, tokenizer, MAX_LEN)

        trainer2 = Trainer(
            model=model2,
            args=stage2_args,
            train_dataset=train_manual_dataset,
            eval_dataset=val_manual_dataset,
            compute_metrics=compute_metrics,
        )

        start2 = time.time()
        trainer2.train()
        stage2_time = time.time() - start2

        stage2_best = os.path.join(stage2_dir, "best")
        trainer2.save_model(stage2_best)
        tokenizer.save_pretrained(stage2_best)

        # Final evaluation
        test_manual_dataset = TextDataset(X_test_manual, y_test_manual, tokenizer, MAX_LEN)
        test_full_dataset = TextDataset(X_test, y_test, tokenizer, MAX_LEN)

        stage2_manual = evaluate_model(trainer2, test_manual_dataset, y_test_manual)
        stage2_full = evaluate_model(trainer2, test_full_dataset, y_test)
        print(f"  Stage 2 -> Manual F1: {stage2_manual['f1_macro']:.2f}%")
        print(f"  Stage 2 -> Full F1:   {stage2_full['f1_macro']:.2f}%")

        del model2, trainer2
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        results[model_key] = {
            "stage1_synthetic_only": {
                "manual_test": stage1_manual,
                "full_test": stage1_full,
                "train_samples": len(X_train_synth),
                "epochs": 3,
                "learning_rate": 2e-5,
                "train_time_seconds": round(stage1_time, 0),
            },
            "stage2_manual_finetune": {
                "manual_test": stage2_manual,
                "full_test": stage2_full,
                "train_samples": len(X_train_manual),
                "epochs": 5,
                "learning_rate": 5e-6,
                "train_time_seconds": round(stage2_time, 0),
            },
        }

    # Save
    output = {
        "metadata": {
            "dataset": DATA_PATH,
            "synthetic_train_count": len(X_train_synth),
            "manual_train_count": len(X_train_manual),
            "manual_test_count": len(X_test_manual),
            "full_test_count": len(X_test),
            "seed": SEED,
            "description": "Two-stage curriculum learning: synthetic → manual fine-tuning",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print("TWO-STAGE TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Model':<15} {'Stage':<20} {'Manual F1':<12} {'Full F1':<12}")
    print("-" * 59)
    for model_key, r in results.items():
        s1 = r["stage1_synthetic_only"]
        s2 = r["stage2_manual_finetune"]
        print(f"{model_key:<15} {'Stage1 (synth)':<20} {s1['manual_test']['f1_macro']:<12.2f} {s1['full_test']['f1_macro']:<12.2f}")
        print(f"{'':15} {'Stage2 (manual)':<20} {s2['manual_test']['f1_macro']:<12.2f} {s2['full_test']['f1_macro']:<12.2f}")

    # Compare with baseline (single-stage)
    print(f"\nComparison with single-stage training:")
    print(f"  XLM-R single-stage manual F1: 53.89% (from augmentation_impact.json)")
    print(f"  XLM-R manual-only F1:         14.64% (collapsed)")
    if "xlmr_large" in results:
        s2_manual = results["xlmr_large"]["stage2_manual_finetune"]["manual_test"]["f1_macro"]
        print(f"  XLM-R two-stage manual F1:    {s2_manual:.2f}%")
        delta = s2_manual - 53.89
        print(f"  Delta vs single-stage:        {delta:+.2f} pp")

    print(f"\nResults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
