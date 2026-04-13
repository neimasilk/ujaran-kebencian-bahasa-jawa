#!/usr/bin/env python3
"""
Step 2: Augmentation Impact Analysis

Evaluates whether LLM-based data augmentation actually helps by:
1. Evaluating the full-dataset XLM-R Large model on manual-only test subset
2. Training XLM-R Large on manual-only data and evaluating on manual-only test
3. Computing augmentation benefit (delta F1)

Input:
    data/cleaned/final_dataset_with_source.csv
    models/comparative/xlmr_large/best/ (existing checkpoint)

Output:
    results/augmentation_impact.json
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
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "comparative", "xlmr_large", "best")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "augmentation_impact.json")

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
    """Get full metrics dict including per-class."""
    f1_macro = f1_score(y_true, y_pred, average="macro") * 100
    f1_weighted = f1_score(y_true, y_pred, average="weighted") * 100
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)

    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
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


def evaluate_checkpoint_on_subset(checkpoint_path, texts, labels, tokenizer):
    """Load a checkpoint and evaluate on given data."""
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    dataset = TextDataset(texts, labels, tokenizer, MAX_LEN)

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )

    output = trainer.predict(dataset)
    y_pred = np.argmax(output.predictions, axis=-1)
    y_true = np.array(labels)

    metrics = get_detailed_metrics(y_true, y_pred)

    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return metrics


def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, output_dir):
    """Train XLM-R Large from scratch and evaluate."""
    pretrained = "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained, num_labels=NUM_LABELS
    )

    train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = TextDataset(X_val, y_val, tokenizer, MAX_LEN)
    test_dataset = TextDataset(X_test, y_test, tokenizer, MAX_LEN)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    # Save best
    best_dir = os.path.join(output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Evaluate on test
    test_output = trainer.predict(test_dataset)
    y_pred = np.argmax(test_output.predictions, axis=-1)
    y_true = np.array(y_test)
    metrics = get_detailed_metrics(y_true, y_pred)
    metrics["train_time_seconds"] = round(train_time, 0)

    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return metrics


def main():
    print("=" * 60)
    print("STEP 2: AUGMENTATION IMPACT ANALYSIS")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load full dataset with source
    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset: {len(df)} samples")
    print(f"  Manual:    {(df['source'] == 'manual').sum()}")
    print(f"  Synthetic: {(df['source'] == 'synthetic').sum()}")

    # === Full dataset split (same as original experiments) ===
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    sources = df["source"].tolist()

    X_train_full, X_temp, y_train_full, y_temp, s_train, s_temp = train_test_split(
        texts, labels, sources, test_size=0.2, random_state=SEED, stratify=labels
    )
    X_val_full, X_test_full, y_val_full, y_test_full, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    print(f"\nFull dataset split:")
    print(f"  Train: {len(X_train_full)} | Val: {len(X_val_full)} | Test: {len(X_test_full)}")

    # Get manual-only test subset
    manual_test_idx = [i for i, s in enumerate(s_test) if s == "manual"]
    X_test_manual = [X_test_full[i] for i in manual_test_idx]
    y_test_manual = [y_test_full[i] for i in manual_test_idx]

    synth_test_idx = [i for i, s in enumerate(s_test) if s == "synthetic"]
    X_test_synth = [X_test_full[i] for i in synth_test_idx]
    y_test_synth = [y_test_full[i] for i in synth_test_idx]

    print(f"\nTest set composition:")
    print(f"  Manual:    {len(X_test_manual)} ({len(X_test_manual)/len(X_test_full)*100:.1f}%)")
    print(f"  Synthetic: {len(X_test_synth)} ({len(X_test_synth)/len(X_test_full)*100:.1f}%)")

    results = {}

    # --- 2a. Evaluate existing checkpoint on manual-only test ---
    print(f"\n{'='*60}")
    print("2a. Evaluate full-dataset model on manual-only test")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    results["full_model_on_full_test"] = evaluate_checkpoint_on_subset(
        CHECKPOINT_PATH, X_test_full, y_test_full, tokenizer
    )
    print(f"  Full test: F1={results['full_model_on_full_test']['f1_macro']:.2f}%")

    results["full_model_on_manual_test"] = evaluate_checkpoint_on_subset(
        CHECKPOINT_PATH, X_test_manual, y_test_manual, tokenizer
    )
    print(f"  Manual-only test: F1={results['full_model_on_manual_test']['f1_macro']:.2f}%")

    results["full_model_on_synthetic_test"] = evaluate_checkpoint_on_subset(
        CHECKPOINT_PATH, X_test_synth, y_test_synth, tokenizer
    )
    print(f"  Synthetic-only test: F1={results['full_model_on_synthetic_test']['f1_macro']:.2f}%")

    # --- 2b. Train XLM-R Large on manual-only data ---
    print(f"\n{'='*60}")
    print("2b. Train XLM-R Large on manual-only data")
    print(f"{'='*60}")

    # Filter to manual-only
    df_manual = df[df["source"] == "manual"]
    texts_m = df_manual["text"].tolist()
    labels_m = df_manual["label"].tolist()

    X_train_m, X_temp_m, y_train_m, y_temp_m = train_test_split(
        texts_m, labels_m, test_size=0.2, random_state=SEED, stratify=labels_m
    )
    X_val_m, X_test_m, y_val_m, y_test_m = train_test_split(
        X_temp_m, y_temp_m, test_size=0.5, random_state=SEED, stratify=y_temp_m
    )

    print(f"Manual-only split:")
    print(f"  Train: {len(X_train_m)} | Val: {len(X_val_m)} | Test: {len(X_test_m)}")

    manual_model_dir = os.path.join(PROJECT_ROOT, "models", "augmentation_impact", "xlmr_manual_only")
    results["manual_model_on_manual_test"] = train_and_evaluate(
        X_train_m, X_val_m, X_test_m,
        y_train_m, y_val_m, y_test_m,
        manual_model_dir,
    )
    print(f"  Manual model on manual test: F1={results['manual_model_on_manual_test']['f1_macro']:.2f}%")

    # Also evaluate manual model on the full test set manual subset
    # to compare on the same test data
    manual_best = os.path.join(manual_model_dir, "best")
    results["manual_model_on_full_test_manual_subset"] = evaluate_checkpoint_on_subset(
        manual_best, X_test_manual, y_test_manual, tokenizer
    )
    print(f"  Manual model on full-test manual subset: "
          f"F1={results['manual_model_on_full_test_manual_subset']['f1_macro']:.2f}%")

    # --- 2c. Augmentation benefit analysis ---
    print(f"\n{'='*60}")
    print("2c. AUGMENTATION BENEFIT ANALYSIS")
    print(f"{'='*60}")

    # Compare on same test data (manual subset of full test)
    f1_full_model = results["full_model_on_manual_test"]["f1_macro"]
    f1_manual_model = results["manual_model_on_full_test_manual_subset"]["f1_macro"]
    delta = f1_full_model - f1_manual_model

    results["augmentation_benefit"] = {
        "full_model_manual_test_f1": f1_full_model,
        "manual_model_manual_test_f1": f1_manual_model,
        "delta_f1": round(delta, 2),
        "augmentation_helps": bool(delta > 0),
        "comparison_basis": "Both models evaluated on manual-only subset of full test set",
    }

    print(f"  Full-dataset model (on manual test):    F1 = {f1_full_model:.2f}%")
    print(f"  Manual-only model (on manual test):     F1 = {f1_manual_model:.2f}%")
    print(f"  Delta F1 (augmentation benefit):        {delta:+.2f}%")
    print(f"  Augmentation helps: {'YES' if delta > 0 else 'NO'}")

    # Save
    output = {
        "metadata": {
            "dataset": DATA_PATH,
            "full_dataset_size": len(df),
            "manual_samples": int((df["source"] == "manual").sum()),
            "synthetic_samples": int((df["source"] == "synthetic").sum()),
            "full_test_size": len(X_test_full),
            "manual_test_size": len(X_test_manual),
            "synthetic_test_size": len(X_test_synth),
            "seed": SEED,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
