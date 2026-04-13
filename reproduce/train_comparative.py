#!/usr/bin/env python3
"""
Fase 2: Comparative Model Training on Clean Dataset

Trains 3 models from scratch on the cleaned dataset:
1. IndoBERT (indobenchmark/indobert-base-p1)
2. XLM-RoBERTa Large (xlm-roberta-large)
3. IndoBERT + Label Smoothing (epsilon=0.1)

Input:  data/cleaned/final_dataset.csv
Output: results/comparative_results.json
        models/comparative/{model_name}/best/
"""

import os
import sys
import json
import time
import argparse

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
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "final_dataset.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "comparative_results.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "comparative")

LABEL_NAMES = [
    "Bukan Ujaran Kebencian",
    "Ujaran Kebencian - Ringan",
    "Ujaran Kebencian - Sedang",
    "Ujaran Kebencian - Berat",
]

NUM_LABELS = 4
SEED = 42
MAX_LEN = 128

# Model configurations
MODELS_CONFIG = {
    "indobert": {
        "pretrained": "indobenchmark/indobert-base-p1",
        "batch_size": 16,
        "lr": 2e-5,
        "epochs": 5,
        "label_smoothing": 0.0,
        "description": "IndoBERT base (no label smoothing)",
    },
    "xlmr_large": {
        "pretrained": "xlm-roberta-large",
        "batch_size": 8,
        "lr": 2e-5,
        "epochs": 5,
        "label_smoothing": 0.0,
        "description": "XLM-RoBERTa Large",
    },
    "indobert_ls": {
        "pretrained": "indobenchmark/indobert-base-p1",
        "batch_size": 16,
        "lr": 2e-5,
        "epochs": 5,
        "label_smoothing": 0.1,
        "description": "IndoBERT base + Label Smoothing (eps=0.1)",
    },
}


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
    """Compute metrics for HuggingFace Trainer."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    return {
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
    }


def load_and_split_data(seed=SEED):
    """Load cleaned dataset and create 80/10/10 stratified split."""
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_single_model(model_key, config, X_train, X_val, X_test, y_train, y_val, y_test):
    """Train a single model and evaluate on test set."""
    print(f"\n{'='*60}")
    print(f"Training: {config['description']}")
    print(f"Model:    {config['pretrained']}")
    print(f"Batch:    {config['batch_size']}")
    print(f"LR:       {config['lr']}")
    print(f"Epochs:   {config['epochs']}")
    print(f"LS:       {config['label_smoothing']}")
    print(f"{'='*60}")

    output_dir = os.path.join(MODELS_DIR, model_key)
    best_dir = os.path.join(output_dir, "best")

    tokenizer = AutoTokenizer.from_pretrained(config["pretrained"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["pretrained"], num_labels=NUM_LABELS
    )

    train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = TextDataset(X_val, y_val, tokenizer, MAX_LEN)
    test_dataset = TextDataset(X_test, y_test, tokenizer, MAX_LEN)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=32,
        learning_rate=config["lr"],
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        label_smoothing_factor=config["label_smoothing"],
        logging_dir=os.path.join(output_dir, "logs"),
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

    # Train
    start = time.time()
    train_result = trainer.train()
    train_time = time.time() - start
    print(f"\nTraining completed in {train_time:.0f}s")

    # Save best model
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Best model saved: {best_dir}")

    # Evaluate on validation set
    val_metrics = trainer.evaluate(val_dataset)
    print(f"\nValidation Results:")
    print(f"  F1-Macro:  {val_metrics['eval_f1_macro']*100:.2f}%")
    print(f"  Accuracy:  {val_metrics['eval_accuracy']*100:.2f}%")

    # Evaluate on test set
    test_metrics = trainer.evaluate(test_dataset)
    print(f"\nTest Results:")
    print(f"  F1-Macro:  {test_metrics['eval_f1_macro']*100:.2f}%")
    print(f"  Accuracy:  {test_metrics['eval_accuracy']*100:.2f}%")

    # Get detailed predictions for test set
    test_output = trainer.predict(test_dataset)
    y_pred = np.argmax(test_output.predictions, axis=-1)
    y_true = np.array(y_test)

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

    # Clean up GPU
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "model_key": model_key,
        "description": config["description"],
        "pretrained": config["pretrained"],
        "hyperparameters": {
            "batch_size": config["batch_size"],
            "learning_rate": config["lr"],
            "epochs": config["epochs"],
            "label_smoothing": config["label_smoothing"],
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "max_len": MAX_LEN,
            "seed": SEED,
        },
        "validation": {
            "f1_macro": round(val_metrics["eval_f1_macro"] * 100, 2),
            "f1_weighted": round(val_metrics.get("eval_f1_weighted", 0) * 100, 2),
            "accuracy": round(val_metrics["eval_accuracy"] * 100, 2),
        },
        "test": {
            "f1_macro": round(test_metrics["eval_f1_macro"] * 100, 2),
            "f1_weighted": round(test_metrics.get("eval_f1_weighted", 0) * 100, 2),
            "accuracy": round(test_metrics["eval_accuracy"] * 100, 2),
            "precision_macro": round(test_metrics.get("eval_precision_macro", 0) * 100, 2),
            "recall_macro": round(test_metrics.get("eval_recall_macro", 0) * 100, 2),
        },
        "confusion_matrix": cm,
        "per_class": per_class,
        "train_time_seconds": round(train_time, 0),
        "best_model_path": best_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Train comparative models")
    parser.add_argument("--model", type=str, default=None,
                        help="Train only a specific model (indobert, xlmr_large, indobert_ls)")
    args = parser.parse_args()

    print("=" * 60)
    print("FASE 2: COMPARATIVE MODEL TRAINING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    print(f"\nDataset: {DATA_PATH}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Determine which models to train
    if args.model:
        if args.model not in MODELS_CONFIG:
            print(f"Error: Unknown model '{args.model}'")
            print(f"Available: {list(MODELS_CONFIG.keys())}")
            sys.exit(1)
        models_to_train = {args.model: MODELS_CONFIG[args.model]}
    else:
        models_to_train = MODELS_CONFIG

    # Load existing results if any
    all_results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            existing = json.load(f)
            all_results = existing.get("models", {})

    # Train each model
    for model_key, config in models_to_train.items():
        result = train_single_model(
            model_key, config,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
        )
        all_results[model_key] = result

    # Find best model
    best_key = max(all_results, key=lambda k: all_results[k]["test"]["f1_macro"])
    best = all_results[best_key]

    # Save results
    output = {
        "metadata": {
            "dataset": DATA_PATH,
            "total_samples": len(X_train) + len(X_val) + len(X_test),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "seed": SEED,
            "split": "80/10/10 stratified",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
        },
        "models": all_results,
        "best_model": {
            "key": best_key,
            "description": best["description"],
            "test_f1_macro": best["test"]["f1_macro"],
            "test_accuracy": best["test"]["accuracy"],
        },
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print("COMPARATIVE RESULTS SUMMARY")
    print(f"{'='*60}")
    for key, r in sorted(all_results.items(), key=lambda x: x[1]["test"]["f1_macro"], reverse=True):
        marker = " <-- BEST" if key == best_key else ""
        print(f"  {r['description']:<45s}")
        print(f"    Val  F1={r['validation']['f1_macro']:.2f}%  Acc={r['validation']['accuracy']:.2f}%")
        print(f"    Test F1={r['test']['f1_macro']:.2f}%  Acc={r['test']['accuracy']:.2f}%{marker}")

    print(f"\nResults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
