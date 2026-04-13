#!/usr/bin/env python3
"""
Fase 4: Multi-Seed Statistical Significance Evaluation

Trains the best model with 5 different seeds to report mean +/- std.

Seeds: [42, 123, 456, 789, 1024]

Input:  data/cleaned/final_dataset.csv
        results/comparative_results.json (to determine best model)
Output: results/multi_seed_results.json
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
    confusion_matrix,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "final_dataset.csv")
COMPARATIVE_RESULTS = os.path.join(PROJECT_ROOT, "results", "comparative_results.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "multi_seed_results.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "multi_seed")

LABEL_NAMES = [
    "Bukan Ujaran Kebencian",
    "Ujaran Kebencian - Ringan",
    "Ujaran Kebencian - Sedang",
    "Ujaran Kebencian - Berat",
]

NUM_LABELS = 4
MAX_LEN = 128
SEEDS = [42, 123, 456, 789, 1024]


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


def determine_best_model():
    """Determine best model config from comparative results, or use default."""
    default = {
        "pretrained": "indobenchmark/indobert-base-p1",
        "batch_size": 16,
        "lr": 2e-5,
        "epochs": 5,
        "label_smoothing": 0.1,
        "description": "IndoBERT + Label Smoothing (eps=0.1)",
    }

    if not os.path.exists(COMPARATIVE_RESULTS):
        print("Comparative results not found. Using default config (IndoBERT + LS).")
        return default

    with open(COMPARATIVE_RESULTS) as f:
        data = json.load(f)

    best_key = data.get("best_model", {}).get("key", None)
    if best_key and best_key in data.get("models", {}):
        model_data = data["models"][best_key]
        hp = model_data.get("hyperparameters", {})
        return {
            "pretrained": model_data.get("pretrained", default["pretrained"]),
            "batch_size": hp.get("batch_size", default["batch_size"]),
            "lr": hp.get("learning_rate", default["lr"]),
            "epochs": hp.get("epochs", default["epochs"]),
            "label_smoothing": hp.get("label_smoothing", default["label_smoothing"]),
            "description": model_data.get("description", best_key),
        }

    return default


def train_with_seed(config, seed, df):
    """Train model with a specific seed and return test metrics."""
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # Split with this seed
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    tokenizer = AutoTokenizer.from_pretrained(config["pretrained"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["pretrained"], num_labels=NUM_LABELS
    )

    train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = TextDataset(X_val, y_val, tokenizer, MAX_LEN)
    test_dataset = TextDataset(X_test, y_test, tokenizer, MAX_LEN)

    output_dir = os.path.join(MODELS_DIR, f"seed_{seed}")

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
        save_total_limit=1,
        seed=seed,
        label_smoothing_factor=config["label_smoothing"],
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
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

    # Evaluate
    val_metrics = trainer.evaluate(val_dataset)
    test_metrics = trainer.evaluate(test_dataset)

    # Detailed predictions
    test_output = trainer.predict(test_dataset)
    y_pred = np.argmax(test_output.predictions, axis=-1)
    y_true = np.array(y_test)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Clean up
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "seed": seed,
        "validation": {
            "f1_macro": round(val_metrics["eval_f1_macro"] * 100, 2),
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
        "train_time_seconds": round(train_time, 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-seed evaluation")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model (e.g., xlm-roberta-large)")
    parser.add_argument("--label-smoothing", type=float, default=None,
                        help="Override label smoothing epsilon")
    args = parser.parse_args()

    print("=" * 60)
    print("FASE 4: MULTI-SEED STATISTICAL SIGNIFICANCE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Determine model config
    config = determine_best_model()
    if args.model:
        config["pretrained"] = args.model
        config["description"] = args.model
    if args.label_smoothing is not None:
        config["label_smoothing"] = args.label_smoothing

    print(f"\nModel:  {config['pretrained']}")
    print(f"LS:     {config['label_smoothing']}")
    print(f"Seeds:  {SEEDS}")

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset: {len(df)} samples")

    # Train with each seed
    all_results = []
    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{len(SEEDS)}: seed={seed}")
        print(f"{'='*60}")

        result = train_with_seed(config, seed, df)
        all_results.append(result)

        print(f"  Test F1={result['test']['f1_macro']:.2f}%  Acc={result['test']['accuracy']:.2f}%")

    # Compute statistics
    f1_scores = [r["test"]["f1_macro"] for r in all_results]
    acc_scores = [r["test"]["accuracy"] for r in all_results]
    prec_scores = [r["test"]["precision_macro"] for r in all_results]
    rec_scores = [r["test"]["recall_macro"] for r in all_results]

    stats = {
        "f1_macro": {
            "mean": round(np.mean(f1_scores), 2),
            "std": round(np.std(f1_scores), 2),
            "min": round(np.min(f1_scores), 2),
            "max": round(np.max(f1_scores), 2),
            "values": f1_scores,
        },
        "accuracy": {
            "mean": round(np.mean(acc_scores), 2),
            "std": round(np.std(acc_scores), 2),
            "min": round(np.min(acc_scores), 2),
            "max": round(np.max(acc_scores), 2),
            "values": acc_scores,
        },
        "precision_macro": {
            "mean": round(np.mean(prec_scores), 2),
            "std": round(np.std(prec_scores), 2),
            "values": prec_scores,
        },
        "recall_macro": {
            "mean": round(np.mean(rec_scores), 2),
            "std": round(np.std(rec_scores), 2),
            "values": rec_scores,
        },
    }

    # Save results
    output = {
        "metadata": {
            "model": config["pretrained"],
            "description": config["description"],
            "label_smoothing": config["label_smoothing"],
            "seeds": SEEDS,
            "dataset": DATA_PATH,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "runs": all_results,
        "statistics": stats,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print("STATISTICAL SIGNIFICANCE RESULTS")
    print(f"{'='*60}")
    print(f"Model: {config['description']}")
    print(f"Seeds: {SEEDS}")
    print(f"\n  F1-Macro:   {stats['f1_macro']['mean']:.2f}% +/- {stats['f1_macro']['std']:.2f}%")
    print(f"  Accuracy:   {stats['accuracy']['mean']:.2f}% +/- {stats['accuracy']['std']:.2f}%")
    print(f"  Precision:  {stats['precision_macro']['mean']:.2f}% +/- {stats['precision_macro']['std']:.2f}%")
    print(f"  Recall:     {stats['recall_macro']['mean']:.2f}% +/- {stats['recall_macro']['std']:.2f}%")
    print(f"\n  Range F1:   [{stats['f1_macro']['min']:.2f}%, {stats['f1_macro']['max']:.2f}%]")
    print(f"\nPer-seed breakdown:")
    for r in all_results:
        print(f"  Seed {r['seed']:>4d}: F1={r['test']['f1_macro']:.2f}%  Acc={r['test']['accuracy']:.2f}%")

    print(f"\nResults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
