#!/usr/bin/env python3
"""
Fase 3: Real Label Smoothing Ablation Study

Trains IndoBERT with different label smoothing epsilon values
to measure the actual effect of label smoothing.

Model:    indobenchmark/indobert-base-p1
Epsilons: [0.0, 0.05, 0.1, 0.15, 0.2]

Input:  data/cleaned/final_dataset.csv
Output: results/ablation_results.json
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
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "final_dataset.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "ablation_results.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "ablation")

LABEL_NAMES = [
    "Bukan Ujaran Kebencian",
    "Ujaran Kebencian - Ringan",
    "Ujaran Kebencian - Sedang",
    "Ujaran Kebencian - Berat",
]

MODEL_NAME = "indobenchmark/indobert-base-p1"
NUM_LABELS = 4
SEED = 42
MAX_LEN = 128
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 5

EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2]


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


def main():
    print("=" * 60)
    print("FASE 3: LABEL SMOOTHING ABLATION STUDY")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    print(f"Dataset: {DATA_PATH}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"\nModel: {MODEL_NAME}")
    print(f"Epsilons: {EPSILONS}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = TextDataset(X_val, y_val, tokenizer, MAX_LEN)
    test_dataset = TextDataset(X_test, y_test, tokenizer, MAX_LEN)

    results = []

    for epsilon in EPSILONS:
        print(f"\n{'='*60}")
        print(f"Training with epsilon = {epsilon}")
        print(f"{'='*60}")

        output_dir = os.path.join(MODELS_DIR, f"eps_{epsilon}")

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=32,
            learning_rate=LR,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=1,
            seed=SEED,
            label_smoothing_factor=epsilon,
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

        start = time.time()
        trainer.train()
        train_time = time.time() - start

        # Evaluate on validation
        val_metrics = trainer.evaluate(val_dataset)

        # Evaluate on test
        test_metrics = trainer.evaluate(test_dataset)

        # Detailed test predictions
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

        result = {
            "epsilon": epsilon,
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
            "per_class": per_class,
            "train_time_seconds": round(train_time, 0),
        }
        results.append(result)

        print(f"\n  Epsilon={epsilon}:")
        print(f"    Val  F1={result['validation']['f1_macro']:.2f}%  Acc={result['validation']['accuracy']:.2f}%")
        print(f"    Test F1={result['test']['f1_macro']:.2f}%  Acc={result['test']['accuracy']:.2f}%")

        # Clean up
        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Find optimal epsilon
    best_idx = max(range(len(results)), key=lambda i: results[i]["test"]["f1_macro"])
    best = results[best_idx]

    # Save results
    output = {
        "metadata": {
            "model": MODEL_NAME,
            "dataset": DATA_PATH,
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "max_len": MAX_LEN,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results,
        "optimal": {
            "epsilon": best["epsilon"],
            "test_f1_macro": best["test"]["f1_macro"],
            "test_accuracy": best["test"]["accuracy"],
        },
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*60}")
    print(f"{'Epsilon':>8s} | {'Val F1':>8s} | {'Test F1':>8s} | {'Test Acc':>8s}")
    print("-" * 42)
    for r in results:
        marker = " <--" if r["epsilon"] == best["epsilon"] else ""
        print(f"  {r['epsilon']:>5.2f}  | {r['validation']['f1_macro']:>7.2f}% | "
              f"{r['test']['f1_macro']:>7.2f}% | {r['test']['accuracy']:>7.2f}%{marker}")

    print(f"\nOptimal epsilon: {best['epsilon']}")
    print(f"Best Test F1-Macro: {best['test']['f1_macro']:.2f}%")
    print(f"\nResults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
