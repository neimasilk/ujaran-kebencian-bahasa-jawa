#!/usr/bin/env python3
"""
Fase 1: Honest Test Set Evaluation

Evaluates ALL existing model checkpoints on the cleaned test set
with real metrics. No synthetic data, no fabrication.

Input:  data/cleaned/final_dataset.csv
Output: results/honest_evaluation_results.json
"""

import os
import sys
import json
import time

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "final_dataset.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "honest_evaluation_results.json")

LABEL_NAMES = [
    "Bukan Ujaran Kebencian",
    "Ujaran Kebencian - Ringan",
    "Ujaran Kebencian - Sedang",
    "Ujaran Kebencian - Berat",
]

# All checkpoints to evaluate (relative to models/)
CHECKPOINTS = {
    "XLM-RoBERTa Large (exp17, cp5010)": "experiment_17_xlmr_large/checkpoint-5010",
    "XLM-RoBERTa Large (exp17, cp4008)": "experiment_17_xlmr_large/checkpoint-4008",
    "XLM-RoBERTa Large (exp17, cp3006)": "experiment_17_xlmr_large/checkpoint-3006",
    "BERT Focal Loss (exp6a, cp2505)": "experiment_6a_focal_loss/checkpoint-2505",
    "BERT Focal Loss (exp6a, cp1503)": "experiment_6a_focal_loss/checkpoint-1503",
    "IndoBERT Phase5 (exp14, cp502)": "experiment_14_indobert_phase5/checkpoint-502",
    "IndoBERT Phase5 (exp14, cp1004)": "experiment_14_indobert_phase5/checkpoint-1004",
    "IndoBERT Phase5 (exp14, cp1255)": "experiment_14_indobert_phase5/checkpoint-1255",
    "Custom BERT Phase5 (exp13, cp3507)": "experiment_13_phase5_custom_bert/checkpoint-3507",
    "Custom BERT Phase5 (exp13, cp3006)": "experiment_13_phase5_custom_bert/checkpoint-3006",
    "Custom Javanese BERT v2": "custom_javanese_bert_v2",
    "Improved Model (Legacy)": "improved_model",
    "IndoBERT Baseline (exp16, cp2505)": "experiment_16_baseline_verify/checkpoint-2505",
    "IndoBERT Large Dataset (exp15, cp4980)": "experiment_15_large_dataset/checkpoint-4980",
}

SEED = 42
MAX_LEN = 128
BATCH_SIZE = 32


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
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_and_split_data(seed=SEED):
    """Load cleaned dataset and create 80/10/10 stratified split."""
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # First split: 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    # Second split: 50/50 of temp = 10% val, 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    print(f"Dataset split (seed={seed}):")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_checkpoint(model_path, X_test, y_test, device):
    """Evaluate a single checkpoint on the test set."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        return {"error": str(e)}

    dataset = TextDataset(X_test, y_test, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Compute metrics
    f1_macro = f1_score(y_true, y_pred, average="macro") * 100
    f1_weighted = f1_score(y_true, y_pred, average="weighted") * 100
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100

    cm = confusion_matrix(y_true, y_pred).tolist()

    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)
    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        per_class[name] = {
            "precision": round(report[name]["precision"] * 100, 2),
            "recall": round(report[name]["recall"] * 100, 2),
            "f1": round(report[name]["f1-score"] * 100, 2),
            "support": int(report[name]["support"]),
        }

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "f1_macro": round(f1_macro, 2),
        "f1_weighted": round(f1_weighted, 2),
        "accuracy": round(accuracy, 2),
        "precision_macro": round(precision_macro, 2),
        "recall_macro": round(recall_macro, 2),
        "confusion_matrix": cm,
        "per_class": per_class,
        "test_samples": len(y_true),
    }


def main():
    print("=" * 60)
    print("FASE 1: HONEST TEST SET EVALUATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

    results = {
        "metadata": {
            "dataset": DATA_PATH,
            "total_samples": len(X_train) + len(X_val) + len(X_test),
            "test_samples": len(X_test),
            "seed": SEED,
            "split": "80/10/10 stratified",
            "max_len": MAX_LEN,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
        },
        "models": {},
    }

    models_dir = os.path.join(PROJECT_ROOT, "models")

    for name, checkpoint in CHECKPOINTS.items():
        model_path = os.path.join(models_dir, checkpoint)

        if not os.path.exists(model_path):
            print(f"\n[SKIP] {name} — path not found: {model_path}")
            results["models"][name] = {"error": "checkpoint not found"}
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"Path: {model_path}")
        print(f"{'='*60}")

        start = time.time()
        result = evaluate_checkpoint(model_path, X_test, y_test, device)
        elapsed = time.time() - start

        if "error" in result:
            print(f"  [ERROR] {result['error']}")
        else:
            print(f"  F1-Macro:   {result['f1_macro']:.2f}%")
            print(f"  F1-Weight:  {result['f1_weighted']:.2f}%")
            print(f"  Accuracy:   {result['accuracy']:.2f}%")
            print(f"  Time:       {elapsed:.1f}s")

        result["elapsed_seconds"] = round(elapsed, 1)
        result["checkpoint_path"] = checkpoint
        results["models"][name] = result

    # Find best model
    valid_results = {
        k: v for k, v in results["models"].items() if "error" not in v
    }
    if valid_results:
        best_name = max(valid_results, key=lambda k: valid_results[k]["f1_macro"])
        best = valid_results[best_name]
        results["best_model"] = {
            "name": best_name,
            "f1_macro": best["f1_macro"],
            "accuracy": best["accuracy"],
        }

        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_name}")
        print(f"  F1-Macro:  {best['f1_macro']:.2f}%")
        print(f"  Accuracy:  {best['accuracy']:.2f}%")
        print(f"{'='*60}")

    # Ranking table
    print(f"\n{'='*60}")
    print("RANKING (by F1-Macro)")
    print(f"{'='*60}")
    ranked = sorted(valid_results.items(), key=lambda x: x[1]["f1_macro"], reverse=True)
    for i, (name, r) in enumerate(ranked, 1):
        print(f"  {i:2d}. {name:<45s} F1={r['f1_macro']:.2f}%  Acc={r['accuracy']:.2f}%")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
