#!/usr/bin/env python3
"""
P0 Fix: Evaluate ALL models on manual-only test subset.

Evaluates 5 models on the 451-sample manual-only test subset:
  - SVM + TF-IDF
  - Logistic Regression + TF-IDF
  - IndoBERT base
  - IndoBERT + Label Smoothing (eps=0.1)
  - XLM-RoBERTa Large

Uses the SAME stratified split (seed=42) on the full dataset,
then filters the test set to source == "manual" only.

Input:
    data/cleaned/final_dataset_with_source.csv
    models/comparative/{indobert,indobert_ls,xlmr_large}/best/

Output:
    results/manual_only_results.json
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
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "final_dataset_with_source.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "manual_only_results.json")
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

# Transformer checkpoints to evaluate
TRANSFORMER_MODELS = {
    "indobert": {
        "path": os.path.join(MODELS_DIR, "indobert", "best"),
        "description": "IndoBERT base (no label smoothing)",
    },
    "indobert_ls": {
        "path": os.path.join(MODELS_DIR, "indobert_ls", "best"),
        "description": "IndoBERT base + Label Smoothing (eps=0.1)",
    },
    "xlmr_large": {
        "path": os.path.join(MODELS_DIR, "xlmr_large", "best"),
        "description": "XLM-RoBERTa Large",
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


def main():
    print("=" * 60)
    print("P0 FIX: EVALUATE ALL MODELS ON MANUAL-ONLY TEST DATA")
    print("=" * 60)

    # ── Load dataset with source tracking ──
    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset: {len(df)} samples")
    print(f"  Manual:    {(df['source'] == 'manual').sum()}")
    print(f"  Synthetic: {(df['source'] == 'synthetic').sum()}")

    # ── Same stratified split as all other experiments ──
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    sources = df["source"].tolist()

    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        texts, labels, sources, test_size=0.2, random_state=SEED, stratify=labels
    )
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    print(f"\nFull split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    # ── Filter test set to manual-only ──
    manual_idx = [i for i, s in enumerate(s_test) if s == "manual"]
    X_test_manual = [X_test[i] for i in manual_idx]
    y_test_manual = [y_test[i] for i in manual_idx]

    print(f"Manual-only test: {len(X_test_manual)} samples")
    print(f"  Class distribution: {dict(pd.Series(y_test_manual).value_counts().sort_index())}")

    results = {}

    # ══════════════════════════════════════════════════════════
    # BASELINE MODELS (SVM, LR) — retrain on full train, eval on manual test
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("BASELINE MODELS (TF-IDF)")
    print(f"{'='*60}")

    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_manual_tfidf = tfidf.transform(X_test_manual)

    # Also compute on full test for comparison
    X_test_full_tfidf = tfidf.transform(X_test)

    # SVM
    print("\nTraining SVM...")
    svm = SVC(kernel="linear", C=1.0, random_state=SEED)
    svm.fit(X_train_tfidf, y_train)

    svm_manual = get_detailed_metrics(y_test_manual, svm.predict(X_test_manual_tfidf))
    svm_full = get_detailed_metrics(y_test, svm.predict(X_test_full_tfidf))
    print(f"  SVM Manual-only F1: {svm_manual['f1_macro']:.2f}%")
    print(f"  SVM Full test F1:   {svm_full['f1_macro']:.2f}%")

    results["svm_tfidf"] = {
        "description": "SVM + TF-IDF (Linear)",
        "manual_only": svm_manual,
        "full_test": svm_full,
    }

    # LR
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        multi_class="multinomial", random_state=SEED,
    )
    lr.fit(X_train_tfidf, y_train)

    lr_manual = get_detailed_metrics(y_test_manual, lr.predict(X_test_manual_tfidf))
    lr_full = get_detailed_metrics(y_test, lr.predict(X_test_full_tfidf))
    print(f"  LR Manual-only F1:  {lr_manual['f1_macro']:.2f}%")
    print(f"  LR Full test F1:    {lr_full['f1_macro']:.2f}%")

    results["lr_tfidf"] = {
        "description": "Logistic Regression + TF-IDF",
        "manual_only": lr_manual,
        "full_test": lr_full,
    }

    # ══════════════════════════════════════════════════════════
    # TRANSFORMER MODELS — load checkpoints, eval on manual test
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("TRANSFORMER MODELS (from checkpoints)")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for key, cfg in TRANSFORMER_MODELS.items():
        checkpoint = cfg["path"]
        print(f"\n--- {cfg['description']} ---")
        print(f"  Loading: {checkpoint}")

        if not os.path.exists(checkpoint):
            print(f"  SKIP: checkpoint not found!")
            results[key] = {
                "description": cfg["description"],
                "error": "checkpoint not found",
            }
            continue

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

        # Manual-only test
        manual_dataset = TextDataset(X_test_manual, y_test_manual, tokenizer, MAX_LEN)
        trainer = Trainer(model=model)
        output = trainer.predict(manual_dataset)
        y_pred_manual = np.argmax(output.predictions, axis=-1)
        manual_metrics = get_detailed_metrics(np.array(y_test_manual), y_pred_manual)

        # Full test (for comparison / consistency check)
        full_dataset = TextDataset(X_test, y_test, tokenizer, MAX_LEN)
        output_full = trainer.predict(full_dataset)
        y_pred_full = np.argmax(output_full.predictions, axis=-1)
        full_metrics = get_detailed_metrics(np.array(y_test), y_pred_full)

        print(f"  Manual-only F1: {manual_metrics['f1_macro']:.2f}%")
        print(f"  Full test F1:   {full_metrics['f1_macro']:.2f}%")

        results[key] = {
            "description": cfg["description"],
            "manual_only": manual_metrics,
            "full_test": full_metrics,
        }

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════
    output = {
        "metadata": {
            "description": "All models evaluated on manual-only test subset",
            "dataset": DATA_PATH,
            "total_dataset": len(df),
            "manual_samples": int((df["source"] == "manual").sum()),
            "synthetic_samples": int((df["source"] == "synthetic").sum()),
            "train_samples": len(X_train),
            "full_test_samples": len(X_test),
            "manual_test_samples": len(X_test_manual),
            "seed": SEED,
            "split": "80/10/10 stratified on full dataset, test filtered to manual",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "models": results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("MANUAL-ONLY TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<30s} {'Manual F1':>10s} {'Full F1':>10s} {'Delta':>8s}")
    print("-" * 60)
    for key, r in results.items():
        if "error" in r:
            print(f"  {r['description']:<28s}  {'ERROR':>10s}")
            continue
        m_f1 = r["manual_only"]["f1_macro"]
        f_f1 = r["full_test"]["f1_macro"]
        delta = m_f1 - f_f1
        print(f"  {r['description']:<28s} {m_f1:>9.2f}% {f_f1:>9.2f}% {delta:>+7.2f}")

    print(f"\nResults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
