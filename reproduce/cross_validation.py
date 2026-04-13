#!/usr/bin/env python3
"""
NEW EXPERIMENT: Stratified K-Fold Cross-Validation

Provides more robust performance estimates than single train/test split.
Reports mean ± std across folds for all models.

Input:  data/cleaned/final_dataset_with_source.csv
Output: results/cross_validation_results.json
"""

import os
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
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "final_dataset_with_source.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "cross_validation_results.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "cv")

LABEL_NAMES = [
    "Bukan Ujaran Kebencian",
    "Ujaran Kebencian - Ringan",
    "Ujaran Kebencian - Sedang",
    "Ujaran Kebencian - Berat",
]
NUM_LABELS = 4
SEED = 42
MAX_LEN = 128
N_FOLDS = 5


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
    preds = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.labels
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
    }


def cv_svm(texts, labels, source_filter=None, sources=None):
    """Run K-fold CV for SVM."""
    if source_filter and sources is not None:
        mask = [s == source_filter for s in sources]
        texts = [t for t, m in zip(texts, mask) if m]
        labels = [l for l, m in zip(labels, mask) if m]

    texts_arr = np.array(texts)
    labels_arr = np.array(labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(texts_arr, labels_arr)):
        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        X_train = tfidf.fit_transform(texts_arr[train_idx])
        X_test = tfidf.transform(texts_arr[test_idx])
        y_train = labels_arr[train_idx]
        y_test = labels_arr[test_idx]

        svm = SVC(kernel="linear", C=1.0, random_state=SEED)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)

        fold_results.append({
            "fold": fold + 1,
            "f1_macro": round(f1_score(y_test, y_pred, average="macro") * 100, 2),
            "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "test_size": len(test_idx),
        })

    f1s = [r["f1_macro"] for r in fold_results]
    accs = [r["accuracy"] for r in fold_results]
    return {
        "folds": fold_results,
        "mean_f1": round(np.mean(f1s), 2),
        "std_f1": round(np.std(f1s), 2),
        "mean_accuracy": round(np.mean(accs), 2),
        "std_accuracy": round(np.std(accs), 2),
        "total_samples": len(texts),
    }


def cv_transformer(texts, labels, pretrained, batch_size, source_filter=None, sources=None):
    """Run K-fold CV for a transformer model."""
    if source_filter and sources is not None:
        mask = [s == source_filter for s in sources]
        texts = [t for t, m in zip(texts, mask) if m]
        labels = [l for l, m in zip(labels, mask) if m]

    texts_arr = np.array(texts)
    labels_arr = np.array(labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(texts_arr, labels_arr)):
        print(f"    Fold {fold+1}/{N_FOLDS}...")
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        model = AutoModelForSequenceClassification.from_pretrained(pretrained, num_labels=NUM_LABELS)

        X_train = [texts_arr[i] for i in train_idx]
        y_train = [int(labels_arr[i]) for i in train_idx]
        X_test = [texts_arr[i] for i in test_idx]
        y_test = [int(labels_arr[i]) for i in test_idx]

        train_ds = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
        test_ds = TextDataset(X_test, y_test, tokenizer, MAX_LEN)

        fold_dir = os.path.join(MODELS_DIR, f"fold_{fold}")
        args = TrainingArguments(
            output_dir=fold_dir,
            num_train_epochs=5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="no",
            save_strategy="no",
            seed=SEED,
            logging_steps=100,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model, args=args,
            train_dataset=train_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        output = trainer.predict(test_ds)
        y_pred = np.argmax(output.predictions, axis=-1)

        fold_results.append({
            "fold": fold + 1,
            "f1_macro": round(f1_score(y_test, y_pred, average="macro") * 100, 2),
            "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "test_size": len(test_idx),
        })
        print(f"      F1={fold_results[-1]['f1_macro']:.2f}%")

        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    f1s = [r["f1_macro"] for r in fold_results]
    accs = [r["accuracy"] for r in fold_results]
    return {
        "folds": fold_results,
        "mean_f1": round(np.mean(f1s), 2),
        "std_f1": round(np.std(f1s), 2),
        "mean_accuracy": round(np.mean(accs), 2),
        "std_accuracy": round(np.std(accs), 2),
        "total_samples": len(texts),
    }


def main():
    print("=" * 60)
    print(f"STRATIFIED {N_FOLDS}-FOLD CROSS-VALIDATION")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    sources = df["source"].tolist()

    print(f"Dataset: {len(texts)} samples")
    print(f"  Manual: {sum(1 for s in sources if s == 'manual')}")
    print(f"  Synthetic: {sum(1 for s in sources if s == 'synthetic')}")

    results = {}

    # --- SVM on full dataset ---
    print(f"\n--- SVM (full dataset) ---")
    results["svm_full"] = cv_svm(texts, labels)
    print(f"  F1: {results['svm_full']['mean_f1']:.2f} ± {results['svm_full']['std_f1']:.2f}")

    # --- SVM on manual-only ---
    print(f"\n--- SVM (manual-only) ---")
    results["svm_manual"] = cv_svm(texts, labels, source_filter="manual", sources=sources)
    print(f"  F1: {results['svm_manual']['mean_f1']:.2f} ± {results['svm_manual']['std_f1']:.2f}")

    # --- IndoBERT on full dataset ---
    print(f"\n--- IndoBERT (full dataset) ---")
    results["indobert_full"] = cv_transformer(
        texts, labels, "indobenchmark/indobert-base-p1", batch_size=16
    )
    print(f"  F1: {results['indobert_full']['mean_f1']:.2f} ± {results['indobert_full']['std_f1']:.2f}")

    # --- IndoBERT on manual-only ---
    print(f"\n--- IndoBERT (manual-only) ---")
    results["indobert_manual"] = cv_transformer(
        texts, labels, "indobenchmark/indobert-base-p1", batch_size=16,
        source_filter="manual", sources=sources,
    )
    print(f"  F1: {results['indobert_manual']['mean_f1']:.2f} ± {results['indobert_manual']['std_f1']:.2f}")

    # Save
    output = {
        "metadata": {
            "dataset": DATA_PATH,
            "n_folds": N_FOLDS,
            "seed": SEED,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Data':<12} {'F1-Macro':<20} {'Accuracy':<20}")
    print("-" * 77)
    for key, r in results.items():
        parts = key.rsplit("_", 1)
        model = parts[0].upper()
        data = parts[1].capitalize()
        print(f"{model:<25} {data:<12} {r['mean_f1']:.2f} ± {r['std_f1']:.2f}{'':8} {r['mean_accuracy']:.2f} ± {r['std_accuracy']:.2f}")

    print(f"\nResults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
