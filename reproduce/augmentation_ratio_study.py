#!/usr/bin/env python3
"""
NEW EXPERIMENT: Augmentation Ratio Study

Trains models with varying proportions of synthetic data (0%, 25%, 50%, 75%, 100%)
and evaluates ALL on the same manual-only test set.

This is the KEY new experiment that demonstrates:
1. How augmentation ratio affects real-world performance
2. Whether there's an optimal ratio
3. That the pattern holds across model architectures (SVM vs Transformer)

Input:  data/cleaned/final_dataset_with_source.csv
Output: results/augmentation_ratio_results.json
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "final_dataset_with_source.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "augmentation_ratio_results.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "ratio_study")

LABEL_NAMES = [
    "Bukan Ujaran Kebencian",
    "Ujaran Kebencian - Ringan",
    "Ujaran Kebencian - Sedang",
    "Ujaran Kebencian - Berat",
]

NUM_LABELS = 4
SEED = 42
MAX_LEN = 128
RATIOS = [0.0, 0.25, 0.50, 0.75, 1.0]


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


def prepare_data():
    """Load data and create splits with source tracking."""
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

    # Separate train into manual and synthetic
    train_manual_idx = [i for i, s in enumerate(s_train) if s == "manual"]
    train_synth_idx = [i for i, s in enumerate(s_train) if s == "synthetic"]

    X_train_manual = [X_train[i] for i in train_manual_idx]
    y_train_manual = [y_train[i] for i in train_manual_idx]
    X_train_synth = [X_train[i] for i in train_synth_idx]
    y_train_synth = [y_train[i] for i in train_synth_idx]

    # Manual-only test set (for evaluation)
    test_manual_idx = [i for i, s in enumerate(s_test) if s == "manual"]
    X_test_manual = [X_test[i] for i in test_manual_idx]
    y_test_manual = [y_test[i] for i in test_manual_idx]

    # Manual-only val set
    val_manual_idx = [i for i, s in enumerate(s_val) if s == "manual"]
    X_val_manual = [X_val[i] for i in val_manual_idx]
    y_val_manual = [y_val[i] for i in val_manual_idx]

    return {
        "X_train_manual": X_train_manual,
        "y_train_manual": y_train_manual,
        "X_train_synth": X_train_synth,
        "y_train_synth": y_train_synth,
        "X_val": X_val,
        "y_val": y_val,
        "X_val_manual": X_val_manual,
        "y_val_manual": y_val_manual,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_manual": X_test_manual,
        "y_test_manual": y_test_manual,
        "X_test_full": X_test,
        "y_test_full": y_test,
    }


def build_ratio_dataset(data, ratio):
    """Build training set with given ratio of synthetic data."""
    X_manual = data["X_train_manual"]
    y_manual = data["y_train_manual"]
    X_synth = data["X_train_synth"]
    y_synth = data["y_train_synth"]

    if ratio == 0.0:
        return X_manual[:], y_manual[:]

    # Sample synthetic data at the given ratio
    n_synth = int(len(X_synth) * ratio)
    np.random.seed(SEED)
    indices = np.random.choice(len(X_synth), size=n_synth, replace=False)
    X_synth_sampled = [X_synth[i] for i in indices]
    y_synth_sampled = [y_synth[i] for i in indices]

    X_combined = X_manual + X_synth_sampled
    y_combined = y_manual + y_synth_sampled

    return X_combined, y_combined


def train_svm_at_ratio(data, ratio):
    """Train SVM + TF-IDF at given augmentation ratio."""
    X_train, y_train = build_ratio_dataset(data, ratio)

    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)

    svm = SVC(kernel="linear", C=1.0, random_state=SEED)
    start = time.time()
    svm.fit(X_train_tfidf, y_train)
    train_time = time.time() - start

    # Evaluate on manual-only test
    X_test_tfidf = tfidf.transform(data["X_test_manual"])
    y_pred = svm.predict(X_test_tfidf)
    metrics_manual = get_detailed_metrics(data["y_test_manual"], y_pred)

    # Also evaluate on full test for comparison
    X_test_full_tfidf = tfidf.transform(data["X_test_full"])
    y_pred_full = svm.predict(X_test_full_tfidf)
    metrics_full = get_detailed_metrics(data["y_test_full"], y_pred_full)

    return {
        "manual_test": metrics_manual,
        "full_test": metrics_full,
        "train_samples": len(X_train),
        "train_time_seconds": round(train_time, 1),
    }


def train_transformer_at_ratio(data, ratio, model_name, pretrained, batch_size, lr):
    """Train a transformer at given augmentation ratio."""
    X_train, y_train = build_ratio_dataset(data, ratio)

    output_dir = os.path.join(MODELS_DIR, f"{model_name}_ratio_{int(ratio*100)}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained, num_labels=NUM_LABELS
    )

    train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = TextDataset(data["X_val"], data["y_val"], tokenizer, MAX_LEN)
    test_manual_dataset = TextDataset(
        data["X_test_manual"], data["y_test_manual"], tokenizer, MAX_LEN
    )
    test_full_dataset = TextDataset(
        data["X_test_full"], data["y_test_full"], tokenizer, MAX_LEN
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        learning_rate=lr,
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

    # Evaluate on manual-only test
    manual_output = trainer.predict(test_manual_dataset)
    y_pred_manual = np.argmax(manual_output.predictions, axis=-1)
    metrics_manual = get_detailed_metrics(
        np.array(data["y_test_manual"]), y_pred_manual
    )

    # Evaluate on full test
    full_output = trainer.predict(test_full_dataset)
    y_pred_full = np.argmax(full_output.predictions, axis=-1)
    metrics_full = get_detailed_metrics(
        np.array(data["y_test_full"]), y_pred_full
    )

    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "manual_test": metrics_manual,
        "full_test": metrics_full,
        "train_samples": len(X_train),
        "train_time_seconds": round(train_time, 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="svm,indobert",
                        help="Comma-separated: svm,indobert,xlmr")
    parser.add_argument("--ratios", type=str, default=None,
                        help="Comma-separated ratios, e.g., 0.0,0.5,1.0")
    args = parser.parse_args()

    requested_models = args.models.split(",")
    ratios = [float(r) for r in args.ratios.split(",")] if args.ratios else RATIOS

    print("=" * 60)
    print("AUGMENTATION RATIO STUDY")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Prepare data
    data = prepare_data()
    print(f"\nData prepared:")
    print(f"  Manual train: {len(data['X_train_manual'])}")
    print(f"  Synthetic train: {len(data['X_train_synth'])}")
    print(f"  Val: {len(data['X_val'])}")
    print(f"  Manual test (evaluation): {len(data['X_test_manual'])}")
    print(f"  Full test: {len(data['X_test_full'])}")
    print(f"\nRatios to test: {ratios}")
    print(f"Models: {requested_models}")

    # Load existing results if any
    all_results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            existing = json.load(f)
            all_results = existing.get("results", {})

    for ratio in ratios:
        ratio_key = f"ratio_{int(ratio*100)}"
        if ratio_key not in all_results:
            all_results[ratio_key] = {}

        # SVM
        if "svm" in requested_models:
            print(f"\n{'='*60}")
            print(f"SVM @ ratio={ratio:.0%}")
            print(f"{'='*60}")
            svm_result = train_svm_at_ratio(data, ratio)
            all_results[ratio_key]["svm"] = svm_result
            print(f"  Train samples: {svm_result['train_samples']}")
            print(f"  Manual test F1: {svm_result['manual_test']['f1_macro']:.2f}%")
            print(f"  Full test F1:   {svm_result['full_test']['f1_macro']:.2f}%")

        # IndoBERT
        if "indobert" in requested_models:
            print(f"\n{'='*60}")
            print(f"IndoBERT @ ratio={ratio:.0%}")
            print(f"{'='*60}")
            ib_result = train_transformer_at_ratio(
                data, ratio, "indobert",
                "indobenchmark/indobert-base-p1",
                batch_size=16, lr=2e-5,
            )
            all_results[ratio_key]["indobert"] = ib_result
            print(f"  Train samples: {ib_result['train_samples']}")
            print(f"  Manual test F1: {ib_result['manual_test']['f1_macro']:.2f}%")
            print(f"  Full test F1:   {ib_result['full_test']['f1_macro']:.2f}%")

        # XLM-R Large (optional, expensive)
        if "xlmr" in requested_models:
            print(f"\n{'='*60}")
            print(f"XLM-R Large @ ratio={ratio:.0%}")
            print(f"{'='*60}")
            xlmr_result = train_transformer_at_ratio(
                data, ratio, "xlmr_large",
                "xlm-roberta-large",
                batch_size=8, lr=2e-5,
            )
            all_results[ratio_key]["xlmr_large"] = xlmr_result
            print(f"  Train samples: {xlmr_result['train_samples']}")
            print(f"  Manual test F1: {xlmr_result['manual_test']['f1_macro']:.2f}%")
            print(f"  Full test F1:   {xlmr_result['full_test']['f1_macro']:.2f}%")

        # Save after each ratio (checkpoint)
        output = {
            "metadata": {
                "dataset": DATA_PATH,
                "ratios": ratios,
                "models": requested_models,
                "manual_train_count": len(data["X_train_manual"]),
                "synthetic_train_count": len(data["X_train_synth"]),
                "manual_test_count": len(data["X_test_manual"]),
                "full_test_count": len(data["X_test_full"]),
                "seed": SEED,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "results": all_results,
        }
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  [Checkpoint saved to {RESULTS_PATH}]")

    # Print summary table
    print(f"\n{'='*60}")
    print("AUGMENTATION RATIO STUDY - SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Ratio':<8}", end="")
    for model in requested_models:
        print(f"  {model:<20s}", end="")
    print()
    print(f"{'':8}", end="")
    for model in requested_models:
        print(f"  {'Manual F1':<10s}{'Full F1':<10s}", end="")
    print()
    print("-" * (8 + 20 * len(requested_models)))

    for ratio in ratios:
        ratio_key = f"ratio_{int(ratio*100)}"
        print(f"{ratio:<8.0%}", end="")
        for model in requested_models:
            if model in all_results.get(ratio_key, {}):
                r = all_results[ratio_key][model]
                mf1 = r["manual_test"]["f1_macro"]
                ff1 = r["full_test"]["f1_macro"]
                print(f"  {mf1:<10.2f}{ff1:<10.2f}", end="")
            else:
                print(f"  {'N/A':<10s}{'N/A':<10s}", end="")
        print()

    print(f"\nResults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
