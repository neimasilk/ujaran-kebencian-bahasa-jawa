#!/usr/bin/env python3
"""
Step 3: Train Traditional Baseline Models

Trains SVM and Logistic Regression with TF-IDF features as baselines.
Uses the same 80/10/10 stratified split (seed=42) as transformer experiments.

Input:  data/cleaned/final_dataset_with_source.csv
Output: results/baseline_results.json
"""

import os
import sys
import json
import time

import pandas as pd
import numpy as np
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
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "baseline_results.json")

LABEL_NAMES = [
    "Bukan Ujaran Kebencian",
    "Ujaran Kebencian - Ringan",
    "Ujaran Kebencian - Sedang",
    "Ujaran Kebencian - Berat",
]

SEED = 42


def load_and_split_data(seed=SEED):
    """Load dataset and create 80/10/10 stratified split (same as transformers)."""
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


def evaluate_model(name, model, X_test_tfidf, y_test):
    """Evaluate a model and return metrics dict."""
    y_pred = model.predict(X_test_tfidf)

    f1_macro = f1_score(y_test, y_pred, average="macro") * 100
    f1_weighted = f1_score(y_test, y_pred, average="weighted") * 100
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0) * 100
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0) * 100

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=LABEL_NAMES, output_dict=True)

    per_class = {}
    for i, label_name in enumerate(LABEL_NAMES):
        per_class[label_name] = {
            "precision": round(report[label_name]["precision"] * 100, 2),
            "recall": round(report[label_name]["recall"] * 100, 2),
            "f1": round(report[label_name]["f1-score"] * 100, 2),
            "support": int(report[label_name]["support"]),
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
    print("STEP 3: TRAIN BASELINE MODELS (SVM + LR)")
    print("=" * 60)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    print(f"\nDataset: {DATA_PATH}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # TF-IDF Vectorization
    print("\nFitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

    results = {}

    # --- SVM ---
    print(f"\n{'='*60}")
    print("Training: SVM + TF-IDF (Linear Kernel)")
    print(f"{'='*60}")
    start = time.time()
    svm = SVC(kernel="linear", C=1.0, random_state=SEED)
    svm.fit(X_train_tfidf, y_train)
    svm_time = time.time() - start
    print(f"Training time: {svm_time:.1f}s")

    svm_val = evaluate_model("SVM", svm, X_val_tfidf, y_val)
    svm_test = evaluate_model("SVM", svm, X_test_tfidf, y_test)
    print(f"Val  F1-Macro: {svm_val['f1_macro']:.2f}%  Acc: {svm_val['accuracy']:.2f}%")
    print(f"Test F1-Macro: {svm_test['f1_macro']:.2f}%  Acc: {svm_test['accuracy']:.2f}%")

    results["svm_tfidf"] = {
        "description": "SVM + TF-IDF (Linear)",
        "hyperparameters": {
            "kernel": "linear",
            "C": 1.0,
            "tfidf_max_features": 10000,
            "tfidf_ngram_range": [1, 2],
        },
        "validation": svm_val,
        "test": svm_test,
        "train_time_seconds": round(svm_time, 1),
    }

    # --- Logistic Regression ---
    print(f"\n{'='*60}")
    print("Training: Logistic Regression + TF-IDF")
    print(f"{'='*60}")
    start = time.time()
    lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=SEED,
    )
    lr.fit(X_train_tfidf, y_train)
    lr_time = time.time() - start
    print(f"Training time: {lr_time:.1f}s")

    lr_val = evaluate_model("LR", lr, X_val_tfidf, y_val)
    lr_test = evaluate_model("LR", lr, X_test_tfidf, y_test)
    print(f"Val  F1-Macro: {lr_val['f1_macro']:.2f}%  Acc: {lr_val['accuracy']:.2f}%")
    print(f"Test F1-Macro: {lr_test['f1_macro']:.2f}%  Acc: {lr_test['accuracy']:.2f}%")

    results["lr_tfidf"] = {
        "description": "Logistic Regression + TF-IDF",
        "hyperparameters": {
            "max_iter": 1000,
            "C": 1.0,
            "solver": "lbfgs",
            "tfidf_max_features": 10000,
            "tfidf_ngram_range": [1, 2],
        },
        "validation": lr_val,
        "test": lr_test,
        "train_time_seconds": round(lr_time, 1),
    }

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
        },
        "models": results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    for key, r in results.items():
        print(f"  {r['description']:<35s}")
        print(f"    Val  F1={r['validation']['f1_macro']:.2f}%  Acc={r['validation']['accuracy']:.2f}%")
        print(f"    Test F1={r['test']['f1_macro']:.2f}%  Acc={r['test']['accuracy']:.2f}%")

    print(f"\nResults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
