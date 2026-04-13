#!/usr/bin/env python3
"""
NEW EXPERIMENT: Error Analysis

Addresses editor complaint: "no analysis of your proposed methods"

Analyzes misclassifications to understand:
1. Which class pairs are most confused (confusion patterns)
2. Text length vs prediction quality
3. Per-class error characteristics
4. Specific examples of common error types
5. Comparison of error patterns between SVM and XLM-R

Input:  data/cleaned/final_dataset_with_source.csv
        models/comparative/xlmr_large/best/
Output: results/error_analysis.json
"""

import os
import sys
import json
import time
import re
from collections import Counter, defaultdict

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
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "final_dataset_with_source.csv")
XLMR_CHECKPOINT = os.path.join(PROJECT_ROOT, "models", "comparative", "xlmr_large", "best")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "error_analysis.json")

LABEL_NAMES = [
    "Bukan Ujaran Kebencian",
    "Ujaran Kebencian - Ringan",
    "Ujaran Kebencian - Sedang",
    "Ujaran Kebencian - Berat",
]
LABEL_SHORT = ["Not Hate", "Light", "Moderate", "Severe"]
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


def get_text_features(text):
    """Extract linguistic features from a text."""
    words = text.split()
    chars = len(text)

    # Detect possible code-switching (Latin + Javanese script patterns)
    has_numbers = bool(re.search(r'\d', text))
    has_urls = bool(re.search(r'http|www|\.com|\.id', text, re.I))
    has_mentions = bool(re.search(r'@\w+', text))
    has_hashtags = bool(re.search(r'#\w+', text))

    # Exclamation/emphasis markers
    exclamation_count = text.count('!') + text.count('?')
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    return {
        "word_count": len(words),
        "char_count": chars,
        "has_numbers": has_numbers,
        "has_urls": has_urls,
        "has_mentions": has_mentions,
        "has_hashtags": has_hashtags,
        "exclamation_count": exclamation_count,
        "caps_ratio": round(caps_ratio, 4),
    }


def analyze_confusion_patterns(y_true, y_pred, texts, model_name):
    """Analyze confusion patterns and error types."""
    analysis = {}

    # 1. Overall accuracy and confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred) * 100

    # 2. Adjacent vs non-adjacent confusion
    adjacent_errors = 0
    non_adjacent_errors = 0
    total_errors = 0

    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            total_errors += 1
            diff = abs(y_true[i] - y_pred[i])
            if diff == 1:
                adjacent_errors += 1
            else:
                non_adjacent_errors += 1

    analysis["total_samples"] = len(y_true)
    analysis["total_errors"] = total_errors
    analysis["error_rate"] = round(total_errors / len(y_true) * 100, 2)
    analysis["adjacent_class_errors"] = adjacent_errors
    analysis["non_adjacent_errors"] = non_adjacent_errors
    analysis["adjacent_error_ratio"] = round(
        adjacent_errors / max(total_errors, 1) * 100, 2
    )

    # 3. Most common error pairs
    error_pairs = Counter()
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            pair = f"{LABEL_SHORT[y_true[i]]} → {LABEL_SHORT[y_pred[i]]}"
            error_pairs[pair] += 1

    analysis["top_error_pairs"] = [
        {"pair": pair, "count": count, "pct_of_errors": round(count / max(total_errors, 1) * 100, 1)}
        for pair, count in error_pairs.most_common(10)
    ]

    # 4. Text length analysis
    correct_lengths = []
    error_lengths = []
    for i in range(len(y_true)):
        wc = len(texts[i].split())
        if y_true[i] == y_pred[i]:
            correct_lengths.append(wc)
        else:
            error_lengths.append(wc)

    analysis["length_analysis"] = {
        "correct_mean_words": round(np.mean(correct_lengths), 1) if correct_lengths else 0,
        "correct_median_words": round(np.median(correct_lengths), 1) if correct_lengths else 0,
        "error_mean_words": round(np.mean(error_lengths), 1) if error_lengths else 0,
        "error_median_words": round(np.median(error_lengths), 1) if error_lengths else 0,
    }

    # 5. Per-class error analysis
    per_class = {}
    for cls in range(NUM_LABELS):
        cls_mask = np.array(y_true) == cls
        cls_total = cls_mask.sum()
        if cls_total == 0:
            continue

        cls_correct = ((np.array(y_true) == cls) & (np.array(y_pred) == cls)).sum()
        cls_errors = cls_total - cls_correct

        # Where are errors going?
        misclass_dist = Counter()
        for i in range(len(y_true)):
            if y_true[i] == cls and y_pred[i] != cls:
                misclass_dist[LABEL_SHORT[y_pred[i]]] += 1

        per_class[LABEL_SHORT[cls]] = {
            "total": int(cls_total),
            "correct": int(cls_correct),
            "accuracy": round(cls_correct / cls_total * 100, 1),
            "misclassified_as": dict(misclass_dist.most_common()),
        }

    analysis["per_class_errors"] = per_class

    # 6. Feature analysis of errors
    correct_features = defaultdict(list)
    error_features = defaultdict(list)
    for i in range(len(y_true)):
        feats = get_text_features(texts[i])
        target = correct_features if y_true[i] == y_pred[i] else error_features
        for k, v in feats.items():
            if isinstance(v, bool):
                target[k].append(int(v))
            elif isinstance(v, (int, float)):
                target[k].append(v)

    feature_comparison = {}
    for feat in correct_features:
        c_vals = correct_features[feat]
        e_vals = error_features.get(feat, [0])
        feature_comparison[feat] = {
            "correct_mean": round(np.mean(c_vals), 3),
            "error_mean": round(np.mean(e_vals), 3),
        }
    analysis["feature_comparison"] = feature_comparison

    # 7. Example errors (anonymized/truncated for paper)
    examples = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i] and len(examples) < 15:
            text_preview = texts[i][:100] + ("..." if len(texts[i]) > 100 else "")
            examples.append({
                "text_preview": text_preview,
                "true_label": LABEL_SHORT[y_true[i]],
                "predicted": LABEL_SHORT[y_pred[i]],
                "word_count": len(texts[i].split()),
            })
    analysis["example_errors"] = examples

    return analysis


def main():
    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    sources = df["source"].tolist()

    # Same split
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        texts, labels, sources, test_size=0.2, random_state=SEED, stratify=labels
    )
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    # Manual-only test
    test_manual_idx = [i for i, s in enumerate(s_test) if s == "manual"]
    X_test_manual = [X_test[i] for i in test_manual_idx]
    y_test_manual = [y_test[i] for i in test_manual_idx]

    print(f"Manual test samples: {len(X_test_manual)}")

    results = {}

    # === SVM Predictions ===
    print(f"\n{'='*60}")
    print("Analyzing SVM predictions...")
    print(f"{'='*60}")

    tfidf = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2),
        sublinear_tf=True, min_df=2,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_manual_tfidf = tfidf.transform(X_test_manual)

    svm = SVC(kernel="linear", C=1.0, random_state=SEED)
    svm.fit(X_train_tfidf, y_train)
    svm_preds = svm.predict(X_test_manual_tfidf).tolist()

    results["svm"] = analyze_confusion_patterns(
        y_test_manual, svm_preds, X_test_manual, "SVM"
    )
    print(f"  SVM errors: {results['svm']['total_errors']}/{len(y_test_manual)}")
    print(f"  Adjacent errors: {results['svm']['adjacent_error_ratio']:.1f}%")

    # === XLM-R Predictions ===
    print(f"\n{'='*60}")
    print("Analyzing XLM-R Large predictions...")
    print(f"{'='*60}")

    if os.path.exists(XLMR_CHECKPOINT):
        tokenizer = AutoTokenizer.from_pretrained(XLMR_CHECKPOINT)
        model = AutoModelForSequenceClassification.from_pretrained(XLMR_CHECKPOINT)

        dataset = TextDataset(X_test_manual, y_test_manual, tokenizer, MAX_LEN)
        trainer = Trainer(model=model)
        output = trainer.predict(dataset)
        xlmr_preds = np.argmax(output.predictions, axis=-1).tolist()

        # Also get prediction confidence (softmax probabilities)
        probs = torch.softmax(torch.tensor(output.predictions), dim=-1).numpy()
        max_probs = probs.max(axis=1)

        results["xlmr_large"] = analyze_confusion_patterns(
            y_test_manual, xlmr_preds, X_test_manual, "XLM-R Large"
        )

        # Add confidence analysis
        correct_mask = np.array(y_test_manual) == np.array(xlmr_preds)
        results["xlmr_large"]["confidence_analysis"] = {
            "correct_mean_confidence": round(float(max_probs[correct_mask].mean()), 4),
            "error_mean_confidence": round(float(max_probs[~correct_mask].mean()), 4),
            "correct_median_confidence": round(float(np.median(max_probs[correct_mask])), 4),
            "error_median_confidence": round(float(np.median(max_probs[~correct_mask])), 4),
        }

        print(f"  XLM-R errors: {results['xlmr_large']['total_errors']}/{len(y_test_manual)}")
        print(f"  Adjacent errors: {results['xlmr_large']['adjacent_error_ratio']:.1f}%")
        print(f"  Correct confidence: {results['xlmr_large']['confidence_analysis']['correct_mean_confidence']:.3f}")
        print(f"  Error confidence: {results['xlmr_large']['confidence_analysis']['error_mean_confidence']:.3f}")

        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print(f"  WARNING: XLM-R checkpoint not found at {XLMR_CHECKPOINT}")
        print(f"  Skipping XLM-R analysis")

    # === Cross-model agreement ===
    if "xlmr_large" in results:
        print(f"\n{'='*60}")
        print("Cross-model agreement analysis...")
        print(f"{'='*60}")

        both_correct = sum(
            1 for t, s, x in zip(y_test_manual, svm_preds, xlmr_preds)
            if s == t and x == t
        )
        svm_only_correct = sum(
            1 for t, s, x in zip(y_test_manual, svm_preds, xlmr_preds)
            if s == t and x != t
        )
        xlmr_only_correct = sum(
            1 for t, s, x in zip(y_test_manual, svm_preds, xlmr_preds)
            if s != t and x == t
        )
        both_wrong = sum(
            1 for t, s, x in zip(y_test_manual, svm_preds, xlmr_preds)
            if s != t and x != t
        )

        # Same wrong answer?
        both_wrong_same_pred = sum(
            1 for t, s, x in zip(y_test_manual, svm_preds, xlmr_preds)
            if s != t and x != t and s == x
        )

        agreement = {
            "both_correct": both_correct,
            "svm_only_correct": svm_only_correct,
            "xlmr_only_correct": xlmr_only_correct,
            "both_wrong": both_wrong,
            "both_wrong_same_prediction": both_wrong_same_pred,
            "ensemble_potential": both_correct + svm_only_correct + xlmr_only_correct,
            "ensemble_potential_pct": round(
                (both_correct + svm_only_correct + xlmr_only_correct) / len(y_test_manual) * 100, 2
            ),
        }
        results["cross_model_agreement"] = agreement

        print(f"  Both correct: {both_correct} ({both_correct/len(y_test_manual)*100:.1f}%)")
        print(f"  SVM only correct: {svm_only_correct}")
        print(f"  XLM-R only correct: {xlmr_only_correct}")
        print(f"  Both wrong: {both_wrong}")
        print(f"  Ensemble upper bound: {agreement['ensemble_potential_pct']:.1f}%")

    # Save
    output = {
        "metadata": {
            "dataset": DATA_PATH,
            "manual_test_count": len(X_test_manual),
            "seed": SEED,
            "description": "Error analysis comparing SVM and XLM-R on manual test data",
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
