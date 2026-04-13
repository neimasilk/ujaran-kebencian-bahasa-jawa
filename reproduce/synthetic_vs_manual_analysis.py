#!/usr/bin/env python3
"""
NEW EXPERIMENT: Synthetic vs Manual Data Quality Analysis

Answers the critical question: WHY does the model score 99.41% on synthetic
but only 53.89% on manual data? What makes synthetic data "easier"?

Analyzes:
1. Text length distributions (synthetic vs manual)
2. Vocabulary overlap and unique terms
3. Lexical diversity (type-token ratio)
4. Per-class text characteristics
5. N-gram distinctiveness (what patterns are unique to synthetic?)
6. TF-IDF feature analysis (what features distinguish synthetic from manual?)

Input:  data/cleaned/final_dataset_with_source.csv
Output: results/data_quality_analysis.json
"""

import os
import json
import re
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "final_dataset_with_source.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "data_quality_analysis.json")


def text_stats(texts):
    """Compute basic text statistics."""
    lengths_words = [len(t.split()) for t in texts]
    lengths_chars = [len(t) for t in texts]

    # Vocabulary
    all_words = []
    for t in texts:
        all_words.extend(t.lower().split())
    vocab = set(all_words)
    word_freq = Counter(all_words)

    # Type-token ratio (lexical diversity)
    ttr = len(vocab) / max(len(all_words), 1)

    # Hapax legomena (words appearing only once)
    hapax = sum(1 for w, c in word_freq.items() if c == 1)

    return {
        "count": len(texts),
        "word_count_mean": round(np.mean(lengths_words), 2),
        "word_count_median": round(np.median(lengths_words), 2),
        "word_count_std": round(np.std(lengths_words), 2),
        "word_count_min": int(np.min(lengths_words)),
        "word_count_max": int(np.max(lengths_words)),
        "char_count_mean": round(np.mean(lengths_chars), 2),
        "char_count_median": round(np.median(lengths_chars), 2),
        "vocab_size": len(vocab),
        "total_tokens": len(all_words),
        "type_token_ratio": round(ttr, 4),
        "hapax_count": hapax,
        "hapax_ratio": round(hapax / max(len(vocab), 1), 4),
    }


def vocabulary_overlap(texts_a, texts_b):
    """Compute vocabulary overlap between two text sets."""
    vocab_a = set()
    vocab_b = set()
    for t in texts_a:
        vocab_a.update(t.lower().split())
    for t in texts_b:
        vocab_b.update(t.lower().split())

    overlap = vocab_a & vocab_b
    only_a = vocab_a - vocab_b
    only_b = vocab_b - vocab_a

    return {
        "vocab_a_size": len(vocab_a),
        "vocab_b_size": len(vocab_b),
        "overlap_size": len(overlap),
        "overlap_ratio_a": round(len(overlap) / max(len(vocab_a), 1), 4),
        "overlap_ratio_b": round(len(overlap) / max(len(vocab_b), 1), 4),
        "jaccard_similarity": round(
            len(overlap) / max(len(vocab_a | vocab_b), 1), 4
        ),
        "only_in_a_count": len(only_a),
        "only_in_b_count": len(only_b),
        "sample_only_a": sorted(list(only_a))[:30],
        "sample_only_b": sorted(list(only_b))[:30],
    }


def source_distinguishability(texts, sources):
    """Can a simple classifier distinguish synthetic from manual?
    If yes, the distributions are very different."""
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X = tfidf.fit_transform(texts)
    y = [1 if s == "synthetic" else 0 for s in sources]

    # Cross-validation score
    lr = LogisticRegression(max_iter=500, random_state=42)
    scores = cross_val_score(lr, X, y, cv=5, scoring="f1_macro")

    # Fit full model to get most distinctive features
    lr.fit(X, y)
    feature_names = tfidf.get_feature_names_out()
    coefs = lr.coef_[0]

    # Top features indicating synthetic
    synth_idx = np.argsort(coefs)[-20:][::-1]
    synth_features = [(feature_names[i], round(float(coefs[i]), 4)) for i in synth_idx]

    # Top features indicating manual
    manual_idx = np.argsort(coefs)[:20]
    manual_features = [(feature_names[i], round(float(coefs[i]), 4)) for i in manual_idx]

    return {
        "cv_f1_macro_mean": round(np.mean(scores) * 100, 2),
        "cv_f1_macro_std": round(np.std(scores) * 100, 2),
        "interpretation": (
            "HIGH (>80%): synthetic and manual are very distinguishable — "
            "model can easily learn source-specific patterns"
            if np.mean(scores) > 0.8 else
            "MODERATE (60-80%): some distributional differences"
            if np.mean(scores) > 0.6 else
            "LOW (<60%): synthetic and manual are similar"
        ),
        "top_synthetic_indicators": synth_features,
        "top_manual_indicators": manual_features,
    }


def per_class_source_distribution(df):
    """Distribution of manual vs synthetic per class."""
    result = {}
    for label in sorted(df["label"].unique()):
        subset = df[df["label"] == label]
        manual_count = (subset["source"] == "manual").sum()
        synth_count = (subset["source"] == "synthetic").sum()
        total = len(subset)
        result[f"class_{label}"] = {
            "total": int(total),
            "manual": int(manual_count),
            "synthetic": int(synth_count),
            "synthetic_ratio": round(synth_count / max(total, 1), 4),
        }
    return result


def sentence_pattern_analysis(texts_manual, texts_synthetic):
    """Analyze sentence-level patterns."""
    def get_patterns(texts):
        patterns = {
            "starts_with_capital": sum(1 for t in texts if t[0].isupper()) / len(texts),
            "ends_with_period": sum(1 for t in texts if t.rstrip().endswith('.')) / len(texts),
            "ends_with_exclamation": sum(1 for t in texts if t.rstrip().endswith('!')) / len(texts),
            "has_quotes": sum(1 for t in texts if '"' in t or "'" in t) / len(texts),
            "has_emoji_chars": sum(1 for t in texts if any(ord(c) > 127 for c in t)) / len(texts),
            "all_lowercase": sum(1 for t in texts if t == t.lower()) / len(texts),
            "has_repetition": sum(1 for t in texts if re.search(r'(.)\1{2,}', t)) / len(texts),
        }
        return {k: round(v, 4) for k, v in patterns.items()}

    return {
        "manual": get_patterns(texts_manual),
        "synthetic": get_patterns(texts_synthetic),
    }


def main():
    print("=" * 60)
    print("SYNTHETIC vs MANUAL DATA QUALITY ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    df_manual = df[df["source"] == "manual"]
    df_synth = df[df["source"] == "synthetic"]

    texts_manual = df_manual["text"].tolist()
    texts_synth = df_synth["text"].tolist()

    print(f"Total: {len(df)} | Manual: {len(df_manual)} | Synthetic: {len(df_synth)}")

    results = {}

    # 1. Basic text statistics
    print("\n1. Computing text statistics...")
    results["manual_stats"] = text_stats(texts_manual)
    results["synthetic_stats"] = text_stats(texts_synth)

    print(f"   Manual — words: {results['manual_stats']['word_count_mean']:.1f} ± "
          f"{results['manual_stats']['word_count_std']:.1f}, "
          f"vocab: {results['manual_stats']['vocab_size']}, "
          f"TTR: {results['manual_stats']['type_token_ratio']:.4f}")
    print(f"   Synthetic — words: {results['synthetic_stats']['word_count_mean']:.1f} ± "
          f"{results['synthetic_stats']['word_count_std']:.1f}, "
          f"vocab: {results['synthetic_stats']['vocab_size']}, "
          f"TTR: {results['synthetic_stats']['type_token_ratio']:.4f}")

    # 2. Vocabulary overlap
    print("\n2. Computing vocabulary overlap...")
    results["vocabulary_overlap"] = vocabulary_overlap(texts_manual, texts_synth)
    print(f"   Jaccard similarity: {results['vocabulary_overlap']['jaccard_similarity']:.4f}")
    print(f"   Only in manual: {results['vocabulary_overlap']['only_in_a_count']}")
    print(f"   Only in synthetic: {results['vocabulary_overlap']['only_in_b_count']}")

    # 3. Source distinguishability
    print("\n3. Testing source distinguishability (can a classifier tell them apart?)...")
    results["source_distinguishability"] = source_distinguishability(
        df["text"].tolist(), df["source"].tolist()
    )
    print(f"   F1 for source classification: "
          f"{results['source_distinguishability']['cv_f1_macro_mean']:.2f}% ± "
          f"{results['source_distinguishability']['cv_f1_macro_std']:.2f}%")
    print(f"   {results['source_distinguishability']['interpretation']}")

    # 4. Per-class source distribution
    print("\n4. Per-class source distribution...")
    results["per_class_distribution"] = per_class_source_distribution(df)
    for cls, info in results["per_class_distribution"].items():
        print(f"   {cls}: {info['manual']} manual + {info['synthetic']} synthetic "
              f"({info['synthetic_ratio']:.0%} synthetic)")

    # 5. Sentence patterns
    print("\n5. Analyzing sentence patterns...")
    results["sentence_patterns"] = sentence_pattern_analysis(texts_manual, texts_synth)
    print("   Pattern              Manual    Synthetic")
    for key in results["sentence_patterns"]["manual"]:
        m = results["sentence_patterns"]["manual"][key]
        s = results["sentence_patterns"]["synthetic"][key]
        diff = "***" if abs(m - s) > 0.1 else ""
        print(f"   {key:<22} {m:.3f}     {s:.3f} {diff}")

    # 6. Most distinctive n-grams
    print("\n6. Top synthetic indicators (TF-IDF features):")
    for feat, score in results["source_distinguishability"]["top_synthetic_indicators"][:10]:
        print(f"   + {feat}: {score:.4f}")
    print("\n   Top manual indicators:")
    for feat, score in results["source_distinguishability"]["top_manual_indicators"][:10]:
        print(f"   - {feat}: {score:.4f}")

    # Save
    output = {
        "metadata": {
            "dataset": DATA_PATH,
            "manual_count": len(df_manual),
            "synthetic_count": len(df_synth),
            "description": "Comparative analysis of manual vs synthetic data quality",
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved: {RESULTS_PATH}")

    # Summary
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    dist_f1 = results["source_distinguishability"]["cv_f1_macro_mean"]
    if dist_f1 > 80:
        print(f"  ⚠ Synthetic data is HIGHLY distinguishable from manual (F1={dist_f1:.1f}%)")
        print(f"    → Model can exploit source-specific patterns instead of learning hate speech")
        print(f"    → This explains the 99.41% synthetic F1 vs 53.89% manual F1")
    elif dist_f1 > 60:
        print(f"  ! Synthetic data is MODERATELY distinguishable (F1={dist_f1:.1f}%)")
        print(f"    → Some distributional differences exist")
    else:
        print(f"  ✓ Synthetic data is similar to manual (F1={dist_f1:.1f}%)")


if __name__ == "__main__":
    main()
