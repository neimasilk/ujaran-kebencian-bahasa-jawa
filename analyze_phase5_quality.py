"""
Data Quality Analysis untuk Phase 5 DeepSeek Re-labeled Dataset
===============================================================

Analisis kualitas dataset setelah LLM re-labeling.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from collections import Counter
import re


def analyze_phase5():
    """Analisis kualitas dataset Phase 5"""

    print("=" * 70)
    print("DATA QUALITY ANALYSIS: Phase 5 DeepSeek Re-labeled Dataset")
    print("=" * 70)

    # Load dataset
    phase5_path = "data/improved/phase5_deepseek_relabeled.csv"
    df = pd.read_csv(phase5_path)

    print(f"\nDataset: {phase5_path}")
    print(f"Total samples: {len(df)}")

    # 1. Label Distribution
    print("\n" + "-" * 70)
    print("1. DISTRIBUSI LABEL")
    print("-" * 70)

    label_names = {
        0: "Neutral (Non-Hate)",
        1: "Light Hate",
        2: "Moderate Hate",
        3: "Severe Hate"
    }

    label_dist = df['label'].value_counts().sort_index()
    total = len(df)

    print(f"\n{'Label':<30} {'Count':>10} {'Percentage':>12}")
    print("-" * 55)

    for label, count in label_dist.items():
        pct = count / total * 100
        print(f"{label} - {label_names[label]:<23} {count:>10} {pct:>11.2f}%")

    # Class balance metrics
    max_count = label_dist.max()
    min_count = label_dist.min()
    imbalance_ratio = max_count / min_count

    print(f"\nClass Balance Metrics:")
    print(f"  - Majority class: {max_count} samples")
    print(f"  - Minority class: {min_count} samples")
    print(f"  - Imbalance ratio: {imbalance_ratio:.2f}:1")

    if imbalance_ratio < 1.5:
        balance_status = "EXCELLENT - Well balanced"
    elif imbalance_ratio < 2.0:
        balance_status = "GOOD - Acceptable balance"
    elif imbalance_ratio < 3.0:
        balance_status = "MODERATE - Some imbalance"
    else:
        balance_status = "POOR - High imbalance"

    print(f"  - Status: {balance_status}")

    # 2. Text Quality Analysis
    print("\n" + "-" * 70)
    print("2. KUALITAS TEKS")
    print("-" * 70)

    texts = df['text'].astype(str)
    text_lengths = texts.str.len()

    print(f"\nText Length Statistics:")
    print(f"  - Mean: {text_lengths.mean():.1f} characters")
    print(f"  - Median: {text_lengths.median():.1f} characters")
    print(f"  - Min: {text_lengths.min()} characters")
    print(f"  - Max: {text_lengths.max()} characters")
    print(f"  - Std Dev: {text_lengths.std():.1f}")

    # Very short texts (potential noise)
    very_short = (text_lengths < 15).sum()
    print(f"\n  - Very short texts (<15 chars): {very_short} ({very_short/total*100:.2f}%)")

    if very_short > 0:
        print("\n    Examples of very short texts:")
        for idx, row in df[text_lengths < 15].head(5).iterrows():
            print(f"      [{row['label']}] {row['text'][:50]}")

    # Empty texts
    empty = texts.str.strip().eq('').sum()
    print(f"\n  - Empty/whitespace texts: {empty}")

    # 3. Duplicate Analysis
    print("\n" + "-" * 70)
    print("3. ANALISIS DUPLIKAT")
    print("-" * 70)

    unique_texts = df['text'].nunique()
    exact_duplicates = len(df) - unique_texts

    print(f"\n  - Total samples: {len(df)}")
    print(f"  - Unique texts: {unique_texts}")
    print(f"  - Exact duplicates: {exact_duplicates} ({exact_duplicates/total*100:.2f}%)")

    if exact_duplicates > 0:
        dup_texts = df[df.duplicated(subset=['text'], keep=False)].sort_values('text')
        print(f"\n  Duplicate pairs with DIFFERENT labels:")
        dup_groups = dup_texts.groupby('text')['label'].nunique()
        conflicting_dups = dup_groups[dup_groups > 1]
        print(f"    - Total: {len(conflicting_dups)} texts with conflicting labels")

        for text in conflicting_dups.head(3).index:
            labels = df[df['text'] == text]['label'].tolist()
            print(f"      Text: {text[:50]}...")
            print(f"      Labels: {labels}")

    # 4. Per-Class Text Statistics
    print("\n" + "-" * 70)
    print("4. STATISTIK TEKS PER KELAS")
    print("-" * 70)

    print(f"\n{'Class':<25} {'Avg Len':>10} {'Median':>10} {'Samples':>10}")
    print("-" * 60)

    for label in range(4):
        class_texts = df[df['label'] == label]['text'].astype(str)
        avg_len = class_texts.str.len().mean()
        median_len = class_texts.str.len().median()
        count = len(class_texts)
        print(f"{label_names[label]:<25} {avg_len:>9.1f} {median_len:>10.1f} {count:>10}")

    # 5. Potential Issues Detection
    print("\n" + "-" * 70)
    print("5. DETEKSI MASALAH POTENSIAL")
    print("-" * 70)

    issues = []

    # Check for very long texts (might be malformed)
    very_long = (text_lengths > 500).sum()
    if very_long > 0:
        issues.append(f"Very long texts (>500 chars): {very_long}")

    # Check for texts with only special characters
    special_only = texts.str.match(r'^[^a-zA-Z0-9\s]+$').sum()
    if special_only > 0:
        issues.append(f"Texts with only special characters: {special_only}")

    # Check for texts with excessive repetition
    repetitive = 0
    for text in texts:
        if len(text) > 20:
            words = text.split()
            if len(set(words)) / len(words) < 0.3:  # Less than 30% unique words
                repetitive += 1
    if repetitive > 0:
        issues.append(f"Highly repetitive texts: {repetitive}")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo significant issues detected!")

    # 6. Comparison with Previous Phases
    print("\n" + "-" * 70)
    print("6. PERBANDINGAN DENGAN PHASE SEBELUMNYA")
    print("-" * 70)

    phase3_path = "data/improved/phase3_relabeled.csv"
    phase4_path = "data/improved/phase4_generated.csv"

    if os.path.exists(phase3_path):
        phase3 = pd.read_csv(phase3_path)
        phase3 = phase3[['text', 'new_label']].copy()
        phase3 = phase3.rename(columns={'new_label': 'label'})

        print(f"\nPhase 3 (Original Relabeled):")
        p3_dist = phase3['label'].value_counts().sort_index()
        for label, count in p3_dist.items():
            pct = count / len(phase3) * 100
            print(f"  {label_names[label]}: {count} ({pct:.2f}%)")

    if os.path.exists(phase4_path):
        phase4 = pd.read_csv(phase4_path)

        print(f"\nPhase 4 (Generated):")
        p4_dist = phase4['label'].value_counts().sort_index()
        for label, count in p4_dist.items():
            pct = count / len(phase4) * 100
            print(f"  {label_names[label]}: {count} ({pct:.2f}%)")

    print(f"\nPhase 5 (DeepSeek Re-labeled):")
    for label, count in label_dist.items():
        pct = count / total * 100
        print(f"  {label_names[label]}: {count} ({pct:.2f}%)")

    # 7. Summary & Recommendations
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print(f"\nDataset Quality Score: {calculate_quality_score(df)}/100")

    recommendations = []

    if imbalance_ratio > 2.0:
        recommendations.append("Consider class weights or focal loss during training")

    if very_short > 10:
        recommendations.append(f"Review {very_short} very short texts for potential noise")

    if exact_duplicates > 0:
        recommendations.append(f"Remove or review {exact_duplicates} duplicate samples")

    if not recommendations:
        recommendations.append("Dataset quality is GOOD - Ready for training!")
        recommendations.append("Recommended next steps:")
        recommendations.append("  1. Train with Phase 5 dataset on GPU")
        recommendations.append("  2. Evaluate improvement vs baseline")
        recommendations.append("  3. Apply Threshold Optimization")
        recommendations.append("  4. Apply Test-Time Augmentation")
    else:
        recommendations.append("Recommended next steps:")
        recommendations.append("  1. Address the issues above")
        recommendations.append("  2. Then proceed to GPU training")

    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    # Save results
    results = {
        'total_samples': int(total),
        'label_distribution': {int(k): int(v) for k, v in label_dist.items()},
        'imbalance_ratio': float(imbalance_ratio),
        'avg_text_length': float(text_lengths.mean()),
        'unique_texts': int(unique_texts),
        'duplicates': int(exact_duplicates),
        'quality_score': calculate_quality_score(df),
        'recommendations': recommendations
    }

    output_path = "results/experiment_11_deepseek/quality_analysis.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


def calculate_quality_score(df):
    """Calculate overall quality score (0-100)"""
    score = 100

    # Deduction for imbalance
    label_dist = df['label'].value_counts()
    imbalance = label_dist.max() / label_dist.min()
    if imbalance > 3:
        score -= 10
    elif imbalance > 2:
        score -= 5

    # Deduction for duplicates
    unique_ratio = df['text'].nunique() / len(df)
    if unique_ratio < 0.95:
        score -= 10
    elif unique_ratio < 0.98:
        score -= 5

    # Deduction for very short texts
    very_short = (df['text'].astype(str).str.len() < 15).sum()
    if very_short / len(df) > 0.01:
        score -= 5

    return max(score, 0)


if __name__ == "__main__":
    analyze_phase5()
