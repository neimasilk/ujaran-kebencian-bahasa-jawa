#!/usr/bin/env python3
"""
Dataset Quality Analysis for Javanese Hate Speech Detection
Investigates data quality issues that may affect model performance
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import json

def load_dataset(path):
    """Load dataset with error handling"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def analyze_basic_stats(df, name="Dataset"):
    """Basic statistics"""
    print(f"\n{'='*60}")
    print(f"ANALISIS: {name}")
    print(f"{'='*60}")

    print(f"\n1. STATISTIK DASAR:")
    print(f"   - Total sampel: {len(df)}")
    print(f"   - Kolom: {list(df.columns)}")
    print(f"   - Missing values: {df.isnull().sum().to_dict()}")

    return len(df)

def analyze_label_distribution(df):
    """Analyze label distribution"""
    print(f"\n2. DISTRIBUSI LABEL:")

    label_map = {
        0: "Bukan Ujaran Kebencian",
        1: "Ujaran Kebencian - Ringan",
        2: "Ujaran Kebencian - Sedang",
        3: "Ujaran Kebencian - Berat"
    }

    dist = df['label'].value_counts().sort_index()
    total = len(df)

    for label, count in dist.items():
        pct = count / total * 100
        label_name = label_map.get(label, f"Unknown({label})")
        print(f"   - {label} ({label_name}): {count} ({pct:.2f}%)")

    # Imbalance ratio
    max_count = dist.max()
    min_count = dist.min()
    imbalance = max_count / min_count
    print(f"\n   Imbalance ratio: {imbalance:.2f}:1")

    return dist

def analyze_text_quality(df):
    """Analyze text quality issues"""
    print(f"\n3. KUALITAS TEKS:")

    texts = df['text'].astype(str)

    # Length analysis
    lengths = texts.str.len()
    print(f"   - Panjang rata-rata: {lengths.mean():.1f} karakter")
    print(f"   - Panjang min: {lengths.min()}")
    print(f"   - Panjang max: {lengths.max()}")
    print(f"   - Median: {lengths.median():.1f}")

    # Very short texts (< 10 chars)
    very_short = (lengths < 10).sum()
    print(f"\n   - Teks sangat pendek (<10 char): {very_short} ({very_short/len(df)*100:.2f}%)")

    # Very long texts (> 500 chars)
    very_long = (lengths > 500).sum()
    print(f"   - Teks sangat panjang (>500 char): {very_long} ({very_long/len(df)*100:.2f}%)")

    # Empty or whitespace only
    empty = texts.str.strip().eq('').sum()
    print(f"   - Teks kosong/whitespace: {empty}")

    return lengths

def analyze_duplicates(df):
    """Analyze duplicate texts"""
    print(f"\n4. ANALISIS DUPLIKAT:")

    # Exact duplicates
    exact_dups = df.duplicated(subset=['text'], keep=False).sum()
    unique_texts = df['text'].nunique()

    print(f"   - Total teks: {len(df)}")
    print(f"   - Teks unik: {unique_texts}")
    print(f"   - Baris dengan duplikat: {exact_dups}")
    print(f"   - Persentase duplikat: {exact_dups/len(df)*100:.2f}%")

    # Duplicates with different labels
    if exact_dups > 0:
        dup_texts = df[df.duplicated(subset=['text'], keep=False)]
        conflicting = dup_texts.groupby('text')['label'].nunique()
        conflicts = (conflicting > 1).sum()
        print(f"\n   - Teks duplikat dengan label BERBEDA: {conflicts}")

        if conflicts > 0:
            print(f"\n   CONTOH KONFLIK LABEL:")
            conflict_texts = conflicting[conflicting > 1].index[:3]
            for text in conflict_texts:
                subset = df[df['text'] == text][['text', 'label']]
                print(f"\n   Teks: '{text[:50]}...'")
                print(f"   Label: {subset['label'].tolist()}")

    return exact_dups, unique_texts

def analyze_language_quality(df):
    """Analyze if text is actually Javanese"""
    print(f"\n5. ANALISIS BAHASA:")

    texts = df['text'].astype(str)

    # Common Javanese words
    javanese_markers = ['ora', 'iku', 'sing', 'kang', 'kanggo', 'yen', 'wong', 'aku', 'kowe', 'dheweke',
                        'ana', 'dudu', 'wis', 'durung', 'arep', 'bisa', 'kudu', 'mung', 'kabeh', 'piye']

    # Common Indonesian words (that shouldn't dominate)
    indonesian_markers = ['yang', 'adalah', 'untuk', 'dengan', 'dari', 'ini', 'itu', 'akan', 'sudah', 'belum']

    # Common English words
    english_markers = ['the', 'is', 'are', 'you', 'they', 'this', 'that', 'have', 'has', 'will']

    def count_markers(text, markers):
        text_lower = text.lower()
        return sum(1 for m in markers if re.search(r'\b' + m + r'\b', text_lower))

    jav_counts = texts.apply(lambda x: count_markers(x, javanese_markers))
    ind_counts = texts.apply(lambda x: count_markers(x, indonesian_markers))
    eng_counts = texts.apply(lambda x: count_markers(x, english_markers))

    # Texts with NO Javanese markers
    no_javanese = (jav_counts == 0).sum()
    mostly_indonesian = ((ind_counts > jav_counts) & (jav_counts < 2)).sum()
    has_english = (eng_counts > 0).sum()

    print(f"   - Teks tanpa kata Jawa: {no_javanese} ({no_javanese/len(df)*100:.2f}%)")
    print(f"   - Teks dominan Indonesia: {mostly_indonesian} ({mostly_indonesian/len(df)*100:.2f}%)")
    print(f"   - Teks mengandung Inggris: {has_english} ({has_english/len(df)*100:.2f}%)")

    # Sample texts without Javanese
    if no_javanese > 0:
        print(f"\n   CONTOH TEKS TANPA KATA JAWA:")
        no_jav_samples = df[jav_counts == 0]['text'].head(5)
        for i, text in enumerate(no_jav_samples):
            print(f"   {i+1}. '{text[:80]}...'")

    return no_javanese, mostly_indonesian

def analyze_label_noise(df):
    """Analyze potential label noise"""
    print(f"\n6. ANALISIS NOISE LABEL:")

    # Keywords that typically indicate hate speech
    hate_keywords = ['bodoh', 'goblok', 'tolol', 'anjing', 'bangsat', 'kontol', 'memek',
                     'bajingan', 'kampret', 'tai', 'babi', 'monyet', 'idiot', 'stupid']

    # Keywords that typically indicate non-hate
    neutral_keywords = ['terima kasih', 'matur nuwun', 'apik', 'bagus', 'seneng', 'happy']

    def has_keywords(text, keywords):
        text_lower = str(text).lower()
        return any(k in text_lower for k in keywords)

    # Non-hate (label 0) with hate keywords
    non_hate = df[df['label'] == 0]
    non_hate_with_hate_words = non_hate['text'].apply(lambda x: has_keywords(x, hate_keywords)).sum()

    # Severe hate (label 3) with neutral keywords
    severe_hate = df[df['label'] == 3]
    severe_with_neutral = severe_hate['text'].apply(lambda x: has_keywords(x, neutral_keywords)).sum()

    print(f"   - 'Bukan Ujaran Kebencian' dengan kata kasar: {non_hate_with_hate_words}/{len(non_hate)} ({non_hate_with_hate_words/len(non_hate)*100:.2f}%)")
    print(f"   - 'Ujaran Berat' dengan kata netral: {severe_with_neutral}/{len(severe_hate)} ({severe_with_neutral/len(severe_hate)*100:.2f}%)")

    # Sample suspicious labels
    if non_hate_with_hate_words > 0:
        print(f"\n   CONTOH LABEL MENCURIGAKAN (non-hate tapi ada kata kasar):")
        suspicious = non_hate[non_hate['text'].apply(lambda x: has_keywords(x, hate_keywords))]['text'].head(5)
        for i, text in enumerate(suspicious):
            print(f"   {i+1}. '{text[:80]}...'")

    return non_hate_with_hate_words

def analyze_synthetic_patterns(df):
    """Detect potential synthetic/generated text patterns"""
    print(f"\n7. ANALISIS POLA SINTETIS:")

    texts = df['text'].astype(str)

    # Patterns that suggest synthetic generation
    patterns = {
        'numbered_list': r'^\d+\.',
        'bullet_points': r'^[-•*]',
        'repetitive_structure': r'(.{20,})\1',  # Repeated 20+ char sequences
        'formal_phrases': r'(dalam konteks|berdasarkan|menurut|sebagaimana)',
        'ai_markers': r'(sebagai AI|sebagai asisten|saya tidak bisa)',
    }

    for name, pattern in patterns.items():
        matches = texts.str.contains(pattern, regex=True, na=False).sum()
        if matches > 0:
            print(f"   - Pola '{name}': {matches} ({matches/len(df)*100:.2f}%)")

    # Check for overly similar texts
    from difflib import SequenceMatcher

    # Sample similarity check (expensive, so only check subset)
    sample_size = min(1000, len(df))
    sample_texts = texts.sample(sample_size, random_state=42).tolist()

    similar_pairs = 0
    for i in range(min(100, len(sample_texts))):
        for j in range(i+1, min(i+10, len(sample_texts))):
            ratio = SequenceMatcher(None, sample_texts[i], sample_texts[j]).ratio()
            if ratio > 0.8 and ratio < 1.0:  # Very similar but not exact
                similar_pairs += 1

    print(f"   - Pasangan teks sangat mirip (>80% similarity): {similar_pairs}")

def compare_datasets(df1, df2, name1, name2):
    """Compare two datasets"""
    print(f"\n{'='*60}")
    print(f"PERBANDINGAN: {name1} vs {name2}")
    print(f"{'='*60}")

    # Size comparison
    print(f"\n   {name1}: {len(df1)} sampel")
    print(f"   {name2}: {len(df2)} sampel")

    # Label distribution comparison
    dist1 = df1['label'].value_counts(normalize=True).sort_index()
    dist2 = df2['label'].value_counts(normalize=True).sort_index()

    print(f"\n   Perbandingan distribusi label:")
    for label in range(4):
        pct1 = dist1.get(label, 0) * 100
        pct2 = dist2.get(label, 0) * 100
        diff = pct2 - pct1
        print(f"   Label {label}: {name1}={pct1:.1f}% | {name2}={pct2:.1f}% | Diff={diff:+.1f}%")

    # Overlap analysis
    texts1 = set(df1['text'].astype(str))
    texts2 = set(df2['text'].astype(str))

    overlap = texts1 & texts2
    only_in_1 = texts1 - texts2
    only_in_2 = texts2 - texts1

    print(f"\n   Overlap teks:")
    print(f"   - Teks di kedua dataset: {len(overlap)}")
    print(f"   - Hanya di {name1}: {len(only_in_1)}")
    print(f"   - Hanya di {name2}: {len(only_in_2)}")

def main():
    print("="*60)
    print("AUDIT KUALITAS DATASET")
    print("Deteksi Ujaran Kebencian Bahasa Jawa")
    print("="*60)

    # Load datasets
    balanced = load_dataset("data/standardized/balanced_dataset.csv")
    train = load_dataset("data/standardized/train_dataset.csv")
    test = load_dataset("data/standardized/test_dataset.csv")
    augmented = load_dataset("data/standardized/augmented_dataset.csv")

    results = {}

    # Analyze main dataset
    if balanced is not None:
        analyze_basic_stats(balanced, "balanced_dataset.csv")
        results['label_dist'] = analyze_label_distribution(balanced)
        results['lengths'] = analyze_text_quality(balanced)
        results['duplicates'] = analyze_duplicates(balanced)
        results['language'] = analyze_language_quality(balanced)
        results['label_noise'] = analyze_label_noise(balanced)
        analyze_synthetic_patterns(balanced)

    # Compare train and test
    if train is not None and test is not None:
        compare_datasets(train, test, "train", "test")

        # Check for data leak
        train_texts = set(train['text'].astype(str))
        test_texts = set(test['text'].astype(str))
        leak = train_texts & test_texts

        print(f"\n   DATA LEAK CHECK:")
        print(f"   - Teks yang ada di KEDUA train dan test: {len(leak)}")
        if len(leak) > 0:
            print(f"   - PERINGATAN: Kemungkinan DATA LEAK!")

    # Analyze augmented if exists
    if augmented is not None:
        analyze_basic_stats(augmented, "augmented_dataset.csv")
        analyze_label_distribution(augmented)

    print(f"\n{'='*60}")
    print("AUDIT SELESAI")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
