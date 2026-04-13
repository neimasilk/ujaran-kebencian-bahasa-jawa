#!/usr/bin/env python3
"""
Fase 0: Aggressive Dataset Cleanup

Cleans phase5_deepseek_relabeled.csv by:
1. Removing exact-match duplicates
2. Removing texts < 20 characters
3. Detecting and removing non-Javanese texts (heuristic-based)
4. Fixing encoding issues
5. Validating label distribution

Input:  data/improved/phase5_deepseek_relabeled.csv (10,019 rows)
Output: data/cleaned/final_dataset.csv (~9,500-9,800 rows)
        data/cleaned/cleaning_report.json
"""

import os
import sys
import json
import re
from collections import Counter

import pandas as pd
import numpy as np

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "improved", "phase5_deepseek_relabeled.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_dataset.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "cleaning_report.json")

# Label names
LABEL_NAMES = {
    0: "Bukan Ujaran Kebencian",
    1: "Ujaran Kebencian - Ringan",
    2: "Ujaran Kebencian - Sedang",
    3: "Ujaran Kebencian - Berat",
}

# Common Javanese words for language detection heuristic.
# Includes function words, pronouns, conjunctions, common verbs/adjectives.
JAVANESE_WORDS = {
    # Pronouns
    "aku", "kowe", "dheweke", "sampeyan", "panjenengan", "kula", "ingsun",
    "awake", "deweke", "kita", "padha", "kabeh",
    # Function words / particles
    "iku", "iki", "kuwi", "kui", "kae", "sing", "kang", "ingkang",
    "ora", "mboten", "dudu", "aja", "gak", "ra", "nggak",
    "wis", "uwis", "durung", "dereng", "lagi", "isih", "tansah",
    "ana", "wonten", "nang", "ing", "saka", "saking", "menyang",
    "lan", "karo", "kalih", "kanggo", "supaya", "amarga", "merga", "mergo",
    "yen", "menawa", "nalika", "sawise", "sakdurunge",
    "mung", "namung", "wae", "bae", "pisan", "banget",
    # Common verbs
    "ngomong", "kandha", "ngerti", "weruh", "ndelok", "ndeleng",
    "lunga", "teka", "mlebu", "metu", "mangan", "ngombe",
    "turu", "tangi", "gawe", "nggawe", "njupuk", "nggawa",
    "seneng", "sengit", "wedi", "isin", "nesu", "susah",
    "bisa", "saged", "kudu", "perlu", "pengen", "arep",
    "dadi", "dados", "melu", "nderek",
    # Common adjectives/adverbs
    "apik", "elek", "gedhe", "cilik", "akeh", "sithik", "sitik",
    "adoh", "cedhak", "dhuwur", "endhek", "ayu", "bagus",
    "larang", "murah", "angel", "gampang", "cepet", "alon",
    "anyar", "lawas", "enom", "tuwa", "ireng", "putih", "abang",
    # Common nouns
    "wong", "uwong", "bocah", "anak", "bapak", "ibu", "sedulur",
    "omah", "dalan", "banyu", "pangan", "dhuwit", "ati",
    "dina", "wayah", "taun", "sasi",
    # Javanese-specific
    "monggo", "mangga", "nuwun", "matur", "ndherek",
    "ngoko", "krama", "madya",
    # Slang / common in social media
    "jancok", "jancuk", "asu", "tempek", "cuk", "goblok",
    "cok", "kon", "lur", "bro", "rek",
    # Negation / affirmation
    "iya", "iyo", "yo", "enggih", "nggih", "ora", "mboten",
    # Question words
    "apa", "sapa", "piye", "kapan", "ngendi", "pira",
    # Additional common
    "bab", "babagan", "prekara", "masalah",
    "kudu", "mesti", "pancen",
    "dhewe", "dewe", "bareng",
    "niku", "niko", "niki",
    "mpun", "sampun",
    "nduk", "ndak", "ndhuk",
    "kok", "lho", "lha", "tha", "tho",
    # More common Javanese missed in initial list
    "nanging", "bakal", "malah", "tambah", "gede", "regane",
    "mumbul", "ambruk", "pecah", "cures", "pecek", "pejah",
    "dicekik", "dipenjara", "dicekel", "dibebas", "ditangkep",
    "sumangga", "sami", "rembugan", "kosok", "wangsul",
    "kasucian", "panyembah", "dadekke", "didadekke",
    "diunekne", "diomongke", "dikarepke", "dikandakke",
    "saiki", "mbesuk", "wingi", "sesuk", "mengko",
    "mbok", "aja", "monggo", "ndang", "cepet",
    "temen", "tenan", "tenan", "temenan",
    "nganti", "tekane", "tekan", "nganti",
    "mlebu", "metu", "mulih", "bali", "budhal",
    "tuku", "adol", "bayar", "utang",
    "seneng", "susah", "bungah", "sedhih",
    "waton", "sokur", "untung", "rugi",
    "rakyat", "negara", "korupsi", "pemerintah",  # common loanwords in Javanese discourse
    "reformasi", "politisi", "pemimpin", "menteri",
}

# Words that strongly indicate NON-Javanese text (English/Indonesian only, not code-switching)
NON_JAVANESE_INDICATORS = {
    "the", "is", "are", "was", "were", "have", "has", "been", "will",
    "would", "should", "could", "with", "from", "that", "this", "they",
    "their", "them", "you", "your", "what", "when", "where", "which",
    "because", "about", "after", "before", "between",
}


def compute_javanese_ratio(text: str) -> float:
    """Compute the ratio of Javanese words in the text."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) == 0:
        return 0.0
    jav_count = sum(1 for w in words if w in JAVANESE_WORDS)
    return jav_count / len(words)


def compute_english_ratio(text: str) -> float:
    """Compute the ratio of English indicator words in the text."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) == 0:
        return 0.0
    eng_count = sum(1 for w in words if w in NON_JAVANESE_INDICATORS)
    return eng_count / len(words)


def is_likely_non_javanese(text: str) -> bool:
    """
    Heuristic to detect non-Javanese text.
    Returns True if the text is likely NOT Javanese.
    """
    # Full CAPSLOCK texts that are Indonesian/English
    if text == text.upper() and len(text) > 30:
        jav_ratio = compute_javanese_ratio(text)
        if jav_ratio < 0.15:
            return True

    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 3:
        return False  # Too short to judge

    jav_ratio = compute_javanese_ratio(text)
    eng_ratio = compute_english_ratio(text)

    # If high English content and very low Javanese
    if eng_ratio > 0.3 and jav_ratio < 0.1:
        return True

    # If text has zero Javanese words and is long enough
    if jav_ratio == 0.0 and len(words) >= 8:
        return True

    return False


def fix_encoding(text: str) -> str:
    """Fix common encoding issues in text."""
    if not isinstance(text, str):
        return text
    # Replace replacement character
    text = text.replace("\ufffd", "")
    # Fix common mojibake for Javanese diacritics
    text = text.replace("Ã©", "é")
    text = text.replace("Ã¨", "è")
    text = text.replace("Ã‰", "É")
    # Remove zero-width characters
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = text.replace("\ufeff", "")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataset():
    """Main cleaning pipeline."""
    print("=" * 60)
    print("FASE 0: AGGRESSIVE DATASET CLEANUP")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    initial_count = len(df)
    print(f"Initial samples: {initial_count}")
    print(f"Initial label distribution:")
    for label, name in LABEL_NAMES.items():
        count = (df["label"] == label).sum()
        print(f"  {name}: {count} ({count/initial_count*100:.1f}%)")

    report = {
        "initial_count": initial_count,
        "initial_distribution": df["label"].value_counts().sort_index().to_dict(),
        "steps": [],
    }

    # Step 1: Fix encoding
    print("\n--- Step 1: Fix Encoding Issues ---")
    df["text"] = df["text"].apply(fix_encoding)
    # Remove rows where text became empty after fixing
    empty_after_fix = (df["text"].str.len() == 0).sum()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    print(f"  Fixed encoding in all texts")
    print(f"  Removed {empty_after_fix} empty texts after fix")
    report["steps"].append({
        "step": "fix_encoding",
        "removed": int(empty_after_fix),
        "remaining": len(df),
    })

    # Step 2: Remove exact duplicates
    print("\n--- Step 2: Remove Exact Duplicates ---")
    dup_mask = df.duplicated(subset=["text"], keep="first")
    n_duplicates = dup_mask.sum()
    df = df[~dup_mask].reset_index(drop=True)
    print(f"  Removed {n_duplicates} exact duplicates")
    print(f"  Remaining: {len(df)}")
    report["steps"].append({
        "step": "remove_duplicates",
        "removed": int(n_duplicates),
        "remaining": len(df),
    })

    # Step 3: Remove short texts (< 20 characters)
    print("\n--- Step 3: Remove Short Texts (< 20 chars) ---")
    short_mask = df["text"].str.len() < 20
    n_short = short_mask.sum()
    # Show some examples
    short_examples = df[short_mask].head(10)[["text", "label"]].values.tolist()
    print(f"  Examples of removed short texts:")
    for text, label in short_examples[:5]:
        print(f"    [{label}] \"{text}\"")
    df = df[~short_mask].reset_index(drop=True)
    print(f"  Removed {n_short} short texts")
    print(f"  Remaining: {len(df)}")
    report["steps"].append({
        "step": "remove_short_texts",
        "threshold": 20,
        "removed": int(n_short),
        "remaining": len(df),
        "examples": short_examples[:5],
    })

    # Step 4: Detect and remove non-Javanese texts
    print("\n--- Step 4: Remove Non-Javanese Texts ---")
    non_jav_mask = df["text"].apply(is_likely_non_javanese)
    n_non_jav = non_jav_mask.sum()
    non_jav_examples = df[non_jav_mask].head(15)[["text", "label"]].values.tolist()
    print(f"  Examples of flagged non-Javanese texts:")
    for text, label in non_jav_examples[:8]:
        print(f"    [{label}] \"{text[:80]}...\"" if len(text) > 80 else f"    [{label}] \"{text}\"")
    df = df[~non_jav_mask].reset_index(drop=True)
    print(f"  Removed {n_non_jav} non-Javanese texts")
    print(f"  Remaining: {len(df)}")
    report["steps"].append({
        "step": "remove_non_javanese",
        "removed": int(n_non_jav),
        "remaining": len(df),
        "examples": [[t[:100], l] for t, l in non_jav_examples[:8]],
    })

    # Step 5: Remove near-duplicate spaced-out texts (e.g., "T R A N I E S A R E S H I T")
    print("\n--- Step 5: Remove Spaced-Out/Obfuscated Texts ---")
    spaced_pattern = re.compile(r"^([A-Z] ){5,}")
    spaced_mask = df["text"].apply(lambda t: bool(spaced_pattern.match(t)))
    n_spaced = spaced_mask.sum()
    if n_spaced > 0:
        spaced_examples = df[spaced_mask].head(5)[["text", "label"]].values.tolist()
        for text, label in spaced_examples:
            print(f"    [{label}] \"{text[:60]}\"")
    df = df[~spaced_mask].reset_index(drop=True)
    print(f"  Removed {n_spaced} spaced-out texts")
    print(f"  Remaining: {len(df)}")
    report["steps"].append({
        "step": "remove_spaced_out",
        "removed": int(n_spaced),
        "remaining": len(df),
    })

    # Final validation
    final_count = len(df)
    total_removed = initial_count - final_count
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Initial:  {initial_count}")
    print(f"Final:    {final_count}")
    print(f"Removed:  {total_removed} ({total_removed/initial_count*100:.1f}%)")
    print(f"\nFinal label distribution:")
    final_dist = {}
    for label, name in LABEL_NAMES.items():
        count = (df["label"] == label).sum()
        pct = count / final_count * 100
        final_dist[label] = count
        print(f"  {name}: {count} ({pct:.1f}%)")

    # Balance check
    min_count = min(final_dist.values())
    max_count = max(final_dist.values())
    print(f"\nBalance ratio (min/max): {min_count/max_count:.3f}")

    # Save cleaned dataset
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned dataset: {OUTPUT_PATH}")

    # Save report
    report["final_count"] = final_count
    report["total_removed"] = total_removed
    report["final_distribution"] = {int(k): int(v) for k, v in final_dist.items()}
    report["removal_percentage"] = round(total_removed / initial_count * 100, 2)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved cleaning report: {REPORT_PATH}")

    return df


if __name__ == "__main__":
    df = clean_dataset()
