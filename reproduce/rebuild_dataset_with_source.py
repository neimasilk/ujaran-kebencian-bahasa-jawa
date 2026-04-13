#!/usr/bin/env python3
"""
Step 1: Rebuild Dataset with Source Tracking

Adds a 'source' column to final_dataset.csv by matching texts
to phase3 (manual) or phase4 (synthetic) origins.

Input:
    data/improved/phase3_relabeled.csv     (manual data)
    data/improved/phase4_generated.csv     (synthetic data)
    data/cleaned/final_dataset.csv         (cleaned combined dataset)

Output:
    data/cleaned/final_dataset_with_source.csv
"""

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE3_PATH = os.path.join(PROJECT_ROOT, "data", "improved", "phase3_relabeled.csv")
PHASE4_PATH = os.path.join(PROJECT_ROOT, "data", "improved", "phase4_generated.csv")
FINAL_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "final_dataset.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "final_dataset_with_source.csv")


def normalize_text(text):
    """Normalize text for matching (strip whitespace, collapse spaces)."""
    if not isinstance(text, str):
        return ""
    import re
    text = re.sub(r"\s+", " ", text).strip()
    # Remove zero-width chars
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    return text


def main():
    print("=" * 60)
    print("STEP 1: REBUILD DATASET WITH SOURCE TRACKING")
    print("=" * 60)

    # Load source datasets
    print(f"\nLoading phase3 (manual): {PHASE3_PATH}")
    df_phase3 = pd.read_csv(PHASE3_PATH)
    print(f"  Rows: {len(df_phase3)}")

    print(f"Loading phase4 (synthetic): {PHASE4_PATH}")
    df_phase4 = pd.read_csv(PHASE4_PATH)
    print(f"  Rows: {len(df_phase4)}")

    # Build lookup sets of normalized texts
    manual_texts = set(normalize_text(t) for t in df_phase3["text"].tolist())
    synthetic_texts = set(normalize_text(t) for t in df_phase4["text"].tolist())

    print(f"\nUnique manual texts: {len(manual_texts)}")
    print(f"Unique synthetic texts: {len(synthetic_texts)}")

    # Check overlap
    overlap = manual_texts & synthetic_texts
    print(f"Overlap (in both): {len(overlap)}")

    # Load final dataset
    print(f"\nLoading final dataset: {FINAL_PATH}")
    df_final = pd.read_csv(FINAL_PATH)
    print(f"  Rows: {len(df_final)}")

    # Match each text to source
    sources = []
    n_manual = 0
    n_synthetic = 0
    n_unknown = 0

    for text in df_final["text"]:
        norm = normalize_text(text)
        if norm in manual_texts:
            sources.append("manual")
            n_manual += 1
        elif norm in synthetic_texts:
            sources.append("synthetic")
            n_synthetic += 1
        else:
            # Could be from either source but modified during cleaning
            # Try substring matching as fallback
            sources.append("unknown")
            n_unknown += 1

    df_final["source"] = sources

    # Report
    print(f"\n--- Source Attribution Results ---")
    print(f"  Manual:    {n_manual} ({n_manual/len(df_final)*100:.1f}%)")
    print(f"  Synthetic: {n_synthetic} ({n_synthetic/len(df_final)*100:.1f}%)")
    print(f"  Unknown:   {n_unknown} ({n_unknown/len(df_final)*100:.1f}%)")

    # If there are unknowns, try fuzzy matching
    if n_unknown > 0:
        print(f"\n  Attempting fuzzy match for {n_unknown} unknown texts...")
        unknown_idx = df_final[df_final["source"] == "unknown"].index
        resolved = 0
        for idx in unknown_idx:
            text = df_final.loc[idx, "text"]
            norm = normalize_text(text)
            # Check if text starts with any manual/synthetic text or vice versa
            found = False
            for mt in manual_texts:
                if norm in mt or mt in norm:
                    df_final.loc[idx, "source"] = "manual"
                    resolved += 1
                    found = True
                    break
            if not found:
                for st in synthetic_texts:
                    if norm in st or st in norm:
                        df_final.loc[idx, "source"] = "synthetic"
                        resolved += 1
                        break
        print(f"  Resolved {resolved} via substring matching")

        # Remaining unknowns - label as "manual" since they likely come
        # from the original collection phases before phase4
        remaining = (df_final["source"] == "unknown").sum()
        if remaining > 0:
            print(f"  Defaulting {remaining} remaining unknowns to 'manual'")
            df_final.loc[df_final["source"] == "unknown", "source"] = "manual"

    # Final counts
    final_manual = (df_final["source"] == "manual").sum()
    final_synthetic = (df_final["source"] == "synthetic").sum()
    print(f"\n--- Final Source Distribution ---")
    print(f"  Manual:    {final_manual} ({final_manual/len(df_final)*100:.1f}%)")
    print(f"  Synthetic: {final_synthetic} ({final_synthetic/len(df_final)*100:.1f}%)")
    print(f"  Total:     {len(df_final)}")

    # Verify total
    assert final_manual + final_synthetic == len(df_final), "Source counts don't add up!"

    # Distribution by source and label
    print(f"\n--- Label Distribution by Source ---")
    label_names = {0: "Not Hate", 1: "Light", 2: "Moderate", 3: "Severe"}
    for source in ["manual", "synthetic"]:
        subset = df_final[df_final["source"] == source]
        print(f"\n  {source.upper()} ({len(subset)} samples):")
        for label in sorted(subset["label"].unique()):
            count = (subset["label"] == label).sum()
            print(f"    {label_names.get(label, label)}: {count}")

    # Save
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Columns: {list(df_final.columns)}")

    return df_final


if __name__ == "__main__":
    main()
