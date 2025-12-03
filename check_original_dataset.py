#!/usr/bin/env python3
"""
Cek dataset original yang bagus untuk eksperimen tanpa augmentasi
"""

import pandas as pd
import os

print("=== MENCARI DATASET ORIGINAL YANG BAGUS ===")
print()

# Cek semua dataset yang tersedia
datasets_to_check = [
    'data/standardized/balanced_dataset.csv',
    'data/standardized/train_dataset.csv', 
    'data/standardized/test_dataset.csv',
    'data/processed/balanced_evaluation_set.csv'
]

for dataset_path in datasets_to_check:
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            print(f"📁 {dataset_path}")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            
            # Cek kolom label yang mungkin
            label_cols = [col for col in df.columns if 'label' in col.lower()]
            if label_cols:
                main_label = label_cols[0]
                print(f"   Label column: {main_label}")
                print(f"   Label distribution:")
                for label, count in df[main_label].value_counts().items():
                    print(f"     {label}: {count}")
                
                # Cek balance ratio
                counts = df[main_label].value_counts()
                balance_ratio = counts.min() / counts.max()
                print(f"   Balance ratio: {balance_ratio:.3f}")
                
                # Cek NaN
                nan_count = df.isnull().sum().sum()
                print(f"   Total NaN: {nan_count}")
                
                # Cek sample data
                print(f"   Sample text length: {df['text'].str.len().mean():.1f} chars")
                
            print()
            
        except Exception as e:
            print(f"❌ Error reading {dataset_path}: {e}")
            print()
    else:
        print(f"❌ File not found: {dataset_path}")
        print()

print("=== REKOMENDASI DATASET TERBAIK ===")
print("Pilih dataset dengan:")
print("• Balance ratio > 0.8")
print("• Minimal NaN values")
print("• Ukuran yang cukup besar (>10k samples)")
print("• Panjang teks yang wajar (50-300 chars)")