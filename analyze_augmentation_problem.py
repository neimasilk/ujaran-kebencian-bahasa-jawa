#!/usr/bin/env python3
"""
Analisis masalah augmentasi data dan hasil eksperimen
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score

print("=== ANALISIS MASALAH AUGMENTASI DATA ===")
print()

# 1. Cek hasil eksperimen stable push
print("1. HASIL EKSPERIMEN STABLE PUSH:")
results_files = glob.glob('*results*.txt') + glob.glob('*stable*.txt') + glob.glob('results/*stable*')
print(f"   Files found: {results_files}")

# Cek folder tmp_stable untuk hasil training
if os.path.exists('tmp_stable_model_0'):
    print("   ✓ Model 0 training completed")
if os.path.exists('tmp_stable_model_1'):
    print("   ✓ Model 1 training completed")

print()

# 2. Analisis dataset augmented vs original
print("2. PERBANDINGAN DATASET:")

try:
    # Dataset original
    df_orig = pd.read_csv('data/processed/balanced_dataset.csv')
    print(f"   Original dataset shape: {df_orig.shape}")
    print(f"   Original label distribution:")
    print(f"   {df_orig['label'].value_counts()}")
    print(f"   Original balance ratio: {df_orig['label'].value_counts().min() / df_orig['label'].value_counts().max():.3f}")
    
except Exception as e:
    print(f"   ❌ Error reading original dataset: {e}")

print()

try:
    # Dataset augmented
    df_aug = pd.read_csv('data/augmented/augmented_dataset.csv')
    print(f"   Augmented dataset shape: {df_aug.shape}")
    print(f"   Augmented label distribution:")
    print(f"   {df_aug['final_label'].value_counts()}")
    
    # Cek NaN values
    nan_count = df_aug.isnull().sum().sum()
    print(f"   Total NaN values: {nan_count}")
    
    if nan_count > 0:
        print(f"   NaN per column:")
        for col in df_aug.columns:
            nan_col = df_aug[col].isnull().sum()
            if nan_col > 0:
                print(f"     {col}: {nan_col} NaN values")
    
    # Cek balance ratio setelah augmentasi
    clean_df = df_aug.dropna(subset=['final_label'])
    print(f"   Clean augmented shape: {clean_df.shape}")
    print(f"   Clean label distribution:")
    print(f"   {clean_df['final_label'].value_counts()}")
    balance_ratio = clean_df['final_label'].value_counts().min() / clean_df['final_label'].value_counts().max()
    print(f"   Clean balance ratio: {balance_ratio:.3f}")
    
    # Cek kualitas augmentasi
    if 'augmented' in df_aug.columns:
        aug_count = df_aug['augmented'].sum() if df_aug['augmented'].dtype == bool else len(df_aug[df_aug['augmented'] == True])
        print(f"   Augmented samples: {aug_count}")
        print(f"   Original samples: {len(df_aug) - aug_count}")
        print(f"   Augmentation ratio: {aug_count / (len(df_aug) - aug_count):.2f}")
    
except Exception as e:
    print(f"   ❌ Error reading augmented dataset: {e}")

print()

# 3. Analisis masalah potensial
print("3. ANALISIS MASALAH POTENSIAL:")

try:
    if 'df_orig' in locals() and 'clean_df' in locals():
        # Bandingkan distribusi panjang teks
        orig_lengths = df_orig['text'].str.len()
        aug_lengths = clean_df['text'].str.len()
        
        print(f"   Original text length - Mean: {orig_lengths.mean():.1f}, Std: {orig_lengths.std():.1f}")
        print(f"   Augmented text length - Mean: {aug_lengths.mean():.1f}, Std: {aug_lengths.std():.1f}")
        
        # Cek apakah ada teks yang terlalu pendek atau panjang
        short_texts = len(clean_df[clean_df['text'].str.len() < 10])
        long_texts = len(clean_df[clean_df['text'].str.len() > 500])
        print(f"   Texts < 10 chars: {short_texts}")
        print(f"   Texts > 500 chars: {long_texts}")
        
        # Cek duplikasi
        duplicates = clean_df['text'].duplicated().sum()
        print(f"   Duplicate texts: {duplicates}")
        
except Exception as e:
    print(f"   ❌ Error in analysis: {e}")

print()

# 4. Rekomendasi
print("4. REKOMENDASI:")
print("   • Jika balance ratio < 0.8, augmentasi mungkin tidak seimbang")
print("   • Jika ada banyak NaN, proses augmentasi bermasalah")
print("   • Jika ada banyak duplikasi, kualitas augmentasi rendah")
print("   • Jika panjang teks berubah drastis, augmentasi mungkin merusak konteks")
print()
print("=== SELESAI ===")