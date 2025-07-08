import pandas as pd
import numpy as np
from collections import Counter

def verify_dataset_balance():
    """
    Verifikasi distribusi label dalam dataset yang sudah dibalance
    """
    print("=== VERIFIKASI DATASET YANG SUDAH DIBALANCE ===")
    
    # Load dataset yang sudah dibalance
    balanced_df = pd.read_csv('data/standardized/balanced_dataset.csv')
    train_df = pd.read_csv('data/standardized/train_dataset.csv')
    test_df = pd.read_csv('data/standardized/test_dataset.csv')
    
    print(f"\n1. UKURAN DATASET:")
    print(f"   - Balanced Dataset: {len(balanced_df):,} sampel")
    print(f"   - Train Dataset: {len(train_df):,} sampel")
    print(f"   - Test Dataset: {len(test_df):,} sampel")
    
    print(f"\n2. DISTRIBUSI LABEL - BALANCED DATASET:")
    label_counts = balanced_df['final_label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(balanced_df)) * 100
        print(f"   - {label}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n3. DISTRIBUSI LABEL NUMERIK - BALANCED DATASET:")
    numeric_counts = balanced_df['label_numeric'].value_counts().sort_index()
    for label_num, count in numeric_counts.items():
        percentage = (count / len(balanced_df)) * 100
        print(f"   - Label {label_num}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n4. DISTRIBUSI LABEL - TRAIN DATASET:")
    train_label_counts = train_df['final_label'].value_counts()
    for label, count in train_label_counts.items():
        percentage = (count / len(train_df)) * 100
        print(f"   - {label}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n5. DISTRIBUSI LABEL - TEST DATASET:")
    test_label_counts = test_df['final_label'].value_counts()
    for label, count in test_label_counts.items():
        percentage = (count / len(test_df)) * 100
        print(f"   - {label}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n6. CONTOH DATA PER KATEGORI:")
    for label_num in sorted(balanced_df['label_numeric'].unique()):
        label_name = balanced_df[balanced_df['label_numeric'] == label_num]['final_label'].iloc[0]
        sample_texts = balanced_df[balanced_df['label_numeric'] == label_num]['text'].head(3).tolist()
        print(f"\n   Label {label_num} ({label_name}):")
        for i, text in enumerate(sample_texts, 1):
            print(f"   {i}. {text[:100]}...")
    
    print(f"\n7. STATISTIK PANJANG TEKS:")
    balanced_df['text_length'] = balanced_df['text'].str.len()
    print(f"   - Rata-rata: {balanced_df['text_length'].mean():.1f} karakter")
    print(f"   - Median: {balanced_df['text_length'].median():.1f} karakter")
    print(f"   - Min: {balanced_df['text_length'].min()} karakter")
    print(f"   - Max: {balanced_df['text_length'].max()} karakter")
    
    print(f"\n8. VALIDASI KONSISTENSI:")
    # Cek apakah label_binary konsisten dengan label_numeric
    binary_check = (balanced_df['label_binary'] == (balanced_df['label_numeric'] > 0).astype(int)).all()
    print(f"   - Label binary konsisten: {binary_check}")
    
    # Cek apakah ada missing values
    missing_text = balanced_df['text'].isna().sum()
    missing_label = balanced_df['final_label'].isna().sum()
    print(f"   - Missing text: {missing_text}")
    print(f"   - Missing label: {missing_label}")
    
    # Cek duplikasi
    duplicates = balanced_df.duplicated(subset=['text']).sum()
    print(f"   - Duplikasi teks: {duplicates}")
    
    print(f"\n=== DATASET SIAP UNTUK EKSPERIMEN ===")
    print(f"Dataset telah dibalance dan distandardisasi dengan sukses!")
    print(f"Total sampel: {len(balanced_df):,}")
    print(f"Distribusi seimbang: {len(set(numeric_counts.values)) == 1}")
    
if __name__ == "__main__":
    verify_dataset_balance()