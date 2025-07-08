#!/usr/bin/env python3
"""
Script untuk menganalisis distribusi dataset hasil labeling
dan membuat strategi balancing dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

def analyze_dataset_distribution(csv_path):
    """
    Menganalisis distribusi label dalam dataset
    """
    print(f"Menganalisis dataset: {csv_path}")
    
    # Baca dataset
    df = pd.read_csv(csv_path)
    print(f"Total data: {len(df)}")
    print(f"Kolom: {list(df.columns)}")
    
    # Analisis distribusi final_label
    print("\n=== DISTRIBUSI LABEL ===")
    label_counts = df['final_label'].value_counts()
    print(label_counts)
    
    # Hitung persentase
    print("\n=== PERSENTASE LABEL ===")
    label_percentages = df['final_label'].value_counts(normalize=True) * 100
    for label, percentage in label_percentages.items():
        print(f"{label}: {percentage:.2f}%")
    
    # Analisis distribusi label asli
    print("\n=== DISTRIBUSI LABEL ASLI ===")
    if 'label' in df.columns:
        original_label_counts = df['label'].value_counts()
        print(original_label_counts)
    
    # Analisis metode labeling
    print("\n=== METODE LABELING ===")
    if 'labeling_method' in df.columns:
        method_counts = df['labeling_method'].value_counts()
        print(method_counts)
    
    # Visualisasi
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Distribusi final_label
    plt.subplot(2, 2, 1)
    label_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribusi Final Label')
    plt.xlabel('Label')
    plt.ylabel('Jumlah')
    plt.xticks(rotation=45)
    
    # Plot 2: Pie chart
    plt.subplot(2, 2, 2)
    plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
    plt.title('Proporsi Label')
    
    # Plot 3: Distribusi confidence score
    if 'confidence_score' in df.columns:
        plt.subplot(2, 2, 3)
        df['confidence_score'].hist(bins=20, alpha=0.7, color='lightgreen')
        plt.title('Distribusi Confidence Score')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frekuensi')
    
    # Plot 4: Distribusi response time
    if 'response_time' in df.columns:
        plt.subplot(2, 2, 4)
        df['response_time'].hist(bins=20, alpha=0.7, color='orange')
        plt.title('Distribusi Response Time')
        plt.xlabel('Response Time (detik)')
        plt.ylabel('Frekuensi')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df, label_counts

def create_balanced_dataset(df, target_samples_per_class=None, strategy='undersample'):
    """
    Membuat dataset yang seimbang
    
    Args:
        df: DataFrame asli
        target_samples_per_class: Jumlah target sampel per kelas
        strategy: 'undersample', 'oversample', atau 'hybrid'
    """
    print(f"\n=== MEMBUAT DATASET SEIMBANG (Strategy: {strategy}) ===")
    
    # Hitung distribusi saat ini
    label_counts = df['final_label'].value_counts()
    print("Distribusi saat ini:")
    print(label_counts)
    
    # Tentukan target samples per class
    if target_samples_per_class is None:
        if strategy == 'undersample':
            target_samples_per_class = label_counts.min()
        elif strategy == 'oversample':
            target_samples_per_class = label_counts.max()
        else:  # hybrid
            target_samples_per_class = int(label_counts.median())
    
    print(f"\nTarget sampel per kelas: {target_samples_per_class}")
    
    balanced_dfs = []
    
    for label in label_counts.index:
        label_df = df[df['final_label'] == label].copy()
        current_count = len(label_df)
        
        if current_count > target_samples_per_class:
            # Undersample
            sampled_df = label_df.sample(n=target_samples_per_class, random_state=42)
            print(f"{label}: {current_count} -> {target_samples_per_class} (undersampled)")
        elif current_count < target_samples_per_class:
            # Oversample dengan duplikasi
            n_duplicates = target_samples_per_class - current_count
            duplicated_df = label_df.sample(n=n_duplicates, replace=True, random_state=42)
            sampled_df = pd.concat([label_df, duplicated_df], ignore_index=True)
            print(f"{label}: {current_count} -> {target_samples_per_class} (oversampled)")
        else:
            sampled_df = label_df
            print(f"{label}: {current_count} (tidak berubah)")
        
        balanced_dfs.append(sampled_df)
    
    # Gabungkan semua
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nDataset seimbang:")
    print(f"Total data: {len(balanced_df)}")
    print(balanced_df['final_label'].value_counts())
    
    return balanced_df

def create_standardized_dataset(df, output_dir='data/standardized'):
    """
    Membuat dataset yang sudah distandarisasi untuk eksperimen
    """
    print(f"\n=== MEMBUAT DATASET STANDAR ===")
    
    # Buat direktori output
    os.makedirs(output_dir, exist_ok=True)
    
    # Standarisasi kolom
    standardized_df = df[['text', 'final_label']].copy()
    
    # Mapping label ke numerik
    label_mapping = {
        'Bukan Ujaran Kebencian': 0,
        'Ujaran Kebencian - Ringan': 1,
        'Ujaran Kebencian - Sedang': 2,
        'Ujaran Kebencian - Berat': 3
    }
    
    standardized_df['label_numeric'] = standardized_df['final_label'].map(label_mapping)
    
    # Buat label binary juga (ujaran kebencian vs bukan)
    standardized_df['label_binary'] = (standardized_df['label_numeric'] > 0).astype(int)
    
    # Simpan dataset standar
    output_path = os.path.join(output_dir, 'balanced_dataset.csv')
    standardized_df.to_csv(output_path, index=False)
    print(f"Dataset standar disimpan: {output_path}")
    
    # Buat train-test split
    from sklearn.model_selection import train_test_split
    
    # Split untuk multi-class
    train_df, test_df = train_test_split(
        standardized_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=standardized_df['label_numeric']
    )
    
    # Simpan train-test split
    train_path = os.path.join(output_dir, 'train_dataset.csv')
    test_path = os.path.join(output_dir, 'test_dataset.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train dataset: {train_path} ({len(train_df)} sampel)")
    print(f"Test dataset: {test_path} ({len(test_df)} sampel)")
    
    # Analisis distribusi train-test
    print("\nDistribusi Train:")
    print(train_df['final_label'].value_counts())
    print("\nDistribusi Test:")
    print(test_df['final_label'].value_counts())
    
    # Simpan label mapping
    import json
    mapping_path = os.path.join(output_dir, 'label_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Label mapping disimpan: {mapping_path}")
    
    return standardized_df, train_df, test_df

def main():
    # Path ke dataset
    csv_path = 'src/data_collection/hasil-labeling.csv'
    
    if not os.path.exists(csv_path):
        print(f"File tidak ditemukan: {csv_path}")
        return
    
    # Analisis distribusi
    df, label_counts = analyze_dataset_distribution(csv_path)
    
    # Buat dataset seimbang dengan strategi undersample
    balanced_df = create_balanced_dataset(df, strategy='undersample')
    
    # Buat dataset standar untuk eksperimen
    standardized_df, train_df, test_df = create_standardized_dataset(balanced_df)
    
    print("\n=== RINGKASAN ===")
    print(f"Dataset asli: {len(df)} sampel")
    print(f"Dataset seimbang: {len(balanced_df)} sampel")
    print(f"Train set: {len(train_df)} sampel")
    print(f"Test set: {len(test_df)} sampel")
    
    print("\nDataset siap untuk eksperimen!")
    print("File yang dihasilkan:")
    print("- data/standardized/balanced_dataset.csv")
    print("- data/standardized/train_dataset.csv")
    print("- data/standardized/test_dataset.csv")
    print("- data/standardized/label_mapping.json")

if __name__ == '__main__':
    main()