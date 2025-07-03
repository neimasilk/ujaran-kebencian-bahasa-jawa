#!/usr/bin/env python3
"""
Script untuk menganalisis distribusi dataset dan melakukan evaluasi model yang lebih representatif
dengan data yang diacak.
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import json
import os

def analyze_dataset_distribution():
    """Analisis distribusi dataset hasil-labeling.csv"""
    print("=== ANALISIS DISTRIBUSI DATASET ===")
    
    # Load dataset
    data_path = "src/data_collection/hasil-labeling.csv"
    print(f"Loading dataset dari: {data_path}")
    
    try:
        # Baca CSV dengan header yang benar
        df = pd.read_csv(data_path, header=None)
        print(f"Total data: {len(df)} baris")
        
        # Tentukan nama kolom berdasarkan struktur yang terlihat
        df.columns = ['text', 'sentiment', 'final_label', 'confidence_score', 'cost', 'method', 'extra1', 'extra2']
        
        # Konversi confidence_score ke numeric, handle string values
        df['confidence_score'] = pd.to_numeric(df['confidence_score'], errors='coerce')
        
        # Filter data dengan confidence >= 0.7 (seperti di training)
        df_filtered = df[df['confidence_score'] >= 0.7].copy()
        print(f"Data setelah filter confidence >= 0.7: {len(df_filtered)} baris")
        
        # Analisis distribusi label
        print("\n=== DISTRIBUSI LABEL KESELURUHAN ===")
        label_counts = df_filtered['final_label'].value_counts()
        print(label_counts)
        
        # Persentase
        print("\n=== PERSENTASE DISTRIBUSI ===")
        label_percentages = df_filtered['final_label'].value_counts(normalize=True) * 100
        for label, pct in label_percentages.items():
            print(f"{label}: {pct:.2f}%")
        
        # Analisis distribusi berdasarkan posisi dalam dataset
        print("\n=== ANALISIS DISTRIBUSI BERDASARKAN POSISI ===")
        
        # Bagi dataset menjadi 4 kuartil
        n = len(df_filtered)
        q1 = n // 4
        q2 = n // 2
        q3 = 3 * n // 4
        
        quarters = {
            'Q1 (0-25%)': df_filtered.iloc[:q1],
            'Q2 (25-50%)': df_filtered.iloc[q1:q2],
            'Q3 (50-75%)': df_filtered.iloc[q2:q3],
            'Q4 (75-100%)': df_filtered.iloc[q3:]
        }
        
        for quarter_name, quarter_data in quarters.items():
            print(f"\n{quarter_name}:")
            quarter_dist = quarter_data['final_label'].value_counts(normalize=True) * 100
            for label, pct in quarter_dist.items():
                print(f"  {label}: {pct:.2f}%")
        
        return df_filtered
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def create_balanced_evaluation_set(df, sample_size=2000):
    """Membuat set evaluasi yang seimbang dengan sampling stratified"""
    print(f"\n=== MEMBUAT SET EVALUASI SEIMBANG ({sample_size} samples) ===")
    
    # Hitung distribusi target yang lebih seimbang
    label_counts = df['final_label'].value_counts()
    print("Distribusi asli:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(df)*100:.2f}%)")
    
    # Stratified sampling untuk mendapatkan representasi yang lebih baik
    # Minimal 200 sample per kelas jika memungkinkan
    min_per_class = min(200, sample_size // len(label_counts))
    
    sampled_dfs = []
    remaining_samples = sample_size
    
    for label in label_counts.index:
        label_data = df[df['final_label'] == label]
        
        if len(label_data) >= min_per_class:
            # Ambil minimal sample per kelas
            n_samples = min(min_per_class, len(label_data), remaining_samples)
        else:
            # Jika data kurang dari minimal, ambil semua
            n_samples = min(len(label_data), remaining_samples)
        
        if n_samples > 0:
            sampled = label_data.sample(n=n_samples, random_state=42)
            sampled_dfs.append(sampled)
            remaining_samples -= n_samples
            print(f"  {label}: {n_samples} samples")
    
    # Gabungkan dan acak
    balanced_df = pd.concat(sampled_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nTotal samples dalam set evaluasi: {len(balanced_df)}")
    print("Distribusi final:")
    final_dist = balanced_df['final_label'].value_counts()
    for label, count in final_dist.items():
        print(f"  {label}: {count} ({count/len(balanced_df)*100:.2f}%)")
    
    return balanced_df

def save_evaluation_dataset(df, output_path="data/processed/balanced_evaluation_set.csv"):
    """Simpan dataset evaluasi yang seimbang"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nDataset evaluasi disimpan ke: {output_path}")
    return output_path

if __name__ == "__main__":
    # Analisis distribusi dataset
    df = analyze_dataset_distribution()
    
    if df is not None:
        # Buat set evaluasi yang seimbang
        balanced_eval_df = create_balanced_evaluation_set(df, sample_size=2000)
        
        # Simpan dataset evaluasi
        eval_path = save_evaluation_dataset(balanced_eval_df)
        
        print("\n=== REKOMENDASI ===")
        print("1. Dataset asli memiliki bias urutan - data awal mayoritas 'Bukan Ujaran Kebencian'")
        print("2. Evaluasi sebelumnya tidak representatif karena hanya mengambil 1000 data pertama")
        print("3. Gunakan dataset evaluasi yang sudah diseimbangkan untuk evaluasi yang lebih akurat")
        print("4. Pertimbangkan untuk melakukan stratified split saat training")
        print(f"5. Dataset evaluasi seimbang tersedia di: {eval_path}")