#!/usr/bin/env python3
"""
Script untuk menghitung jumlah data dalam dataset
"""

import pandas as pd
import os

def count_dataset():
    dataset_path = "src/data_collection/raw-dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"File tidak ditemukan: {dataset_path}")
        return
    
    print("=== ANALISIS DATASET ===")
    print(f"File: {dataset_path}")
    print()
    
    # Baca dataset dengan header
    try:
        df_with_header = pd.read_csv(dataset_path)
        print(f"Dengan header:")
        print(f"  - Total baris: {len(df_with_header):,}")
        print(f"  - Kolom: {list(df_with_header.columns)}")
        print(f"  - Shape: {df_with_header.shape}")
        print()
    except Exception as e:
        print(f"Error membaca dengan header: {e}")
    
    # Baca dataset tanpa header (assume text,label)
    try:
        df_no_header = pd.read_csv(dataset_path, names=['text', 'label'])
        print(f"Tanpa header (assume text,label):")
        print(f"  - Total baris: {len(df_no_header):,}")
        print(f"  - Shape: {df_no_header.shape}")
        print()
    except Exception as e:
        print(f"Error membaca tanpa header: {e}")
    
    # Hitung distribusi label
    try:
        if 'label' in df_with_header.columns:
            label_dist = df_with_header['label'].value_counts()
        else:
            # Gunakan kolom kedua sebagai label
            label_dist = df_with_header.iloc[:, 1].value_counts()
        
        print("Distribusi Label:")
        for label, count in label_dist.items():
            percentage = (count / len(df_with_header)) * 100
            print(f"  - {label}: {count:,} ({percentage:.1f}%)")
        print()
    except Exception as e:
        print(f"Error menghitung distribusi: {e}")
    
    # Cek beberapa baris pertama
    print("5 Baris Pertama:")
    try:
        if 'label' in df_with_header.columns:
            display_df = df_with_header
        else:
            display_df = df_no_header
        
        for i, row in display_df.head().iterrows():
            print(f"  {i+1}. Text: '{row.iloc[0][:50]}...' | Label: '{row.iloc[1]}'")
    except Exception as e:
        print(f"Error menampilkan sample: {e}")
    
    print()
    print("=== ESTIMASI UNTUK LABELING ===")
    
    # Gunakan jumlah yang benar
    actual_count = len(df_with_header) if 'df_with_header' in locals() else len(df_no_header)
    
    # Berdasarkan testing: 3.71 detik per sampel
    avg_response_time = 3.71
    batch_size = 10
    
    total_batches = (actual_count + batch_size - 1) // batch_size  # Ceiling division
    total_time_seconds = total_batches * batch_size * avg_response_time
    total_time_hours = total_time_seconds / 3600
    
    print(f"Data aktual: {actual_count:,} sampel")
    print(f"Batch size: {batch_size}")
    print(f"Total batch: {total_batches:,}")
    print(f"Estimasi waktu: {total_time_hours:.1f} jam ({total_time_seconds/60:.0f} menit)")
    
    # Estimasi biaya
    tokens_per_sample = 275  # Average dari testing
    input_tokens = actual_count * 200
    output_tokens = actual_count * 75
    
    # Standard price
    input_cost_std = (input_tokens / 1_000_000) * 0.27
    output_cost_std = (output_tokens / 1_000_000) * 1.10
    total_cost_std = input_cost_std + output_cost_std
    
    # Discount price (50% off)
    input_cost_disc = (input_tokens / 1_000_000) * 0.135
    output_cost_disc = (output_tokens / 1_000_000) * 0.550
    total_cost_disc = input_cost_disc + output_cost_disc
    
    print(f"\nEstimasi Biaya:")
    print(f"  - Input tokens: {input_tokens/1_000_000:.2f}M")
    print(f"  - Output tokens: {output_tokens/1_000_000:.2f}M")
    print(f"  - Standard price: ${total_cost_std:.2f}")
    print(f"  - Discount price (50% off): ${total_cost_disc:.2f}")

if __name__ == "__main__":
    count_dataset()