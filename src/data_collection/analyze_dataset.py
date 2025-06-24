#!/usr/bin/env python3
"""
Script untuk menganalisis dataset dan mempersiapkan struktur pelabelan
sesuai dengan pedoman pelabelan ujaran kebencian Bahasa Jawa.
"""

import pandas as pd
import os
from pathlib import Path

def analyze_raw_dataset():
    """
    Menganalisis dataset mentah dan memberikan statistik dasar.
    """
    dataset_path = "src/data_collection/raw-dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset tidak ditemukan di {dataset_path}")
        return None
    
    try:
        # Baca dataset
        df = pd.read_csv(dataset_path, header=None, names=['text', 'label'])
        
        print("=== ANALISIS DATASET MENTAH ===")
        print(f"Total baris: {len(df)}")
        print(f"Kolom: {df.columns.tolist()}")
        print(f"\nDistribusi label saat ini:")
        print(df['label'].value_counts())
        
        print(f"\nContoh data (5 baris pertama):")
        for i, row in df.head().iterrows():
            print(f"{i+1}. Text: {row['text'][:50]}...")
            print(f"   Label: {row['label']}\n")
        
        # Cek missing values
        print(f"Missing values:")
        print(df.isnull().sum())
        
        return df
        
    except Exception as e:
        print(f"Error saat membaca dataset: {e}")
        return None

def create_labeling_template(df, sample_size=500):
    """
    Membuat template untuk pelabelan manual dengan sampel data.
    """
    if df is None:
        print("Dataset tidak tersedia untuk membuat template")
        return
    
    # Ambil sampel data untuk pelabelan manual
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Buat struktur baru sesuai pedoman pelabelan
    labeling_df = pd.DataFrame({
        'id': range(1, len(sample_df) + 1),
        'text': sample_df['text'].values,
        'old_label': sample_df['label'].values,
        'new_label': '',  # Akan diisi manual
        'confidence': '',  # Tingkat keyakinan pelabel (1-5)
        'notes': ''  # Catatan tambahan
    })
    
    # Simpan template
    output_path = "data/processed/labeling_template.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    labeling_df.to_csv(output_path, index=False)
    print(f"\n=== TEMPLATE PELABELAN DIBUAT ===")
    print(f"File: {output_path}")
    print(f"Jumlah sampel: {len(labeling_df)}")
    
    return labeling_df

def create_labeling_guidelines():
    """
    Membuat file panduan pelabelan yang mudah diakses.
    """
    guidelines = """
# PANDUAN PELABELAN UJARAN KEBENCIAN BAHASA JAWA

## Kategori Label:

1. **bukan_ujaran_kebencian**
   - Teks netral, positif, atau kritik membangun
   - Tidak mengandung hinaan, provokasi, atau ancaman

2. **ujaran_kebencian_ringan**
   - Sindiran halus, ejekan terselubung
   - Metafora budaya yang menyiratkan ketidaksukaan
   - Memerlukan pemahaman konteks budaya Jawa

3. **ujaran_kebencian_sedang**
   - Hinaan langsung, cercaan, bahasa kasar
   - Penggunaan ngoko yang tidak pantas
   - Lebih eksplisit dari kategori ringan

4. **ujaran_kebencian_berat**
   - Ancaman kekerasan fisik
   - Hasutan untuk melakukan kekerasan
   - Dehumanisasi, diskriminasi sistematis
   - Penghinaan ekstrem terkait SARA

## Tingkat Keyakinan (1-5):
1 = Sangat tidak yakin
2 = Tidak yakin
3 = Netral
4 = Yakin
5 = Sangat yakin

## Tips Pelabelan:
- Perhatikan tingkatan bahasa Jawa (Ngoko, Krama)
- Pertimbangkan konteks budaya dan metafora lokal
- Fokus pada maksud (intent) di balik ujaran
- Catat kasus sulit di kolom notes
"""
    
    guidelines_path = "data/processed/panduan_pelabelan.md"
    os.makedirs(os.path.dirname(guidelines_path), exist_ok=True)
    
    with open(guidelines_path, 'w', encoding='utf-8') as f:
        f.write(guidelines)
    
    print(f"\n=== PANDUAN PELABELAN DIBUAT ===")
    print(f"File: {guidelines_path}")

if __name__ == "__main__":
    print("Memulai analisis dataset dan persiapan pelabelan...\n")
    
    # Analisis dataset
    df = analyze_raw_dataset()
    
    if df is not None:
        # Buat template pelabelan
        create_labeling_template(df, sample_size=500)
        
        # Buat panduan pelabelan
        create_labeling_guidelines()
        
        print("\n=== SELESAI ===")
        print("Langkah selanjutnya:")
        print("1. Buka file data/processed/labeling_template.csv")
        print("2. Baca panduan di data/processed/panduan_pelabelan.md")
        print("3. Mulai proses pelabelan manual")
        print("4. Isi kolom 'new_label', 'confidence', dan 'notes'")
    else:
        print("Gagal menganalisis dataset.")