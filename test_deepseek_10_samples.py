#!/usr/bin/env python3
"""
Script untuk menguji pelabelan DeepSeek dengan 10 sampel data
dan menampilkan hasil untuk review kualitas.
"""

import sys
import os
sys.path.append('src')

from data_collection.deepseek_labeling import DeepSeekLabeler
import pandas as pd
from dotenv import load_dotenv

def test_deepseek_labeling():
    """Test pelabelan dengan 10 sampel dan tampilkan hasil."""
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('DEEPSEEK_API_KEY')
    
    if not api_key:
        print("❌ API key tidak ditemukan di file .env")
        return
    
    print(f"✅ API key ditemukan: {api_key[:8]}...")
    print("✅ Menggunakan model: deepseek-chat (DeepSeek-V3)")
    print("✅ Base URL: https://api.deepseek.com")
    print()
    
    # Test koneksi API
    print("Testing koneksi API...")
    labeler = DeepSeekLabeler(api_key)
    test_result = labeler.call_deepseek_api("Test koneksi")
    
    if not test_result:
        print("❌ Test koneksi API gagal. Periksa API key dan koneksi internet.")
        return
    
    print("✅ Koneksi API berhasil!")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset_path = "src/data_collection/raw-dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset tidak ditemukan di {dataset_path}")
        print("Mencoba lokasi alternatif...")
        
        # Coba lokasi alternatif
        alt_paths = [
            "data/raw/raw-dataset.csv",
            "data/processed/labeling_template.csv"
        ]
        
        dataset_path = None
        for path in alt_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if dataset_path is None:
            print("❌ Dataset tidak ditemukan di lokasi manapun")
            return
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset dimuat dari: {dataset_path}")
    except Exception as e:
        print(f"❌ Gagal memuat dataset: {e}")
        return
    
    # Ambil 10 sampel
    sample_df = df.sample(n=min(10, len(df)), random_state=42).reset_index(drop=True)
    print(f"✅ Dataset dimuat: {len(df)} total, menggunakan {len(sample_df)} sampel")
    print()
    
    # Proses pelabelan
    print("Memulai pelabelan dengan 10 sampel...")
    results = []
    
    for idx, row in sample_df.iterrows():
        text = row['review']
        print(f"Processing {idx+1}/10: {text[:50]}...")
        
        result = labeler.call_deepseek_api(text)
        
        if result:
            results.append({
                'text': text,
                'deepseek_label': result.get('label', 'unknown'),
                'deepseek_confidence': result.get('confidence', 0),
                'deepseek_reasoning': result.get('reasoning', 'No reasoning provided')
            })
        else:
            print(f"❌ Gagal memproses data ke-{idx+1}")
            results.append({
                'text': text,
                'deepseek_label': 'error',
                'deepseek_confidence': 0,
                'deepseek_reasoning': 'API call failed'
            })
    
    # Buat DataFrame hasil
    result_df = pd.DataFrame(results)
    
    # Tampilkan hasil
    print("\n" + "=" * 80)
    print("=== HASIL PELABELAN (10 SAMPEL) ===")
    print("Format: [Text] -> [Label] (Confidence: X%) [Reasoning]")
    print("=" * 80)
    
    for idx, row in result_df.iterrows():
        text = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
        label = row['deepseek_label']
        confidence = row['deepseek_confidence']
        reasoning = row['deepseek_reasoning'][:150] + "..." if len(row['deepseek_reasoning']) > 150 else row['deepseek_reasoning']
        
        print(f"\n{idx+1}. [{text}]")
        print(f"   -> {label} (Confidence: {confidence}%)")
        print(f"   Reasoning: {reasoning}")
        print("-" * 80)
    
    # Analisis kualitas
    print("\n=== ANALISIS KUALITAS ===")
    label_counts = result_df['deepseek_label'].value_counts()
    avg_confidence = result_df['deepseek_confidence'].mean()
    print(f"Distribusi Label: {dict(label_counts)}")
    print(f"Rata-rata Confidence: {avg_confidence:.1f}%")
    
    # Simpan hasil
    os.makedirs('data/processed', exist_ok=True)
    result_df.to_csv('data/processed/deepseek_test_10_samples.csv', index=False)
    print(f"\n✅ Hasil test tersimpan di: data/processed/deepseek_test_10_samples.csv")
    
    print("\n=== LANGKAH SELANJUTNYA ===")
    print("1. Review hasil pelabelan di atas")
    print("2. Jika kualitas memuaskan, proses dataset lengkap")
    print("3. Jika perlu perbaikan, sesuaikan system prompt")
    print("4. Update training pipeline dengan dataset berlabel")

if __name__ == "__main__":
    test_deepseek_labeling()