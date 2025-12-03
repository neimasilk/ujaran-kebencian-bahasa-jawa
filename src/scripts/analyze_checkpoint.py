#!/usr/bin/env python3
"""
Skrip untuk menganalisis checkpoint hasil labeling
Menampilkan statistik dan sample data yang sudah diproses
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
import datetime

def analyze_checkpoint_file(checkpoint_path):
    """
    Menganalisis file checkpoint JSON
    """
    print("=== ANALISIS CHECKPOINT HASIL LABELING ===")
    print(f"File: {checkpoint_path}")
    print()
    
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Basic info
        checkpoint_id = data.get('checkpoint_id', 'Unknown')
        timestamp = data.get('timestamp', 0)
        processed_indices = data.get('processed_indices', [])
        labeled_data = data.get('labeled_data', [])
        metadata = data.get('metadata', {})
        
        print(f"📋 Checkpoint ID: {checkpoint_id}")
        
        # Convert timestamp
        if timestamp:
            dt = datetime.datetime.fromtimestamp(timestamp)
            print(f"🕒 Timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"📊 Total processed indices: {len(processed_indices):,}")
        print(f"📝 Labeled data entries: {len(labeled_data):,}")
        
        # Metadata info
        total_processed = metadata.get('total_processed', 0)
        output_file = metadata.get('output_file', 'Unknown')
        interrupted = metadata.get('interrupted', False)
        
        print(f"✅ Total processed: {total_processed:,}")
        print(f"📁 Output file: {output_file}")
        print(f"⚠️ Interrupted: {'Ya' if interrupted else 'Tidak'}")
        print()
        
        if labeled_data:
            analyze_labeled_data(labeled_data)
        else:
            print("⚠️ Tidak ada data berlabel ditemukan")
        
    except Exception as e:
        print(f"❌ Error reading checkpoint: {e}")

def analyze_labeled_data(labeled_data):
    """
    Menganalisis data yang sudah diberi label
    """
    print("=== ANALISIS DATA BERLABEL ===")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(labeled_data)
    
    print(f"📊 Total entries: {len(df):,}")
    print()
    
    # Analyze columns
    print("📋 Kolom yang tersedia:")
    for col in df.columns:
        print(f"  - {col}")
    print()
    
    # Analyze labels
    if 'label' in df.columns:
        print("=== DISTRIBUSI LABEL AWAL ===")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{label}: {count:,} ({percentage:.1f}%)")
        print()
    
    # Analyze final labels
    if 'final_label' in df.columns:
        print("=== DISTRIBUSI KATEGORI DETAIL ===")
        final_label_counts = df['final_label'].value_counts()
        for label, count in final_label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{label}: {count:,} ({percentage:.1f}%)")
        print()
    
    # Analyze confidence scores
    if 'confidence_score' in df.columns:
        print("=== STATISTIK CONFIDENCE SCORE ===")
        confidence_stats = df['confidence_score'].describe()
        print(f"Rata-rata: {confidence_stats['mean']:.3f}")
        print(f"Median: {confidence_stats['50%']:.3f}")
        print(f"Minimum: {confidence_stats['min']:.3f}")
        print(f"Maksimum: {confidence_stats['max']:.3f}")
        print(f"Std Dev: {confidence_stats['std']:.3f}")
        
        # Confidence distribution
        print()
        print("=== DISTRIBUSI CONFIDENCE ===")
        high_conf = len(df[df['confidence_score'] >= 0.8])
        med_conf = len(df[(df['confidence_score'] >= 0.6) & (df['confidence_score'] < 0.8)])
        low_conf = len(df[df['confidence_score'] < 0.6])
        
        print(f"Tinggi (≥0.8): {high_conf:,} ({(high_conf/len(df)*100):.1f}%)")
        print(f"Sedang (0.6-0.8): {med_conf:,} ({(med_conf/len(df)*100):.1f}%)")
        print(f"Rendah (<0.6): {low_conf:,} ({(low_conf/len(df)*100):.1f}%)")
        print()
    
    # Analyze response times
    if 'response_time' in df.columns:
        print("=== STATISTIK RESPONSE TIME ===")
        response_stats = df['response_time'].describe()
        print(f"Rata-rata: {response_stats['mean']:.3f} detik")
        print(f"Median: {response_stats['50%']:.3f} detik")
        print(f"Minimum: {response_stats['min']:.3f} detik")
        print(f"Maksimum: {response_stats['max']:.3f} detik")
        print()
    
    # Show sample data
    print("=== SAMPLE DATA (10 ENTRI PERTAMA) ===")
    for i, row in df.head(10).iterrows():
        print(f"{i+1}. Text: {row.get('text', '')[:60]}...")
        print(f"   Label awal: {row.get('label', 'N/A')}")
        print(f"   Kategori detail: {row.get('final_label', 'N/A')}")
        print(f"   Confidence: {row.get('confidence_score', 0):.3f}")
        print(f"   Response time: {row.get('response_time', 0):.3f}s")
        print()
    
    # Quality assessment
    print("=== PENILAIAN KUALITAS ===")
    
    if 'confidence_score' in df.columns:
        avg_confidence = df['confidence_score'].mean()
        if avg_confidence >= 0.8:
            quality = "🟢 SANGAT BAIK"
        elif avg_confidence >= 0.6:
            quality = "🟡 BAIK"
        else:
            quality = "🔴 PERLU PERHATIAN"
        
        print(f"Kualitas labeling: {quality}")
        print(f"Rata-rata confidence: {avg_confidence:.3f}")
    
    # Check for errors
    if 'error' in df.columns:
        error_count = df['error'].notna().sum()
        if error_count > 0:
            print(f"⚠️ Error ditemukan: {error_count} entri")
        else:
            print("✅ Tidak ada error ditemukan")
    
    print()

def main():
    """
    Fungsi utama
    """
    print("ANALISIS CHECKPOINT HASIL LABELING")
    print("Memverifikasi kualitas dan progress labeling")
    print("=" * 60)
    print()
    
    # Check for checkpoint files
    checkpoint_dir = Path('src/checkpoints')
    if not checkpoint_dir.exists():
        print("❌ Folder checkpoints tidak ditemukan")
        return
    
    checkpoint_files = list(checkpoint_dir.glob('*.json'))
    if not checkpoint_files:
        print("❌ Tidak ada file checkpoint ditemukan")
        return
    
    print(f"📋 Ditemukan {len(checkpoint_files)} file checkpoint:")
    for i, file in enumerate(checkpoint_files, 1):
        print(f"  {i}. {file.name}")
    print()
    
    # Analyze the main checkpoint file
    main_checkpoint = None
    for file in checkpoint_files:
        if 'raw-dataset' in file.name:
            main_checkpoint = file
            break
    
    if main_checkpoint:
        analyze_checkpoint_file(main_checkpoint)
    else:
        print("⚠️ File checkpoint utama tidak ditemukan")
        print("💡 Menganalisis file checkpoint pertama...")
        analyze_checkpoint_file(checkpoint_files[0])
    
    print()
    print("=== KESIMPULAN ===")
    print("✅ Sistem labeling berjalan dengan baik jika:")
    print("   - Ada data berlabel dengan confidence tinggi (>0.8)")
    print("   - Distribusi kategori masuk akal")
    print("   - Response time stabil")
    print("   - Tidak ada error")
    print()
    print("⚠️ Perlu perhatian jika:")
    print("   - Confidence rata-rata rendah (<0.6)")
    print("   - Banyak error")
    print("   - Response time terlalu lambat (>5 detik)")
    print()
    print("💡 Untuk monitoring real-time: python calculate_progress.py")

if __name__ == "__main__":
    main()