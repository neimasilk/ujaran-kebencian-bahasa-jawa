#!/usr/bin/env python3
"""
Skrip untuk mengecek hasil labeling di Google Drive
Memverifikasi apakah hasil sudah sesuai dengan ekspektasi
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from utils.cloud_checkpoint_manager import CloudCheckpointManager
from utils.logger import setup_logger

def check_google_drive_results():
    """
    Mengecek hasil labeling di Google Drive
    """
    logger = setup_logger('gdrive_checker')
    
    print("=== PENGECEKAN HASIL LABELING DI GOOGLE DRIVE ===")
    print()
    
    try:
        # Setup Google Drive manager
        print("🔐 Menghubungkan ke Google Drive...")
        checkpoint_manager = CloudCheckpointManager()
        
        # Check if authenticated
        if not checkpoint_manager.is_authenticated():
            print("❌ Google Drive tidak terautentikasi")
            print("💡 Jalankan setup Google Drive terlebih dahulu")
            return
        
        print("✅ Google Drive terkoneksi")
        print()
        
        # List files in the project folder
        print("📂 Mengecek file di Google Drive/ujaran-kebencian-datasets/...")
        
        # Check for checkpoint files
        checkpoint_files = checkpoint_manager.list_checkpoint_files()
        print(f"📋 Ditemukan {len(checkpoint_files)} file checkpoint:")
        
        for i, file_info in enumerate(checkpoint_files, 1):
            print(f"  {i}. {file_info['name']} (Modified: {file_info.get('modified', 'Unknown')})")
        
        print()
        
        # Check for hasil-labeling.csv
        print("🔍 Mencari file hasil-labeling.csv...")
        result_files = checkpoint_manager.list_result_files()
        
        if result_files:
            print(f"✅ Ditemukan {len(result_files)} file hasil:")
            for i, file_info in enumerate(result_files, 1):
                print(f"  {i}. {file_info['name']} (Size: {file_info.get('size', 'Unknown')} bytes)")
        else:
            print("⚠️ File hasil-labeling.csv belum ditemukan")
            print("💡 Proses mungkin masih berjalan atau belum selesai")
        
        print()
        
        # Download and analyze latest checkpoint
        print("📥 Mengunduh checkpoint terbaru untuk analisis...")
        
        # Try to get the latest checkpoint
        checkpoint_data = checkpoint_manager.download_checkpoint('labeling_raw-dataset_hasil-labeling')
        
        if checkpoint_data:
            print("✅ Checkpoint berhasil diunduh")
            analyze_checkpoint_data(checkpoint_data)
        else:
            print("⚠️ Tidak ada checkpoint yang ditemukan")
            print("💡 Proses mungkin baru dimulai")
        
    except Exception as e:
        logger.error(f"Error checking Google Drive: {e}")
        print(f"❌ Error: {e}")
        print("💡 Pastikan Google Drive sudah disetup dengan benar")

def analyze_checkpoint_data(checkpoint_data):
    """
    Menganalisis data checkpoint
    """
    print()
    print("=== ANALISIS DATA CHECKPOINT ===")
    
    try:
        # Extract information from checkpoint
        total_samples = checkpoint_data.get('total_samples', 0)
        processed_samples = checkpoint_data.get('processed_samples', 0)
        current_batch = checkpoint_data.get('current_batch', 0)
        total_batches = checkpoint_data.get('total_batches', 0)
        
        print(f"📊 Total sampel: {total_samples:,}")
        print(f"✅ Sampel diproses: {processed_samples:,}")
        print(f"🔄 Batch saat ini: {current_batch}/{total_batches}")
        
        if total_samples > 0:
            progress = (processed_samples / total_samples) * 100
            print(f"📈 Progress: {progress:.2f}%")
        
        # Check if there's labeled data
        labeled_data = checkpoint_data.get('labeled_data', [])
        if labeled_data:
            print(f"📝 Data berlabel: {len(labeled_data)} entri")
            
            # Analyze sample of labeled data
            print()
            print("=== SAMPLE DATA BERLABEL ===")
            
            # Show first 5 entries
            for i, entry in enumerate(labeled_data[:5], 1):
                text = entry.get('text', '')[:50] + '...' if len(entry.get('text', '')) > 50 else entry.get('text', '')
                category = entry.get('detailed_category', 'N/A')
                confidence = entry.get('confidence', 0)
                
                print(f"{i}. Text: {text}")
                print(f"   Kategori: {category}")
                print(f"   Confidence: {confidence:.3f}")
                print()
            
            # Analyze categories
            categories = {}
            confidences = []
            
            for entry in labeled_data:
                cat = entry.get('detailed_category', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1
                
                conf = entry.get('confidence', 0)
                if conf > 0:
                    confidences.append(conf)
            
            print("=== DISTRIBUSI KATEGORI ===")
            for cat, count in sorted(categories.items()):
                percentage = (count / len(labeled_data)) * 100
                print(f"{cat}: {count} ({percentage:.1f}%)")
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                min_confidence = min(confidences)
                max_confidence = max(confidences)
                
                print()
                print("=== STATISTIK CONFIDENCE ===")
                print(f"Rata-rata: {avg_confidence:.3f}")
                print(f"Minimum: {min_confidence:.3f}")
                print(f"Maksimum: {max_confidence:.3f}")
        
        else:
            print("⚠️ Belum ada data berlabel dalam checkpoint")
        
        # Check timestamp
        timestamp = checkpoint_data.get('timestamp')
        if timestamp:
            print()
            print(f"🕒 Checkpoint terakhir: {timestamp}")
        
    except Exception as e:
        print(f"❌ Error menganalisis checkpoint: {e}")

def check_local_results():
    """
    Mengecek hasil lokal jika ada
    """
    print()
    print("=== PENGECEKAN HASIL LOKAL ===")
    
    # Check for local hasil-labeling.csv
    local_result_file = Path('hasil-labeling.csv')
    if local_result_file.exists():
        print(f"✅ File lokal ditemukan: {local_result_file}")
        
        try:
            df = pd.read_csv(local_result_file)
            print(f"📊 Jumlah baris: {len(df):,}")
            print(f"📋 Kolom: {list(df.columns)}")
            
            # Show sample
            print()
            print("=== SAMPLE DATA LOKAL ===")
            print(df.head())
            
        except Exception as e:
            print(f"❌ Error membaca file lokal: {e}")
    else:
        print("⚠️ File hasil lokal belum ada")
    
    # Check for local checkpoints
    checkpoint_dir = Path('src/checkpoints')
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob('*.json'))
        print(f"📋 Checkpoint lokal: {len(checkpoint_files)} file")
        
        for file in checkpoint_files:
            print(f"  - {file.name}")
    else:
        print("📋 Tidak ada checkpoint lokal")

def main():
    """
    Fungsi utama
    """
    print("PENGECEKAN HASIL LABELING GOOGLE DRIVE")
    print("Memverifikasi hasil dan progress labeling")
    print("=" * 60)
    print()
    
    # Check Google Drive results
    check_google_drive_results()
    
    # Check local results
    check_local_results()
    
    print()
    print("=== KESIMPULAN ===")
    print("1. Jika ada data berlabel dengan confidence tinggi (>0.8), sistem bekerja dengan baik")
    print("2. Jika progress masih rendah (<1%), proses masih di tahap awal")
    print("3. Jika ada error, periksa koneksi Google Drive dan API key")
    print("4. File hasil final akan muncul setelah semua batch selesai")
    print()
    print("💡 Untuk monitoring real-time, gunakan: python calculate_progress.py")

if __name__ == "__main__":
    main()