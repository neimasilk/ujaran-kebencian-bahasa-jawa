#!/usr/bin/env python3
"""
Skrip untuk memonitor proses labeling yang sedang berjalan
Mengecek file hasil dan checkpoint terbaru
"""

import os
import json
import pandas as pd
from pathlib import Path
import datetime
import time

def check_current_files():
    """
    Mengecek file-file yang ada saat ini
    """
    print("=== MONITORING PROSES LABELING AKTIF ===")
    print(f"Waktu check: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check for hasil-labeling.csv
    result_file = Path('hasil-labeling.csv')
    print(f"📄 File hasil utama (hasil-labeling.csv): {'✅ ADA' if result_file.exists() else '❌ BELUM ADA'}")
    
    if result_file.exists():
        stat = result_file.stat()
        size_mb = stat.st_size / (1024 * 1024)
        modified = datetime.datetime.fromtimestamp(stat.st_mtime)
        print(f"   📊 Ukuran: {size_mb:.2f} MB")
        print(f"   🕒 Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Try to read and analyze
        try:
            df = pd.read_csv(result_file)
            print(f"   📋 Jumlah baris: {len(df):,}")
            print(f"   📋 Kolom: {list(df.columns)}")
            
            if len(df) > 0:
                print("   📝 Sample data (3 baris pertama):")
                for i, row in df.head(3).iterrows():
                    text = str(row.get('text', ''))[:40] + '...' if len(str(row.get('text', ''))) > 40 else str(row.get('text', ''))
                    print(f"      {i+1}. {text} -> {row.get('detailed_category', 'N/A')}")
        except Exception as e:
            print(f"   ⚠️ Error membaca file: {e}")
    
    print()
    
    # Check for checkpoint files
    checkpoint_dirs = ['src/checkpoints']
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_path = Path(checkpoint_dir)
        if checkpoint_path.exists():
            print(f"📁 Checkpoint directory: {checkpoint_dir}")
            
            checkpoint_files = list(checkpoint_path.glob('*.json'))
            if checkpoint_files:
                print(f"   📋 Ditemukan {len(checkpoint_files)} file checkpoint:")
                
                for file in sorted(checkpoint_files, key=lambda x: x.stat().st_mtime, reverse=True):
                    stat = file.stat()
                    size_kb = stat.st_size / 1024
                    modified = datetime.datetime.fromtimestamp(stat.st_mtime)
                    
                    print(f"   📄 {file.name}")
                    print(f"      📊 Ukuran: {size_kb:.1f} KB")
                    print(f"      🕒 Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Try to read checkpoint info
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        processed = data.get('metadata', {}).get('total_processed', 0)
                        labeled_count = len(data.get('labeled_data', []))
                        
                        print(f"      ✅ Processed: {processed:,}")
                        print(f"      📝 Labeled: {labeled_count:,}")
                        
                        if labeled_count > 0:
                            # Show sample labeled data
                            labeled_data = data.get('labeled_data', [])
                            print(f"      📋 Sample labeled data (2 entri):")
                            
                            for i, entry in enumerate(labeled_data[-2:], 1):  # Show last 2 entries
                                text = entry.get('text', '')[:30] + '...' if len(entry.get('text', '')) > 30 else entry.get('text', '')
                                category = entry.get('final_label', 'N/A')
                                confidence = entry.get('confidence_score', 0)
                                
                                print(f"         {i}. {text}")
                                print(f"            Kategori: {category}")
                                print(f"            Confidence: {confidence:.3f}")
                    
                    except Exception as e:
                        print(f"      ⚠️ Error membaca checkpoint: {e}")
                    
                    print()
            else:
                print(f"   ❌ Tidak ada file checkpoint")
        else:
            print(f"📁 Checkpoint directory: {checkpoint_dir} - ❌ TIDAK ADA")
        
        print()
    
    # Check for any CSV files that might be results
    print("📄 File CSV lainnya:")
    csv_files = list(Path('.').glob('*.csv'))
    
    if csv_files:
        for csv_file in csv_files:
            if csv_file.name != 'hasil-labeling.csv':  # Skip the main result file
                stat = csv_file.stat()
                size_kb = stat.st_size / 1024
                modified = datetime.datetime.fromtimestamp(stat.st_mtime)
                
                print(f"   📄 {csv_file.name}")
                print(f"      📊 Ukuran: {size_kb:.1f} KB")
                print(f"      🕒 Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("   ❌ Tidak ada file CSV lainnya")
    
    print()

def estimate_progress_from_log():
    """
    Estimasi progress berdasarkan log yang diberikan user
    """
    print("=== ESTIMASI PROGRESS BERDASARKAN LOG ===")
    
    # Data dari log yang diberikan user
    log_info = {
        'total_batches': 2285,
        'completed_batches': 12,  # Batch 1-12 selesai berdasarkan log
        'samples_processed_before': 18907,  # Dari checkpoint lama
        'negative_samples_remaining': 22850,  # Dari log
        'avg_batch_time': 48.4,  # Dari analisis sebelumnya
        'avg_confidence': 0.87  # Estimasi dari log
    }
    
    print(f"📊 Total batch yang harus diproses: {log_info['total_batches']:,}")
    print(f"✅ Batch yang sudah selesai: {log_info['completed_batches']}")
    print(f"📈 Progress batch: {(log_info['completed_batches'] / log_info['total_batches'] * 100):.2f}%")
    print()
    
    print(f"📊 Sampel negatif yang harus diproses: {log_info['negative_samples_remaining']:,}")
    print(f"✅ Sampel yang sudah diproses sebelumnya: {log_info['samples_processed_before']:,}")
    
    # Estimate current processed samples
    samples_per_batch = 10  # Berdasarkan konfigurasi
    current_processed = log_info['completed_batches'] * samples_per_batch
    total_current = log_info['samples_processed_before'] + current_processed
    
    print(f"🔄 Sampel baru yang diproses: ~{current_processed}")
    print(f"📊 Total sampel diproses: ~{total_current:,}")
    print()
    
    # Time estimates
    remaining_batches = log_info['total_batches'] - log_info['completed_batches']
    remaining_time_seconds = remaining_batches * log_info['avg_batch_time']
    remaining_hours = remaining_time_seconds / 3600
    
    print(f"⏱️ Batch tersisa: {remaining_batches:,}")
    print(f"⏱️ Estimasi waktu tersisa: {remaining_hours:.1f} jam")
    
    # Estimated completion
    now = datetime.datetime.now()
    completion_time = now + datetime.timedelta(seconds=remaining_time_seconds)
    print(f"🎯 Estimasi selesai: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_system_status():
    """
    Mengecek status sistem
    """
    print("=== STATUS SISTEM ===")
    
    # Check if labeling process is running
    print("🔍 Mengecek proses yang berjalan...")
    
    # Check recent file modifications
    recent_files = []
    
    # Check all directories for recent files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith(('.json', '.csv', '.log')):
                file_path = Path(root) / file
                try:
                    stat = file_path.stat()
                    modified = datetime.datetime.fromtimestamp(stat.st_mtime)
                    
                    # Check if modified in last hour
                    if (datetime.datetime.now() - modified).total_seconds() < 3600:
                        recent_files.append((file_path, modified))
                except:
                    pass
    
    if recent_files:
        print(f"📝 File yang dimodifikasi dalam 1 jam terakhir: {len(recent_files)}")
        for file_path, modified in sorted(recent_files, key=lambda x: x[1], reverse=True):
            print(f"   📄 {file_path} - {modified.strftime('%H:%M:%S')}")
    else:
        print("⚠️ Tidak ada file yang dimodifikasi dalam 1 jam terakhir")
        print("💡 Proses mungkin sudah berhenti atau belum dimulai")
    
    print()

def main():
    """
    Fungsi utama
    """
    print("MONITORING PROSES LABELING AKTIF")
    print("Mengecek status dan hasil labeling real-time")
    print("=" * 60)
    print()
    
    # Check current files
    check_current_files()
    
    # Estimate progress
    estimate_progress_from_log()
    
    # Check system status
    check_system_status()
    
    print("=== KESIMPULAN ===")
    print("✅ Proses berjalan dengan baik jika:")
    print("   - Ada file checkpoint yang baru dimodifikasi")
    print("   - File hasil-labeling.csv mulai muncul")
    print("   - Checkpoint menunjukkan data berlabel dengan confidence tinggi")
    print()
    print("⚠️ Perlu perhatian jika:")
    print("   - Tidak ada file yang dimodifikasi dalam 1 jam terakhir")
    print("   - File hasil belum muncul setelah beberapa batch")
    print("   - Checkpoint tidak bertambah")
    print()
    print("💡 Untuk melihat log real-time, cek terminal tempat labeling.py berjalan")
    print("💡 Untuk menghentikan: Ctrl+C di terminal labeling")
    print("💡 Untuk melanjutkan: python labeling.py lagi")

if __name__ == "__main__":
    main()