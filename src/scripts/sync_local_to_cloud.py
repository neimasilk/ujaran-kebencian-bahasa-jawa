#!/usr/bin/env python3
"""
Script untuk sinkronisasi data lokal ke Google Drive setelah cloud dibersihkan

Usage:
    python sync_local_to_cloud.py
    python sync_local_to_cloud.py --checkpoint-only
    python sync_local_to_cloud.py --results-only
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from config.settings import Settings
from utils.cloud_checkpoint_manager import CloudCheckpointManager
import json

def find_local_checkpoints(checkpoint_dir="src/checkpoints"):
    """
    Cari semua checkpoint lokal yang ada
    
    Returns:
        List[str]: List path ke checkpoint files
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory tidak ditemukan: {checkpoint_dir}")
        return []
    
    checkpoint_files = list(checkpoint_path.glob("*.json"))
    print(f"📁 Ditemukan {len(checkpoint_files)} checkpoint lokal:")
    for file in checkpoint_files:
        print(f"   - {file.name}")
    
    return checkpoint_files

def find_local_results(results_pattern="hasil-labeling*.csv"):
    """
    Cari semua file hasil labeling lokal
    
    Returns:
        List[str]: List path ke result files
    """
    current_dir = Path(".")
    result_files = list(current_dir.glob(results_pattern))
    
    print(f"📊 Ditemukan {len(result_files)} file hasil labeling:")
    for file in result_files:
        print(f"   - {file.name}")
    
    return result_files

def upload_checkpoints(cloud_manager, checkpoint_files):
    """
    Upload semua checkpoint lokal ke cloud
    
    Args:
        cloud_manager: CloudCheckpointManager instance
        checkpoint_files: List checkpoint files
        
    Returns:
        int: Jumlah checkpoint yang berhasil diupload
    """
    print("\n🔄 Mengupload checkpoints ke Google Drive...")
    
    success_count = 0
    for checkpoint_file in checkpoint_files:
        try:
            # Load checkpoint data
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # Extract checkpoint_id from filename or data
            checkpoint_id = checkpoint_data.get('checkpoint_id', checkpoint_file.stem)
            
            # Upload to cloud
            success = cloud_manager.save_checkpoint(checkpoint_data, checkpoint_id)
            if success:
                print(f"   ✅ {checkpoint_file.name} -> {checkpoint_id}")
                success_count += 1
            else:
                print(f"   ❌ Gagal upload {checkpoint_file.name}")
                
        except Exception as e:
            print(f"   ❌ Error processing {checkpoint_file.name}: {e}")
    
    print(f"\n📤 Berhasil upload {success_count}/{len(checkpoint_files)} checkpoints")
    return success_count

def upload_results(cloud_manager, result_files):
    """
    Upload semua file hasil labeling ke cloud
    
    Args:
        cloud_manager: CloudCheckpointManager instance
        result_files: List result files
        
    Returns:
        int: Jumlah file hasil yang berhasil diupload
    """
    print("\n🔄 Mengupload hasil labeling ke Google Drive...")
    
    success_count = 0
    for result_file in result_files:
        try:
            # Generate cloud filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cloud_filename = f"{result_file.stem}_{timestamp}.csv"
            
            # Upload to cloud
            success = cloud_manager.upload_dataset(str(result_file), cloud_filename)
            if success:
                print(f"   ✅ {result_file.name} -> {cloud_filename}")
                success_count += 1
            else:
                print(f"   ❌ Gagal upload {result_file.name}")
                
        except Exception as e:
            print(f"   ❌ Error uploading {result_file.name}: {e}")
    
    print(f"\n📤 Berhasil upload {success_count}/{len(result_files)} file hasil")
    return success_count

def main():
    parser = argparse.ArgumentParser(description='Sinkronisasi data lokal ke Google Drive')
    parser.add_argument('--checkpoint-only', action='store_true', 
                       help='Hanya upload checkpoints')
    parser.add_argument('--results-only', action='store_true', 
                       help='Hanya upload file hasil labeling')
    
    args = parser.parse_args()
    
    print("🔄 SINKRONISASI DATA LOKAL KE GOOGLE DRIVE")
    print("=" * 50)
    
    try:
        # Initialize settings and cloud manager
        print("🔧 Inisialisasi...")
        settings = Settings()
        cloud_manager = CloudCheckpointManager(settings)
        
        # Authenticate
        print("🔐 Autentikasi Google Drive...")
        if not cloud_manager.authenticate():
            print("❌ Gagal autentikasi Google Drive")
            return 1
        
        print("✅ Berhasil terhubung ke Google Drive")
        
        total_uploaded = 0
        
        # Upload checkpoints
        if not args.results_only:
            checkpoint_files = find_local_checkpoints()
            if checkpoint_files:
                uploaded_checkpoints = upload_checkpoints(cloud_manager, checkpoint_files)
                total_uploaded += uploaded_checkpoints
            else:
                print("📁 Tidak ada checkpoint lokal yang ditemukan")
        
        # Upload results
        if not args.checkpoint_only:
            result_files = find_local_results()
            if result_files:
                uploaded_results = upload_results(cloud_manager, result_files)
                total_uploaded += uploaded_results
            else:
                print("📊 Tidak ada file hasil labeling yang ditemukan")
        
        print("\n" + "=" * 50)
        print(f"🎉 SINKRONISASI SELESAI")
        print(f"📤 Total file yang diupload: {total_uploaded}")
        print("\n💡 Tips:")
        print("   - Checkpoint akan otomatis tersinkron saat labeling berjalan")
        print("   - Gunakan 'python src/google_drive_labeling.py --status' untuk cek status")
        print("   - File hasil akan dibackup otomatis saat proses labeling selesai")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())