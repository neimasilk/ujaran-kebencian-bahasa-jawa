#!/usr/bin/env python3
"""
Script untuk menghapus semua file hasil labeling dari Google Drive
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from utils.cloud_checkpoint_manager import CloudCheckpointManager
from pathlib import Path

def delete_gdrive_results():
    """Hapus semua file hasil labeling dari Google Drive"""
    
    # Initialize checkpoint manager
    checkpoint_manager = CloudCheckpointManager(
        project_folder="ujaran-kebencian-datasets",
        local_cache_dir="src/checkpoints"
    )
    
    # Authenticate with Google Drive
    print("🔐 Melakukan autentikasi Google Drive...")
    if not checkpoint_manager.authenticate():
        print("❌ Gagal autentikasi Google Drive")
        return False
    
    print("✅ Autentikasi berhasil")
    
    # List of result files to delete from Google Drive
    result_files = [
        "hasil-labeling.csv",
        "demo-persistent-results.csv", 
        "quick-demo-results.csv",
        "test_deepseek_negative_10_results.csv"
    ]
    
    deleted_count = 0
    
    for filename in result_files:
        print(f"\n🔍 Mencari file: {filename}")
        
        try:
            # Search for the file in Google Drive
            if hasattr(checkpoint_manager, 'drive_service') and checkpoint_manager.drive_service:
                query = f"name='{filename}' and trashed=false"
                results = checkpoint_manager.drive_service.files().list(
                    q=query,
                    fields="files(id, name, parents)"
                ).execute()
                
                files = results.get('files', [])
                
                if files:
                    for file_info in files:
                        file_id = file_info['id']
                        file_name = file_info['name']
                        
                        print(f"📁 Ditemukan: {file_name} (ID: {file_id})")
                        
                        # Confirm deletion
                        confirm = input(f"❓ Hapus file '{file_name}' dari Google Drive? (y/N): ")
                        if confirm.lower() in ['y', 'yes']:
                            try:
                                checkpoint_manager.drive_service.files().delete(fileId=file_id).execute()
                                print(f"✅ File '{file_name}' berhasil dihapus dari Google Drive")
                                deleted_count += 1
                            except Exception as e:
                                print(f"❌ Gagal menghapus file '{file_name}': {e}")
                        else:
                            print(f"⏭️ Melewati file '{file_name}'")
                else:
                    print(f"ℹ️ File '{filename}' tidak ditemukan di Google Drive")
                    
        except Exception as e:
            print(f"❌ Error saat mencari file '{filename}': {e}")
    
    print(f"\n📊 Total file yang dihapus: {deleted_count}")
    return deleted_count > 0

if __name__ == "__main__":
    print("🗑️ Script Penghapusan File Hasil Labeling dari Google Drive")
    print("=" * 60)
    
    success = delete_gdrive_results()
    
    if success:
        print("\n✅ Proses penghapusan selesai")
    else:
        print("\n⚠️ Tidak ada file yang dihapus atau terjadi error")
    
    print("\n🔄 Sekarang sistem akan mulai dari awal yang bersih saat menjalankan labeling.py")