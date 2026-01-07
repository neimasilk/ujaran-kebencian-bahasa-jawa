#!/usr/bin/env python3
"""
Script untuk membersihkan lock yang stuck
Gunakan script ini jika yakin tidak ada proses labeling yang sedang berjalan
"""

import sys
import os
sys.path.append('src')

from utils.cloud_checkpoint_manager import CloudCheckpointManager
from config.settings import Settings

def main():
    print("🔧 Clearing stuck labeling lock...")
    print("⚠️  PERINGATAN: Hanya gunakan jika yakin tidak ada proses labeling yang sedang berjalan!")
    
    # Konfirmasi dari user
    response = input("\nApakah Anda yakin ingin menghapus lock? (y/N): ")
    if response.lower() != 'y':
        print("❌ Dibatalkan")
        return
    
    try:
        # Setup settings
        settings = Settings()
        
        # Setup cloud manager
        cloud_manager = CloudCheckpointManager(
            credentials_file='credentials.json',
            token_file='token.json',
            project_folder='ujaran-kebencian-labeling'
        )
        
        # Setup authentication
        if not cloud_manager.authenticate():
            print("❌ Gagal autentikasi Google Drive")
            return
        
        # Force release lock
        success = cloud_manager.force_release_lock()
        
        if success:
            print("✅ Lock berhasil dihapus!")
            print("💡 Sekarang Anda bisa menjalankan 'python labeling.py' lagi")
        else:
            print("❌ Gagal menghapus lock")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Pastikan credentials.json sudah ada dan valid")

if __name__ == "__main__":
    main()