#!/usr/bin/env python3
"""
Script untuk menghapus checkpoint dari Google Drive secara manual
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.cloud_checkpoint_manager import CloudCheckpointManager

def delete_cloud_checkpoints():
    """
    Menghapus checkpoint dari Google Drive secara manual
    """
    print("🔍 Mencari checkpoint di Google Drive...")
    
    # Setup manager
    manager = CloudCheckpointManager()
    
    try:
        # Authenticate dan setup
        if not manager.authenticate():
            print("❌ Gagal autentikasi Google Drive")
            return
        
        # Cari semua file checkpoint di Google Drive
        if not manager.checkpoint_folder_id:
            print("❌ Folder checkpoint tidak ditemukan di Google Drive")
            return
            
        # List semua file di folder checkpoint
        query = f"'{manager.checkpoint_folder_id}' in parents and trashed=false"
        results = manager.service.files().list(
            q=query,
            fields="files(id, name, createdTime)"
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            print("✅ Tidak ada checkpoint di Google Drive")
            return
            
        print(f"📄 Ditemukan {len(files)} file checkpoint di Google Drive:")
        for file in files:
            print(f"   - {file['name']} (ID: {file['id']})")
            
        # Konfirmasi
        confirm = input(f"\n❓ Yakin ingin menghapus SEMUA {len(files)} checkpoint dari Google Drive? (y/N): ")
        if confirm.lower() != 'y':
            print("❌ Dibatalkan")
            return
            
        # Hapus semua file
        deleted_count = 0
        for file in files:
            try:
                manager.service.files().delete(fileId=file['id']).execute()
                print(f"   ✅ Dihapus: {file['name']}")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Gagal menghapus {file['name']}: {e}")
                
        print(f"\n✅ Berhasil menghapus {deleted_count}/{len(files)} checkpoint dari Google Drive")
        print("🚀 Sekarang sistem akan benar-benar mulai dari awal")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Pastikan credentials.json dan token.json sudah dikonfigurasi dengan benar")

if __name__ == "__main__":
    delete_cloud_checkpoints()