#!/usr/bin/env python3
"""
Script untuk menghapus semua checkpoint dari Google Drive dan lokal
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.cloud_checkpoint_manager import CloudCheckpointManager

def clear_all_checkpoints():
    """
    Menghapus semua checkpoint dari Google Drive dan lokal
    """
    print("🗑️ Menghapus semua checkpoint...")
    
    # Setup
    manager = CloudCheckpointManager()
    
    # List checkpoint yang ada
    checkpoints = manager.list_checkpoints()
    print(f"📄 Ditemukan {len(checkpoints)} checkpoint:")
    for checkpoint in checkpoints:
        print(f"   - {checkpoint['id']} ({checkpoint['source']})")
    
    if len(checkpoints) == 0:
        print("✅ Tidak ada checkpoint yang perlu dihapus")
        return
    
    # Konfirmasi
    confirm = input(f"\n❓ Yakin ingin menghapus SEMUA {len(checkpoints)} checkpoint? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Dibatalkan")
        return
    
    # Hapus semua checkpoint (keep_count=0 berarti hapus semua)
    deleted_count = manager.cleanup_old_checkpoints(keep_count=0)
    
    print(f"\n✅ Berhasil menghapus {deleted_count} checkpoint")
    print("🚀 Sekarang sistem akan mulai dari awal saat menjalankan labeling.py")

if __name__ == "__main__":
    clear_all_checkpoints()