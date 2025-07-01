#!/usr/bin/env python3
"""
Javanese Hate Speech Labeling Tool
Alat Labeling Ujaran Kebencian Bahasa Jawa

Perintah sederhana untuk melakukan labeling dataset dengan DeepSeek API + Google Drive backup.

Usage:
    python labeling.py
    
Author: AI Assistant
Date: 2025-01-01
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """
    Main function untuk menjalankan labeling pipeline
    """
    print("🚀 Javanese Hate Speech Labeling Tool")
    print("="*50)
    
    # Path ke script utama
    script_path = Path(__file__).parent / "src" / "google_drive_labeling.py"
    dataset_path = "src/data_collection/raw-dataset.csv"
    output_name = "hasil-labeling"
    
    # Command untuk menjalankan labeling
    cmd = [
        sys.executable,
        str(script_path),
        "--dataset", dataset_path,
        "--output", output_name,
        "--no-promo-wait"  # Langsung mulai tanpa menunggu jam promo
    ]
    
    # Check jika user ingin force override lock
    if len(sys.argv) > 1 and "--force" in sys.argv:
        cmd.append("--force")
        print("⚠️  FORCE MODE: Override existing locks")
    
    print(f"📂 Dataset: {dataset_path}")
    print(f"📁 Output: {output_name}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print("\n⚡ Starting labeling process...")
    print("💡 Tekan Ctrl+C untuk STOP dan SIMPAN (bukan pause)")
    print("💡 Untuk melanjutkan: jalankan 'python labeling.py' lagi")
    print("="*50)
    
    try:
        # Jalankan command
        result = subprocess.run(cmd, check=True)
        print("\n🎉 Labeling selesai dengan sukses!")
        print("📁 Hasil tersimpan di Google Drive dan lokal")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Cek log untuk detail error")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n🛑 Process DIHENTIKAN oleh user (Ctrl+C)")
        print("💾 Progress TERSIMPAN dan di-sync ke Google Drive")
        print("🔄 Jalankan 'python labeling.py' lagi untuk melanjutkan")
        sys.exit(0)

if __name__ == '__main__':
    main()