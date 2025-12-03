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
import argparse
from pathlib import Path

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Javanese Hate Speech Labeling Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python labeling.py                    # Serial labeling (default)
  python labeling.py --parallel         # PRODUCTION parallel labeling (real API, 20x faster)
  python labeling.py --parallel --workers 5  # Custom worker count for production
  python labeling.py --force            # Override existing locks
  python labeling.py --parallel --mock  # Test with mock API (no cost)
        """
    )
    
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Enable parallel processing for PRODUCTION (20x+ faster, uses real API by default)"
    )
    
    parser.add_argument(
        "--workers", 
        type=int, 
        default=5,
        help="Number of parallel workers (default: 5, optimal: 3-5)"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Override existing locks"
    )
    
    parser.add_argument(
        "--mock", 
        action="store_true",
        help="Use mock API for testing only (no cost, overrides production mode)"
    )
    
    parser.add_argument(
        "--dataset", 
        default="src/data_collection/raw-dataset.csv",
        help="Path to input dataset (default: src/data_collection/raw-dataset.csv)"
    )
    
    parser.add_argument(
        "--output", 
        default="hasil-labeling",
        help="Output filename prefix (default: hasil-labeling)"
    )
    
    return parser.parse_args()

def main():
    """
    Main function untuk menjalankan labeling pipeline
    """
    args = parse_arguments()
    
    print("🚀 Javanese Hate Speech Labeling Tool")
    print("="*50)
    
    # Tentukan script yang akan digunakan
    if args.parallel:
        print("⚡ PARALLEL MODE: 20x+ faster processing!")
        script_path = Path(__file__).parent / "production_parallel_labeling.py"
        
        # Command untuk parallel labeling
        cmd = [
            sys.executable,
            str(script_path),
            "--input", args.dataset,
            "--output", f"{args.output}.csv",
            "--workers", str(args.workers)
        ]
        
        if args.mock:
            cmd.append("--mock")
            print("🧪 MOCK MODE: Testing without API cost")
        else:
            cmd.append("--real")
            print("💰 PRODUCTION MODE: Using real DeepSeek API (promo period!)")
            print("🎯 Ready for production labeling with cost optimization")
            
    else:
        print("🐌 SERIAL MODE: Traditional processing")
        script_path = Path(__file__).parent / "src" / "google_drive_labeling.py"
        
        # Command untuk serial labeling
        cmd = [
            sys.executable,
            str(script_path),
            "--dataset", args.dataset,
            "--output", args.output,
            "--no-promo-wait"  # Langsung mulai tanpa menunggu jam promo
        ]
    
    # Add force flag if specified
    if args.force:
        cmd.append("--force")
        print("⚠️  FORCE MODE: Override existing locks")
    
    print(f"📂 Dataset: {args.dataset}")
    print(f"📁 Output: {args.output}")
    if args.parallel:
        print(f"👥 Workers: {args.workers}")
        print(f"🔧 Mode: {'Mock' if args.mock else 'Real API'}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print("\n⚡ Starting labeling process...")
    print("💡 Tekan Ctrl+C untuk STOP dan SIMPAN (bukan pause)")
    print("💡 Untuk melanjutkan: jalankan command yang sama lagi")
    print("="*50)
    
    try:
        # Jalankan command
        result = subprocess.run(cmd, check=True)
        print("\n🎉 Labeling selesai dengan sukses!")
        if args.parallel:
            print("📁 Hasil tersimpan di file lokal")
            print("💡 Upload ke Google Drive secara manual jika diperlukan")
        else:
            print("📁 Hasil tersimpan di Google Drive dan lokal")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Cek log untuk detail error")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n🛑 Process DIHENTIKAN oleh user (Ctrl+C)")
        if args.parallel:
            print("💾 Progress TERSIMPAN di file lokal")
            print("🔄 Jalankan command yang sama untuk melanjutkan")
        else:
            print("💾 Progress TERSIMPAN dan di-sync ke Google Drive")
            print("🔄 Jalankan 'python labeling.py' lagi untuk melanjutkan")
        sys.exit(0)

if __name__ == '__main__':
    main()