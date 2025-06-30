#!/usr/bin/env python3
"""
Demo script untuk menunjukkan penggunaan Persistent Labeling Pipeline.

Script ini mendemonstrasikan:
1. Menjalankan labeling dengan checkpoint
2. Simulasi interupsi (Ctrl+C)
3. Resume dari checkpoint
4. Melihat daftar checkpoint

Usage:
    python demo_persistent_labeling.py
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collection.persistent_labeling_pipeline import PersistentLabelingPipeline, CheckpointManager
from config.settings import Settings
from utils.logger import setup_logger

# Setup logging
logger = setup_logger("demo_persistent")

def demo_checkpoint_functionality():
    """Demo fungsi checkpoint dan resume."""
    print("\n" + "="*60)
    print("    DEMO PERSISTENT LABELING PIPELINE")
    print("="*60)
    
    # Setup
    settings = Settings()
    pipeline = PersistentLabelingPipeline(
        mock_mode=True,  # Gunakan mock untuk demo
        settings=settings,
        checkpoint_interval=2  # Save checkpoint setiap 2 batch untuk demo
    )
    
    input_file = "src/data_collection/raw-dataset.csv"
    output_file = "demo-persistent-results.csv"
    
    print(f"\n🎯 DEMO SCENARIO:")
    print(f"   Input: {input_file}")
    print(f"   Output: {output_file}")
    print(f"   Mode: Mock (tidak menggunakan API real)")
    print(f"   Checkpoint interval: 2 batch")
    
    # Check existing checkpoints
    checkpoint_manager = CheckpointManager()
    existing_checkpoints = checkpoint_manager.list_checkpoints()
    
    if existing_checkpoints:
        print(f"\n📁 CHECKPOINT YANG ADA:")
        for checkpoint_id in existing_checkpoints:
            checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_id)
            if checkpoint_data:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', 
                                       time.localtime(checkpoint_data['timestamp']))
                processed = len(checkpoint_data.get('processed_indices', []))
                print(f"   {checkpoint_id}: {processed} samples ({timestamp})")
    
    print(f"\n🚀 MEMULAI DEMO...")
    print(f"   Tip: Tekan Ctrl+C untuk simulasi interupsi")
    print(f"   Pipeline akan save checkpoint secara otomatis")
    
    try:
        # Run pipeline
        report = pipeline.run_pipeline(input_file, output_file, resume=False)
        
        print(f"\n✅ DEMO SELESAI TANPA INTERUPSI")
        print(f"   Total samples: {report['summary']['total_samples']}")
        print(f"   Waktu: {report['summary']['processing_time_seconds']:.2f} detik")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  DEMO INTERUPSI BERHASIL!")
        print(f"   Progress sudah disimpan di checkpoint")
        print(f"   Jalankan demo_resume() untuk melanjutkan")
        return True
    
    return False

def demo_resume_functionality():
    """Demo resume dari checkpoint."""
    print(f"\n🔄 DEMO RESUME DARI CHECKPOINT")
    
    # Setup
    settings = Settings()
    pipeline = PersistentLabelingPipeline(
        mock_mode=True,
        settings=settings,
        checkpoint_interval=2
    )
    
    input_file = "src/data_collection/raw-dataset.csv"
    output_file = "demo-persistent-results.csv"
    
    try:
        # Resume dari checkpoint
        report = pipeline.run_pipeline(input_file, output_file, resume=True)
        
        print(f"\n✅ RESUME BERHASIL!")
        print(f"   Total samples: {report['summary']['total_samples']}")
        print(f"   Waktu: {report['summary']['processing_time_seconds']:.2f} detik")
        
        # Show final results
        if Path(output_file).exists():
            import pandas as pd
            df = pd.read_csv(output_file)
            print(f"\n📊 HASIL AKHIR:")
            print(f"   File: {output_file}")
            print(f"   Total rows: {len(df)}")
            if 'final_label' in df.columns:
                label_dist = df['final_label'].value_counts()
                for label, count in label_dist.items():
                    print(f"   {label}: {count}")
        
    except Exception as e:
        print(f"\n❌ Error during resume: {e}")

def demo_checkpoint_management():
    """Demo manajemen checkpoint."""
    print(f"\n📁 DEMO MANAJEMEN CHECKPOINT")
    
    checkpoint_manager = CheckpointManager()
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if not checkpoints:
        print(f"   Tidak ada checkpoint yang tersedia")
        print(f"   Jalankan demo_checkpoint_functionality() terlebih dahulu")
        return
    
    print(f"\n📋 DAFTAR CHECKPOINT:")
    for i, checkpoint_id in enumerate(checkpoints, 1):
        checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_id)
        if checkpoint_data:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', 
                                   time.localtime(checkpoint_data['timestamp']))
            processed = len(checkpoint_data.get('processed_indices', []))
            metadata = checkpoint_data.get('metadata', {})
            
            print(f"\n   {i}. {checkpoint_id}")
            print(f"      Timestamp: {timestamp}")
            print(f"      Processed: {processed} samples")
            print(f"      Output: {metadata.get('output_file', 'N/A')}")
            
            if metadata.get('interrupted'):
                print(f"      Status: ⚠️  Interrupted")
            elif metadata.get('error'):
                print(f"      Status: ❌ Error - {metadata['error']}")
            else:
                print(f"      Status: ✅ Normal")
    
    # Show latest checkpoint
    latest = checkpoint_manager.get_latest_checkpoint()
    if latest:
        print(f"\n🕐 CHECKPOINT TERBARU: {latest}")

def interactive_demo():
    """Demo interaktif."""
    print("\n" + "="*60)
    print("    DEMO INTERAKTIF PERSISTENT PIPELINE")
    print("="*60)
    
    while True:
        print(f"\n🎮 PILIH DEMO:")
        print(f"   1. Demo Checkpoint Functionality (dengan simulasi interupsi)")
        print(f"   2. Demo Resume dari Checkpoint")
        print(f"   3. Demo Manajemen Checkpoint")
        print(f"   4. Keluar")
        
        try:
            choice = input(f"\nPilihan (1-4): ").strip()
            
            if choice == "1":
                interrupted = demo_checkpoint_functionality()
                if interrupted:
                    print(f"\n💡 TIP: Sekarang coba pilihan 2 untuk resume!")
            elif choice == "2":
                demo_resume_functionality()
            elif choice == "3":
                demo_checkpoint_management()
            elif choice == "4":
                print(f"\n👋 Terima kasih! Demo selesai.")
                break
            else:
                print(f"\n❌ Pilihan tidak valid. Coba lagi.")
                
        except KeyboardInterrupt:
            print(f"\n\n👋 Demo dihentikan. Sampai jumpa!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def quick_demo():
    """Demo cepat untuk testing."""
    print("\n🚀 QUICK DEMO - PERSISTENT LABELING")
    
    # Check if dataset exists
    dataset_path = Path("src/data_collection/raw-dataset.csv")
    if not dataset_path.exists():
        print(f"\n❌ Dataset tidak ditemukan: {dataset_path}")
        print(f"   Pastikan file raw-dataset.csv ada di src/data_collection/")
        return
    
    print(f"\n✅ Dataset ditemukan: {dataset_path}")
    
    # Setup pipeline
    settings = Settings()
    pipeline = PersistentLabelingPipeline(
        mock_mode=True,  # Mock mode untuk demo
        settings=settings,
        checkpoint_interval=5  # Save setiap 5 batch
    )
    
    output_file = "quick-demo-results.csv"
    
    print(f"\n⚙️  KONFIGURASI:")
    print(f"   Mode: Mock (tidak pakai API real)")
    print(f"   Checkpoint interval: 5 batch")
    print(f"   Output: {output_file}")
    
    print(f"\n🎯 Memproses 50 sample pertama...")
    
    try:
        # Load dan ambil sample kecil
        import pandas as pd
        df = pd.read_csv(dataset_path, names=['text', 'label'])
        sample_df = df.head(50)  # Ambil 50 sample untuk demo cepat
        
        sample_file = "quick-demo-input.csv"
        sample_df.to_csv(sample_file, index=False)
        
        # Run pipeline
        report = pipeline.run_pipeline(sample_file, output_file, resume=False)
        
        print(f"\n✅ DEMO SELESAI!")
        print(f"   Processed: {report['summary']['total_samples']} samples")
        print(f"   Time: {report['summary']['processing_time_seconds']:.2f} seconds")
        print(f"   Output: {output_file}")
        
        # Show results
        if Path(output_file).exists():
            result_df = pd.read_csv(output_file)
            print(f"\n📊 HASIL:")
            if 'final_label' in result_df.columns:
                for label, count in result_df['final_label'].value_counts().items():
                    print(f"   {label}: {count}")
        
        # Cleanup
        Path(sample_file).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

def main():
    """Main function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            quick_demo()
        elif sys.argv[1] == "--interactive":
            interactive_demo()
        else:
            print(f"Usage: python {sys.argv[0]} [--quick|--interactive]")
    else:
        # Default: interactive demo
        interactive_demo()

if __name__ == "__main__":
    main()