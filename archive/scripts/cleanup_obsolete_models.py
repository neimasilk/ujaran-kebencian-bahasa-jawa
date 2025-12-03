#!/usr/bin/env python3
"""
Script untuk membersihkan model-model obsolete dan menjaga hanya model yang sukses
Berdasarkan analisis eksperimen, model terbaik adalah dari Ultimate Optimization (89.22% F1-Macro)
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

class ModelCleanup:
    def __init__(self):
        self.project_root = Path(".")
        self.total_space_freed = 0
        
        # Model yang HARUS DIPERTAHANKAN berdasarkan dokumentasi eksperimen
        self.keep_models = {
            # Model terbaik dari Ultimate Optimization (89.22% F1-Macro)
            "results/xlm-roberta-base_final",  # XLM-RoBERTa final model
            "results/indoroberta_optimized",   # IndoRoBERTa optimized
            "results/xlm_roberta_optimized",   # XLM-RoBERTa optimized  
            "results/bert_multilingual_optimized",  # BERT multilingual optimized
            
            # Hasil eksperimen penting
            "results/ultimate_90_percent_results.json",
            "results/final_90_percent_results.json",
            "results/threshold_tuning",
        }
        
        # Direktori yang bisa dihapus (model sampah/obsolete)
        self.cleanup_dirs = [
            # Temporary model directories
            "tmp_90_percent_indobert-base-uncased",
            "tmp_meta_bert_multilingual", 
            "tmp_meta_indoroberta",
            "tmp_meta_xlm_roberta",
            "tmp_original_model_0",
            "tmp_original_model_1", 
            "tmp_stable_model_0",
            "tmp_stable_model_1",
            "tmp_trainer",
            
            # Obsolete results dari eksperimen gagal
            "results/advanced_techniques_flax-community_indonesian-roberta-base",
            "results/advanced_techniques_indobenchmark_indobert-base-p1",
            "results/bert-base-multilingual-cased_final",
            "results/indolem_indobert-base-uncased_final",
            "results/hyperparameter_optimization",  # Banyak trial yang tidak terpakai
        ]
        
        # Log directories yang bisa dihapus (hanya log, bukan model)
        self.cleanup_logs = [
            "logs_90_percent_indobert-base-uncased",
            "logs_meta_bert_multilingual",
            "logs_meta_indoroberta", 
            "logs_meta_xlm_roberta",
            "logs_original_0",
            "logs_original_1",
            "logs_stable_0", 
            "logs_stable_1",
        ]

    def get_directory_size(self, path):
        """Menghitung ukuran direktori dalam GB"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except (OSError, FileNotFoundError):
            pass
        return total_size / (1024**3)  # Convert to GB

    def backup_important_results(self):
        """Backup hasil penting sebelum cleanup"""
        backup_dir = Path("backup_important_results")
        backup_dir.mkdir(exist_ok=True)
        
        important_files = [
            "results/ultimate_90_percent_results.json",
            "results/final_90_percent_results.json", 
            "FINAL_EXPERIMENT_STATUS.md",
            "REKAP_SEMUA_EKSPERIMEN.md"
        ]
        
        for file_path in important_files:
            if Path(file_path).exists():
                shutil.copy2(file_path, backup_dir / Path(file_path).name)
                print(f"✅ Backed up: {file_path}")

    def analyze_current_usage(self):
        """Analisis penggunaan disk saat ini"""
        print("🔍 ANALISIS PENGGUNAAN DISK SAAT INI")
        print("=" * 60)
        
        total_model_size = 0
        total_tmp_size = 0
        total_log_size = 0
        
        # Analisis direktori model/tmp/log
        for item in self.project_root.iterdir():
            if item.is_dir():
                size_gb = self.get_directory_size(item)
                if size_gb > 0.1:  # Hanya tampilkan yang > 100MB
                    print(f"{item.name}: {size_gb:.2f} GB")
                    
                    if "model" in item.name.lower():
                        total_model_size += size_gb
                    elif "tmp" in item.name.lower():
                        total_tmp_size += size_gb
                    elif "log" in item.name.lower():
                        total_log_size += size_gb
        
        print(f"\n📊 RINGKASAN:")
        print(f"Total Model Size: {total_model_size:.2f} GB")
        print(f"Total Tmp Size: {total_tmp_size:.2f} GB") 
        print(f"Total Log Size: {total_log_size:.2f} GB")
        print(f"Total: {total_model_size + total_tmp_size + total_log_size:.2f} GB")

    def safe_remove_directory(self, dir_path):
        """Hapus direktori dengan aman"""
        path = Path(dir_path)
        if not path.exists():
            print(f"⚠️  Directory tidak ditemukan: {dir_path}")
            return 0
            
        # Hitung ukuran sebelum dihapus
        size_gb = self.get_directory_size(path)
        
        try:
            shutil.rmtree(path)
            print(f"🗑️  Dihapus: {dir_path} ({size_gb:.2f} GB)")
            return size_gb
        except Exception as e:
            print(f"❌ Error menghapus {dir_path}: {str(e)}")
            return 0

    def cleanup_obsolete_models(self):
        """Hapus model-model obsolete"""
        print("\n🧹 MEMBERSIHKAN MODEL OBSOLETE")
        print("=" * 60)
        
        space_freed = 0
        
        # Hapus temporary directories
        print("\n📁 Menghapus temporary directories:")
        for tmp_dir in self.cleanup_dirs:
            space_freed += self.safe_remove_directory(tmp_dir)
        
        # Hapus log directories (opsional, bisa dipertahankan jika diperlukan)
        print("\n📋 Menghapus log directories:")
        for log_dir in self.cleanup_logs:
            space_freed += self.safe_remove_directory(log_dir)
        
        self.total_space_freed = space_freed
        return space_freed

    def verify_important_models_intact(self):
        """Verifikasi model penting masih ada"""
        print("\n✅ VERIFIKASI MODEL PENTING")
        print("=" * 60)
        
        all_intact = True
        for model_path in self.keep_models:
            path = Path(model_path)
            if path.exists():
                if path.is_file():
                    size_mb = path.stat().st_size / (1024**2)
                    print(f"✅ {model_path} ({size_mb:.1f} MB)")
                else:
                    size_gb = self.get_directory_size(path)
                    print(f"✅ {model_path} ({size_gb:.2f} GB)")
            else:
                print(f"❌ MISSING: {model_path}")
                all_intact = False
        
        return all_intact

    def generate_cleanup_report(self):
        """Generate laporan cleanup"""
        report = {
            "cleanup_date": datetime.now().isoformat(),
            "space_freed_gb": round(self.total_space_freed, 2),
            "directories_removed": self.cleanup_dirs + self.cleanup_logs,
            "models_preserved": list(self.keep_models),
            "status": "completed"
        }
        
        with open("cleanup_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Laporan cleanup disimpan ke: cleanup_report.json")

    def run_cleanup(self, dry_run=True):
        """Jalankan proses cleanup"""
        print("🚀 MODEL CLEANUP UTILITY")
        print("=" * 60)
        print(f"Mode: {'DRY RUN (simulasi)' if dry_run else 'LIVE CLEANUP'}")
        print()
        
        # Backup file penting
        if not dry_run:
            self.backup_important_results()
        
        # Analisis penggunaan saat ini
        self.analyze_current_usage()
        
        if dry_run:
            print("\n🔍 DRY RUN - Simulasi cleanup:")
            total_to_free = 0
            for dir_name in self.cleanup_dirs + self.cleanup_logs:
                if Path(dir_name).exists():
                    size = self.get_directory_size(dir_name)
                    total_to_free += size
                    print(f"Will remove: {dir_name} ({size:.2f} GB)")
            
            print(f"\n💾 Total space yang akan dibebaskan: {total_to_free:.2f} GB")
            print("\n⚠️  Untuk menjalankan cleanup sesungguhnya, gunakan: python cleanup_obsolete_models.py --live")
        else:
            # Cleanup sesungguhnya
            space_freed = self.cleanup_obsolete_models()
            
            # Verifikasi model penting
            models_intact = self.verify_important_models_intact()
            
            # Generate report
            self.generate_cleanup_report()
            
            print(f"\n🎉 CLEANUP SELESAI!")
            print(f"💾 Total space dibebaskan: {space_freed:.2f} GB")
            print(f"✅ Model penting: {'Aman' if models_intact else 'Ada yang hilang!'}")

if __name__ == "__main__":
    import sys
    
    cleanup = ModelCleanup()
    
    # Default adalah dry run untuk keamanan
    live_mode = "--live" in sys.argv
    
    if live_mode:
        response = input("⚠️  Anda yakin ingin menghapus model obsolete? (yes/no): ")
        if response.lower() == "yes":
            cleanup.run_cleanup(dry_run=False)
        else:
            print("Cleanup dibatalkan.")
    else:
        cleanup.run_cleanup(dry_run=True)