#!/usr/bin/env python3
"""
Script untuk membuat proyek siap GitHub dengan menghapus checkpoint dan model besar
Hanya menyimpan model final dan file penting untuk reproduksi
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

class GitHubReadyCleanup:
    def __init__(self):
        self.project_root = Path(".")
        self.total_space_freed = 0
        
        # File yang HARUS DIPERTAHANKAN untuk GitHub
        self.keep_files = {
            # Script dan kode
            "*.py", "*.md", "*.txt", "*.json", "*.yml", "*.yaml",
            
            # Hasil eksperimen penting (hanya JSON, bukan model)
            "results/ultimate_90_percent_results.json",
            "results/final_90_percent_results.json",
            "results/threshold_tuning/threshold_analysis.png",
            
            # Dokumentasi
            "FINAL_EXPERIMENT_STATUS.md",
            "REKAP_SEMUA_EKSPERIMEN.md",
            "README.md",
            
            # Data kecil (jika ada)
            "data/standardized/*.csv"  # Dataset yang sudah diproses
        }
        
        # Direktori yang HARUS DIHAPUS untuk GitHub
        self.remove_completely = [
            "models",  # 338 GB - terlalu besar!
            "experiments/models",  # Model dalam eksperimen
            ".git",  # Jika ada, akan dibuat ulang
            "__pycache__",
            ".pytest_cache",
            "*.pyc",
            "optuna_studies",
            "optuna_study.db"
        ]
        
        # Direktori results - hanya simpan JSON, hapus checkpoint
        self.clean_results = True

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
        return total_size / (1024**3)

    def clean_results_directory(self, results_dir="results"):
        """Bersihkan results directory - hanya simpan JSON dan file kecil"""
        results_path = Path(results_dir)
        if not results_path.exists():
            return 0
            
        space_freed = 0
        print(f"\n🧹 Membersihkan directory {results_dir}:")
        
        for item in results_path.rglob("*"):
            if item.is_file():
                # Hapus checkpoint files dan model files besar
                if (item.name.startswith("pytorch_model") or 
                    item.name.startswith("optimizer.pt") or
                    item.name.startswith("scheduler.pt") or
                    item.name.startswith("trainer_state.json") or
                    item.name.startswith("training_args.bin") or
                    item.suffix in ['.bin', '.safetensors']):
                    
                    size_gb = item.stat().st_size / (1024**3)
                    try:
                        item.unlink()
                        space_freed += size_gb
                        if size_gb > 0.01:  # Hanya log file > 10MB
                            print(f"  🗑️  {item.name}: {size_gb:.2f} GB")
                    except Exception as e:
                        print(f"  ❌ Error deleting {item}: {e}")
            
            elif item.is_dir() and item.name.startswith("checkpoint-"):
                # Hapus semua checkpoint directories
                size_gb = self.get_directory_size(item)
                try:
                    shutil.rmtree(item)
                    space_freed += size_gb
                    print(f"  🗑️  {item.name}/: {size_gb:.2f} GB")
                except Exception as e:
                    print(f"  ❌ Error deleting {item}: {e}")
        
        return space_freed

    def remove_large_directories(self):
        """Hapus direktori besar yang tidak diperlukan untuk GitHub"""
        space_freed = 0
        
        print("\n🗑️  Menghapus direktori besar:")
        
        for dir_pattern in self.remove_completely:
            if "*" in dir_pattern:
                # Handle wildcard patterns
                continue
            
            dir_path = Path(dir_pattern)
            if dir_path.exists():
                size_gb = self.get_directory_size(dir_path)
                try:
                    if dir_path.is_file():
                        dir_path.unlink()
                    else:
                        shutil.rmtree(dir_path)
                    space_freed += size_gb
                    print(f"  🗑️  {dir_pattern}: {size_gb:.2f} GB")
                except Exception as e:
                    print(f"  ❌ Error deleting {dir_pattern}: {e}")
        
        return space_freed

    def create_github_gitignore(self):
        """Buat .gitignore yang tepat untuk GitHub"""
        gitignore_content = """# Model files (too large for GitHub)
models/
*.bin
*.safetensors
*.pt
*.pth
*.ckpt

# Temporary files
tmp*/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.pytest_cache/

# Logs
logs*/
*.log

# Optuna
optuna_studies/
optuna_study.db

# Large result files
results/*/checkpoint-*/
results/*/*.bin
results/*/*.safetensors

# Environment
.env
.venv
env/
venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Large data files (use Git LFS if needed)
*.csv
data/raw/
data/processed/
data/augmented/

# Backup files
backup_*/
"""
        
        with open(".gitignore", "w", encoding="utf-8") as f:
            f.write(gitignore_content)
        
        print("✅ Created GitHub-ready .gitignore")

    def create_github_readme(self):
        """Update README untuk GitHub"""
        readme_content = """# Ujaran Kebencian Bahasa Jawa - Detection System

## 🎯 Project Overview
Sistem deteksi ujaran kebencian dalam Bahasa Jawa menggunakan ensemble model transformer.

## 🏆 Best Results
- **F1-Macro**: 89.22%
- **Accuracy**: 89.24%
- **Models**: IndoRoBERTa + BERT Multilingual + XLM-RoBERTa ensemble

## 📁 Project Structure
```
├── src/                    # Source code
├── results/               # Experiment results (JSON only)
├── docs/                  # Documentation
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train_model.py
```

### Evaluation
```bash
python evaluate_model.py
```

## 📊 Results
Detailed results available in:
- `results/ultimate_90_percent_results.json`
- `FINAL_EXPERIMENT_STATUS.md`

## 📝 Documentation
- [Final Experiment Status](FINAL_EXPERIMENT_STATUS.md)
- [Complete Experiment Summary](REKAP_SEMUA_EKSPERIMEN.md)

## ⚠️ Note
Model files are not included due to GitHub size limits. 
Contact authors for trained models.

## 📄 License
MIT License

## 👥 Authors
Research Team - Ujaran Kebencian Bahasa Jawa Project
"""
        
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        print("✅ Updated README.md for GitHub")

    def analyze_final_size(self):
        """Analisis ukuran final setelah cleanup"""
        print("\n📊 ANALISIS UKURAN FINAL:")
        print("=" * 50)
        
        total_size = 0
        large_files = []
        
        for item in self.project_root.rglob("*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024**2)
                total_size += size_mb
                
                if size_mb > 10:  # File > 10MB
                    large_files.append((str(item), size_mb))
        
        total_size_gb = total_size / 1024
        
        print(f"Total project size: {total_size_gb:.2f} GB")
        
        if large_files:
            print(f"\nFile besar yang tersisa (>10MB):")
            for filepath, size_mb in sorted(large_files, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {filepath}: {size_mb:.1f} MB")
        
        # GitHub recommendations
        if total_size_gb < 1:
            print("✅ PERFECT: Ukuran < 1GB, sangat cocok untuk GitHub")
        elif total_size_gb < 5:
            print("✅ GOOD: Ukuran < 5GB, cocok untuk GitHub")
        elif total_size_gb < 50:
            print("⚠️  WARNING: Ukuran > 5GB, pertimbangkan Git LFS")
        else:
            print("❌ TOO LARGE: Masih terlalu besar untuk GitHub")
        
        return total_size_gb

    def run_github_cleanup(self, dry_run=True):
        """Jalankan cleanup untuk GitHub"""
        print("🚀 GITHUB-READY CLEANUP")
        print("=" * 50)
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE CLEANUP'}")
        
        if not dry_run:
            # Cleanup besar-besaran
            space_freed = 0
            
            # 1. Hapus direktori model besar
            space_freed += self.remove_large_directories()
            
            # 2. Bersihkan results directory
            space_freed += self.clean_results_directory("results")
            space_freed += self.clean_results_directory("experiments/results")
            
            # 3. Buat file GitHub
            self.create_github_gitignore()
            self.create_github_readme()
            
            # 4. Analisis final
            final_size = self.analyze_final_size()
            
            print(f"\n🎉 CLEANUP SELESAI!")
            print(f"💾 Space dibebaskan: {space_freed:.2f} GB")
            print(f"📦 Ukuran final: {final_size:.2f} GB")
            
            # Generate report
            report = {
                "cleanup_date": datetime.now().isoformat(),
                "space_freed_gb": round(space_freed, 2),
                "final_size_gb": round(final_size, 2),
                "github_ready": final_size < 50,
                "status": "completed"
            }
            
            with open("github_cleanup_report.json", "w") as f:
                json.dump(report, f, indent=2)
            
            print(f"📄 Report: github_cleanup_report.json")
            
        else:
            print("\n🔍 DRY RUN - Estimasi cleanup:")
            
            # Estimasi models directory
            models_size = self.get_directory_size("models") if Path("models").exists() else 0
            experiments_models_size = self.get_directory_size("experiments/models") if Path("experiments/models").exists() else 0
            
            print(f"Will remove models/: {models_size:.2f} GB")
            print(f"Will remove experiments/models/: {experiments_models_size:.2f} GB")
            
            # Estimasi results cleanup
            results_checkpoints = 0
            for result_dir in ["results", "experiments/results"]:
                path = Path(result_dir)
                if path.exists():
                    for item in path.rglob("checkpoint-*"):
                        if item.is_dir():
                            results_checkpoints += self.get_directory_size(item)
                    
                    # Also count large files in results
                    for item in path.rglob("*"):
                        if item.is_file() and (
                            item.name.startswith("pytorch_model") or 
                            item.suffix in ['.bin', '.safetensors']
                        ):
                            results_checkpoints += item.stat().st_size / (1024**3)
            
            print(f"Will clean results/experiments checkpoints/models: {results_checkpoints:.2f} GB")
            
            total_to_free = models_size + experiments_models_size + results_checkpoints
            print(f"\n💾 Total space to free: {total_to_free:.2f} GB")
            
            # Estimasi ukuran final
            current_total = self.get_directory_size(".")
            estimated_final = current_total - total_to_free
            print(f"📦 Estimated final size: {estimated_final:.2f} GB")
            
            if estimated_final < 1:
                print("✅ Will be PERFECT for GitHub!")
            elif estimated_final < 5:
                print("✅ Will be GOOD for GitHub")
            else:
                print("⚠️  May still need Git LFS for some files")
            
            print("\n⚠️  Untuk cleanup sesungguhnya: python cleanup_github_ready.py --live")

if __name__ == "__main__":
    import sys
    
    cleanup = GitHubReadyCleanup()
    
    live_mode = "--live" in sys.argv
    
    if live_mode:
        response = input("⚠️  Ini akan menghapus SEMUA model files! Yakin? (yes/no): ")
        if response.lower() == "yes":
            cleanup.run_github_cleanup(dry_run=False)
        else:
            print("Cleanup dibatalkan.")
    else:
        cleanup.run_github_cleanup(dry_run=True)