# Workflow Dua Komputer - Deteksi Ujaran Kebencian Bahasa Jawa

**Update Terakhir:** 6 Januari 2026
**Tujuan:** Pembagian kerja yang efisien antara GPU kuat dan komputer biasa

---

## Overview

Proyek ini menggunakan dua komputer dengan spesifikasi berbeda untuk memaksimalkan efisiensi:

| Komputer | Spesifikasi | Fokus Utama |
|----------|-------------|-------------|
| **Komputer 1** | GPU RTX 4080, Memory besar | Training model berat |
| **Komputer 2** | Komputer biasa (tanpa GPU kuat) | Dataset optimization, dokumentasi, writing |

---

## Komputer 1: GPU Kuat (Training Machine)

### Spesifikasi
- **GPU:** NVIDIA GeForce RTX 4080 (16GB VRAM)
- **Memory:** RAM besar
- **Cocok untuk:** Training model, fine-tuning, experiment berat

### Task yang Dilakukan

1. **Training Model**
   - Fine-tuning IndoBERT, mBERT, XLM-RoBERTa
   - Custom BERT pre-training (DAPT)
   - Ensemble training

2. **Experiment Berat**
   - Hyperparameter tuning
   - Multi-model training
   - Large-scale evaluation

3. **File Operations di Komputer Ini**
   - Model checkpoints disimpan lokal
   - Training logs dan metrics
   - Results dari experiments

### File yang TIDAK di-push ke GitHub
```
models/
checkpoints/
results/*.json
logs/
```

---

## Komputer 2: Komputer Biasa (Non-GPU)

### Spesifikasi
- Tanpa GPU kuat atau GPU kecil
- Fokus pada task non-computational

### Task yang Dilakukan

1. **Dataset Optimization dengan DeepSeek API**
   - Data filtering dan cleaning
   - Naturalization (Western → Indonesian context)
   - Re-labeling dengan Chain-of-Thought
   - Synthetic data generation

2. **Dokumentasi dan Writing**
   - Update README, EXPERIMENT_TIMELINE.md
   - Write experiment results documentation
   - Paper preparation

3. **Code Development**
   - Script development dan debugging
   - Configuration updates
   - Review dan refactoring

### File yang di-push ke GitHub
```
data/*.csv
data/improved/*.csv
experiments/*.py
*.md
```

---

## Workflow Git

### .gitignore Configuration

File yang di-exclude dari git (tidak di-push):
```
# Model checkpoints dan trained models
models/
checkpoints/
*.pt
*.pth
*.bin

# Training artifacts
results/*.json
logs/
wandb/
*.log

# Temporary files
__pycache__/
*.pyc
.cache/
```

File yang DI-INCLUDE ke git:
- Dataset CSV (`data/`, `data/improved/`)
- Python scripts (`experiments/`)
- Dokumentasi (`*.md`)
- Configuration files

---

## Workflow Penggunaan

### Langkah 1: Sync dari GitHub

Di **kedua komputer**, lakukan:
```bash
git pull origin main
```

### Langkah 2: Beritahu Claude Komputer yang Digunakan

Gunakan prompt berikut untuk memberitahu Claude komputer mana yang sedang digunakan:

#### Untuk Komputer Biasa (Non-GPU)
```
ini pakai komputer biasa, lanjutkan dan lakukan untuk komputer biasa
```

#### Untuk Komputer GPU Kuat
```
ini pakai komputer GPU kuat, lakukan dan lanjutkan eksperimen yang pakai komputer kuat
```

### Langkah 3: Eksekusi Task

Claude akan otomatis menyesuaikan task berdasarkan komputer yang digunakan:

**Komputer Biasa:**
- Fokus pada dataset optimization dengan DeepSeek API
- Update dokumentasi
- Code review dan development

**Komputer GPU Kuat:**
- Jalankan training experiments
- Simpan model checkpoints
- Generate results

### Langkah 4: Commit dan Push

Setelah selesai:
```bash
git add .
git commit -m "feat: description singkat"
git push origin main
```

---

## Contoh Use Cases

### Use Case 1: Dataset Improvement

**Komputer Biasa:**
1. `git pull`
2. Prompt: "ini pakai komputer biasa..."
3. Run dataset improvement pipeline dengan DeepSeek
4. Commit dan push dataset baru

**Komputer GPU Kuat:**
1. `git pull` (dataset baru sudah tersedia)
2. Prompt: "ini pakai komputer GPU kuat..."
3. Train model dengan dataset baru
4. Results disimpan lokal, dokumentasi di-push

### Use Case 2: Experiment Cycle

**Komputer Biasa:**
- Menulis experiment script
- Update dokumentasi
- Review hasil

**Komputer GPU Kuat:**
- Run experiment
- Train model
- Evaluate

---

## Task Mapping

| Task | Komputer 1 (GPU) | Komputer 2 (Biasa) |
|------|------------------|---------------------|
| Dataset cleaning | | ✅ |
| Synthetic data generation | | ✅ |
| Script development | | ✅ |
| Documentation | | ✅ |
| Paper writing | | ✅ |
| Model training | ✅ | |
| Hyperparameter tuning | ✅ | |
| Model evaluation | ✅ | |
| Experiment results | ✅ | (doc only) |

---

## Best Practices

1. **Selalu git pull sebelum memulai** di kedua komputer
2. **Gunakan prompt yang jelas** untuk menunjukkan komputer yang digunakan
3. **Commit sering** dengan pesan yang deskriptif
4. **Jangan push model checkpoints** - hanya dataset dan kode
5. **Sinkronisasi dokumentasi** di kedua komputer

---

## Troubleshooting

### Conflict saat git pull
```bash
git stash
git pull origin main
git stash pop
```

### Dataset terlalu besar untuk GitHub
- Gunakan Git LFS jika perlu
- Atau simpan di external storage dan share manual

### Model perlu dipindah antar komputer
- Gunakan external storage (USB, cloud storage)
- Atau re-train di komputer GPU (recommended)

---

## File Referensi

- `EXPERIMENT_TIMELINE.md` - Status eksperimen terkini
- `ROADMAP_PENELITIAN.md` - Roadmap dan next steps
- `DATASET_IMPROVEMENT_REPORT.md` - Dataset improvement pipeline

---

**Status:** READY FOR TWO-COMPUTER WORKFLOW
**Last Update:** 6 Januari 2026
