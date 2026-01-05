# ROADMAP PERBAIKAN PROYEK
## Deteksi Ujaran Kebencian Bahasa Jawa

**Tanggal**: 6 Januari 2026
**Status**: DATASET IMPROVEMENT COMPLETED - READY FOR TRAINING

---

## LATEST UPDATE - JANUARI 2026

### ✅ COMPLETED: Dataset Improvement Pipeline

Dataset berhasil diperluas menjadi **10,019 records** dengan cost hanya **$0.40** menggunakan DeepSeek API.

| Status | Item | Details |
|--------|------|---------|
| ✅ | Phase 1: Filtering | Remove Western-specific content |
| ✅ | Phase 2: Naturalization | Adapt to Indonesian context |
| ✅ | Phase 3: Re-labeling | 4,779 records with quality verification |
| ✅ | Phase 4: Generation | 5,240 new records generated |
| ✅ | **TOTAL** | **10,019 records** (Target: 10,000) |

**Key Results:**
- Class Balance: 1.38:1 (near-optimal)
- Label Confidence: 86.6% average
- Context: Fully Indonesian/Javanese
- Cost Efficiency: ~$0.04 per 1,000 records

**Files:**
- `data/improved/phase3_relabeled.csv` - 4,779 quality-verified records
- `data/improved/phase4_generated.csv` - 5,240 generated records
- `data/improved/README.md` - Documentation
- `DATASET_IMPROVEMENT_REPORT.md` - Full technical report
- `dataset_improvement_deepseek.py` - Pipeline script

---

## NEXT STEPS (PRIORITAS)

### Priority 1: Training dengan Dataset Baru ⭐

**Status**: SIAP DILAKUKAN
**Waktu**: 1-2 hari
**Script**: `train_custom_bert_v2.py` atau buat script baru

```bash
# Training dengan dataset improved 10K+
python train_with_improved_dataset.py \
    --train-data data/improved/phase3_relabeled.csv \
    --augment-data data/improved/phase4_generated.csv \
    --output-dir models/production/dataset_10k_v1
```

**Expected Results:**
- Baseline F1-Macro: 65-70% (dengan dataset 10K+)
- Perbaikan signifikan dari baseline 62.55%

### Priority 2: Evaluasi & Paper Update

**Status**: WAITING TRAINING RESULTS
**Waktu**: 1 hari setelah training selesai

Update paper dengan:
1. Hasil training dengan dataset 10K+
2. Dokumentasi pipeline dataset improvement
3. Cost-benefit analysis ($0.40 untuk 10K records)
4. Section tentang AI-assisted data generation

### Priority 3: Final Optimization (Opsional)

**Status**: NICE TO HAVE
**Waktu**: 3-5 hari

Jika hasil Priority 1 belum memuaskan:
1. Filter tambahan untuk hapus ~250 Western references
2. Tambah 500-1000 Neutral class samples
3. Hyperparameter tuning dengan dataset baru
4. Ensemble optimization

---

## ROADMAP LENGKAP

### Opsi A: Training Langsung dengan Dataset 10K+ (DIREKOMENDASIKAN) ⭐

**Waktu Estimasi**: 1-2 hari

**Langkah**:
1. [ ] Buat script training gabungan Phase 3 + Phase 4
2. [ ] Train Custom BERT v2 dengan dataset 10K+
3. [ ] Train mBERT dan XLM-RoBERTa dengan dataset 10K+
4. [ ] Ensemble ketiga model
5. [ ] Evaluasi dan bandingkan dengan baseline

**Expected Target**: F1-Macro 68-72%

---

### Opsi B: Revisi Paper dengan Hasil Reproducible

**Waktu Estimasi**: Langsung bisa dilakukan

**Langkah**:
1. Update paper dengan angka reproducible:
   - F1-Macro: 60.77% (Ensemble) atau 62.55% (Custom BERT v2)
   - Tambahkan section tentang dataset improvement

2. Dokumentasikan pipeline improvement:
   - 4-phase AI-assisted pipeline
   - Cost efficiency analysis
   - Quality control measures

---

### Opsi C: Investigasi dan Perbaikan Menyeluruh (FUTURE)

**Waktu Estimasi**: 2-4 minggu

#### Fase 1: Investigasi (3-5 hari)
- [ ] Cari training logs di `logs/` folder
- [ ] Verifikasi dataset yang digunakan
- [ ] Check apakah ada data leak

#### Fase 2: Optimasi dengan Dataset 10K+ (1-2 minggu)
- [ ] Hyperparameter tuning sistematis
- [ ] Perbaiki XLM-RoBERTa training
- [ ] Coba arsitektur lain (ELECTRA, DeBERTa)
- [ ] Implementasi teknik anti-overfitting

#### Fase 3: Ensemble Optimization (3-5 hari)
- [ ] Weighted ensemble dengan bobot optimal
- [ ] Stacking dengan berbagai meta-learner
- [ ] Cross-validation untuk semua komponen

**Target**: F1-Macro 75-80%

---

## API & COST STATUS

### DeepSeek API (Current)

| Item | Status |
|------|--------|
| Initial Balance | $3.90 |
| Used for Dataset Improvement | $0.40 |
| **Remaining** | **~$3.50** |
| Cost per 1K records | ~$0.04 |

**Budget for:**
- Training experiments: ~50K tokens = ~$0.10
- Future data generation: ~10K more records = ~$0.40
- Total remaining budget sufficient for next phase

---

## REKOMENDASI PENATAAN FOLDER

### Current Structure

```
ujaran-kebencian-bahasa-jawa/
├── data/
│   ├── improved/              # ★ NEW: 10K+ dataset
│   │   ├── phase3_relabeled.csv      # 4,779 records
│   │   ├── phase4_generated.csv      # 5,240 records
│   │   ├── checkpoints/              # Pipeline checkpoints
│   │   └── README.md                 # Documentation
│   ├── standardized/          # Dataset utama (39,841 sampel)
│   └── corpus/                # Corpus untuk DAPT
├── models/
│   ├── production/            # Model siap pakai
│   ├── integrated_*/          # Model ensemble components
│   └── experimental/          # Eksperimen model
├── results/
│   ├── reproducible/          # Hasil yang dapat direproduksi
│   └── historical/            # Hasil eksperimen lama
├── dataset_improvement_deepseek.py  # ★ NEW: Pipeline script
├── DATASET_IMPROVEMENT_REPORT.md     # ★ NEW: Full report
├── AUDIT_EKSPERIMEN_KOMPREHENSIF.md  # Audit status
└── ROADMAP_PERBAIKAN.md              # This file
```

---

## CHECKLIST SEBELUM TRAINING DENGAN DATASET 10K+

### Wajib Dilakukan:

- [x] Dataset improvement selesai (10,019 records)
- [x] Quality verification dilakukan
- [x] Class balance tercapai (1.38:1)
- [x] Dokumentasi lengkap dibuat
- [ ] Gabungkan Phase 3 + Phase 4 untuk training
- [ ] Split train/val/test (80/10/10)
- [ ] Set random seed untuk reproducibility

### Opsional tapi Direkomendasikan:

- [ ] Filter Western references (~250 records)
- [ ] Tambah Neutral class samples
- [ ] Cross-validation (5-fold)
- [ ] Ablation study

---

## PANDUAN TRAINING DENGAN DATASET BARU

### Step 1: Persiapan Data

```python
import pandas as pd

# Load datasets
df_phase3 = pd.read_csv('data/improved/phase3_relabeled.csv')
df_phase4 = pd.read_csv('data/improved/phase4_generated.csv')

# Prepare Phase 3
df_phase3_train = df_phase3[['text', 'new_label']].rename(columns={'new_label': 'label'})

# Prepare Phase 4
df_phase4_train = df_phase4[['text', 'label']]

# Combine
df_train = pd.concat([df_phase3_train, df_phase4_train], ignore_index=True)

# Shuffle
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# Split train/val/test (80/10/10)
train_size = int(0.8 * len(df_train))
val_size = int(0.1 * len(df_train))

df_train_set = df_train[:train_size]
df_val_set = df_train[train_size:train_size+val_size]
df_test_set = df_train[train_size+val_size:]

print(f"Train: {len(df_train_set)}, Val: {len(df_val_set)}, Test: {len(df_test_set)}")
```

### Step 2: Training

```bash
# Custom BERT v2 dengan dataset 10K+
python train_custom_bert_v2.py \
    --train-data data/improved/train.csv \
    --val-data data/improved/val.csv \
    --output-dir models/production/dataset_10k_v1 \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 2e-5
```

---

## SCRIPT YANG PERLU DIBUAT

1. **`train_with_improved_dataset.py`** - Training script khusus untuk dataset improved
2. **`evaluate_dataset_10k.py`** - Evaluasi hasil training
3. **`merge_datasets.py`** - Gabungkan dan siapkan data untuk training

---

## CONTACT & NOTES

### Untuk Pindah Komputer

**Pull di komputer baru:**
```bash
git pull origin main
```

**Context yang perlu diketahui:**
1. Dataset improvement sudah SELESAI (10,019 records)
2. DeepSeek API balance: ~$3.50 remaining
3. Langkah berikutnya: Training dengan dataset baru
4. File utama: `data/improved/phase3_relabeled.csv` + `data/improved/phase4_generated.csv`

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Setup .env file
echo "DEEPSEEK_API_KEY=sk-xxxxx" > .env
echo "DEEPSEEK_BASE_URL=https://api.deepseek.com" >> .env
```

---

**Status**: AKTIF - READY FOR TRAINING PHASE
**Update Terakhir**: 6 Januari 2026
**Next Milestone**: Training dengan Dataset 10K+ → Target F1-Macro 68-72%
