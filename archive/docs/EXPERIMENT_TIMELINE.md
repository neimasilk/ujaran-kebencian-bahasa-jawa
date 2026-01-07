# Kronologi Eksperimen - Deteksi Ujaran Kebencian Bahasa Jawa

**Update Terakhir:** 6 Januari 2026
**Status:** FASE 6 BERJALAN - LABEL SMOOTHING CAPAI 81.38% F1-MACRO

---

## Timeline Lengkap

### FASE 0: Data Collection & Preprocessing (Agustus 2025)

| Tanggal | Kegiatan | Hasil |
|---------|----------|-------|
| Awal Agustus 2025 | Pengumpulan dataset awal | ~4,800 records |
| Mid Agustus 2025 | Standardisasi dataset | 39,841 records (balanced) |
| File Referensi | `data/standardized/balanced_dataset.csv` | Dataset utama |

**Catatan:** Dataset ini memiliki masalah:
- Banyak konteks Barat yang tidak relevan
- Terjemahan kaku yang tidak natural
- Distribusi kelas tidak seimbang

---

### FASE 1: Baseline Experiments (Agustus 2025)

#### Experiment 0: Baseline IndoBERT
- **Model:** IndoBERT Base
- **Dataset:** `balanced_dataset.csv` (39,841 records)
- **Hasil:** F1-Macro ~80.36%
- **Status:** ✅ Complete
- **Dokumentasi:** `memory-bank/02-research-active/consolidated-experiments/EXPERIMENT_0_BASELINE_INDOBERT_BALANCED_DOCUMENTATION.md`

#### Experiment 0.1: IndoBERT dengan SMOTE
- **Model:** IndoBERT Base + SMOTE augmentation
- **Hasil:** F1-Macro ~67-75%
- **Status:** ✅ Complete
- **Catatan:** SMOTE tidak memberikan improvement signifikan

---

### FASE 2: Advanced Model Experiments (September - November 2025)

#### Experiment 1: IndoBERT Large
- **Model:** IndoBERT Large (340M parameters)
- **Dataset:** `balanced_dataset.csv`
- **Hasil:** F1-Macro ~75%
- **Status:** ✅ Complete
- **Dokumentasi:** `memory-bank/02-research-active/consolidated-experiments/EXPERIMENT_1_2_INDOBERT_LARGE_RESULTS.md`

#### Experiment 1.2: XLM-RoBERTa
- **Model:** XLM-RoBERTa Base
- **Dataset:** `balanced_dataset.csv`
- **Hasil:** F1-Macro ~55-62%
- **Status:** ✅ Complete
- **Catatan:** Device mismatch error, berhasil diperbaiki
- **Dokumentasi:** `memory-bank/02-research-active/consolidated-experiments/EXPERIMENT_1_2_XLM_ROBERTA_IMPROVED_RESULTS.md`

#### Experiment 1.3: mBERT
- **Model:** Multilingual BERT
- **Dataset:** `balanced_dataset.csv`
- **Hasil:** F1-Macro ~51-65%
- **Status:** ✅ Complete
- **Dokumentasi:** `memory-bank/02-research-active/consolidated-experiments/EXPERIMENT_1_3_MBERT_RESULTS.md`

---

### FASE 3: Custom BERT & Ensemble (Desember 2025)

#### Experiment 2: Custom Javanese BERT v1
- **Model:** Custom BERT dengan DAPT (Domain-Adaptive Pre-Training)
- **Corpus:** Wikipedia Jawa + Dataset + AI Synthetic Data
- **Hasil:** F1-Macro ~59-60%
- **Status:** ✅ Complete
- **Catatan:** DAPT berhasil diimplementasikan

#### Experiment 3: Custom Javanese BERT v2
- **Model:** Custom BERT v2 dengan improved DAPT
- **Hasil:** F1-Macro 62.55% (reproducible)
- **Status:** ✅ Complete & Reproducible
- **Dokumentasi:** `memory-bank/02-research-active/TRAINING_EVALUATION_REPORT.md`

#### Experiment 4: Ensemble Methods
- **Models:** Custom BERT v2 + mBERT + XLM-RoBERTa
- **Method:** Stacking dengan Logistic Regression
- **Hasil:** F1-Macro 60.77%
- **Status:** ✅ Complete & Reproducible
- **Catatan:** Ensemble tidak memberikan improvement di atas single best model

---

### FASE 4: Dataset Improvement (Januari 2026) ⭐ TERBARU

#### Dataset Improvement Pipeline
- **Objective:** Perbaiki dataset agar lebih representatif dengan konteks Indonesia
- **Method:** 4-phase pipeline dengan DeepSeek API
- **Cost:** $0.40 untuk 10,019 records

| Phase | Proses | Records | File |
|-------|--------|---------|------|
| 1 | Filtering (Western content removal) | 5,005 | `phase1_keep.csv`, `phase1_naturalize.csv` |
| 2 | Naturalization (Western → Indonesian context) | 1,409 | `phase2_naturalized.csv` |
| 3 | Re-labeling (Chain-of-Thought) | 4,779 | `phase3_relabeled.csv` |
| 4 | Generation (AI Synthetic) | 5,240 | `phase4_generated.csv` |
| **TOTAL** | | **10,019** | |

**Hasil:**
- Class Balance: 1.38:1 (near-optimal)
- Label Confidence: 86.6% average
- Context: Fully Indonesian/Javanese
- Cost Efficiency: ~$0.04 per 1,000 records

**Label Distribution (Final):**
- Neutral (0): 2,474 (24.7%)
- Light (1): 2,615 (26.1%)
- Moderate (2): 2,862 (28.6%)
- Severe (3): 2,068 (20.6%)

**Status:** ✅ COMPLETE
**Dokumentasi:** `DATASET_IMPROVEMENT_REPORT.md`

---

### FASE 5: Eksperimen dengan Dataset Improved (Januari 2026) 🔥

#### Experiment 5: IndoBERT Base dengan Dataset Improved ✅ **COMPLETED**
- **Model:** IndoBERT Base (indobenchmark/indobert-base-p1)
- **Dataset:** `data/improved/` (10,019 records)
- **Hasil:** F1-Macro **79.19%** 🎉
- **Status:** ✅ COMPLETE - EXCELLENT RESULTS!
- **Dokumentasi:** `experiments/EXPERIMENT_5_RESULTS.md`
- **Improvement:** +16.64% dari baseline 62.55%

**Per-Class F1:**
- Neutral: 77.36%
- Light: 72.73%
- Moderate: 83.66%
- Severe: 83.00%

#### Experiment 6: Multi-Model Training ✅ **COMPLETED**
- **Models:** mBERT + XLM-RoBERTa
- **Dataset:** `data/improved/` (10,019 records)
- **Hasil:**
  - mBERT: F1-Macro **77.93%**
  - XLM-RoBERTa: F1-Macro **78.38%**
- **Status:** ✅ COMPLETE
- **Dokumentasi:** `experiments/experiment_6_7_multi_model_ensemble.py`

#### Experiment 7: Ensemble Methods ✅ **COMPLETED**
- **Method:** Stacking dengan Logistic Regression meta-learner
- **Models:** mBERT + XLM-RoBERTa
- **Hasil:** F1-Macro **78.90%**
- **Status:** ✅ COMPLETE
- **Catatan:** Ensemble memberikan improvement kecil (+0.5%) dari best single model
- **Per-Class F1 (Ensemble):**
  - Neutral: 74.62%
  - Light: 71.30%
  - Moderate: 84.05%
  - Severe: 85.64%

---

### FASE 6: Loss Function Engineering (Januari 2026) 🔥

#### Experiment 6A: Custom Loss Functions ✅ **COMPLETED - NEW BEST!**
- **Objective:** Tingkatkan akurasi dengan custom loss functions
- **Methods Tested:**
  1. Focal Loss (gamma=2.0, alpha=1.0)
  2. Label Smoothing (epsilon=0.1)
  3. Combined (Focal + Label Smoothing)
- **Dataset:** `data/improved/` (10,019 records)
- **Hasil:** F1-Macro **81.38%** dengan Label Smoothing 🎉
- **Status:** ✅ COMPLETE - NEW BEST RESULT!
- **Dokumentasi:** `experiments/EXPERIMENT_6A_RESULTS.md`

**Comparison Table:**
| Loss Function | F1-Macro | Improvement |
|---------------|----------|-------------|
| **Label Smoothing** | **81.38%** | **+2.19%** ⭐ |
| Combined (Focal + LS) | 81.24% | +2.05% |
| Focal Loss Only | 79.11% | -0.08% |
| Baseline (Cross-Entropy) | 79.19% | - |

**Per-Class F1 (Label Smoothing - Best):**
- Neutral: 79.83%
- Light: 74.77%
- Moderate: 85.09%
- Severe: 85.84%

**Key Findings:**
- Label Smoothing adalah pemenang yang jelas
- Focal Loss tidak bekerja karena dataset sudah well-balanced
- Label Smoothing membantu menghandle label noise dari AI-generated data
- Target International Workshop (82%) hampir tercapai (-0.62%)

---

### FASE 6: Roadmap Perbaikan (Januari 2026) 📋

**Roadmap lengkap tersedia di:** `ROADMAP_PENELITIAN.md`

**Target Akurasi:**
- Baseline saat ini: 81.38% (Label Smoothing)
- Target Konservatif: 82-85%
- Target Moderate: 85-88%
- Target Optimistik: 88-90%

**3 Dimensi Perbaikan:**
1. **Dataset Improvements:** Hard example mining, active learning, data augmentation
2. **Custom BERT Pretraining:** Extended DAPT dengan 20-50M tokens
3. **Model Architecture:** Advanced ensemble, focal loss, hierarchical classification

---

## Summary: Hasil Reproducible

| Model | Loss Function | Dataset | F1-Macro | Status |
|-------|---------------|---------|----------|--------|
| **IndoBERT Base + LS** | **Label Smoothing** | **improved (10K)** | **81.38%** | ✅ **BEST RESULT** 🏆 |
| IndoBERT Base | Combined (Focal+LS) | improved (10K) | 81.24% | ✅ Reproducible |
| IndoBERT Base | Cross-Entropy | improved (10K) | 79.19% | ✅ Reproducible |
| XLM-RoBERTa | Cross-Entropy | improved (10K) | 78.38% | ✅ Reproducible |
| Ensemble (Stacking) | Cross-Entropy | improved (10K) | 78.90% | ✅ Reproducible |
| mBERT | Cross-Entropy | improved (10K) | 77.93% | ✅ Reproducible |
| IndoBERT Base | balanced_dataset (39K) | 80.36% | ⚠️ Perlu verifikasi |
| Custom Javanese BERT v2 | balanced_dataset (39K) | 62.55% | ✅ Reproducible |
| mBERT | balanced_dataset (39K) | 54.82% | ✅ Reproducible |
| XLM-RoBERTa | balanced_dataset (39K) | 55.68% | ✅ Reproducible |
| Ensemble (Stacking) | balanced_dataset (39K) | 60.77% | ✅ Reproducible |

**Note:** Hasil IndoBERT Base dengan Label Smoothing (81.38%) adalah hasil terbaik yang reproducible saat ini. Hasil 80.36% dengan dataset 39K perlu diverifikasi ulang.

---

## File Dokumentasi Penting

| File | Deskripsi |
|------|-----------|
| `DATASET_IMPROVEMENT_REPORT.md` | Laporan lengkap dataset improvement |
| `AUDIT_EKSPERIMEN_KOMPREHENSIF.md` | Audit hasil eksperimen dan inkonsistensi |
| `ROADMAP_PENELITIAN.md` | Rencana perbaikan dan next steps |
| `WORKFLOW_DUA_KOMPUTER.md` | Workflow untuk dua komputer (GPU vs biasa) |
| `EXPERIMENT_TIMELINE.md` | File ini - kronologi eksperimen |
| `experiments/EXPERIMENT_5_RESULTS.md` | Hasil Experiment 5 (baseline 79.19%) |
| `experiments/EXPERIMENT_6A_RESULTS.md` | Hasil Experiment 6A (LS 81.38%) |

---

## Next Steps (Prioritas)

1. **[x] Label Smoothing implementation** - COMPLETED (81.38%)
2. **[ ] Weighted Ensemble dengan XGBoost** - Expected +1%
3. **[ ] Contextual Data Augmentation dengan DeepSeek API** - Expected +1%
4. **[ ] Hyperparameter Tuning** - Expected +0.5%
5. **[ ] Extended DAPT** - Expected +2-3%

**Target Short-term:** Capai 82% untuk International Workshop submission (gap: 0.62%)

---

**Status:** PHASE 6 IN PROGRESS - LABEL SMOOTHING SUCCESS
**Last Update:** 6 Januari 2026
