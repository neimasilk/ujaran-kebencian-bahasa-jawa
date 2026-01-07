# EXPERIMENT 5: Training dengan Dataset Improved 10K+

## Overview

**Status:** READY TO RUN
**Date:** 6 Januari 2026
**Priority:** HIGH - Eksperimen pertama dengan dataset improved

---

## Background

Dataset berhasil diperbaiki melalui 4-phase pipeline dengan DeepSeek API:
- **Phase 1-3:** 4,779 records (re-labeled dengan quality verification)
- **Phase 4:** 5,240 records (AI-generated dengan konteks Indonesia)
- **Total:** 10,019 records

**Keunggulan Dataset Improved:**
1. Konteks Indonesia yang realistis (bukan terjemahan kaku dari Barat)
2. Class balance yang lebih baik (1.38:1 vs 3.31:1 sebelumnya)
3. Bahasa Jawa yang lebih natural
4. Label confidence rata-rata 86.6%

---

## Hypothesis

Dataset yang lebih representatif akan meningkatkan performa model karena:
1. Model belajar dari konteks yang relevan dengan budaya Indonesia
2. Class imbalance yang lebih baik mengurangi bias terhadap kelas mayoritas
3. Data yang lebih natural membantu model memahami nuansa bahasa Jawa

**Expected:** F1-Macro 65-70% (improvement dari baseline 62.55%)

---

## Methodology

### Model
- **Base Model:** IndoBERT Base (indobenchmark/indobert-base-p1)
- **Alternatif:** Custom Javanese BERT v2 (jika available)

### Dataset Split
- Train: 80% (~8,015 records)
- Validation: 10% (~1,002 records)
- Test: 10% (~1,002 records)

### Training Configuration
```yaml
max_length: 128
batch_size: 16
learning_rate: 2e-5
num_epochs: 5
warmup_ratio: 0.1
weight_decay: 0.01
gradient_accumulation_steps: 1
seed: 42
```

### Hardware
- **GPU:** NVIDIA GeForce RTX 4080 (16GB VRAM)
- **CUDA:** Version 12.8
- **Expected Training Time:** ~30-60 menit

---

## Evaluation Metrics

### Primary Metrics
- **F1-Score Macro:** Metrik utama untuk class imbalance
- **Accuracy:** Overall performance

### Secondary Metrics
- **F1-Score per Class:** Neutral, Light, Moderate, Severe
- **Precision/Recall Macro**
- **Confusion Matrix**

---

## Expected Results

### Conservative Estimate
| Metric | Baseline (Old Dataset) | Target (Improved Dataset) |
|--------|------------------------|---------------------------|
| F1-Macro | 62.55% | 65-68% |
| Accuracy | ~66% | 68-72% |

### Per-Class Expectations
| Class | Old F1 | Target F1 |
|-------|--------|-----------|
| Neutral | ~60% | 65-70% |
| Light | ~55% | 60-65% |
| Moderate | ~60% | 65-70% |
| Severe | ~65% | 70-75% |

---

## Comparison with Previous Experiments

| Experiment | Dataset | F1-Macro | Notes |
|------------|---------|----------|-------|
| Exp 0: Baseline IndoBERT | 39K (unbalanced) | 80.36% | ⚠️ Perlu verifikasi |
| Exp 3: Custom BERT v2 | 39K (standardized) | 62.55% | ✅ Reproducible |
| Exp 4: Ensemble | 39K (standardized) | 60.77% | ✅ Reproducible |
| **Exp 5: Improved Dataset** | **10K (improved)** | **65-70% (expected)** | 🔥 IN PROGRESS |

---

## Next Steps After Experiment 5

### If Results Meet Expectations (F1 > 65%)
1. **Experiment 6:** Re-train ensemble models dengan dataset improved
2. **Experiment 7:** Hyperparameter tuning untuk optimal performance
3. **Documentation:** Update paper dengan hasil baru

### If Results Exceed Expectations (F1 > 70%)
1. **Production Deployment:** Siapkan model untuk production
2. **Ablation Study:** Analisis komponen mana yang paling berpengaruh
3. **Paper Submission:** Siapkan untuk submission

### If Results Below Expectations (F1 < 65%)
1. **Error Analysis:** Investigasi cases yang model salah prediksi
2. **Data Augmentation:** Tambah data untuk kelas yang underperforming
3. **Architecture Search:** Coba arsitektur model lain

---

## Files

- **Script:** `experiments/experiment_5_dataset_improved.py`
- **Data:** `data/improved/phase3_relabeled.csv` + `data/improved/phase4_generated.csv`
- **Output:** `results/experiment_5_dataset_improved/`
- **Model:** `models/experiment_5_improved/`

---

## Notes

- Dataset improved jauh lebih kecil (10K vs 39K) tapi lebih berkualitas
- Trade-off quantity vs quality akan diuji dalam eksperimen ini
- Jika hasil baik, ini membuktikan bahwa kualitas > kuantitas untuk low-resource languages

---

**Status:** GPU CUDA installation in progress...
**Last Update:** 6 Januari 2026
