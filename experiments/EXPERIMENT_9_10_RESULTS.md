# EXPERIMENTS 9 & 10: Quick Wins - Results Summary

**Date:** 6 Januari 2026
**Status:** ✅ COMPLETED (No improvement)
**Baseline F1-Macro:** 81.24% (checkpoint-1503 from Experiment 6A)

---

## Overview

Dua teknik "Quick Wins" dari ROADMAP V2 diuji untuk mencapai target 82%:
1. **Experiment 9**: Threshold Optimization per-Class
2. **Experiment 10**: Test-Time Augmentation (TTA)

Kedua teknik tidak memerlukan retraining dan seharusnya memberikan improvement cepat.

---

## Experiment 9: Threshold Optimization

### Concept
Optimasi decision threshold per-class menggunakan validation set. Default argmax menggunakan threshold implicit 0.5 untuk semua kelas. Pendekatan ini mencari threshold optimal untuk setiap kelas secara terpisah.

### Method
- Binary one-vs-rest approach untuk setiap kelas
- Search threshold range: 0.2 - 0.8 dengan 61 steps
- Adjusted score: prob / threshold (lebih rendah threshold = lebih mudah predict)

### Results

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| **F1-Macro** | 81.24% | 80.13% | **-1.12%** |
| Neutral | 78.23% | 78.18% | -0.05% |
| Light | 78.33% | 76.40% | -1.93% |
| Moderate | 84.52% | 83.27% | -1.24% |
| Severe | 83.90% | 82.65% | -1.25% |

### Optimal Thresholds Found
- Neutral (Class 0): 0.360
- Light (Class 1): 0.370
- Moderate (Class 2): 0.330
- Severe (Class 3): 0.570

### Conclusion
❌ **Threshold optimization membuat performa lebih buruk.** Model sudah well-calibrated dan argmax adalah pendekatan optimal.

---

## Experiment 10: Test-Time Augmentation (TTA)

### Concept
Buat beberapa versi augmented dari setiap input dan rata-ratakan prediksi.

### Augmentations Used
1. **Random Deletion**: Hapus stopwords dengan probability 0.1
2. **Random Swap**: Tukar posisi 2 random words
3. **Character Noise**: Tambah noise character-level (simulasi typo)
4. **Lowercase Variant**: Convert ke lowercase
5. **Original**: Tanpa modifikasi

### Method
- 5 augmented versions per sample
- Average probabilities dari semua versi
- Argmax pada averaged probabilities

### Results

| Metric | Baseline | TTA (n=5) | Change |
|--------|----------|-----------|--------|
| **F1-Macro** | 81.24% | 80.44% | **-0.81%** |
| Neutral | 78.23% | 77.72% | -0.51% |
| Light | 78.33% | 76.19% | -2.14% |
| Moderate | 84.52% | 84.52% | +0.00% |
| Severe | 83.90% | 83.33% | -0.57% |

### Conclusion
❌ **TTA membuat performa lebih buruk.** Untuk hate speech detection, exact wording sangat penting. Augmentations menghilangkan signals penting.

---

## Analysis: Why Quick Wins Didn't Work

### 1. Model is Already Well-Calibrated
- Model sudah trained dengan Label Smoothing (epsilon=0.1)
- Probabilities sudah well-calibrated
- Argmax adalah decision boundary yang optimal

### 2. Hate Speech Detection is Sensitive to Wording
- Kata-kata kasar/hate speech adalah strong indicators
- Augmentations (deletion, swap, noise) menghilangkan indikator ini
- Tidak seperti image classification dimana augmentations membantu

### 3. Class Imbalance is Already Optimal
- Dataset balance ratio: 1.38:1 (near-optimal)
- Tidak perlu threshold adjustment untuk handling imbalance

---

## Next Steps

### Current Status
- **Best F1-Macro: 81.24%** (Experiment 6A - Combined Loss)
- **Target Workshop (82%):** Gap 0.76%
- **Gap to 85%:** 3.76%

### Recommended Approaches

#### Option A: Cross-Lingual Transfer (High Priority)
Transfer knowledge dari Indonesian hate speech dataset yang lebih besar.

**Expected:** +1-3%
**Effort:** Medium
**Sources:**
- Indonesian Hate Speech & Abusive Language dataset
- NLP Indonesia hate speech datasets

#### Option B: Fine-tune Custom BERT v3 (Currently Running)
Experiment 8 (Extended DAPT) sedang berjalan dengan 3 checkpoints:
- checkpoint-5364 (~epoch 1.5)
- checkpoint-10728 (~epoch 3)
- checkpoint-16092 (~epoch 4.5)

**Expected:** +1-2%
**Status:** Training in progress

#### Option C: LLM-as-Judge Re-labeling
Gunakan LLM (GPT-4/Claude) untuk re-label uncertain samples dari validation set.

**Expected:** +0.5-1%
**Cost:** ~$50-100 untuk 5K samples
**Effort:** Medium

#### Option D: Hierarchical Classification
Two-stage classification:
1. Binary: Hate vs Non-Hate
2. Multi-class: Severity (Light/Moderate/Severe)

**Expected:** +0.5-1%
**Effort:** Medium

---

## Files

- Experiment 9: `experiments/experiment_9_threshold_optimization.py`
- Experiment 10: `experiments/experiment_10_tta.py`
- Results 9: `results/experiment_9_threshold_opt/results.json`
- Results 10: `results/experiment_10_tta/results.json`

---

*Completed on: 6 Januari 2026*
*GPU: NVIDIA GeForce RTX 4080*
