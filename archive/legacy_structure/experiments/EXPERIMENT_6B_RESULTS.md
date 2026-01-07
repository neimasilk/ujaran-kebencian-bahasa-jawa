# EXPERIMENT 6B: Weighted Ensemble dengan XGBoost - Results

**Date:** 6 Januari 2026
**Status:** ✅ COMPLETED - NO IMPROVEMENT
**Objective:** Tingkatkan akurasi dengan advanced ensemble method

---

## Executive Summary

**F1-Macro: 80.83%** dengan XGBoost Meta-Learner

Ini adalah **penurunan** dari baseline:
- Baseline (Label Smoothing): 81.38%
- **XGBoost Ensemble: 80.83% (-0.55%)**

Ensemble tidak memberikan improvement karena model-model yang di-ensemble terlalu mirip.

---

## Results Comparison

| Method | F1-Macro | Accuracy | vs Baseline |
|--------|----------|----------|-------------|
| **Label Smoothing (Single)** | **81.38%** | **81.24%** | **BEST** |
| XGBoost Meta-Learner | 80.83% | 80.64% | -0.55% |
| XGBoost Tuned | 80.23% | 80.04% | -1.15% |
| Simple Weighted Average | 79.89% | 79.74% | -1.49% |

---

## Per-Class Results (XGBoost - Best Ensemble)

| Class | F1-Score | Precision | Recall | Support |
|-------|----------|-----------|--------|--------|
| Neutral | 79.06% | 78.79% | 79.44% | 248 |
| Light | 75.39% | 73.89% | 77.08% | 240 |
| Moderate | 84.42% | 86.28% | 82.80% | 250 |
| Severe | 84.46% | 84.73% | 84.47% | 264 |

---

## Why Ensemble Didn't Work

### 1. Model Homogeneity
Kedua base model sangat mirip:
- Arsitektur: sama (IndoBERT Base)
- Training data: sama (10,019 improved dataset)
- Hanya berbeda loss function

### 2. Lack of Diversity
Ensemble bekerja best ketika model-model memiliki:
- Prediksi yang saling melengkapi
- Error yang tidak berkorelasi
- Kelebihan di area yang berbeda

Dalam kasus ini, kedua model membuat prediksi yang sangat mirip.

### 3. Meta-Learner Overfitting
XGBoost meta-learner overfit pada validation set:
- CV F1-Macro: 77.49% +/- 7.43%
- Test F1-Macro: 80.83%

Banyak variansi dalam cross-validation menunjukkan instability.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Base Models | 2 (Label Smoothing, Combined) |
| Meta-Learner | XGBoost Classifier |
| XGBoost max_depth | 5 |
| XGBoost learning_rate | 0.1 |
| XGBoost rounds | 39 (from CV) |
| Validation | 5-fold Stratified CV |

**Hardware:**
- GPU: NVIDIA GeForce RTX 4080
- Inference Time: ~3 menit untuk predictions
- XGBoost Training: <1 detik

---

## Comparison with Previous Experiments

| Experiment | Method | F1-Macro | Status |
|------------|--------|----------|--------|
| **Exp 6A** | **Label Smoothing** | **81.38%** | **BEST** |
| Exp 6B | XGBoost Ensemble | 80.83% | Worse |
| Exp 6A | Combined Loss | 81.24% | 2nd Best |
| Exp 5 | Baseline CE | 79.19% | Baseline |
| Exp 7 | Logistic Ensemble | 78.90% | Worse |

---

## Key Findings

### 1. Single Model > Ensemble
Untuk kasus ini, single model dengan Label Smoothing mengungguli semua ensemble variants.

### 2. Model Diversity Critical
Ensemble perlu model yang diverse:
- Berbeda arsitektur (BERT, RoBERTa, etc.)
- Berbeda training data
- Berbeda approach (rule-based + ML, etc.)

### 3. Hyperparameter Sensitivity
XGBoost hyperparameter tidak banyak membantu:
- Default params: 80.83%
- Tuned params: 80.23% (worse!)

---

## Lessons Learned

1. **Don't force ensemble** jika tidak ada model diversity yang jelas
2. **Single strong model** bisa mengungguli ensemble
3. **Label Smoothing** adalah improvement yang solid dan reproducible
4. **Next focus** harus pada hal lain: hyperparameter tuning atau data augmentation

---

## Next Steps

1. ✅ Label Smoothing - COMPLETED (81.38%)
2. ❌ Weighted Ensemble - NO IMPROVEMENT
3. [ ] **Hyperparameter Tuning** - Potensi +0.5-1%
4. [ ] **Contextual Data Augmentation** - Potensi +1-2%

**Rekomendasi:** Fokus pada hyperparameter tuning atau data augmentation untuk mencapai 82%+.

---

## Files

- Script: `experiments/experiment_6b_weighted_ensemble.py`
- Results: `results/experiment_6b_weighted_ensemble/results.json`

---

*Experiment completed on: 6 Januari 2026*
*Conclusion: Ensemble tidak membantu, single Label Smoothing model tetap terbaik*
