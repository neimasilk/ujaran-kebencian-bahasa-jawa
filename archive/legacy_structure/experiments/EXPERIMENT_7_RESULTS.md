# EXPERIMENT 7: Simple Ensemble - Results

**Date:** 6 Januari 2026
**Status:** COMPLETED
**Method:** Hard/Soft/Weighted Voting dari 3 Models

---

## Executive Summary

**Best F1-Macro: 79.50% (Hard Voting)**

Ensemble TIDAK memberikan improvement dari single best model.

- Single Best (Focal Loss): 80.73%
- Best Ensemble (Hard Voting): 79.50%
- **Result: Ensemble WORSE than single model**

---

## Models Used

| Model | Individual F1 |
|-------|---------------|
| IndoBERT (Exp 5) | 78.39% |
| Label Smoothing (Exp 6C) | 80.47% |
| Focal Loss (Exp 6A) | 80.73% |

---

## Ensemble Results

| Method | F1-Macro | Accuracy |
|--------|----------|----------|
| **Hard Voting** | **79.50%** | 79.44% |
| Soft Voting | 79.26% | 79.24% |
| Weighted Voting | 79.17% | 79.14% |

---

## Per-Class Results (Best: Hard Voting)

| Class | F1-Score |
|-------|----------|
| Neutral | 75.68% |
| Light | 73.44% |
| Moderate | 85.51% |
| Severe | 83.37% |

---

## Analysis

### Why Ensemble Didn't Work?

1. **Models are too similar** - All 3 models based on IndoBERT base
2. **Predictions are highly correlated** - Ensemble works best when models have uncorrelated errors
3. **Base model sama** - Tidak ada diversity arsitektur

### What Would Work Better?

Untuk ensemble yang effective, perlu:

1. **Different base architectures:**
   - IndoBERT (Indonesia-specific)
   - mBERT (multilingual)
   - XLM-RoBERTa (multilingual large)
   - Custom Javanese BERT

2. **Different training techniques:**
   - Standard cross-entropy
   - Focal Loss
   - Label Smoothing
   - Class-balanced loss

3. **Advanced ensemble methods:**
   - Stacking dengan XGBoost/LightGBM meta-learner
   - Bayesian Model Averaging
   - Cascade ensemble

---

## Conclusion

Ensemble sederhana (voting) tidak efektif untuk kasus ini karena:
- Model-model terlalu similar
- Tidak ada complementary strengths

**Rekomendasi:** Lanjutkan ke metode lain seperti:
1. Extended DAPT untuk Custom Javanese BERT
2. Data augmentation dengan contextual Javanese
3. Cross-domain transfer learning dari Indonesian hate speech

---

**Files:**
- Results: `results/experiment_7/results.json`
- Script: `experiments/experiment_7_simple.py`

---

*Experiment completed on: 6 Januari 2026*
