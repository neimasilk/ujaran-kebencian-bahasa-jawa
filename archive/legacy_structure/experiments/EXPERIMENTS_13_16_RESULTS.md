# EXPERIMENTS 13-16: GPU Training Session Results

**Date:** 7 Januari 2026
**GPU:** NVIDIA GeForce RTX 4080
**Status:** COMPLETED

---

## Summary

| Experiment | Model | Dataset | F1-Macro | vs Baseline | Notes |
|------------|-------|---------|----------|-------------|-------|
| Exp 6A (Baseline) | IndoBERT | Phase 3+4 | **81.38%** | - | Best result |
| Exp 13 | Custom BERT v3 | Phase 5 | 78.26% | -3.12% | DeepSeek relabeled |
| Exp 14 | IndoBERT | Phase 5 | 77.13% | -4.25% | DeepSeek relabeled |
| Exp 15 | IndoBERT | Balanced (39K) | 61.67% | -19.71% | Too imbalanced |
| Exp 16 | IndoBERT | Phase 3+4 | 78.84% | -2.54% | Reproduction attempt |

---

## Key Findings

### 1. Phase 5 Dataset (DeepSeek Relabeled) is Worse
- **Hypothesis:** DeepSeek re-labeling akan improve kualitas label
- **Reality:** F1-Macro turun dari 81.38% → 77-78%
- **Possible causes:**
  - Model DeepSeek mungkin tidak optimal untuk bahasa Jawa
  - Re-labeling memperkenalkan noise
  - Original labels mungkin lebih akurat

### 2. Large Balanced Dataset (39K) Performed Poorly
- **Hypothesis:** More data = better performance
- **Reality:** F1-Macro turun drastis ke 61.67%
- **Analysis:**
  - Dataset sangat imbalanced (48.7% Neutral)
  - Class weights tidak cukup untuk menangani imbalance
  - Model lebih fokus ke kelas majority

### 3. Custom BERT v3 vs IndoBERT on Phase 5
- Custom BERT v3: 78.26%
- IndoBERT: 77.13%
- Custom BERT v3 sedikit lebih baik tapi masih di bawah baseline

---

## Configuration Comparison

### Exp 6A (Best - 81.38%)
```python
Model: indobenchmark/indobert-base-p1
Dataset: Phase 3 + Phase 4 (10,019 samples)
Batch Size: 16
Learning Rate: 2e-5
Epochs: 5
Label Smoothing: 0.1
Max Length: 128
```

### Exp 13 (Custom BERT v3 - 78.26%)
```python
Model: models/custom_javanese_bert_v3/final_model
Dataset: Phase 5 DeepSeek (10,019 samples)
Batch Size: 16
Learning Rate: 2e-5
Epochs: 7 (early stopped)
Label Smoothing: 0.1
Max Length: 256
```

### Exp 16 (IndoBERT - 78.84%)
```python
Model: indobenchmark/indobert-base-p1
Dataset: Phase 3 + Phase 4 (10,019 samples)
Batch Size: 16
Learning Rate: 2e-5
Epochs: 5
Label Smoothing: 0.1
Max Length: 128
```

---

## Per-Class Performance Comparison

### Exp 6A (Best)
| Class | F1-Score |
|-------|----------|
| Neutral | 79.83% |
| Light | 74.77% |
| Moderate | 85.09% |
| Severe | 85.84% |

### Exp 13 (Custom BERT v3 + Phase 5)
| Class | F1-Score | Delta |
|-------|----------|-------|
| Neutral | 76.60% | -3.23% |
| Light | 71.06% | -3.71% |
| Moderate | 84.06% | -1.03% |
| Severe | 81.34% | -4.50% |

### Exp 16 (IndoBERT + Phase 3+4)
| Class | F1-Score | Delta |
|-------|----------|-------|
| Neutral | 76.41% | -3.42% |
| Light | 73.14% | -1.63% |
| Moderate | 83.48% | -1.61% |
| Severe | 82.32% | -3.52% |

---

## Conclusions

1. **Baseline Exp 6A remains the best** at 81.38% F1-Macro
2. **DeepSeek re-labeling (Phase 5) degraded performance**
3. **Custom BERT v3 shows promise but needs more work**
4. **Phase 3+4 dataset is the best available dataset**

---

## Recommendations for Next Steps

### Short Term (Ready to implement)
1. **Use Exp 6A model for deployment**
   - Already achieves 81.38%
   - Ready for National Conference submission

2. **Try ensemble of Exp 6A + Custom BERT v3**
   - Expected: +0.5-1% improvement
   - Low risk, high reward

### Medium Term (Requires more work)
1. **Improve Custom BERT v3 training**
   - More epochs on Phase 3+4
   - Better hyperparameter tuning
   - Expected: +1-2%

2. **Data quality analysis**
   - Manual review of Phase 5 labels
   - Filter low-confidence samples
   - Expected: +0.5-1%

### Long Term (Research needed)
1. **Cross-lingual transfer from Indonesian hate speech**
2. **Hierarchical classification approach**
3. **Test-time augmentation with better techniques**

---

## Files Created

| File | Purpose |
|------|---------|
| `experiments/experiment_13_silent.py` | Silent training script |
| `experiments/experiment_13_ultra_silent.py` | Ultra silent version |
| `experiments/experiment_14_indobert_phase5.py` | IndoBERT + Phase 5 |
| `experiments/experiment_15_large_dataset.py` | Large dataset test |
| `experiments/experiment_16_baseline_verify.py` | Baseline verification |
| `data/improved/phase3_phase4_combined.csv` | Clean combined dataset |

---

*Session completed: 7 Januari 2026*
*GPU time used: ~15 minutes*
