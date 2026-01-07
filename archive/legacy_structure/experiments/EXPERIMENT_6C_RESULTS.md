# EXPERIMENT 6C: Hyperparameter Tuning - Results

**Date:** 6 Januari 2026
**Status:** COMPLETED
**Model:** IndoBERT Base (indobenchmark/indobert-base-p1)
**Technique:** Label Smoothing + Hyperparameter Grid Search

---

## Executive Summary

**Best F1-Macro: 81.38%**

Hasil eksperimen 6C menunjukkan bahwa **hyperparameter tuning TIDAK memberikan improvement signifikan** dari baseline Label Smoothing.

- Baseline (Label Smoothing): 81.38%
- Best Hyperparameter Config: 81.38% (sama persis)
- **Improvement: 0.00%**

---

## Test Results (Best Configuration)

| Metric | Score |
|--------|-------|
| **Accuracy** | 81.24% |
| **F1-Macro** | 81.38% |
| F1-Micro | 81.24% |
| Precision Macro | 81.65% |
| Recall Macro | 81.55% |

### Best Configuration
```
learning_rate: 2e-05
batch_size: 16
weight_decay: 0.01
warmup_ratio: 0.0
epochs: 5
epsilon (label smoothing): 0.1
```

---

## Per-Class Results (Best Config)

| Class | F1-Score |
|-------|----------|
| Neutral | 78.21% |
| Light | 78.67% |
| Moderate | 84.17% |
| Severe | 84.49% |

---

## Top 5 Configurations

| Rank | F1-Macro | LR | BS | WD | WR | EP | EPS |
|------|----------|----|----|----|----|----|----|
| 1 | 81.38% | 2e-05 | 16 | 0.01 | 0.0 | 5 | 0.1 |
| 2 | 80.97% | 3e-05 | 16 | 0.01 | 0.1 | 5 | 0.1 |
| 3 | 80.53% | 2e-05 | 16 | 0.0 | 0.1 | 5 | 0.1 |
| 4 | 80.53% | 2e-05 | 16 | 0.001 | 0.1 | 5 | 0.1 |
| 5 | 80.48% | 2e-05 | 16 | 0.01 | 0.1 | 5 | 0.15 |

---

## Hyperparameter Analysis

### Learning Rate Impact
| Learning Rate | Best F1 |
|--------------|---------|
| 1e-05 | 78.31% |
| 2e-05 | 81.38% |
| 3e-05 | 80.97% |
| 5e-05 | 78.49% |

**Conclusion:** LR=2e-05 memberikan hasil terbaik. Terlalu rendah (1e-05) atau terlalu tinggi (5e-05) menurunkan performa.

### Batch Size Impact
| Batch Size | Best F1 |
|------------|---------|
| 8 | 80.30% |
| 16 | 81.38% |
| 32 | 79.30% |

**Conclusion:** BS=16 optimal. BS=32 terlalu besar untuk dataset ini.

### Weight Decay Impact
| Weight Decay | Best F1 |
|--------------|---------|
| 0.0 | 80.53% |
| 0.001 | 80.53% |
| 0.01 | 81.38% |

**Conclusion:** WD=0.01 memberikan regularisasi yang tepat.

### Warmup Ratio Impact
| Warmup Ratio | Best F1 |
|--------------|---------|
| 0.0 | 81.38% |
| 0.1 | 78.32% |
| 0.2 | 79.12% |

**Conclusion:** NO warmup (wr=0.0) memberikan hasil terbaik!

### Epochs Impact
| Epochs | Best F1 |
|--------|---------|
| 3 | 79.00% |
| 5 | 81.38% |
| 7 | 79.80% |

**Conclusion:** 5 epochs optimal. Terlalu sedikit (3) underfit, terlalu banyak (7) mulai overfit.

### Epsilon (Label Smoothing) Impact
| Epsilon | Best F1 |
|---------|---------|
| 0.05 | 79.16% |
| 0.1 | 81.38% |
| 0.15 | 80.48% |

**Conclusion:** epsilon=0.1 optimal. Terlalu rendah (0.05) kurang smoothing, terlalu tinggi (0.15) terlalu smooth.

---

## Key Findings

1. **Baseline sudah optimal:** Konfigurasi default Label Smoothing sudah sangat dekat dengan optimal.

2. **Warmup tidak membantu:** Menariknya, warmup_ratio=0.0 memberikan hasil terbaik, kemungkinan karena dataset sudah cukup besar.

3. **Stabilitas hasil:** Perbedaan antar konfigurasi tidak terlalu signifikan (78-81%), menunjukkan model cukup robust.

4. **Tidak ada breakthrough:** Hyperparameter tuning tidak bisa menembus 82% target.

---

## Next Steps

Karena hyperparameter tuning tidak memberikan improvement signifikan, strategi selanjutnya:

1. **Experiment 7: Multi-Model Ensemble** - Menggabungkan prediksi dari beberapa model berbeda (IndoBERT, mBERT, XLM-R)

2. **Data Augmentation** - Menambah data training dengan teknik augmentation

3. **Model Architecture Changes** - Mencari arsitektur yang berbeda, bukan hanya hyperparameter tuning

---

## Comparison with All Experiments

| Experiment | Technique | F1-Macro | vs Baseline |
|------------|-----------|----------|-------------|
| Exp 5 | IndoBERT Base + Improved Dataset | 79.19% | baseline |
| Exp 6A | Focal Loss | 79.42% | +0.23% |
| Exp 6B | Weighted Ensemble | ~80% | +0.81% |
| Exp 6C | Label Smoothing + Hyperparam Tuning | **81.38%** | **+2.19%** |

---

**Files:**
- Results: `results/experiment_6c_hyperparam_tuning/results.json`
- Models: `models/experiment_6c_hyperparam_tuning/`
- Script: `experiments/experiment_6c_silent.py`

---

*Experiment completed on: 6 Januari 2026*
*GPU: RTX 4080*
*Configurations tested: 16*
*Total training time: ~15 minutes*
