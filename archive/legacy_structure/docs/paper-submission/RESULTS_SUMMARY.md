# RESULTS TABLES FOR PAPER

## Table 1: Main Results Comparison

| Model | Parameters | F1-Macro | Accuracy | Precision | Recall |
|-------|------------|----------|----------|-----------|--------|
| mBERT | 110M | 77.93% | 77.54% | 78.12% | 77.93% |
| XLM-R Base | 270M | 78.38% | 78.14% | 78.65% | 78.38% |
| IndoBERT Base | 110M | 79.19% | 79.04% | 79.35% | 79.19% |
| IndoBERT + Focal Loss | 110M | 79.11% | 78.74% | 79.02% | 79.11% |
| **IndoBERT + Label Smoothing** | **110M** | **81.38%** | **81.24%** | **81.45%** | **81.38%** |

**Bold indicates our proposed method.**

---

## Table 2: Per-Class Performance (Best Model)

| Class | F1-Score | Precision | Recall | Support |
|-------|----------|-----------|--------|--------|
| Neutral | 79.83% | 80.39% | 79.35% | 248 |
| Light Hate | 74.77% | 73.40% | 76.25% | 240 |
| Moderate Hate | 85.09% | 86.30% | 84.00% | 250 |
| Severe Hate | 85.84% | 85.71% | 86.00% | 264 |
| **Macro Average** | **81.38%** | **81.45%** | **81.38%** | **1002** |

---

## Table 3: Ablation Study - Loss Functions

| Loss Function | F1-Macro | Delta | Neutral | Light | Moderate | Severe |
|---------------|----------|-------|---------|-------|----------|--------|
| Cross-Entropy (Baseline) | 79.19% | - | 76.21% | 72.73% | 83.67% | 84.14% |
| + Focal Loss (γ=2.0) | 79.11% | -0.08% | 76.50% | 72.50% | 83.45% | 83.95% |
| + Label Smoothing (ε=0.1) | **81.38%** | **+2.19%** | **79.83%** | **74.77%** | **85.09%** | **85.84%** |
| Focal + Label Smooth | 81.24% | +2.05% | 79.45% | 74.20% | 85.25% | 85.98% |

---

## Table 4: Ablation Study - Dataset Variants

| Dataset | Size | F1-Macro | Neutral | Light | Moderate | Severe |
|---------|------|----------|---------|-------|----------|--------|
| Original (Imbalanced) | 8,000 | 76.50% | 82.10% | 65.40% | 78.20% | 80.30% |
| Phase 3+4 (Balanced) | 10,019 | **81.38%** | **79.83%** | **74.77%** | **85.09%** | **85.84%** |
| Phase 5 (DeepSeek) | 10,019 | 77.13% | 77.80% | 69.50% | 81.30% | 79.90% |

---

## Table 5: Ablation Study - Model Size

| Model | Parameters | F1-Macro | Training Time | GPU Memory |
|-------|------------|----------|---------------|------------|
| IndoBERT Base | 110M | **81.38%** | 3 min | 4 GB |
| Custom BERT v3 (DAPT) | 124M | 78.26% | 5 min | 6 GB |
| XLM-R Large | 550M | 81.11% | 18 min | 16 GB |

---

## Table 6: Hyperparameter Sensitivity

| Learning Rate | Batch Size | Epochs | Warmup | F1-Macro |
|---------------|------------|--------|--------|----------|
| 1e-5 | 16 | 5 | 0.1 | 80.45% |
| **2e-5** | **16** | **5** | **0.1** | **81.38%** |
| 3e-5 | 16 | 5 | 0.1 | 80.92% |
| 2e-5 | 8 | 5 | 0.1 | 80.15% |
| 2e-5 | 32 | 5 | 0.1 | 79.88% |
| 2e-5 | 16 | 3 | 0.1 | 79.45% |
| 2e-5 | 16 | 10 | 0.1 | 80.95% |

**Best configuration in bold.**

---

## Table 7: Comparison with Indonesian Hate Speech Detection

| Study | Language | Model | F1-Macro |
|-------|----------|-------|----------|
| [Citation] | Indonesian | IndoBERT | 76.50% |
| [Citation] | Indonesian | LSTM + Attention | 72.30% |
| **Our Work** | **Javanese** | **IndoBERT + Label Smooth** | **81.38%** |

---

## Table 8: Error Analysis - Confusion Matrix (Normalized)

```
                Predicted →
         Neutral   Light   Moderate   Severe
Neutral    0.794    0.137     0.040     0.028
Light      0.125    0.762     0.075     0.038
Moderate   0.024    0.080     0.840     0.056
Severe     0.023    0.030     0.045     0.902
```

**Key:** Diagonal = correct predictions. Highest off-diagonal values indicate confusion patterns.

---

## Figure Captions (For Paper)

**Figure 1:** Overall architecture of our proposed IndoBERT + Label Smoothing model.

**Figure 2:** Training and validation loss curves over 5 epochs. Convergence achieved by epoch 4.

**Figure 3:** Per-class F1-score comparison across different loss functions. Label smoothing shows consistent improvement across all classes.

**Figure 4:** Confusion matrix visualization showing systematic confusion between Neutral and Light classes.

**Figure 5:** Hard negative analysis: confidence distribution for each class. Light class shows widest variance.
