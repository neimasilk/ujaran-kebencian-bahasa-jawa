# Experiment 1: IndoBERT Large Fine-tuning Results

## Overview
Eksperimen fine-tuning IndoBERT Large untuk klasifikasi ujaran kebencian dalam bahasa Jawa.

## Dataset Information
- **Dataset:** Balanced dataset (standardized)
- **Path:** `data/standardized/balanced_dataset.csv`
- **Classes:** 4 (Bukan Ujaran Kebencian, Ujaran Kebencian - Ringan/Sedang/Berat)

## Model Configuration
- **Model:** `indobenchmark/indobert-large-p1`
- **Max Length:** 256 tokens
- **Batch Size:** 4 (reduced for memory optimization)
- **Gradient Accumulation Steps:** 4
- **Learning Rate:** 2e-5
- **Epochs:** 5
- **Warmup Ratio:** 0.2
- **Weight Decay:** 0.01

## Training Configuration
- **Loss Function:** Weighted Focal Loss
- **Class Weights:** {0: 1.0, 1: 3.0, 2: 2.5, 3: 3.5}
- **Early Stopping:** Patience 5, Threshold 0.01
- **Mixed Precision:** FP16 enabled

## Results Summary

### Overall Performance
- **Accuracy:** 0.4516 (45.16%)
- **F1-Score Macro:** 0.3884 (38.84%)
- **Training Time:** 1,182.74 seconds (~19.7 minutes)
- **Total Experiment Time:** 1,204.45 seconds (~20.1 minutes)

### Baseline Comparison
- **Baseline F1-Score:** 0.8036 (target)
- **Current F1-Score:** 0.3884
- **Improvement:** -0.4152 (below baseline)

### Per-Class Performance

| Class | F1-Score | Precision | Recall |
|-------|----------|-----------|--------|
| Bukan Ujaran Kebencian | 0.0312 | 0.6061 | 0.0160 |
| Ujaran Kebencian - Ringan | 0.5268 | 0.3776 | 0.8710 |
| Ujaran Kebencian - Sedang | 0.3810 | 0.4369 | 0.3379 |
| Ujaran Kebencian - Berat | 0.6145 | 0.6511 | 0.5817 |

## Technical Details

### Memory Optimization
- Batch size dikurangi dari 8 menjadi 4 untuk menghindari CUDA out of memory
- Gradient accumulation steps ditingkatkan menjadi 4 untuk mempertahankan effective batch size
- Mixed precision (FP16) digunakan untuk efisiensi memori

### Training Progress
- Model berhasil dimuat dan training berjalan tanpa error
- Evaluasi dilakukan setiap 50 steps
- Progress menunjukkan peningkatan F1-macro dari 0.228 (epoch 0.16) menjadi 0.388 (epoch 0.56)

## Issues Identified

### Model Performance
1. **Underperforming:** F1-macro 0.3884 jauh di bawah target 0.8036
2. **Class Imbalance:** Performa sangat buruk untuk kelas "Bukan Ujaran Kebencian" (F1: 0.0312)
3. **Overfitting pada Minority Classes:** Model cenderung memprediksi kelas ujaran kebencian

### Technical Issues
1. **Model Saving:** Model tidak tersimpan di lokasi yang diharapkan (`models/indobert_large_hate_speech`)
2. **Results File:** File hasil JSON tidak ditemukan meskipun log menunjukkan eksperimen selesai
3. **Memory Constraints:** Memerlukan optimisasi batch size untuk IndoBERT Large

## Files Generated
- **Checkpoints:** `experiments/results/experiment_1_indobert_large/checkpoint-*`
- **Confusion Matrix:** `experiments/results/experiment_1_indobert_large/confusion_matrix.png`
- **Expected Results:** `experiment_1_results.json` (tidak ditemukan)
- **Expected Model:** `models/indobert_large_hate_speech` (tidak ditemukan)

## Next Steps

### Immediate Actions
1. **Debug Model Saving:** Investigasi mengapa model dan hasil tidak tersimpan
2. **Class Balance:** Implementasi strategi yang lebih efektif untuk menangani ketidakseimbangan kelas
3. **Hyperparameter Tuning:** Eksperimen dengan learning rate dan class weights yang berbeda

### Future Experiments
1. **Data Augmentation:** Implementasi teknik augmentasi untuk kelas minoritas
2. **Different Architecture:** Coba model yang lebih kecil dengan performa lebih stabil
3. **Ensemble Methods:** Kombinasi multiple models untuk performa yang lebih baik

## Conclusion
Eksperimen IndoBERT Large berhasil dijalankan tanpa error teknis, namun performa model masih jauh dari target. Diperlukan optimisasi lebih lanjut pada strategi handling class imbalance dan hyperparameter tuning.

---
**Date:** 2025-07-07  
**Status:** Completed with suboptimal performance  
**Next Experiment:** Optimization and debugging required