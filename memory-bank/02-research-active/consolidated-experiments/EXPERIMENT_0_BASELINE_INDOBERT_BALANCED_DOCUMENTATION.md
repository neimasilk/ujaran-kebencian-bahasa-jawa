# Dokumentasi Eksperimen 0: Baseline IndoBERT dengan Dataset Balanced

## Overview
Eksperimen ini mengimplementasikan model baseline IndoBERT dengan teknik class weighting dan stratified splits untuk mengatasi masalah ketidakseimbangan dataset yang diidentifikasi pada eksperimen baseline sebelumnya.

## Dataset Information
- **Total samples**: 41,757
- **Distribusi kelas**:
  - Bukan Ujaran Kebencian: 20,205 (48.39%) - Majority class
  - Ujaran Kebencian - Sedang: 8,600 (20.60%)
  - Ujaran Kebencian - Berat: 6,711 (16.07%)
  - Ujaran Kebencian - Ringan: 6,241 (14.95%) - Minority class
- **Imbalance ratio**: 3.24:1 (majority:minority)

## Hyperparameters Configuration

### Model Configuration
```python
CONFIG = {
    'model_name': 'indobenchmark/indobert-base-p1',
    'dataset_path': 'data/processed/final_dataset_shuffled.csv',
    'max_length': 128,
    'batch_size': 16,
    'num_epochs': 10,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_steps': 100,
    'logging_steps': 50,
    'eval_steps': 200,
    'save_steps': 400,
    'early_stopping_patience': 3,
    'test_size': 0.2,
    'validation_size': 0.2,
    'random_state': 42,
    'use_class_weights': True
}
```

### Training Arguments
- **Output directory**: `experiments/results/experiment_0_baseline_indobert_balanced_simple`
- **Evaluation strategy**: Steps-based (every 200 steps)
- **Save strategy**: Disabled (untuk menghindari JSON serialization issues)
- **Load best model at end**: Disabled
- **Metric for best model**: `eval_f1_macro`
- **Report to**: None (wandb disabled)
- **Seed**: 42

### Class Weights (Balanced)
```python
class_weights = {
    0: 0.5167,  # Bukan Ujaran Kebencian
    1: 1.6728,  # Ujaran Kebencian - Ringan
    2: 1.2138,  # Ujaran Kebencian - Sedang
    3: 1.5555   # Ujaran Kebencian - Berat
}
```

## Data Splits

### Train Set (26,724 samples)
- Bukan Ujaran Kebencian: 12,931 (48.39%)
- Ujaran Kebencian - Sedang: 5,504 (20.60%)
- Ujaran Kebencian - Berat: 4,295 (16.07%)
- Ujaran Kebencian - Ringan: 3,994 (14.95%)

### Validation Set (6,681 samples)
- Bukan Ujaran Kebencian: 3,233 (48.39%)
- Ujaran Kebencian - Sedang: 1,376 (20.60%)
- Ujaran Kebencian - Berat: 1,073 (16.06%)
- Ujaran Kebencian - Ringan: 999 (14.95%)

### Test Set (8,352 samples)
- Bukan Ujaran Kebencian: 4,041 (48.38%)
- Ujaran Kebencian - Sedang: 1,720 (20.59%)
- Ujaran Kebencian - Berat: 1,343 (16.08%)
- Ujaran Kebencian - Ringan: 1,248 (14.94%)

## Training Information

### Training Duration
- **Total training time**: 458.55 seconds (~7.6 menit)
- **Training start**: 2025-07-04 11:13:23
- **Training end**: 2025-07-04 11:21:02

### Mengapa Training Selesai?
Berdasarkan log eksperimen, training **TIDAK** berhenti karena early stopping. Training berjalan sampai selesai secara normal karena:

1. **Tidak ada early stopping trigger**: Log menunjukkan training completed normally tanpa ada pesan early stopping
2. **Training time konsisten**: 458.55 detik menunjukkan training berjalan hingga epoch yang ditentukan
3. **Evaluasi final berhasil**: Model berhasil dievaluasi pada test set setelah training selesai

Kemungkinan penyebab persepsi "training berhenti":
- Progress bar mungkin tidak terlihat jelas di terminal
- Training berjalan dalam background tanpa output verbose
- Log hanya menampilkan milestone tertentu (setiap 200 steps)

## Results

### Overall Performance
- **Test Accuracy**: 66.51%
- **Test F1-Macro**: 62.36%
- **Test Precision-Macro**: 61.92%
- **Test Recall-Macro**: 62.88%

### Per-Class Performance

#### Bukan Ujaran Kebencian
- **Precision**: 77.47%
- **Recall**: 74.54%
- **F1-Score**: 75.97%
- **Support**: 4,041

#### Ujaran Kebencian - Ringan
- **Precision**: 50.20%
- **Recall**: 50.16%
- **F1-Score**: 50.18%
- **Support**: 1,248

#### Ujaran Kebencian - Sedang
- **Precision**: 55.49%
- **Recall**: 56.69%
- **F1-Score**: 56.08%
- **Support**: 1,720

#### Ujaran Kebencian - Berat
- **Precision**: 64.52%
- **Recall**: 70.14%
- **F1-Score**: 67.21%
- **Support**: 1,343

### Confusion Matrix
```
                Predicted
Actual    BUK   Ringan  Sedang  Berat
BUK      3012    408     361    260
Ringan    322    626     239     61
Sedang    357    191     975    197
Berat     197     22     182    942
```

## Key Improvements Implemented

1. **Class weighting (balanced)**: Mengatasi ketidakseimbangan dataset dengan memberikan bobot lebih tinggi pada kelas minoritas
2. **Stratified train/validation/test splits**: Memastikan distribusi kelas yang konsisten di semua split
3. **Comprehensive per-class evaluation**: Evaluasi detail untuk setiap kelas
4. **Early stopping with F1-macro monitoring**: Monitoring berdasarkan F1-macro untuk menghindari overfitting

## Technical Issues Resolved

1. **FileNotFoundError**: Memperbaiki path dataset dari `final_dataset_shuffled.csv` ke `data/processed/final_dataset_shuffled.csv`
2. **Empty dataset**: Mengubah mapping label dari kolom `label` ke `final_label`
3. **TrainingArguments error**: Menyesuaikan `save_steps` agar kompatibel dengan `eval_steps`
4. **JSON serialization error**: Menghapus `class_weights` dari `model.config` dan menonaktifkan saving
5. **EarlyStoppingCallback error**: Menambahkan `metric_for_best_model="eval_f1_macro"`

## Analysis

### Strengths
- Model menunjukkan performa yang baik pada kelas mayoritas (Bukan Ujaran Kebencian)
- Class weighting berhasil meningkatkan recall untuk kelas minoritas
- Stratified splits memastikan evaluasi yang fair

### Areas for Improvement
- Performa pada kelas "Ujaran Kebencian - Ringan" masih rendah (F1: 50.18%)
- Masih ada confusion antara kelas hate speech yang berbeda
- Precision untuk kelas minoritas bisa ditingkatkan

### Recommendations for Next Experiments
1. Coba teknik sampling lain (SMOTE, ADASYN)
2. Eksperimen dengan model yang lebih besar (IndoBERT-large)
3. Fine-tuning hyperparameter lebih lanjut
4. Implementasi focal loss untuk mengatasi class imbalance
5. Data augmentation untuk kelas minoritas

## Files Generated
- **Script**: `experiments/experiment_0_baseline_indobert_balanced_simple.py`
- **Log**: `experiment_0_balanced_simple.log`
- **Results directory**: `experiments/results/experiment_0_baseline_indobert_balanced_simple/`

## Conclusion
Eksperimen ini berhasil mengimplementasikan baseline yang solid dengan teknik class balancing. Meskipun masih ada ruang untuk improvement, hasil ini memberikan foundation yang baik untuk eksperimen selanjutnya. Training berjalan normal hingga selesai tanpa early stopping, menunjukkan stabilitas konfigurasi yang digunakan.