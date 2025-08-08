# Analisis Hasil Hyperparameter Optimization

## Ringkasan Eksperimen
- **Tanggal**: 6 Agustus 2025
- **Total Trials**: 119
- **F1-Score Terbaik**: 0.3203 (32.03%)
- **Status**: Gagal mencapai target 90%

## Masalah Utama yang Ditemukan

### 1. Error Teknis yang Menghambat Optimasi

#### A. Error 'stratify' Parameter (Trial 0-49)
- **Masalah**: `NDFrame.sample() got an unexpected keyword argument 'stratify'`
- **Dampak**: 50 trial pertama gagal total
- **Penyebab**: Versi pandas yang tidak mendukung parameter stratify
- **Status**: Sudah diperbaiki dengan train_test_split

#### B. Error Konfigurasi Training Arguments (Trial 50-68)
- **Masalah**: `--load_best_model_at_end requires the save and eval strategy to match`
- **Dampak**: 19 trial gagal karena konfigurasi tidak sesuai
- **Penyebab**: save_strategy='no' tidak cocok dengan eval_strategy='steps'
- **Status**: Sudah diperbaiki dengan save_strategy='steps'

### 2. Performa Model yang Buruk

#### A. Hasil Akhir
- **F1-Score Terbaik**: 32.03% (jauh dari target 90%)
- **Learning Rate Optimal**: 2.64e-06 (sangat kecil)
- **Batch Size Optimal**: 32
- **Weight Decay Optimal**: 0.0015

#### B. Pola Hyperparameter
- Optuna konvergen ke learning rate yang sangat kecil (1-3e-06)
- Semua trial sukses menggunakan batch_size=32
- Weight decay sangat kecil (0.001-0.003)
- Gradient accumulation steps = 4
- Max length = 512

## Analisis Penyebab Performa Buruk

### 1. Learning Rate Terlalu Kecil
- **Range yang diuji**: 1e-6 hingga 5e-5
- **Hasil optimal**: 2.64e-06 (sangat kecil)
- **Dampak**: Model tidak belajar dengan efektif
- **Indikasi**: Konvergensi prematur ke nilai yang sangat kecil

### 2. Dataset Sampling yang Tidak Optimal
- **Ukuran sample**: Hanya 20% dari dataset augmented
- **Dampak**: Representasi data tidak cukup untuk optimasi
- **Masalah**: Stratified sampling mungkin tidak seimbang

### 3. Epoch Training Terlalu Sedikit
- **Jumlah epoch**: Hanya 2 epoch per trial
- **Dampak**: Model tidak sempat belajar dengan baik
- **Masalah**: Trade-off antara kecepatan dan akurasi

### 4. Range Hyperparameter Tidak Optimal
- **Learning rate**: Range terlalu rendah
- **Focal loss parameters**: Mungkin tidak sesuai dengan karakteristik data
- **Warmup ratio**: Range terlalu sempit

## Perbandingan dengan Eksperimen Sebelumnya

| Eksperimen | Akurasi | F1-Score | Keterangan |
|------------|---------|----------|-----------|
| Baseline | 66.00% | 66.00% | IndoBERT standar |
| Data Augmentation | 72.97% | 73.01% | Dengan augmentasi |
| Ensemble Advanced | 86.86% | - | Meta-learner ensemble |
| **Hyperparameter Opt** | **32.03%** | **32.03%** | **Gagal total** |

## Rekomendasi Perbaikan

### 1. Perbaikan Immediate

#### A. Perbaiki Range Hyperparameter
```python
# Range yang disarankan
learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)  # Naikkan range
batch_size = trial.suggest_categorical('batch_size', [8, 16])  # Fokus ke batch kecil
weight_decay = trial.suggest_float('weight_decay', 0.01, 0.1, log=True)  # Naikkan range
warmup_ratio = trial.suggest_float('warmup_ratio', 0.1, 0.3)  # Perluas range
```

#### B. Tingkatkan Epoch Training
```python
num_train_epochs = 3  # Dari 2 menjadi 3
```

#### C. Gunakan Dataset Lebih Besar
```python
# Gunakan 50% dataset untuk optimasi
df_sample = train_test_split(X_full, y_full, test_size=0.5, stratify=y_full)
```

### 2. Perbaikan Advanced

#### A. Multi-Stage Optimization
1. **Stage 1**: Optimasi learning rate dan batch size (10 trials)
2. **Stage 2**: Optimasi focal loss parameters (10 trials)
3. **Stage 3**: Fine-tuning semua parameter (30 trials)

#### B. Early Stopping yang Lebih Agresif
```python
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,  # Dari default
    early_stopping_threshold=0.01
)
```

#### C. Validasi Cross-Validation
- Gunakan 3-fold CV untuk hasil yang lebih robust
- Rata-rata hasil dari multiple folds

### 3. Strategi Alternatif

#### A. Grid Search Terfokus
- Fokus pada range yang sudah terbukti baik dari eksperimen sebelumnya
- Learning rate: 1e-5, 2e-5, 3e-5
- Batch size: 8, 16

#### B. Manual Hyperparameter Tuning
- Mulai dari konfigurasi eksperimen data augmentation yang sukses
- Lakukan fine-tuning manual step-by-step

#### C. Ensemble Hyperparameter
- Kombinasikan beberapa model dengan hyperparameter berbeda
- Voting atau averaging untuk hasil final

## Kesimpulan

### Status Saat Ini
- ❌ **Hyperparameter optimization gagal total**
- ❌ **Performa jauh di bawah baseline (32% vs 66%)**
- ❌ **Tidak mencapai target 90%**

### Langkah Selanjutnya
1. **Prioritas Tinggi**: Perbaiki range hyperparameter dan jalankan ulang
2. **Prioritas Sedang**: Implementasi multi-stage optimization
3. **Prioritas Rendah**: Eksplorasi strategi alternatif

### Rekomendasi Eksperimen Berikutnya
1. **Manual Fine-tuning**: Mulai dari konfigurasi data augmentation yang sukses
2. **Ensemble Method**: Fokus pada multi-architecture ensemble
3. **Advanced Techniques**: Cross-validation dan regularization yang lebih baik

---

**Catatan**: Hasil ini menunjukkan bahwa hyperparameter optimization memerlukan konfigurasi yang sangat hati-hati. Kegagalan ini memberikan pembelajaran berharga untuk eksperimen selanjutnya.