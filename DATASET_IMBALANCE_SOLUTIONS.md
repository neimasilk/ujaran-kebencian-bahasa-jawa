# Solusi Ketidakseimbangan Dataset Ujaran Kebencian Bahasa Jawa

## 📊 Analisis Masalah

### Distribusi Label Dataset
Dataset `final_dataset_shuffled.csv` mengalami ketidakseimbangan yang signifikan:

| Label | Jumlah Sampel | Persentase |
|-------|---------------|------------|
| Bukan Ujaran Kebencian | 20,205 | 48.39% |
| Ujaran Kebencian - Sedang | 8,600 | 20.60% |
| Ujaran Kebencian - Berat | 6,711 | 16.07% |
| Ujaran Kebencian - Ringan | 6,241 | 14.95% |

**Rasio Ketidakseimbangan: 3.24:1**

### Dampak pada Model
1. **Bias terhadap kelas mayoritas**: Model cenderung memprediksi "Bukan Ujaran Kebencian"
2. **Performa buruk pada kelas minoritas**: Terutama "Ujaran Kebencian - Ringan"
3. **F1-Score Macro rendah**: 30.01% pada eksperimen baseline
4. **Early stopping**: Model tidak belajar dengan optimal

## 🛠️ Solusi yang Diimplementasikan

### 1. Class Weighting (Balanced)
**File**: `experiments/experiment_0_baseline_indobert_balanced.py`

#### Fitur:
- ✅ Automatic class weight computation (`class_weight='balanced'`)
- ✅ Stratified train/validation/test splits
- ✅ Comprehensive per-class evaluation metrics
- ✅ Early stopping dengan F1-macro monitoring

#### Cara Kerja:
```python
# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_df['label_id']),
    y=train_df['label_id']
)
```

#### Keunggulan:
- Sederhana dan cepat
- Tidak mengubah ukuran dataset
- Terintegrasi dengan Transformers

#### Kekurangan:
- Mungkin tidak optimal untuk imbalance yang sangat ekstrem
- Bergantung pada implementasi loss function

### 2. SMOTE Resampling
**File**: `experiments/experiment_0_baseline_indobert_smote.py`

#### Fitur:
- ✅ SMOTE oversampling untuk minority classes
- ✅ TF-IDF feature extraction untuk SMOTE
- ✅ Synthetic sample generation
- ✅ Balanced dataset untuk training

#### Cara Kerja:
```python
# Apply SMOTE
smote = SMOTE(
    sampling_strategy='auto',  # Balance all to majority class
    k_neighbors=5,
    random_state=42
)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y_train)
```

#### Keunggulan:
- Menciptakan sampel sintetis yang realistis
- Meningkatkan representasi kelas minoritas
- Terbukti efektif untuk banyak kasus

#### Kekurangan:
- Membutuhkan waktu preprocessing lebih lama
- Ukuran dataset training menjadi lebih besar
- Risiko overfitting pada synthetic samples

## 🚀 Cara Penggunaan

### Menjalankan Eksperimen Class Weighting
```bash
cd d:/documents/ujaran-kebencian-bahasa-jawa
python experiments/experiment_0_baseline_indobert_balanced.py
```

### Menjalankan Eksperimen SMOTE
```bash
cd d:/documents/ujaran-kebencian-bahasa-jawa
python experiments/experiment_0_baseline_indobert_smote.py
```

### Menganalisis Distribusi Dataset
```bash
python analyze_label_distribution.py
```

## 📈 Monitoring dan Evaluasi

### Metrik yang Dipantau
1. **F1-Score Macro**: Rata-rata F1-score semua kelas
2. **Per-class Precision/Recall**: Untuk setiap kategori ujaran kebencian
3. **Confusion Matrix**: Untuk analisis kesalahan klasifikasi
4. **Training Time**: Efisiensi computational

### Output Files
Setiap eksperimen menghasilkan:
- `detailed_results.json`: Hasil lengkap dengan confusion matrix
- `experiment_summary.json`: Ringkasan performa
- `logs/`: TensorBoard logs untuk monitoring training

## 🔧 Konfigurasi

### Parameter Class Weighting
```python
CONFIG = {
    'use_class_weights': True,
    'stratify': True,
    'early_stopping_patience': 3,
    'metric_for_best_model': 'f1_macro'
}
```

### Parameter SMOTE
```python
CONFIG = {
    'smote_k_neighbors': 5,
    'smote_sampling_strategy': 'auto',  # atau dict untuk kontrol manual
    'random_state': 42
}
```

## 📊 Perbandingan Pendekatan

| Aspek | Class Weighting | SMOTE |
|-------|----------------|-------|
| **Kecepatan** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Memory Usage** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Efektivitas** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Kompleksitas** | ⭐⭐ | ⭐⭐⭐⭐ |
| **Risiko Overfitting** | ⭐⭐ | ⭐⭐⭐ |

## 🎯 Rekomendasi

### Untuk Eksperimen Cepat
**Gunakan Class Weighting** (`experiment_0_baseline_indobert_balanced.py`)
- Implementasi sederhana
- Hasil cepat
- Resource efficient

### Untuk Performa Optimal
**Gunakan SMOTE** (`experiment_0_baseline_indobert_smote.py`)
- Potensi performa lebih tinggi
- Dataset lebih seimbang
- Analisis mendalam

### Kombinasi Terbaik
1. **Mulai dengan Class Weighting** untuk baseline cepat
2. **Lanjut dengan SMOTE** jika performa belum optimal
3. **Bandingkan hasil** kedua pendekatan
4. **Pilih yang terbaik** berdasarkan F1-macro dan per-class metrics

## 🔍 Troubleshooting

### Error: KeyError 'label_id'
**Solusi**: Pastikan dataset memiliki kolom `final_label` atau `label`
```python
# Script otomatis membuat label_id dari final_label/label
df['label_id'] = df[label_column].map(LABEL_MAPPING)
```

### Error: SMOTE Memory Error
**Solusi**: Kurangi `max_features` di TfidfVectorizer
```python
vectorizer = TfidfVectorizer(max_features=500)  # Kurangi dari 1000
```

### Warning: UndefinedMetricWarning
**Solusi**: Normal untuk kelas dengan support=0, gunakan `zero_division=0`

## 📚 Referensi

1. **SMOTE**: Chawla, N. V., et al. "SMOTE: synthetic minority oversampling technique." JAIR (2002)
2. **Class Weighting**: King, G., & Zeng, L. "Logistic regression in rare events data." Political analysis (2001)
3. **Stratified Sampling**: Kohavi, R. "A study of cross-validation and bootstrap for accuracy estimation." IJCAI (1995)

## 📝 Changelog

### v1.0.0 (2025-01-02)
- ✅ Implementasi Class Weighting solution
- ✅ Implementasi SMOTE resampling solution
- ✅ Comprehensive evaluation metrics
- ✅ Dokumentasi lengkap
- ✅ Troubleshooting guide

---

**Catatan**: Kedua solusi ini dirancang untuk mengatasi masalah ketidakseimbangan dataset yang menyebabkan F1-Score rendah (30.01%) pada eksperimen baseline. Pilih pendekatan yang sesuai dengan kebutuhan computational dan target performa Anda.