# Laporan Perbandingan Evaluasi Model

## Ringkasan Eksekutif

Analisis ini mengungkap masalah serius dalam evaluasi model sebelumnya yang disebabkan oleh **bias urutan dataset**. Evaluasi ulang dengan dataset seimbang menunjukkan performa model yang jauh lebih realistis dan mengidentifikasi kelemahan signifikan yang sebelumnya tersembunyi.

## Masalah yang Ditemukan

### 1. Bias Urutan Dataset
- **Dataset asli (41,758 samples)** memiliki distribusi yang sebenarnya cukup seimbang:
  - Bukan Ujaran Kebencian: ~20%
  - Ujaran Kebencian - Sedang: ~21%
  - Ujaran Kebencian - Berat: ~16%
  - Ujaran Kebencian - Ringan: ~15%

- **Namun data terurut** dengan mayoritas "Bukan Ujaran Kebencian" di awal file
- **Evaluasi sebelumnya** hanya mengambil 1,000 data pertama (95.5% "Bukan Ujaran Kebencian")

### 2. Hasil Evaluasi Bias vs Seimbang

| Metrik | Evaluasi Bias (1000 samples) | Evaluasi Seimbang (800 samples) | Selisih |
|--------|------------------------------|----------------------------------|----------|
| **Accuracy** | 95.5% | 73.8% | **-21.7%** |
| **F1-Score Macro** | ~2% (estimasi) | 73.7% | **+71.7%** |
| **F1-Score Weighted** | 97.7% | 73.7% | **-24.0%** |
| **Precision Weighted** | 100.0% | 77.6% | **-22.4%** |

## Analisis Performa Per Kelas (Evaluasi Seimbang)

### Performa Terbaik: Ujaran Kebencian - Berat
- **Precision**: 88.2% - Model sangat akurat dalam mengidentifikasi ujaran kebencian berat
- **Recall**: 82.5% - Mendeteksi sebagian besar kasus ujaran kebencian berat
- **F1-Score**: 85.3% - Performa terbaik di antara semua kelas

### Performa Terburuk: Ujaran Kebencian - Ringan
- **Precision**: 83.6% - Cukup akurat tapi banyak false positive
- **Recall**: 58.5% - **Masalah serius**: Gagal mendeteksi 41.5% ujaran kebencian ringan
- **F1-Score**: 68.8% - Performa terburuk

### Masalah Utama: Bukan Ujaran Kebencian
- **Precision**: 57.8% - **Sangat rendah**: 42.2% prediksi "Bukan Ujaran Kebencian" sebenarnya adalah ujaran kebencian
- **Recall**: 93.0% - Sangat tinggi, model cenderung over-predict kelas ini
- **Implikasi**: Model bias terhadap kelas mayoritas dalam training

## Analisis Confusion Matrix

```
Actual \ Predicted:    Bukan    Ringan   Sedang   Berat
Bukan Ujaran Kebencian:  186      6        4        4     (93.0% recall)
Ujaran Kebencian - Ringan: 71    117       10       2     (58.5% recall)
Ujaran Kebencian - Sedang: 45     17      122      16     (61.0% recall)
Ujaran Kebencian - Berat:  20      0       15     165     (82.5% recall)
```

### Temuan Kritis:
1. **71 ujaran kebencian ringan** diprediksi sebagai "Bukan Ujaran Kebencian" (35.5%)
2. **45 ujaran kebencian sedang** diprediksi sebagai "Bukan Ujaran Kebencian" (22.5%)
3. **20 ujaran kebencian berat** diprediksi sebagai "Bukan Ujaran Kebencian" (10.0%)

**Total: 136 dari 600 ujaran kebencian (22.7%) tidak terdeteksi sama sekali**

## Distribusi Prediksi

| Kelas | Prediksi | Persentase |
|-------|----------|------------|
| Bukan Ujaran Kebencian | 322/800 | 40.3% |
| Ujaran Kebencian - Ringan | 140/800 | 17.5% |
| Ujaran Kebencian - Sedang | 151/800 | 18.9% |
| Ujaran Kebencian - Berat | 187/800 | 23.4% |

## Implikasi dan Rekomendasi

### 1. Masalah Keamanan
- **22.7% ujaran kebencian tidak terdeteksi** - risiko tinggi untuk aplikasi produksi
- **42.2% false positive** pada "Bukan Ujaran Kebencian" - banyak konten berbahaya lolos

### 2. Rekomendasi Perbaikan Model

#### A. Data dan Training
1. **Stratified Sampling**: Gunakan stratified split saat training untuk memastikan distribusi seimbang
2. **Data Augmentation**: Tambah data untuk kelas "Ujaran Kebencian - Ringan" yang performanya terburuk
3. **Class Weighting**: Implementasikan class weights untuk mengatasi bias terhadap kelas mayoritas
4. **Threshold Tuning**: Sesuaikan threshold klasifikasi untuk mengurangi false negative

#### B. Arsitektur Model
1. **Focal Loss**: Gunakan focal loss untuk mengatasi class imbalance
2. **Ensemble Methods**: Kombinasikan multiple models untuk performa lebih robust
3. **Fine-tuning Strategy**: Pertimbangkan fine-tuning dengan learning rate berbeda per layer

#### C. Evaluasi
1. **Selalu gunakan dataset seimbang** untuk evaluasi
2. **Prioritaskan Recall** untuk ujaran kebencian (lebih baik false positive daripada false negative)
3. **Monitor per-class metrics** secara teratur

### 3. Deployment Strategy

#### Untuk Produksi:
1. **Conservative Threshold**: Set threshold lebih rendah untuk mengurangi false negative
2. **Human Review**: Implementasikan human review untuk kasus borderline
3. **Continuous Monitoring**: Monitor distribusi prediksi secara real-time
4. **A/B Testing**: Test model performance dengan user feedback

## Kesimpulan

Evaluasi sebelumnya dengan akurasi 95.5% **sangat menyesatkan** karena bias dataset. Evaluasi seimbang mengungkap:

1. **Akurasi sebenarnya hanya 73.8%** - jauh di bawah standar produksi
2. **22.7% ujaran kebencian tidak terdeteksi** - risiko keamanan tinggi
3. **Model bias terhadap kelas "Bukan Ujaran Kebencian"** - perlu rebalancing
4. **Performa bervariasi signifikan antar kelas** - perlu strategi khusus per kelas

**Rekomendasi utama**: Model ini **TIDAK SIAP untuk produksi** dan memerlukan perbaikan signifikan sebelum deployment.

## File Output

- **Evaluasi Bias**: `models/trained_model/evaluation_results.json`
- **Evaluasi Seimbang**: `models/trained_model/balanced_evaluation_results.json`
- **Dataset Seimbang**: `data/processed/balanced_evaluation_set.csv`
- **Script Analisis**: `analyze_dataset_distribution.py`
- **Script Evaluasi**: `balanced_evaluation.py`

---

*Laporan ini dibuat pada: 3 Juli 2025*  
*Model: Javanese Hate Speech Detection*  
*Dataset: hasil-labeling.csv (41,758 samples)*