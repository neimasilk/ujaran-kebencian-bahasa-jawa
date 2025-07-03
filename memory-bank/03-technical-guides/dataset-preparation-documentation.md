# Dokumentasi Persiapan Dataset untuk Eksperimen IndoBERT Large

## Ringkasan Eksekusi

**Tanggal:** 3 Juli 2025  
**Waktu:** 12:17:15 WIB  
**Status:** ✅ Berhasil  
**Dataset Sumber:** `src/data_collection/hasil-labeling.csv`  
**Dataset Output:** `data/processed/final_dataset.csv`  

## Proses Persiapan Dataset

### 1. Sumber Data
- **File:** `hasil-labeling.csv`
- **Lokasi:** `src/data_collection/hasil-labeling.csv`
- **Jumlah Sampel Awal:** 41,757 sampel
- **Kolom:** 8 kolom (text, label, final_label, confidence_score, response_time, labeling_method, error, original_index)

### 2. Distribusi Label Awal

| Label | Jumlah | Persentase |
|-------|--------|------------|
| Bukan Ujaran Kebencian | 20,205 | 48.39% |
| Ujaran Kebencian - Sedang | 8,600 | 20.60% |
| Ujaran Kebencian - Berat | 6,711 | 16.07% |
| Ujaran Kebencian - Ringan | 6,241 | 14.95% |

### 3. Metode Pelabelan

| Metode | Jumlah | Persentase |
|--------|--------|------------|
| deepseek_api_parallel | 22,930 | 54.91% |
| rule_based_positive | 18,827 | 45.09% |

### 4. Statistik Confidence Score

- **Mean:** 0.920
- **Std:** 0.101
- **Min:** 0.100
- **Max:** 1.000
- **Median:** 0.950

## Proses Pembersihan Data

### 1. Penghapusan Data Duplikat
- **Duplikat Ditemukan:** 1,861 teks duplikat
- **Metode:** Berdasarkan kolom `text`
- **Strategi:** Mempertahankan entri pertama (`keep='first'`)

### 2. Pembersihan Missing Values
- **Missing Text:** 0 (tidak ada)
- **Missing Final Label:** 0 (tidak ada)
- **Missing Error:** 41,757 (kolom error kosong untuk semua data)
- **Missing Original Index:** 18,827 (untuk data rule_based_positive)

### 3. Hasil Pembersihan
- **Sampel Akhir:** 39,896 sampel
- **Pengurangan Data:** 1,861 sampel (4.46%)

## Proses Pengacakan Dataset

### Konfigurasi Pengacakan
- **Random Seed:** 42 (untuk reproducibility)
- **Metode:** `pandas.DataFrame.sample(frac=1, random_state=42)`
- **Status:** ✅ Dataset berhasil diacak

### Verifikasi Pengacakan
- **Original Index Setelah Shuffle:** [nan, nan, 1506.0, nan, 40505.0]
- **Konfirmasi:** Urutan data telah berubah dari urutan asli

## Distribusi Label Setelah Pembersihan dan Pengacakan

| Label | Jumlah | Persentase |
|-------|--------|------------|
| Bukan Ujaran Kebencian | 19,479 | 48.82% |
| Ujaran Kebencian - Sedang | 8,077 | 20.25% |
| Ujaran Kebencian - Berat | 6,418 | 16.09% |
| Ujaran Kebencian - Ringan | 5,922 | 14.84% |

### Analisis Class Imbalance
- **Rasio Ketidakseimbangan:** 3.29:1
- **Kelas Mayoritas:** Bukan Ujaran Kebencian (48.82%)
- **Kelas Minoritas:** Ujaran Kebencian - Ringan (14.84%)

## Pembagian Train-Test Split

### Konfigurasi Split
- **Metode:** Stratified sampling
- **Rasio:** 80% training, 20% testing
- **Random State:** 42
- **Stratifikasi:** Berdasarkan `label_id`

### Hasil Split

#### Training Set (31,916 sampel)
| Label | Jumlah | Persentase |
|-------|--------|------------|
| Bukan Ujaran Kebencian | 15,583 | 48.83% |
| Ujaran Kebencian - Sedang | 6,461 | 20.24% |
| Ujaran Kebencian - Berat | 5,134 | 16.09% |
| Ujaran Kebencian - Ringan | 4,738 | 14.85% |

#### Testing Set (7,980 sampel)
| Label | Jumlah | Persentase |
|-------|--------|------------|
| Bukan Ujaran Kebencian | 3,896 | 48.82% |
| Ujaran Kebencian - Sedang | 1,616 | 20.25% |
| Ujaran Kebencian - Berat | 1,284 | 16.09% |
| Ujaran Kebencian - Ringan | 1,184 | 14.84% |

## Label Mapping

```python
label_mapping = {
    'Bukan Ujaran Kebencian': 0,
    'Ujaran Kebencian - Ringan': 1,
    'Ujaran Kebencian - Sedang': 2,
    'Ujaran Kebencian - Berat': 3
}
```

## File Output

### Dataset Files
1. **`data/processed/final_dataset.csv`** - Dataset lengkap yang telah dibersihkan dan diacak
2. **`data/processed/train_set.csv`** - Training set (80%)
3. **`data/processed/test_set.csv`** - Testing set (20%)

### Documentation Files
1. **`data/processed/dataset_analysis.json`** - Analisis komprehensif dataset
2. **`data_preparation.log`** - Log lengkap proses persiapan
3. **`DATASET_PREPARATION_DOCUMENTATION.md`** - Dokumentasi ini

## Struktur Dataset Final

### Kolom Dataset Final
- **`text`** - Teks ujaran dalam bahasa Jawa
- **`label`** - Label kategori hate speech (string)
- **`label_id`** - ID numerik label (0-3)
- **`confidence_score`** - Skor kepercayaan pelabelan (0.1-1.0)
- **`labeling_method`** - Metode pelabelan yang digunakan
- **`response_time`** - Waktu respons pelabelan

### Validasi Kualitas Data
- ✅ Tidak ada missing values pada kolom penting
- ✅ Tidak ada teks kosong
- ✅ Semua label berhasil dimapping
- ✅ Distribusi stratified terjaga pada train-test split
- ✅ Dataset telah diacak dengan random seed yang konsisten

## Persiapan untuk Eksperimen IndoBERT Large

### Target Eksperimen
- **Model:** IndoBERT Large
- **Baseline F1-Score Macro:** 80.36% (dari eksperimen sebelumnya)
- **Target F1-Score Macro:** >83%
- **Target Accuracy:** >85%

### Konfigurasi Training
- **Batch Size:** 8 (disesuaikan untuk IndoBERT Large)
- **Learning Rate:** 1e-5
- **Epochs:** 5
- **Max Length:** 256 tokens
- **Loss Function:** WeightedFocalLoss
- **Class Weights:** Berdasarkan inverse frequency

### Strategi Improvement
1. **Model Architecture:** Upgrade ke IndoBERT Large (334M parameters)
2. **Advanced Loss:** WeightedFocalLoss untuk mengatasi class imbalance
3. **Class Weighting:** Inverse frequency weighting
4. **Stratified Evaluation:** Balanced train-test split
5. **Comprehensive Metrics:** Per-class analysis dan confusion matrix

## Status Eksperimen

- ✅ **Dataset Preparation:** Selesai (3 Juli 2025, 12:17:15)
- 🔄 **Model Training:** Sedang berjalan (IndoBERT Large)
- ⏳ **Evaluation:** Menunggu training selesai
- ⏳ **Results Analysis:** Menunggu evaluation selesai

## Catatan Teknis

### Reproducibility
- **Random Seed:** 42 (konsisten di semua proses)
- **Stratified Split:** Memastikan distribusi kelas terjaga
- **Deterministic Shuffle:** Menggunakan pandas dengan random_state

### Performance Considerations
- **Memory Usage:** Dataset 39,896 sampel dapat dimuat dalam memory
- **Training Time:** Estimasi 2-3 jam untuk IndoBERT Large dengan 5 epochs
- **GPU Requirements:** Minimal 8GB VRAM untuk batch size 8

### Quality Assurance
- **Data Validation:** Semua tahap validasi passed
- **Label Consistency:** Mapping label konsisten dengan eksperimen sebelumnya
- **Distribution Preservation:** Stratified split mempertahankan distribusi asli

---

**Prepared by:** AI Research Assistant  
**Last Updated:** 3 Juli 2025, 12:17:15 WIB  
**Next Step:** Monitor IndoBERT Large training progress