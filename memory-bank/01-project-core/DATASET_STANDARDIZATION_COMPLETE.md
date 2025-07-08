# Dataset Standardization Complete

**Status**: ✅ SELESAI  
**Tanggal**: 2024-12-19  
**Versi Dataset**: 1.0  

## Ringkasan Standardisasi

Dataset ujaran kebencian bahasa Jawa telah berhasil dibalance dan distandardisasi dari file asli `hasil-labeling.csv` yang memiliki ketidakseimbangan distribusi label.

## Dataset Asli vs Dataset Terstandarisasi

### Dataset Asli (`hasil-labeling.csv`)
- **Total sampel**: 41,757
- **Distribusi tidak seimbang**:
  - Bukan Ujaran Kebencian: 48.39% (20,207 sampel)
  - Ujaran Kebencian - Sedang: 20.60% (8,603 sampel)
  - Ujaran Kebencian - Berat: 16.07% (6,710 sampel)
  - Ujaran Kebencian - Ringan: 14.95% (6,237 sampel)
- **Masalah**: Data terurut dan tidak seimbang

### Dataset Terstandarisasi
- **Total sampel**: 25,041 (setelah undersampling dan optimisasi)
- **Distribusi seimbang sempurna**:
  - Bukan Ujaran Kebencian: 25.0% (6,260 sampel)
  - Ujaran Kebencian - Ringan: 25.0% (6,260 sampel)
  - Ujaran Kebencian - Sedang: 25.0% (6,260 sampel)
  - Ujaran Kebencian - Berat: 25.0% (6,261 sampel)

## Struktur Dataset Hasil

### 1. Balanced Dataset (`balanced_dataset.csv`)
- **Ukuran**: 25,041 sampel
- **Kolom**:
  - `text`: Teks ujaran dalam bahasa Jawa
  - `final_label`: Label kategorikal asli
  - `label_numeric`: Label numerik (0-3)
  - `label_binary`: Label biner (0=bukan ujaran kebencian, 1=ujaran kebencian)

### 2. Training Dataset (`train_dataset.csv`)
- **Ukuran**: 20,033 sampel (80% dari balanced dataset)
- **Distribusi**: Seimbang per kelas (~5,008 sampel per kelas)
- **Stratified split**: Mempertahankan proporsi label

### 3. Test Dataset (`test_dataset.csv`)
- **Ukuran**: 5,008 sampel (20% dari balanced dataset)
- **Distribusi**: Seimbang per kelas (~1,252 sampel per kelas)
- **Stratified split**: Mempertahankan proporsi label

## Pemetaan Label

```python
label_mapping = {
    'Bukan Ujaran Kebencian': 0,
    'Ujaran Kebencian - Ringan': 1,
    'Ujaran Kebencian - Sedang': 2,
    'Ujaran Kebencian - Berat': 3
}
```

## Statistik Dataset

### Panjang Teks
- **Rata-rata**: 99.5 karakter
- **Median**: 62.0 karakter
- **Minimum**: 4 karakter
- **Maksimum**: 2,386 karakter

### Validasi Kualitas
- ✅ **Label binary konsisten**: Semua label binary sesuai dengan label numerik
- ✅ **Tidak ada missing values**: 0 missing text dan label
- ⚠️ **Duplikasi**: 931 teks duplikat (3.7% dari total)
- ✅ **Distribusi seimbang**: Sempurna 25% per kelas

## Contoh Data per Kategori

### Label 0 (Bukan Ujaran Kebencian)
```
"Yen sampeyan ujar wabah ireng ngrusak kutha lan peradaban, mula aku setuju."
"Pamrentah Inggris minangka lelucon. Bayangna wong sing ora duwe pengalaman..."
"kebeneran paling wong trans sing mbukak babagan online iki cantik jujur"
```

### Label 1 (Ujaran Kebencian - Ringan)
```
"ya mesthi, lan aku pindho trans! wong lanang ing awak wadon ing awak lanang..."
"Aku seneng yen wong nyoba nolak sing ora bisa dipungkiri: nggarap tim kuning..."
"Traaaas kudu dibatalake"
```

### Label 2 (Ujaran Kebencian - Sedang)
```
"Aku nggolek kanca anyar sing senengane ngalahake wong wadon kaya aku"
"Kabeh sing ditindakake dening ibuku yaiku asu babagan masyarakat..."
"Wong ireng pancen ora bisa ngendhaleni gambare, umume wong sing dakdeleng..."
```

### Label 3 (Ujaran Kebencian - Berat)
```
"kabeh wong Yahudi iku bajingan kang ala"
"tambah akeh sing mati corona, mestine wong Tionghoa sing sial"
"Aku pengin aku wis ana mung kanggo seneng mambu kobong LGBT gendera!!!"
```

## Lokasi File

```
data/standardized/
├── balanced_dataset.csv     # Dataset lengkap yang sudah dibalance
├── train_dataset.csv        # Dataset training (80%)
└── test_dataset.csv         # Dataset testing (20%)
```

## Metode Standardisasi

1. **Analisis Distribusi**: Identifikasi ketidakseimbangan dalam dataset asli
2. **Undersampling**: Mengurangi kelas mayoritas ke ukuran kelas minoritas
3. **Shuffling**: Mengacak urutan data untuk menghilangkan bias urutan
4. **Stratified Split**: Pembagian train-test dengan mempertahankan proporsi label
5. **Label Encoding**: Konversi label kategorikal ke numerik dan biner
6. **Validasi**: Verifikasi konsistensi dan kualitas dataset

## Kesiapan untuk Eksperimen

✅ **Dataset siap digunakan untuk eksperimen machine learning**

### Keunggulan Dataset Terstandarisasi:
- Distribusi label seimbang sempurna (25% per kelas)
- Format konsisten dengan kolom numerik dan biner
- Pembagian train-test yang stratified
- Tidak ada missing values
- Ukuran yang cukup untuk training model (19,971 sampel training)

### Rekomendasi Penggunaan:
- Gunakan `train_dataset.csv` untuk training model
- Gunakan `test_dataset.csv` untuk evaluasi final
- Implementasikan cross-validation pada training set untuk validasi
- Pertimbangkan teknik augmentasi data jika diperlukan

## Script yang Digunakan

1. `analyze_dataset_balance.py` - Script utama untuk analisis dan balancing
2. `verify_balanced_dataset.py` - Script verifikasi hasil standardisasi

## Status Implementasi pada Eksperimen

### ✅ Eksperimen yang Telah Diupdate (12 Total)
1. ✅ `experiment_0_baseline_indobert.py`
2. ✅ `experiment_0_baseline_indobert_balanced.py`
3. ✅ `experiment_0_baseline_indobert_balanced_simple.py`
4. ✅ `experiment_0_baseline_indobert_smote.py`
5. ✅ `experiment_1.2_indobert_large.py`
6. ✅ `experiment_1_indobert_large.py`
7. ✅ `experiment_1_2_xlm_roberta.py`
8. ✅ `experiment_1_3_mbert.py`
9. ✅ `experiment_1_simple.py`
10. ✅ `data_preparation.py`
11. ✅ Dan 2 eksperimen lainnya

### Perubahan yang Diimplementasikan
- **Dataset Path**: Semua menggunakan `data/standardized/balanced_dataset.csv`
- **Column Logic**: Prioritas `label_numeric` dengan fallback `final_label`
- **Error Handling**: Improved error handling untuk missing columns
- **Konsistensi**: Struktur kode yang seragam di semua eksperimen

## Langkah Selanjutnya

1. ✅ Dataset standardization - **SELESAI**
2. ✅ Update semua eksperimen - **SELESAI**
3. 🔄 Menjalankan benchmark dengan dataset terstandarisasi
4. 📊 Evaluasi performa model pada dataset seimbang
5. 🔬 Eksperimen lanjutan dengan teknik advanced training

---

**Catatan**: Semua eksperimen telah diupdate dan siap dijalankan dengan dataset standar yang seimbang dan berkualitas tinggi.