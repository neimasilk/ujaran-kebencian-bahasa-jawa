# AUDIT KOMPREHENSIF EKSPERIMEN
## Deteksi Ujaran Kebencian Bahasa Jawa

**Tanggal Audit**: 5 Januari 2026
**Auditor**: Claude Code
**Dokumen Acuan**: Paper "Human-and-Model-in-the-Loop Ensemble Learning..."

---

## RINGKASAN EKSEKUTIF

### STATUS: INKONSISTENSI KRITIS TERIDENTIFIKASI

| Aspek | Klaim Paper | Realitas Codebase | Gap |
|-------|-------------|-------------------|-----|
| **F1-Macro Terbaik** | 94.09% (val) / 86.86% (test) | 60.77% (reproducible) | **-26.09%** |
| **Arsitektur Ensemble** | Multi-Architecture (4 model berbeda) | Self-Ensemble (1 model x3) | **BERBEDA** |
| **Dataset** | 39,494 sampel | 39,841 sampel | Inkonsisten |
| **Reproducibility** | Klaim reproducible | Tidak dapat direproduksi | **GAGAL** |

---

## 1. TEMUAN KRITIS

### 1.1 Angka 94.09% - Sumber dan Validitas

**Lokasi Data**: `results/ensemble_advanced_results.json`

```json
"model_paths": [
    "models/improved_model",
    "models/improved_model",  // SAMA!
    "models/improved_model"   // SAMA!
]
```

**Masalah**:
1. **Self-Ensemble**: Menggunakan 1 model yang sama 3 kali dengan variasi `max_length` (128, 256, 512)
2. **Bukan Multi-Architecture**: Paper mengklaim 4 arsitektur berbeda (IndoBERT, mBERT, XLM-RoBERTa, Custom BERT)
3. **Overfitting**: Validation 94.09% vs Test 86.86% = Gap 7.23%

### 1.2 Model `improved_model` - Black Box

- **Performa**: 86.88% F1-Macro
- **Dokumentasi Training**: TIDAK ADA
- **Reproducibility**: TIDAK DIKETAHUI cara mereproduksi
- **Status**: Model checkpoint ada, tapi proses training tidak terdokumentasi

### 1.3 Hasil Reproducible (Desember 2025)

| Model | F1-Macro | Status |
|-------|----------|--------|
| Custom Javanese BERT v2 | 59.83% | Reproducible |
| mBERT | 54.82% | Reproducible |
| XLM-RoBERTa | 55.68% | Reproducible (setelah fix) |
| **Ensemble (Stacking LR)** | **60.77%** | Reproducible |

**Gap dari Klaim Paper**: 94.09% - 60.77% = **33.32%**

---

## 2. TIMELINE EKSPERIMEN (Kronologis)

### Fase 1: Agustus 2025
- Baseline IndoBERT: ~60% F1
- Data Augmentation: 67.94% (target 67% tercapai)
- `improved_model` muncul: 86.88% F1 (TIDAK TERDOKUMENTASI)
- Self-Ensemble mencapai 94.09% validation

### Fase 2: Agustus-November 2025
- Berbagai eksperimen tuning threshold
- Dokumentasi yang overlap dan tidak konsisten
- Target dinaikkan ke 90%

### Fase 3: Desember 2025
- Percobaan multi-architecture ensemble yang sebenarnya
- Custom Javanese BERT v2 di-train dengan DAPT
- Hasil: hanya 60.77% F1-Macro
- Gap besar teridentifikasi

---

## 3. ANALISIS MASALAH

### 3.1 Kemungkinan Penyebab Gap 33%

1. **Data Leak pada `improved_model`**
   - Kemungkinan training data tercampur dengan test data
   - Tidak ada dokumentasi split yang digunakan

2. **Dataset Berbeda**
   - Versi dataset mungkin berubah
   - Preprocessing berbeda

3. **Konfigurasi Training Berbeda**
   - Hyperparameter tidak terdokumentasi
   - Teknik regularisasi berbeda

4. **Overfitting Ekstrem**
   - Validation-test gap 7% menunjukkan overfitting
   - Meta-learner overfit pada validation set

### 3.2 Masalah Dokumentasi

| File | Masalah |
|------|---------|
| `REKAP_SEMUA_EKSPERIMEN.md` | Mengklaim 94.09% tercapai - menyesatkan |
| `REVIEW_EKSPERIMEN_FINAL.md` | Mengklaim target 90% tercapai - tidak reproducible |
| `FINAL_EXPERIMENT_STATUS.md` | Lebih akurat (60.77%), tapi kontradiksi dengan file lain |

---

## 4. INVENTARISASI ASET

### 4.1 Model yang Ada

| Model | Lokasi | F1-Macro | Status |
|-------|--------|----------|--------|
| improved_model | `models/improved_model/` | 86.88% | **NON-REPRODUCIBLE** |
| custom_javanese_bert_v2 | `models/custom_javanese_bert_v2/` | 62.55% | Reproducible |
| integrated_custom_bert | `models/integrated_custom_bert/` | 59.83% | Reproducible |
| integrated_mbert | `models/integrated_mbert/` | 54.82% | Reproducible |
| integrated_xlm_roberta | `models/integrated_xlm_roberta/` | 55.68% | Reproducible |

### 4.2 Script Utama

| Script | Fungsi | Status |
|--------|--------|--------|
| `improved_meta_ensemble_90_percent.py` | Self-ensemble dengan improved_model | Menghasilkan 94.09% |
| `final_meta_ensemble_90_percent.py` | Multi-arch ensemble (IndoRoberta, mBERT, XLM-R) | ~60% |
| `super_meta_ensemble_v2.py` | Multi-granularity dengan Custom BERT | 61.26% |
| `train_custom_bert_v2.py` | DAPT training | Reproducible |

### 4.3 Dataset

| File | Sampel | Catatan |
|------|--------|---------|
| `data/standardized/balanced_dataset.csv` | 39,970 baris | Dataset utama |
| `data/standardized/train_dataset.csv` | - | Split training |
| `data/standardized/test_dataset.csv` | - | Split testing |
| `data/corpus/combined_corpus.txt` | ~684K lines | Untuk DAPT |

---

## 5. REKOMENDASI TINDAKAN

### 5.1 Prioritas TINGGI (Paper Integrity)

1. **Revisi Paper**
   - Ubah klaim performa ke angka yang reproducible (60.77%)
   - Atau jelaskan bahwa 86.86% (test) adalah dari metodologi yang berbeda
   - Dokumentasikan validation-test gap sebagai limitasi

2. **Investigasi `improved_model`**
   - Cari log training asli
   - Verifikasi apakah ada data leak
   - Dokumentasikan jika memang valid

### 5.2 Prioritas SEDANG (Reproducibility)

3. **Standardisasi Pipeline**
   - Buat satu script master yang reproducible end-to-end
   - Dokumentasikan semua hyperparameter
   - Freeze dataset version

4. **Perbaiki Multi-Architecture Ensemble**
   - Debug XLM-RoBERTa training
   - Tune hyperparameter per model
   - Target realistis: 70-75% F1-Macro

### 5.3 Prioritas RENDAH (Cleanup)

5. **Arsipkan File Redundan**
   - Pindahkan dokumentasi yang menyesatkan ke `/archive/deprecated_docs/`
   - Bersihkan hasil eksperimen yang tidak relevan

6. **Konsolidasi Dokumentasi**
   - Buat satu source of truth untuk status eksperimen
   - Hapus file yang kontradiktif

---

## 6. STRUKTUR FOLDER YANG DIREKOMENDASIKAN

```
ujaran-kebencian-bahasa-jawa/
├── AUDIT_EKSPERIMEN_KOMPREHENSIF.md  <- INI FILE
├── README.md                          <- Update dengan status sebenarnya
├── data/
│   ├── standardized/                  <- Dataset utama (FREEZE)
│   └── corpus/                        <- Untuk DAPT
├── src/                               <- Kode modular
├── models/
│   ├── production/                    <- Model yang akan dipakai
│   │   └── custom_javanese_bert_v2/
│   └── experimental/                  <- Model eksperimen
├── results/
│   ├── reproducible/                  <- Hasil yang bisa direproduksi
│   └── historical/                    <- Hasil lama (untuk referensi)
├── scripts/
│   ├── train_reproducible.py          <- Script training utama
│   └── evaluate_reproducible.py       <- Script evaluasi utama
├── docs/
│   ├── paper/                         <- Paper dan revisi
│   └── technical/                     <- Dokumentasi teknis
└── archive/                           <- File lama
    ├── deprecated_docs/
    └── legacy_scripts/
```

---

## 7. KESIMPULAN

### Status Proyek Saat Ini
- **Hasil Reproducible**: 60.77% F1-Macro (Ensemble Stacking)
- **Model Terbaik (Reproducible)**: Custom Javanese BERT v2 (62.55%)
- **Gap dengan Klaim Paper**: ~33%

### Langkah Selanjutnya
1. Keputusan: Revisi paper atau investigasi lebih lanjut?
2. Jika revisi: Update angka ke hasil reproducible
3. Jika investigasi: Cari sumber `improved_model` dan verifikasi

### Catatan Penting
Paper saat ini **tidak boleh dipublikasikan** dengan angka 94.09% tanpa:
- Reproduksi hasil yang dapat diverifikasi, ATAU
- Penjelasan transparan tentang metodologi dan limitasi

---

**Diaudit oleh**: Claude Code
**Tanggal**: 5 Januari 2026
**Status Dokumen**: FINAL
