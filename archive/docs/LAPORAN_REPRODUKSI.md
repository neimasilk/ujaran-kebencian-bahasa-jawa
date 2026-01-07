# 📘 PANDUAN REPRODUKSI EKSPERIMEN (Reproduction Guide)
## Deteksi Ujaran Kebencian Bahasa Jawa

Dokumen ini berisi panduan lengkap untuk mereproduksi hasil eksperimen deteksi ujaran kebencian bahasa Jawa, mulai dari persiapan lingkungan, pembersihan data, hingga pelatihan model.

---

### 1. 🛠️ Persiapan Lingkungan (Environment Setup)

Pastikan Anda memiliki Python 3.8+ (Disarankan 3.10 atau 3.13) dan GPU NVIDIA (CUDA) untuk pelatihan yang efisien.

**Install Dependensi:**
```bash
pip install -r requirements.txt
pip install optuna optuna-integration[pytorch_lightning] accelerate>=0.26.0
```

**Dependensi Kunci:**
*   `torch` (PyTorch)
*   `transformers` (Hugging Face)
*   `scikit-learn`
*   `pandas`, `numpy`
*   `optuna` (untuk optimasi hyperparameter)

---

### 2. 🧹 Persiapan Data (Dataset Preparation)

Eksperimen ini menggunakan dataset yang telah dibersihkan dan distandarisasi.

**Sumber Data:**
*   File mentah: `hasil-labeling_20250702_055252.csv`
*   Lokasi standar: `data/standardized/balanced_dataset.csv`

**Proses Pembersihan (Sudah dilakukan):**
1.  **Mapping Label:**
    *   `Bukan Ujaran Kebencian` → 0
    *   `Ujaran Kebencian - Ringan` → 1
    *   `Ujaran Kebencian - Sedang` → 2
    *   `Ujaran Kebencian - Berat` → 3
2.  **Deduplikasi:** Menghapus ~1.800 duplikat teks untuk mencegah *data leakage*.
3.  **Filtering:** Menghapus teks sampah (< 10 karakter).
4.  **Balancing:** Menghitung ulang *Class Weights* untuk menangani ketidakseimbangan kelas.

**Statistik Dataset Bersih:**
*   Total Sampel: **39,841**
*   Sebaran Kelas:
    *   Class 0: 19,416 (48.7%)
    *   Class 1: 5,926 (14.9%)
    *   Class 2: 8,075 (20.3%)
    *   Class 3: 6,424 (16.1%)

---

### 3. 🚀 Menjalankan Eksperimen Utama (Ultimate Optimization)

Ini adalah skrip utama yang menghasilkan model ensemble terbaik (Baseline Jujur: ~65-68% F1-Macro).

**Perintah:**
```bash
python ultimate_90_percent_optimization.py
```

**Apa yang dilakukan skrip ini?**
1.  Memuat dataset bersih.
2.  Melatih 3 model Transformer secara independen:
    *   `indolem/indobert-base-uncased`
    *   `bert-base-multilingual-cased`
    *   `xlm-roberta-base`
3.  Menggunakan **Focal Loss** (Gamma=3.0) dan **Class Weights** untuk menangani imbalance.
4.  Membuat **Meta-Learner Ensemble** (Stacking) menggunakan Logistic Regression/Random Forest.
5.  Menyimpan hasil evaluasi ke `results/ultimate_90_percent_results.json`.

---

### 4. 🔬 Eksperimen Lanjutan (Advanced Hyperparameter Tuning)

Jika ingin meningkatkan performa lebih jauh menggunakan pencarian parameter otomatis (Optuna).

**Perintah:**
```bash
python advanced_hyperparameter_optimization.py
```

**Fitur:**
*   Mencari *Learning Rate*, *Batch Size*, *Dropout*, dll. secara otomatis.
*   Menyimpan trial terbaik ke database SQLite lokal (`optuna_studies/`).
*   Hasil disimpan di folder `results/`.

---

### 5. 📊 Hasil Eksperimen Terakhir (06 Des 2025)

Setelah pembersihan data (penghapusan duplikat):

*   **Metode:** Ultimate Ensemble (Stacking)
*   **Akurasi:** **67.69%**
*   **F1-Macro:** **64.76%**
*   **Performa per Kelas:**
    *   Normal: 74.5%
    *   Ringan: 53.8% (Paling sulit)
    *   Sedang: 60.2%
    *   Berat: 70.5%

**Catatan:** Penurunan dari skor 90%+ sebelumnya disebabkan oleh penghapusan duplikat (*data leakage*). Skor 64-67% ini adalah performa yang **jujur dan valid** untuk dataset ini.

---

### 📂 Struktur Folder Penting

*   `data/standardized/` : Dataset bersih (`balanced_dataset.csv`) dan metadata.
*   `results/` : File JSON hasil evaluasi model.
*   `models/` : (Akan dibuat) Checkpoint model yang telah dilatih.
*   `optuna_studies/` : Database histori optimasi hyperparameter.