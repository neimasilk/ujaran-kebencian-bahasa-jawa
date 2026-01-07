# Analisis Keselarasan: Paper vs. Implementasi Kode
**Tanggal:** 03 Desember 2025
**Dokumen Acuan:** `Human-and-Model-in-the-Loop Ensemble Learning for Javanese Hate Speech Detection_ A Sociolinguistically-Informed Approach.md`

Dokumen ini menguraikan hasil audit terhadap codebase untuk memverifikasi klaim "Standar Emas" yang tertulis di dalam paper akademik.

## 1. Ringkasan Temuan Utama (Executive Summary)

| Komponen | Klaim di Paper (Gold Standard) | Realitas di Codebase (Implementasi) | Status |
| :--- | :--- | :--- | :--- |
| **Metode Ensemble** | Stacked Transformer Ensemble (4 Model Berbeda) | **Self-Ensemble** (1 Model dengan variasi konfigurasi) | ⚠️ **DIVERGENSI** |
| **Model Dasar** | IndoBERT, mBERT, XLM-RoBERTa, Custom Javanese BERT | `models/improved_model` (kemungkinan Custom BERT) x3 | ⚠️ **DIVERGENSI** |
| **Meta-Learner** | XGBoost / Weighted Voting | XGBoost / Random Forest (diimplementasikan di skrip berbeda) | ✅ Selaras (Parsial) |
| **Akurasi Validasi** | **94.09%** | **94.09%** (ditemukan di `ensemble_advanced_results.json`) | ✅ **TERKONFIRMASI** |
| **Akurasi Test** | **86.86%** (F1-Macro) | **86.86%** (Accuracy), **86.92%** (F1-Macro) | ✅ **TERKONFIRMASI** |

**Kesimpulan:** Angka performa tinggi (94.09%) **valid dan dapat dilacak** ke file hasil eksperimen, namun metodologi yang menghasilkannya berbeda dari narasi "Multi-Architecture" di paper. Angka tersebut dihasilkan oleh teknik *Self-Ensemble* (variasi input pada model yang sama), bukan penggabungan 4 arsitektur berbeda.

---

## 2. Detail Investigasi

### A. Verifikasi Angka Hasil (The "Numbers")
Paper mengklaim:
- **Validation F1:** 94.09%
- **Test F1:** 86.86% (±0.12)

**Bukti di Codebase:**
File: `results/ensemble_advanced_results.json`
```json
"validation_results": {
    "meta_learner": {
      "accuracy": 0.9409261576971214,
      "f1_macro": 0.9409427656298969
    }
},
"final_test_results": {
    "accuracy": 0.8686160624874825,
    "f1_macro": 0.8692591043943694
}
```
**Status:** Angka cocok persis. Ini adalah file "Ground Truth".

### B. Verifikasi Arsitektur (The "Architecture")
Paper mengklaim penggunaan 4 model berbeda:
1. IndoBERT
2. mBERT
3. XLM-RoBERTa
4. Custom Javanese BERT

**Bukti di File Hasil (`ensemble_advanced_results.json`):**
```json
"model_paths": [
    "models/improved_model",
    "models/improved_model",
    "models/improved_model"
]
```
File hasil menunjukkan bahwa ensemble dibentuk dari **3 instans model yang sama** (`models/improved_model`).

**Bukti di Skrip Kode:**
1.  **`improved_meta_ensemble_90_percent.py`**:
    - Skrip ini memuat `models/improved_model` dengan 3 variasi `max_length` (128, 256, 512).
    - Ini adalah kandidat paling kuat yang menghasilkan angka 94.09%.
    - **Metode:** Self-Ensemble / Data augmentation ensemble.

2.  **`multi_architecture_ensemble_advanced.py`**:
    - Skrip ini mencoba mengimplementasikan klaim paper (menggabungkan `indobert-base-p1`, `indobert-base-uncased`, `roberta-base-indonesian`).
    - **Hasil:** F1-Macro hanya mencapai **~62.7%** (lihat `results/multi_architecture_ensemble_advanced_results.json`).
    - Ini menunjukkan bahwa pendekatan multi-arsitektur (seperti deskripsi paper) justru performanya **jauh lebih rendah** dibanding self-ensemble.

3.  **`final_meta_ensemble_90_percent.py`**:
    - Menggunakan `indoroberta`, `bert-base-multilingual-cased` (mBERT), `xlm-roberta-base`.
    - Menggunakan Random Forest.
    - Skrip ini ada, tapi tidak ada bukti file hasil yang menunjukkan skrip ini mencapai 94%.

---

## 3. Rekomendasi Penyelarasan

Untuk menyelaraskan Paper dan Codebase, kita memiliki dua opsi:

1.  **Opsi A (Ubah Narasi Paper):** Merevisi bagian metodologi paper untuk secara jujur menyatakan bahwa hasil terbaik dicapai melalui *Self-Ensemble* dengan variasi panjang token (yang merupakan teknik valid untuk *robustness*), bukan multi-arsitektur.
2.  **Opsi B (Perbaiki Kode):** Mencoba men-debug `multi_architecture_ensemble_advanced.py` untuk melihat mengapa performanya (62%) sangat jauh di bawah self-ensemble (94%). Mungkin ada bug pada normalisasi data atau inisialisasi model mBERT/XLM-R.

**Rekomendasi Saat Ini:**
Dokumentasikan `improved_meta_ensemble_90_percent.py` sebagai implementasi *de facto* dari "Standar Emas" saat ini, meskipun namanya di paper berbeda.

---

## 4. Daftar File Penting (Artifacts)

*   **Skrip Utama (Best Performer):** `improved_meta_ensemble_90_percent.py`
*   **Skrip Arsitektur (Sesuai Deskripsi Paper):** `multi_architecture_ensemble_advanced.py`
*   **Hasil Terbaik (JSON):** `results/ensemble_advanced_results.json`
*   **Model Checkpoint:** `models/improved_model` (Harus dipastikan keberadaannya)
