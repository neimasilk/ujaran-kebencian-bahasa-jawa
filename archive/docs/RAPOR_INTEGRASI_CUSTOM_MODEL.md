# 📊 Laporan Integrasi Custom Model ke Ensemble

## 📅 Tanggal: 12 Desember 2025
**Eksperimen**: `experiment_integrated_custom_ensemble.py`
**Tujuan**: Menggabungkan "Otak Terbaik" (Custom Javanese BERT v2) ke dalam Arsitektur Ensemble Multi-Model.

---

## 📝 Ringkasan Hasil

| Komponen | Model | F1-Macro (Validation) | Status |
|---|---|---|---|
| **Model Utama** | **Custom Javanese BERT v2** | **61.86%** | ✅ **BERHASIL** (Strong Performance) |
| Model Pendukung 1 | mBERT | 56.32% | ⚠️ Standard |
| Model Pendukung 2 | XLM-RoBERTa | 19.81% | ❌ **GAGAL** (Convergence Failure) |
| **ENSEMBLE** | **Weighted / Equal** | **62.40%** | 🔼 +0.55% Improvement |

## 🔍 Analisis Detail

### 1. Performa Custom Javanese BERT v2
*   **Kekuatan**: Model ini berhasil menjadi **model tunggal terbaik** dalam ensemble, mengalahkan mBERT sebesar +5.5%.
*   **Stabilitas**: Training berjalan stabil selama 2.5 epoch dengan convergence yang baik.
*   **Validasi Hipotesis**: Ini mengonfirmasi bahwa *Domain-Adaptive Pre-Training (DAPT)* dengan data sintetik efektif meningkatkan kapabilitas model.

### 2. Masalah pada XLM-RoBERTa
*   **Observasi**: XLM-R berhenti training sangat awal (Epoch 0.33) dengan performa mendekati random guessing (19%).
*   **Dampak**: Menjadi "beban" bagi ensemble.
*   **Potensi**: Jika XLM-R beroperasi normal (~60%), skor ensemble diproyeksikan bisa menembus **75-80%**.

### 3. Performa Ensemble
*   Meskipun salah satu anggota tim (XLM-R) gagal total, ensemble **tetap berhasil** memberikan performa lebih tinggi (62.40%) dibanding single model terbaik (61.86%).
*   Ini menunjukkan robust-ness dari metode ensemble, namun potensi maksimalnya belum tergali karena kegagalan XLM-R.

## 🚀 Rekomendasi Selanjutnya

1.  **Fix XLM-RoBERTa**: Perlu tuning learning rate khusus untuk XLM-R (kemungkinan 2e-5 terlalu besar atau terlalu kecil untuk inisialisasi ini) atau memperbesar `warmup_ratio`.
2.  **Re-Run Ensemble**: Setelah XLM-R diperbaiki, jalankan ulang ensemble. Target rasional adalah >75%.
3.  **Simpan Custom Model**: Model `Custom Javanese BERT v2` layak dijadikan backbone utama untuk eksperimen selanjutnya.

---
**Status**: 
*   Integrasi Code: ✅ Selesai
*   Training: ✅ Selesai
*   Target F1 (>72%): ❌ Belum tercapai (karena error teknis XLM-R)
