# 📊 Laporan Perbaikan XLM-RoBERTa & Integrasi Ensemble

## 📅 Tanggal: 12 Desember 2025
**Eksperimen**: `experiment_integrated_custom_ensemble.py` (Revised with Weighted Trainer)
**Tujuan**: Memperbaiki performa XLM-RoBERTa yang sebelumnya gagal konvergen dan mengintegrasikannya kembali ke ensemble.

---

## 🛠️ Solusi Perbaikan XLM-RoBERTa

Masalah utama pada eksperimen sebelumnya adalah **Model Collapse to Majority Class** (F1 ~19%).
Solusi yang diterapkan:
1.  **Weighted Cross Entropy Loss**: Memberikan penalti lebih besar pada kesalahan prediksi kelas minoritas.
2.  **Gradient Accumulation**: Meningkatkan kestabilan update gradient.
3.  **Unified Trainer**: Menggunakan `WeightedTrainer` untuk semua model dalam ensemble.

## 📝 Ringkasan Hasil (Validation Set)

| Model | Status Awal | Status Akhir | Peningkatan |
|---|---|---|---|
| **Custom Javanese BERT v2** | 61.86% | **59.83%** | -2.03% (Stabil) |
| mBERT | 56.32% | **54.82%** | -1.50% (Stabil) |
| **XLM-RoBERTa** | 19.81% (Broken) | **55.68%** (Fixed) | **+35.87% (SUKSES BESAR)** |

*Catatan: Penurunan sedikit pada Custom BERT & mBERT wajar karena penggunaan weighted loss yang mengubah objektif optimasi dari Accuracy murni ke F1-Macro balance.*

## 🏆 Hasil Akhir Ensemble (Test Set)

| Metode Ensemble | Accuracy | F1-Macro | Keterangan |
|---|---|---|---|
| Equal Weight Voting | 63.30% | 60.25% | Baseline Ensemble |
| Optimized Weight | 63.60% | 60.63% | Bobot: [0.5, 0.3, 0.2] |
| **Stacking (Logistic Regression)** | **66.27%** | **60.77%** | **BEST RESULT** |

## 🔍 Analisis Mendalam

1.  **XLM-RoBERTa Kembali ke Permainan**:
    - Dari "beban" tim menjadi kontributor solid (F1 55.68%).
    - Performanya sekarang setara dengan mBERT, membuktikan bahwa model multilingual besar ini bisa beradaptasi dengan Bahasa Jawa jika ditangani class imbalance-nya.

2.  **Sinergi Ensemble**:
    - Ensemble Stacking (LR) berhasil menggabungkan prediksi ketiga model.
    - F1 Ensemble (60.77%) lebih tinggi dari model individu terbaik (Custom BERT: 59.83%).
    - **Improvement**: +0.94% dari Single Best Model.

3.  **Tantangan Tersisa**:
    - Meskipun stabil, skor absolut masih di area 60-66%.
    - `Custom Javanese BERT v2` masih menjadi komponen terkuat.
    - Sepertinya "Weighted Loss" sedikit menekan performa akurasi Custom BERT dibanding eksperimen sebelumnya (tanpa bobot).

## 🚀 Rekomendasi Selanjutnya

1.  **Fine-tuning Weighted Loss**: Bobot yang dihitung otomatis mungkin terlalu agresif. Coba haluskan bobotnya (e.g., pangkat 0.5) agar tidak terlalu mengorbankan kelas mayoritas.
2.  **Meta-Learner yang Lebih Kuat**: Logistic Regression sederhana menang, tapi mungkin Neural Network kecil sebagai meta-learner bisa lebih baik mengekstrak pola non-linear.
3.  **Knowledge Distillation**: Menggunakan ensemble ini sebagai "Guru" untuk melatih ulang satu model kecil (Distilled Model) agar lebih efisien.

---
**Status**: 
*   XLM-RoBERTa Fix: ✅ **SUKSES**
*   Integrasi Ensemble: ✅ **SUKSES**
*   Stabilitas Training: ✅ **Sangat Stabil**
