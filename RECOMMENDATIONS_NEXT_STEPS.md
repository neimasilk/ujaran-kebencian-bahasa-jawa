# 🚀 Rekomendasi Langkah Selanjutnya (Roadmap to 80%+)

Berdasarkan hasil eksperimen terakhir (Ensemble F1: 60.77%), berikut adalah strategi teknis untuk sesi berikutnya:

## 1. Hybrid Loss Strategy
**Masalah**: Penggunaan `WeightedTrainer` sangat efektif untuk XLM-RoBERTa (naik dari 19% ke 55%), namun sedikit menurunkan performa Custom Javanese BERT (turun dari 61.8% ke 59.8%).
**Solusi**: Terapkan strategi training hybrid.
*   **XLM-RoBERTa**: Tetap gunakan **Weighted Cross Entropy** (Wajib).
*   **Custom BERT & mBERT**: Gunakan **Standard Loss** (atau bobot yang sangat halus, e.g., `weights^0.5`).
*   **Tujuannya**: Memaksimalkan kekuatan masing-masing model tanpa mengorbankan model yang sudah stabil.

## 2. Advanced Meta-Learner Tuning
**Masalah**: Saat ini Stacking hanya menggunakan Logistic Regression standar.
**Solusi**: Upgrade "Otak" Ensemble.
*   Gunakan **XGBoost** atau **LightGBM** dengan hyperparameter tuning yang agresif (Optuna).
*   Coba **Neural Meta-Learner** (MLP 2-layer sederhana) untuk menangkap hubungan non-linear antar prediksi model.

## 3. Data Expansion (The "Nuclear" Option)
**Masalah**: Model stuck di 60-66% mungkin karena keterbatasan variasi data training.
**Solusi**:
*   **Generate More Synthetic Data**: Fokus spesifik pada kelas yang paling sering salah (cek confusion matrix).
*   **Back-Translation Augmentation**: Jawa -> Inggris -> Jawa (menggunakan NLLB/Google Translate).

## 4. Fine-Grained Optimization
*   **Learning Rate Scheduler**: Coba `cosine_with_restarts` untuk menghindari local minima.
*   **Layer-wise Learning Rate Decay**: Beri learning rate lebih kecil pada layer awal BERT dan lebih besar pada classifier head.

---
**Prioritas Besok**: Eksekusi Poin 1 (Hybrid Loss) & Poin 2 (Advanced Meta-Learner).
