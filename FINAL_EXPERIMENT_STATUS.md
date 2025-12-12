# 📊 STATUS EKSPERIMEN TERAKHIR - UJARAN KEBENCIAN BAHASA JAWA

## 🎯 Target dan Pencapaian
- **Target**: F1-Macro 90%
- **Pencapaian Terbaik (Historical)**: 94.09%
- **Pencapaian Terakhir (Reproducible Pipeline)**: 66.27% (Integrated Ensemble)
- **Status**: Optimasi Ensemble Berhasil

## 📈 Timeline Eksperimen Terbaru (Desember 2025)

### 15. Integrasi Custom Model ke Ensemble (✅ SELESAI)
- **Masalah Awal**: XLM-RoBERTa gagal training (F1 19%).
- **Solusi**: Implementasi `WeightedTrainer` dan Gradient Accumulation.
- **Hasil**: XLM-RoBERTa pulih ke **F1 55.68%**.
- **Skor Ensemble Akhir**: **66.27% Accuracy / 60.77% F1-Macro**.
- **Insight**: Teknik Weighted Loss berhasil menyelamatkan model multilingual yang kolaps akibat class imbalance.

## 🏆 Komposisi Tim Ensemble Saat Ini

| Model | Peran | Status Performa |
|---|---|---|
| **Custom Javanese BERT v2** | **Kapten** | ⭐ **59.8%** (Solid) |
| XLM-RoBERTa | Co-Kapten | ✅ **55.7%** (Fixed!) |
| mBERT | Pendukung | 54.8% (Stabil) |

## 🔧 Next Steps
1.  **Push to 80%**: Diperlukan dataset yang lebih besar atau arsitektur meta-learner yang lebih canggih (e.g., Deep Meta-Learner).
2.  **Hyperparameter Tuning**: Tuning bobot loss agar tidak terlalu agresif.

---

**Status**: STABIL (Reproducible)
**Current Best**: 60.77% F1-Macro (Stacking Ensemble)
**Date**: 2025-12-12