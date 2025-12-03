# 📚 Dokumentasi Akademik - Sistem Deteksi Ujaran Kebencian Bahasa Jawa

## 🎯 Overview

Dokumentasi ini disusun untuk mendukung penulisan paper akademik tentang sistem deteksi ujaran kebencian bahasa Jawa. Semua eksperimen telah selesai dengan hasil yang komprehensif dan siap untuk dipublikasikan.

## 📊 Status Eksperimen

**Status Keseluruhan:** ✅ COMPLETE (7/9 eksperimen berhasil, 2 dengan masalah teknis)

### 🏆 Model Terbaik
- **XLM-RoBERTa (Improved):** F1-Macro 61.86%, Akurasi 61.95%
- **IndoBERT Large v1.2:** F1-Macro 60.75%, Akurasi 63.05%

## 📁 Struktur Dokumentasi

```
docs/academic-paper/
├── README.md                    # File ini - overview dokumentasi
├── 01-methodology/              # Metodologi penelitian
│   ├── dataset-specification.md
│   ├── model-architecture.md
│   └── evaluation-framework.md
├── 02-experiments/              # Hasil eksperimen
│   ├── experiment-summary.md
│   ├── baseline-results.md
│   ├── advanced-models.md
│   └── comparative-analysis.md
├── 03-results/                  # Analisis hasil
│   ├── performance-metrics.md
│   ├── error-analysis.md
│   └── statistical-significance.md
├── 04-discussion/               # Diskusi dan interpretasi
│   ├── findings-interpretation.md
│   ├── limitations.md
│   └── future-work.md
├── 05-appendices/               # Lampiran
│   ├── technical-details.md
│   ├── hyperparameters.md
│   └── code-repository.md
└── paper-template/              # Template paper
    ├── abstract.md
    ├── introduction.md
    ├── related-work.md
    ├── methodology.md
    ├── results.md
    ├── discussion.md
    └── conclusion.md
```

## 🎯 Kontribusi Penelitian

### 1. **Dataset Contribution**
- Dataset ujaran kebencian bahasa Jawa dengan 4 tingkat klasifikasi
- Standardisasi dan balancing dataset untuk evaluasi yang fair
- Metodologi labeling yang konsisten

### 2. **Model Development**
- Perbandingan 6 model transformer untuk bahasa Jawa
- Optimisasi konfigurasi yang menghasilkan peningkatan 25.47% pada XLM-RoBERTa
- Analisis cross-lingual transfer learning effectiveness

### 3. **Technical Innovation**
- Solusi untuk device mismatch error dalam training pipeline
- Framework evaluasi yang robust untuk class imbalance
- Metodologi improvement yang dapat direplikasi

## 📈 Key Findings

### Performance Rankings
1. **XLM-RoBERTa (Improved):** 61.86% F1-Macro
2. **IndoBERT Large v1.2:** 60.75% F1-Macro  
3. **mBERT:** 51.67% F1-Macro
4. **IndoBERT Base:** 43.22% F1-Macro
5. **IndoBERT Large v1.0:** 38.84% F1-Macro
6. **XLM-RoBERTa (Baseline):** 36.39% F1-Macro

### Technical Insights
- **Configuration optimization** lebih penting daripada pemilihan arsitektur
- **Cross-lingual models** (XLM-RoBERTa) menunjukkan potensi terbaik
- **Device management** menjadi bottleneck utama dalam implementasi

## 🔗 Referensi Cepat

### Dokumentasi Eksperimen Lengkap
- [Status Eksperimen Lengkap](../../FINAL_EXPERIMENT_STATUS_COMPLETE.md)
- [Memory Bank - Research Active](../../memory-bank/02-research-active/)

### Hasil Eksperimen Individual
- [IndoBERT Large v1.2 Results](../../EXPERIMENT_1_2_INDOBERT_LARGE_RESULTS.md)
- [mBERT Results](../../EXPERIMENT_1_3_MBERT_RESULTS.md)
- [XLM-RoBERTa Results](../../EXPERIMENT_1_2_XLM_ROBERTA_RESULTS.md)

### Source Code
- [Experiments Directory](../../experiments/)
- [Model Implementations](../../src/)

## 📝 Panduan Penggunaan

### Untuk Penulis Paper
1. Mulai dengan [experiment-summary.md](02-experiments/experiment-summary.md)
2. Gunakan [paper-template/](paper-template/) sebagai struktur dasar
3. Referensikan hasil spesifik dari folder [03-results/](03-results/)

### Untuk Reviewer
1. Lihat [methodology overview](01-methodology/) untuk memahami pendekatan
2. Periksa [comparative-analysis.md](02-experiments/comparative-analysis.md) untuk perbandingan model
3. Review [limitations.md](04-discussion/limitations.md) untuk transparansi

### Untuk Replikasi
1. Ikuti [technical-details.md](05-appendices/technical-details.md)
2. Gunakan [hyperparameters.md](05-appendices/hyperparameters.md) untuk konfigurasi
3. Akses [code-repository.md](05-appendices/code-repository.md) untuk implementasi

---

**Dibuat oleh:** Tim Peneliti Ujaran Kebencian Bahasa Jawa  
**Tanggal:** 2025-01-06  
**Status:** Ready for Academic Publication  
**Lisensi:** Academic Use Only  

---

*Dokumentasi ini mengikuti standar VIBE Coding Guide v1.4 untuk kolaborasi hibrida manusia-AI dalam penelitian akademik.*