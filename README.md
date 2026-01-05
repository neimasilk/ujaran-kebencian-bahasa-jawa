# Javanese Hate Speech Detection: AI-Augmented Ensemble

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red) ![Status](https://img.shields.io/badge/Status-Research-yellow)

Implementasi sistem deteksi ujaran kebencian untuk bahasa Jawa menggunakan **Ensemble Learning** dengan **Domain-Adaptive Pre-Training (DAPT)**.

## Status Proyek (Januari 2026)

| Metrik | Hasil Reproducible |
|--------|-------------------|
| **F1-Macro (Ensemble)** | 60.77% |
| **F1-Macro (Custom BERT v2)** | 62.55% |
| **Accuracy** | 66.27% |

> **Catatan**: Lihat `AUDIT_EKSPERIMEN_KOMPREHENSIF.md` untuk detail lengkap tentang status eksperimen.

## Key Features

- **Dual-Engine Data Generation**: DeepSeek (Ngoko/Slang) + Gemini (Code-Switching/Krama)
- **Domain-Adaptive Pre-Training (DAPT)**: Custom Javanese BERT dari Wikipedia + Dataset + AI Synthetic Data
- **Multi-Architecture Ensemble**: Custom BERT + mBERT + XLM-RoBERTa

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Train Custom Model
```bash
python train_custom_bert_v2.py
```

### 3. Run Ensemble Experiment
```bash
python super_meta_ensemble_v2.py
```

## Struktur Folder

```
ujaran-kebencian-bahasa-jawa/
├── data/
│   ├── standardized/          # Dataset utama (39,841 sampel)
│   └── corpus/                # Corpus untuk DAPT
├── models/
│   ├── production/            # Model siap pakai
│   │   └── custom_javanese_bert_v2/
│   ├── integrated_*/          # Model ensemble components
│   └── experimental/          # Eksperimen model
├── results/
│   ├── reproducible/          # Hasil yang dapat direproduksi
│   └── historical/            # Hasil eksperimen lama
├── src/                       # Source code modular
├── archive/                   # File-file lama
│   └── deprecated_docs/       # Dokumentasi yang sudah tidak akurat
├── AUDIT_EKSPERIMEN_KOMPREHENSIF.md  # Audit status proyek
└── ROADMAP_PERBAIKAN.md              # Rencana perbaikan
```

## Hasil Eksperimen (Reproducible)

| Model | F1-Macro | Status |
|-------|----------|--------|
| Custom Javanese BERT v2 | 62.55% | Reproducible |
| mBERT (Multilingual) | 54.82% | Reproducible |
| XLM-RoBERTa | 55.68% | Reproducible |
| **Ensemble (Stacking LR)** | **60.77%** | Reproducible |

## Dokumentasi Penting

- `AUDIT_EKSPERIMEN_KOMPREHENSIF.md` - Audit lengkap semua eksperimen
- `ROADMAP_PERBAIKAN.md` - Rencana peningkatan performa
- `FINAL_EXPERIMENT_STATUS.md` - Status eksperimen terakhir
- `Technical_Report_Javanese_AI_Augmentation.md` - Laporan teknis DAPT

## Dataset

### Dataset Utama
- **Total Sampel**: 39,841
- **Kelas**: 4 (Bukan Ujaran Kebencian, Ringan, Sedang, Berat)
- **Imbalance Ratio**: 3.31:1
- **Lokasi**: `data/standardized/balanced_dataset.csv`

### Dataset Improvement (10K+ Records)
Dataset tambahan hasil perbaikan dengan AI augmentation:

| File | Records | Deskripsi |
|------|---------|-----------|
| `data/improved/phase3_relabeled.csv` | 4,779 | Data re-label dengan verifikasi kualitas |
| `data/improved/phase4_generated.csv` | 5,240 | Data baru hasil generate (DeepSeek) |
| **TOTAL** | **10,019** | Target 10,000 tercapai |

Lihat `data/improved/README.md` untuk detail lengkap.

## Paper

Paper utama: `Human-and-Model-in-the-Loop Ensemble Learning for Javanese Hate Speech Detection_ A Sociolinguistically-Informed Approach.md`

> **Catatan Penting**: Paper masih dalam proses revisi untuk menyesuaikan dengan hasil reproducible.

## License

MIT License
