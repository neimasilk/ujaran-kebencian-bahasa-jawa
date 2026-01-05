# Dataset Improvement Pipeline

Pipeline ini memperluas dataset ujaran kebencian bahasa Jawa dari ~4,800 menjadi **10,019 records**.

## Overview

Pipeline ini terdiri dari 4 tahap:

### Phase 1: Filtering (CPU Only)
- Menghapus data yang mengandung referensi Barat (Western-specific)
- Mengategorikan data menjadi: `keep`, `naturalize`, atau `remove`

### Phase 2: Naturalization (DeepSeek API)
- Mengubah terjemahan kaku menjadi Bahasa Jawa yang natural
- Mengadaptasi konteks Barat menjadi konteks Indonesia

### Phase 3: Re-labeling (DeepSeek API)
- Memberi label ulang dengan Chain-of-Thought reasoning
- Verifikasi kualitas bahasa Jawa yang murni

### Phase 4: Generation (DeepSeek API)
- Generate data baru dengan konteks Indonesia yang realistis
- 5 kategori: `politics`, `neighbors`, `neutral`, `severe`, `regional`

## Hasil Akhir

| Phase | File | Records | Deskripsi |
|-------|------|---------|-----------|
| Phase 1 | `phase1_keep.csv` | ~3,370 | Data yang sudah valid |
| Phase 1 | `phase1_naturalize.csv` | ~1,409 | Data perlu dinaturalisasi |
| Phase 1 | `phase1_remove.csv` | ~XXX | Data yang dihapus |
| Phase 2 | `phase2_naturalized.csv` | ~1,409 | Data hasil naturalisasi |
| Phase 3 | `phase3_relabeled.csv` | **4,779** | Data re-label dengan verifikasi |
| Phase 4 | `phase4_generated.csv` | **5,240** | Data baru yang digenerate |
| **TOTAL** | | **10,019** | **Target: 10,000** |

## Label Distribution

- **Label 0 (Neutral)**: Netral, tidak mengandung ujaran kebencian
- **Label 1 (Light)**: Ujaran kebencian ringan - sindiran halus, ejekan ringan
- **Label 2 (Moderate)**: Ujaran kebencian sedang - hinaan langsung, bahasa kasar
- **Label 3 (Severe)**: Ujaran kebencian berat - ancaman, provokasi kekerasan, dehumanisasi

## Category Distribution (Generated)

| Category | Count | Label |
|----------|-------|-------|
| politics | 1,038 | 2 (Moderate) |
| neighbors | 1,332 | 1 (Light) |
| neutral | 530 | 0 (Neutral) |
| severe | 1,265 | 3 (Severe) |
| regional | 1,075 | 2 (Moderate) |

## Usage

```python
import pandas as pd

# Load re-labeled data (Phase 3)
df_relabeled = pd.read_csv('data/improved/phase3_relabeled.csv')

# Load generated data (Phase 4)
df_generated = pd.read_csv('data/improved/phase4_generated.csv')

# Combine for full dataset
df_full = pd.concat([df_relabeled, df_generated], ignore_index=True)
```

## Requirements

- Python 3.8+
- DeepSeek API key
- `pandas`, `openai`, `python-dotenv`, `tqdm`

## Author

Dataset Improvement Team - Januari 2026
