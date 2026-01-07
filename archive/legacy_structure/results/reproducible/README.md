# Reproducible Results

Folder ini berisi hasil eksperimen yang **dapat direproduksi** dengan pipeline saat ini.

## Hasil Valid

| File | Eksperimen | F1-Macro | Script |
|------|------------|----------|--------|
| `integrated_custom_ensemble_results.json` | Custom BERT + mBERT + XLM-R | 60.77% | `experiment_integrated_custom_ensemble.py` |
| `super_ensemble_results.json` | Multi-granularity Custom BERT | 61.26% | `super_meta_ensemble_v2.py` |
| `final_meta_ensemble_90_percent_results.json` | IndoRoberta + mBERT + XLM-R | ~60% | `final_meta_ensemble_90_percent.py` |

## Cara Mereproduksi

```bash
# Untuk integrated ensemble
python experiment_integrated_custom_ensemble.py

# Untuk super ensemble
python super_meta_ensemble_v2.py
```

## Model yang Diperlukan

Pastikan model berikut tersedia di `/models/`:
- `custom_javanese_bert_v2/`
- `integrated_custom_bert/`
- `integrated_mbert/`
- `integrated_xlm_roberta/`

---

**Diverifikasi**: 5 Januari 2026
