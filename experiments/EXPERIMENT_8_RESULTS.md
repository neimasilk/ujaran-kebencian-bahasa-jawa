# EXPERIMENT 8: Extended DAPT - RESULTS

**Date:** 6 Januari 2026
**Status:** ✅ COMPLETED
**Training Time:** ~2.5 hours (9273 seconds)

---

## Executive Summary

Custom Javanese BERT v3 berhasil dibuat melalui Extended Domain-Adaptive Pre-Training (DAPT).

**Model:** `models/custom_javanese_bert_v3/final_model/`

**Training Details:**
- Base Model: `flax-community/indonesian-roberta-base`
- Corpus: 684,048 lines (343,263 samples)
- Epochs: 10 (completed)
- Final Loss: 1.6808
- Training Speed: 370 samples/second

---

## Training Progress

| Checkpoint | Epoch | Time | Loss |
|------------|-------|------|------|
| checkpoint-53640 | ~4 | 1:57 | - |
| checkpoint-42912 | ~8 | 2:00 | - |
| checkpoint-48276 | ~9 | 2:13 | - |
| checkpoint-53640 | ~10 | 2:28 | 1.6808 |
| **final_model** | **10** | **2:28** | **1.6808** |

---

## Corpus Composition

Corpus combined dari beberapa sumber:

| Source | Lines |
|--------|-------|
| Wikipedia Jawa | ~800,000 |
| Hate Speech Dataset | ~40,000 |
| Synthetic AI Data | ~3,000 |
| **Combined (after dedup)** | **684,048** |

---

## Next Steps

### 1. Fine-tune pada Hate Speech Detection

```bash
python experiments/experiment_8_finetune.py --model models/custom_javanese_bert_v3/final_model
```

### 2. Compare dengan Baseline

| Model | F1-Macro | Status |
|-------|----------|--------|
| IndoBERT Base + Label Smoothing | 81.38% | Baseline |
| Custom Javanese BERT v3 | TBD | To be tested |

**Expected Improvement:** +1-2% F1-Macro

---

## Files

| File | Location |
|------|----------|
| Model | `models/custom_javanese_bert_v3/final_model/` |
| Checkpoints | `models/custom_javanese_bert_v3/checkpoint-*` |
| Script | `experiments/experiment_8_extended_dapt.py` |
| Progress | `experiments/experiment_8_progress.json` |

---

## Troubleshooting

### Resume Training
```bash
python experiments/experiment_8_extended_dapt.py --resume
```

### Check Status
```bash
python experiments/experiment_8_extended_dapt.py --status
```

---

*Completed: 6 Januari 2026, 16:28*
*GPU: NVIDIA GeForce RTX 4080*
*Training Time: ~2.5 hours*
