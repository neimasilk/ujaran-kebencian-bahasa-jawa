# EXPERIMENT 8: Extended DAPT - Documentation & Resume Guide

**Status:** IN PROGRESS (Running in background)
**Start Time:** 6 Januari 2026, 13:53
**Estimated Duration:** 1-2 jam untuk 10 epochs

---

## Objective

Membuat **Custom Javanese BERT v3** melalui Extended Domain-Adaptive Pre-Training (DAPT):

- **Base Model:** Indonesian RoBERTa (flax-community/indonesian-roberta-base)
- **Corpus:** 684K lines (343K samples) dari:
  - Wikipedia Jawa (800K lines)
  - Hate Speech Dataset (40K lines)
  - Synthetic AI Data (3K lines)
  - Combined dan deduplicated

---

## Progress Tracking

### Cek Status Saat Ini

```bash
cd D:/documents/ujaran-kebencian-bahasa-jawa
python experiments/experiment_8_extended_dapt.py --status
```

### Cek Progress File

```bash
python -c "import json; print(json.dumps(json.load(open('experiments/experiment_8_progress.json')), indent=2))"
```

---

## Resume Instructions

### Jika Training Terhenti (Shutdown, Error, dll)

**Opsi 1: Resume dari checkpoint terakhir**
```bash
python experiments/experiment_8_extended_dapt.py --resume
```

Script akan:
- Detect checkpoint terakhir di `models/custom_javanese_bert_v3/checkpoint-*`
- Resume training dari epoch tersebut
- Lanjutkan sampai target epochs tercapai

**Opsi 2: Cek dulu status**
```bash
python experiments/experiment_8_extended_dapt.py --status
```

---

## File Locations

| File | Location |
|------|----------|
| **Script** | `experiments/experiment_8_extended_dapt.py` |
| **Progress** | `experiments/experiment_8_progress.json` |
| **Checkpoints** | `models/custom_javanese_bert_v3/checkpoint-*` |
| **Final Model** | `models/custom_javanese_bert_v3/final_model/` |
| **Logs** | `models/custom_javanese_bert_v3/logs/` |

---

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max Epochs | 10 | Bisa ditingkat jika mau lebih lama |
| Batch Size | 32 | Per device |
| Learning Rate | 2e-5 | Standard untuk DAPT |
| Gradient Accumulation | 2 | Effective batch size = 64 |
| Max Length | 128 | Sequence length |
| MLM Probability | 0.15 | Masked token percentage |

---

## Expected Outcome

Setelah training selesai:
- Model Custom Javanese BERT v3 akan tersimpan
- Bisa langsung digunakan untuk fine-tuning hate speech detection
- Expected improvement: +1-2% F1-Macro → target ~82-83%

---

## Next Steps Setelah Selesai

### 1. Fine-tune pada Hate Speech Detection

```bash
python experiments/experiment_8_finetune.py --model models/custom_javanese_bert_v3/final_model
```

### 2. Compare dengan Baseline

- Baseline (Label Smoothing): 81.38%
- Target dengan Custom BERT v3: 82-83%

---

## Troubleshooting

### "CUDA out of memory"
- Kurangi batch size di script
- Atau gunakan gradient_accumulation yang lebih besar

### "Checkpoint not found"
- Pastikan folder `models/custom_javanese_bert_v3/` ada
- Cek subfolder `checkpoint-*`

### "Progress file corrupted"
- Hapus `experiments/experiment_8_progress.json`
- Jalankan ulang script (akan mulai dari awal)

---

## Monitoring Selama Training

### Cek GPU Usage
```bash
nvidia-smi -l 1
```

### Cek Log Files
```bash
tail -f models/custom_javanese_bert_v3/logs/trainer.log
```

### Cek Checkpoint Folder
```bash
ls -la models/custom_javanese_bert_v3/
```

---

## Timeline (Estimasi)

| Phase | Duration | Status |
|-------|----------|--------|
| Dataset Prep | 5-10 menit | ✅ Done |
| Epoch 1 | 10-15 menit | 🔄 In Progress |
| Epoch 2-10 | 90-135 menit | ⏳ Pending |
| **Total** | **~2 jam** | |

---

## Notes

- Training berjalan di background
- Checkpoint disimpan setiap epoch
- Bisa resume kapan saja
- Progress tracking real-time di progress file

---

*Created: 6 Januari 2026*
*Last Updated: 13:54*
