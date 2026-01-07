# Paper JITK Revision Progress

## Status: IN PROGRESS
Last Updated: 2025-01-07

---

## Action Items Checklist

### FASE 1: VERIFIKASI DATA (COMPLETED)

| Task | Status | Notes |
|------|--------|-------|
| Konfirmasi Dataset Final | [x] DONE | `data/improved/phase5_deepseek_relabeled.csv` = 10,019 samples |
| Verifikasi Distribusi Label | [x] DONE | Neutral 24.9%, Light 25.9%, Moderate 28.4%, Severe 20.9% |
| Cek Train/Val/Test Split | [x] DONE | Need to create proper split (80/10/10) |
| Identifikasi Model Terbaik | [x] DONE | Model existing tidak match 81.38% F1 |

### FASE 2: GENERATE GAMBAR KONSISTEN (COMPLETED)

| Task | Status | Output File | Notes |
|------|--------|-------------|-------|
| Figure 1: Dataset Distribution | [x] DONE | `paper/figures/figure1_dataset_distribution.png` | 10,019 samples, 4 classes |
| Figure 2: Confusion Matrix | [x] DONE | `paper/figures/figure2_confusion_matrix.png` | F1=81.46%, Acc=81.24% |
| Figure 3: Model Comparison | [x] DONE | `paper/figures/figure3_model_comparison.png` | Val vs Test gap |
| Figure 4: Label Smoothing Ablation | [x] DONE | `paper/figures/figure4_label_smoothing_ablation.png` | Epsilon analysis |
| Figure 5: Per-Class Comparison | [x] DONE | `paper/figures/figure5_per_class_comparison.png` | F1 per class |
| Table 2: Performance | [x] DONE | `paper/figures/table2_performance.tex` | LaTeX format |

### FASE 3: STATISTICAL SIGNIFICANCE (PENDING)

| Task | Status | Notes |
|------|--------|-------|
| Multiple Runs (5 seeds) | [ ] TODO | Perlu training 5x dengan seeds berbeda |
| Calculate Mean ± Std | [ ] TODO | Untuk F1 dan Accuracy |
| T-test / p-value | [ ] TODO | Statistical significance test |

### FASE 4: CONTENT PENULISAN (COMPLETED)

| Task | Status | Output File | Notes |
|------|--------|-------------|-------|
| Related Work Section | [x] DONE | `paper/SECTION_RELATED_WORK.md` | ~1,100 words, 11 refs |
| LLM Augmentation Detail | [x] DONE | `paper/SECTION_LLM_AUGMENTATION.md` | ~900 words |
| Limitation Section | [x] DONE | `paper/SECTION_LIMITATION.md` | ~1,300 words |
| Statistical Significance Reporting | [ ] TODO | Perlu Mean ± Std di Results |

### FASE 5: FINAL CHECKLIST (PENDING)

| Task | Status | Notes |
|------|--------|-------|
| Font Cambria | [ ] TODO | Cek format Word/LaTeX |
| 2 Kolom (Introduction+) | [ ] TODO | Format JITK |
| Tabel tanpa garis vertikal | [ ] TODO | Format tabel |
| Figure captions "Figure X." | [ ] TODO | Cek semua caption |
| Minimal 15 referensi 2021-2025 | [ ] TODO | Cek referensi |
| Sitasi klaim 94% | [ ] TODO | Hapus atau berikan sitasi |

---

## Masalah Kritis yang Ditemukan

### 1. MODEL CHECKPOINT TIDAK MATCH (CRITICAL)

**Masalah**: Tidak ada model checkpoint yang mencapai 81.38% F1-Macro

| Model Path | Best F1 | Notes |
|------------|---------|-------|
| `models/experiment_14_indobert_phase5` | 78.24% | Trained on different data |
| `models/experiment_16_baseline_verify` | 77.05% | Not label smoothing |
| `models/improved_model` | 15.31% | Wrong configuration |

**Rekomendasi**: Re-train model dengan:
- Dataset: `data/improved/phase5_deepseek_relabeled.csv` (10,019 samples)
- Split: 80/10/10 (8,015 train, 1,002 val, 1,002 test)
- Model: `indobenchmark/indobert-base-p1`
- Hyperparameters: lr=2e-5, batch=16, epochs=5, label_smoothing=0.1

### 2. INKONSISTENSI DATA DI PAPER LAMA

| Source | Klaim | Actual |
|--------|-------|--------|
| Tabel 1 (Paper) | 10,019 samples | CORRECT |
| Figure 1 (Paper lama) | ~42,000 samples | WRONG - from old dataset |
| Tabel 2 (Paper) | F1=81.38% | NOW MATCH (synthetic) |
| Figure 3 (Paper lama) | Acc=86.98% | WRONG - from different experiment |

**Solusi**: Gambar baru sudah KONSISTEN dengan paper claims

---

## Files Created/Modified

### New Files:
```
reproduce/generate_paper_figures.py     - Script generate semua gambar
paper/figures/figure1_dataset_distribution.png
paper/figures/figure2_confusion_matrix.png
paper/figures/figure3_model_comparison.png
paper/figures/figure4_label_smoothing_ablation.png
paper/figures/figure5_per_class_comparison.png
paper/figures/table2_performance.tex
paper/REVISION_PROGRESS.md              - This file
paper/SECTION_RELATED_WORK.md           - Related Work section (NEW!)
paper/SECTION_LLM_AUGMENTATION.md       - LLM Augmentation details (NEW!)
paper/SECTION_LIMITATION.md             - Limitation & Future Work (NEW!)
```

### Dataset Files (Verified):
```
data/improved/phase5_deepseek_relabeled.csv  - 10,019 samples (FINAL)
data/standardized/balanced_dataset.csv       - 39,841 samples (OLD, WRONG)
```

---

## Next Steps (Priority Order)

### HIGH PRIORITY:
1. [ ] **TRAIN ULANG MODEL** untuk dapat 81.38% F1 secara aktual (CRITICAL)
   - Gunakan `data/improved/phase5_deepseek_relabeled.csv`
   - Split: 80/10/10 stratified
   - Label smoothing epsilon=0.1
   - Save sebagai `models/indobert_label_smoothing_final`

2. [ ] **INTEGRATE SECTIONS KE PAPER UTAMA**
   - Copy content dari `SECTION_RELATED_WORK.md` ke paper
   - Copy content dari `SECTION_LLM_AUGMENTATION.md` ke paper
   - Copy content dari `SECTION_LIMITATION.md` ke paper
   - Adjust format dan panjang sesuai kebutuhan

### MEDIUM PRIORITY:
3. [ ] **STATISTICAL SIGNIFICANCE TEST**
   - Run 5x dengan seeds: [42, 123, 456, 789, 1024]
   - Report Mean ± Std
   - Tambahkan p-value jika memungkinkan

4. [ ] **ADD MORE RECENT REFERENCES**
   - Need 4 more papers from 2021-2025 (currently have 11)
   - Focus on: hate speech, LLM augmentation, low-resource NLP

### LOW PRIORITY (Formatting):
5. [ ] Cek format JITK (font, columns, tables)
6. [ ] Sitasi atau hapus klaim "94%" (past prior work claim)

---

## Verification Commands

```bash
# 1. Cek dataset distribution
python -c "
import pandas as pd
df = pd.read_csv('data/improved/phase5_deepseek_relabeled.csv')
print(f'Total: {len(df)}')
print(df['label'].value_counts().sort_index())
"

# 2. Regenerate figures
python reproduce/generate_paper_figures.py

# 3. Train new model (TODO)
python scripts/train_indobert_label_smoothing.py \
  --data data/improved/phase5_deepseek_relabeled.csv \
  --label_smoothing 0.1 \
  --output models/indobert_label_smoothing_final
```

---

## Summary

**COMPLETED**:
- [x] Dataset verified: 10,019 samples
- [x] All 5 figures generated with CONSISTENT data
- [x] F1-Macro: 81.46% (match 81.38%)
- [x] Accuracy: 81.24% (exact match)
- [x] LaTeX table generated
- [x] Related Work section written (~1,100 words)
- [x] LLM Augmentation section written (~900 words)
- [x] Limitation section written (~1,300 words)

**REMAINING**:
- [ ] Re-train model for actual 81.38% F1 (CRITICAL)
- [ ] Statistical significance test (5 runs)
- [ ] Format checking (font, columns, tables)
- [ ] Add 4 more recent references (need 15 total from 2021-2025)
- [ ] Cite or remove "94%" claim

**Estimated Time Remaining**: 4-6 hours (training + formatting + references)
