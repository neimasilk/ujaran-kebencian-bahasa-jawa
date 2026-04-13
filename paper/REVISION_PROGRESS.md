# Paper JITK Revision Progress

## Status: PIVOTED TO HONEST COMPARATIVE STUDY
Last Updated: 2026-02-25

---

## Critical Decision: Integrity Pivot

The original approach used fabricated data (synthetic confusion matrix in `generate_paper_figures.py:397`) to match a claimed F1 of 81.38%. This has been identified and corrected.

**New Approach**: Honest comparative transformer study with:
- Aggressive dataset cleanup (10,019 → 9,775 samples)
- Real model evaluation on test set
- Comparative study of 3 transformer models
- Real label smoothing ablation
- Statistical significance via 5-seed evaluation
- Corrected data composition numbers (52.3% LLM-generated, not 33%)

---

## Completed Steps

### Dataset Cleanup (Fase 0) — DONE
| Task | Status | Details |
|------|--------|---------|
| Remove duplicates | DONE | 33 exact duplicates removed |
| Remove short texts | DONE | 186 texts < 20 chars removed |
| Remove non-Javanese | DONE | 25 non-Javanese texts removed |
| Fix encoding | DONE | Encoding issues fixed |
| **Final dataset** | **DONE** | **9,775 samples** in `data/cleaned/final_dataset.csv` |

### Scripts Created — DONE
| Script | Purpose |
|--------|---------|
| `reproduce/clean_dataset.py` | Dataset cleanup |
| `reproduce/honest_evaluation.py` | Evaluate existing checkpoints |
| `reproduce/train_comparative.py` | Train 3 models from scratch |
| `reproduce/real_ablation_study.py` | Label smoothing ablation |
| `reproduce/multi_seed_eval.py` | 5-seed statistical significance |
| `reproduce/generate_honest_figures.py` | Figures from real data |
| `reproduce/verify_references.py` | Reference verification |

### Paper Section Updates — DONE
| Section | Changes |
|---------|---------|
| SECTION_LLM_AUGMENTATION.md | Fixed: 33% → 52.3%, 3,319 → 5,240, removed fabricated F1 table |
| SECTION_LIMITATION.md | Updated test set size, added synthetic data limitation, updated stat sig section |
| NEXT_ACTIONS.md | Complete rewrite reflecting honest approach |

### Archive — DONE
| File | Reason |
|------|--------|
| `archive/scripts/generate_paper_figures.py` | Contains fabricated confusion matrix |

---

## Execution Results (All Steps DONE)

### Step 1: Honest Evaluation — DONE
- 14 existing checkpoints evaluated on clean test set (978 samples)
- **Best existing checkpoint**: IndoBERT Phase5 (exp14, cp1255) — F1=94.56%, Acc=94.58%
- Output: `results/honest_evaluation_results.json`

### Step 2: Comparative Training — DONE
- 3 models trained from scratch on clean dataset:
  - IndoBERT: F1=76.12%, Acc=76.28%
  - **XLM-RoBERTa Large**: **F1=80.26%**, **Acc=80.27%** ← best from-scratch
  - IndoBERT+LS (eps=0.1): F1=77.36%, Acc=77.51%
- Output: `results/comparative_results.json`

### Step 3: Ablation Study — DONE
- Label smoothing epsilon sweep: [0.0, 0.05, 0.1, 0.15, 0.2]
- **Optimal epsilon**: 0.1 (F1=77.36%, Acc=77.51%)
- Output: `results/ablation_results.json`

### Step 4: Multi-Seed Evaluation — DONE
- XLM-R Large trained with 5 seeds [42, 123, 456, 789, 1024]
- Seed 1024 collapsed (F1=11.07%), 4 successful seeds: F1=80.83% ± 1.74%
- Output: `results/multi_seed_results.json`

### Step 5: Generate Figures — DONE
- All figures generated from real data to `paper/figures/`
- 5 PNG figures + 2 LaTeX tables

### Step 6: Update Paper Sections — DONE
- Updated all numbers with actual experiment results
- No TBD values remain

---

## Verification Checklist

After all steps complete, ALL must be true:
- [x] Every number in paper comes from real experiments
- [x] F1 reported = test set, not validation
- [x] Figures = from real model inference
- [x] References = all verified via Google Scholar
- [x] Dataset stats accurate (9,775 samples, 52.3% synthetic)
- [x] Statistical significance reported (mean ± std)
- [x] No inconsistencies between sections
- [x] `generate_synthetic_predictions()` archived, not used

---

## Key Numbers (Corrected)

| Metric | Old (Fabricated) | New (Honest) |
|--------|-----------------|--------------|
| Dataset size | 10,019 | 9,775 (after cleaning) |
| LLM-generated % | 33% (3,319) | 52.3% (5,240) |
| F1-Macro | 81.38% (synthetic) | **80.26%** (XLM-R Large, real) |
| Models compared | 1 (IndoBERT+LS) | 3 (IndoBERT, XLM-R Large, IndoBERT+LS) |
| Statistical sig. | None | 80.83% ± 1.74% (4 stable seeds) |
| Ablation | Fake curve | Real epsilon sweep (optimal ε=0.1) |
