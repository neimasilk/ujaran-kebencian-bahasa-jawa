# NEXT ACTIONS - Honest Comparative Transformer Study

## Updated: 2026-02-25
## Strategy: Pivot to honest comparative study with real metrics

---

## COMPLETED

| Task | Status | Output |
|------|--------|--------|
| Fase 0: Dataset Cleanup | DONE | `data/cleaned/final_dataset.csv` (9,775 samples) |
| Fase 0: Cleaning Report | DONE | `data/cleaned/cleaning_report.json` |
| Script: Honest Evaluation | DONE | `reproduce/honest_evaluation.py` |
| Script: Comparative Training | DONE | `reproduce/train_comparative.py` |
| Script: Ablation Study | DONE | `reproduce/real_ablation_study.py` |
| Script: Multi-Seed Eval | DONE | `reproduce/multi_seed_eval.py` |
| Script: Honest Figures | DONE | `reproduce/generate_honest_figures.py` |
| Script: Reference Verification | DONE | `reproduce/verify_references.py` |
| Archive fabricated script | DONE | `archive/scripts/generate_paper_figures.py` |
| Update SECTION_LLM_AUGMENTATION.md | DONE | Fixed 33% -> 52.3%, 3,319 -> 5,240 |
| Update SECTION_LIMITATION.md | DONE | Updated numbers, added synthetic data limitation |

---

## ALL EXECUTION STEPS — DONE ✓

| Step | Script | Status | Key Result |
|------|--------|--------|------------|
| 1 | `honest_evaluation.py` | ✓ DONE | Best existing: IndoBERT Phase5, F1=94.56% |
| 2 | `train_comparative.py` | ✓ DONE | Best from-scratch: XLM-R Large, F1=80.26% |
| 3 | `real_ablation_study.py` | ✓ DONE | Optimal ε=0.1, F1=77.36% |
| 4 | `multi_seed_eval.py` | ✓ DONE | XLM-R mean F1=80.83% ± 1.74% (4 stable seeds) |
| 5 | `generate_honest_figures.py` | ✓ DONE | 5 PNGs + 2 LaTeX tables generated |
| 6 | Paper section updates | ✓ DONE | All numbers updated with actual results |

### Remaining: Final Integration (Step 7)
- Integrate all sections into `Paper_JITK_NoFigures_Safe.docx`
- Insert figures from `paper/figures/`
- Final format check (JITK: Cambria font, 2-column, IEEE references)

---

## KEY CHANGES FROM PREVIOUS APPROACH

| Aspect | Before (Fabricated) | After (Honest) |
|--------|-------------------|----------------|
| F1-Macro claim | 81.38% (synthetic) | **80.26%** (XLM-R Large, real test set) |
| Confusion matrix | Hardcoded values | Real model inference on 978 test samples |
| Ablation data | Fake epsilon curve | Real sweep: ε=[0.0,0.05,0.1,0.15,0.2] optimal=0.1 |
| LLM ratio | "33%" (3,319) | 52.3% (5,240) |
| Dataset size | 10,019 (raw) | 9,775 (cleaned) |
| Statistical sig. | None | 80.83% ± 1.74% F1 (5 seeds, 4 stable) |
| Model comparison | Single model only | 3 transformers compared |

---

## FILES REFERENCE

### New Scripts
```
reproduce/
├── clean_dataset.py            # Fase 0: Dataset cleanup
├── honest_evaluation.py        # Fase 1: Eval existing checkpoints
├── train_comparative.py        # Fase 2: Train 3 models
├── real_ablation_study.py      # Fase 3: Label smoothing ablation
├── multi_seed_eval.py          # Fase 4: Statistical significance
├── generate_honest_figures.py  # Fase 5: Real figures
└── verify_references.py        # Fase 6: Reference check
```

### Data
```
data/cleaned/
├── final_dataset.csv           # 9,775 cleaned samples
└── cleaning_report.json        # Cleaning statistics
```

### Results (after running scripts)
```
results/
├── honest_evaluation_results.json
├── comparative_results.json
├── ablation_results.json
├── multi_seed_results.json
└── reference_verification.json
```

### Archived
```
archive/scripts/
└── generate_paper_figures.py   # Contains fabricated confusion matrix
```
