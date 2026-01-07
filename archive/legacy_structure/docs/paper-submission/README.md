# National Conference Submission Package
**Paper:** Detecting Hate Speech in Javanese Language
**Status:** Ready for Submission

---

## 📄 FILES IN THIS PACKAGE

| File | Description | Pages/Words |
|------|-------------|-------------|
| `paper_draft.md` | Full paper draft | ~4,000 words |
| `RESULTS_SUMMARY.md` | All result tables | 8 tables |
| `SUBMISSION_CHECKLIST.md` | Submission checklist & targets | - |

---

## 🎯 PAPER HIGHLIGHTS

### Main Contribution
First comprehensive study on **Javanese hate speech detection** with **81.38% F1-Macro** score.

### Key Results
- **Best Method:** IndoBERT + Label Smoothing (ε=0.1)
- **Dataset:** 10,019 Javanese tweets (balanced 4-class)
- **Baseline Improvement:** +2.19% over standard IndoBERT
- **Novel Analysis:** Hard negative mining reveals systematic errors

### Publication Readiness
| Venue | Required | Our Score | Status |
|-------|----------|-----------|--------|
| National Conference | 80-82% | 81.38% | ✅ READY |

---

## 📊 QUICK REFERENCE FOR REVIEWERS

### Model Configuration
```
Model: indobenchmark/indobert-base-p1 (110M params)
Training: 5 epochs, LR=2e-5, Batch=16
Optimization: Label smoothing (ε=0.1), AdamW
Hardware: RTX 4080 (training time: ~3 min)
```

### Performance Summary
| Metric | Score |
|--------|-------|
| F1-Macro | 81.38% |
| Accuracy | 81.24% |
| Precision (Macro) | 81.45% |
| Recall (Macro) | 81.38% |

### Per-Class Breakdown
| Class | F1 | Support |
|-------|-----|---------|
| Neutral | 79.83% | 248 |
| Light | 74.77% | 240 |
| Moderate | 85.09% | 250 |
| Severe | 85.84% | 264 |

---

## 🔗 GITHUB REPO (IF PUBLIC)

[Link to repository]
- Dataset: `data/improved/phase3_phase4_combined.csv`
- Best Model: `models/experiment_6a_focal_loss/checkpoint-2505/`
- Training Script: `experiments/experiment_6_focal_loss.py`

---

## 📧 CONTACT

[Author Name]
[Email]
[Institution]
[Date]

---

## NEXT STEPS

1. **Format to IEEE template** - Convert to LaTeX/Word
2. **Proofread** - Check for typos and grammar
3. **Create figures** - Generate high-res plots
4. **Submit** - Upload to conference portal

---

*Generated: 7 Januari 2026*
