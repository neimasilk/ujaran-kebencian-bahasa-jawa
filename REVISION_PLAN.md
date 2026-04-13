# REVISION PLAN — Kinetik Resubmission
**Date:** 2026-04-13
**Status:** REJECTED → Planning revision
**Target:** Kinetik (SINTA-2) — resubmission

---

## REJECTION DIAGNOSIS

Editor's complaints mapped to root causes:

| Editor Said | Root Cause | Fix |
|---|---|---|
| "No analysis of proposed methods" | Paper reports results without analyzing WHY models behave differently | Add error analysis, confusion patterns, confidence analysis |
| "No comparison with existing methods" | Only internal comparison (our 5 models). No table vs published papers | Add comparison table: Ibrohim, Alfina, Putri, etc. |
| "No GAP analysis in introduction" | Intro tells narrative but never explicitly identifies research gaps | Rewrite intro with explicit GAP statements |
| "No significant research problem from literature" | Problem not grounded in literature failures | Frame around: "Is LLM-augmented evaluation reliable?" |
| "No significant research contribution" | Using off-the-shelf models isn't novel | Reframe: evaluation protocol + augmentation bias finding = contribution |
| "Follow KINETIK guidelines" | Template/formatting issues | Regenerate from template |

## STRATEGIC PIVOT

**Old framing (REJECTED):**
> "We evaluated transformer models for Javanese hate speech detection"

**New framing:**
> "We investigate evaluation bias from LLM data augmentation in low-resource hate speech detection, revealing a 45.52pp gap between synthetic and manual test performance"

### Why This Works
- The 53.89% vs 99.41% gap becomes THE FINDING, not a limitation
- Clear research question: "Can you trust LLM-augmented evaluation?"
- Relevant to entire low-resource NLP community
- Novel evaluation protocol: dual-track testing
- Comparison with prior work is natural

### Three Clear Contributions
1. **First severity-level (4-class) hate speech dataset for Javanese** (9,775 samples)
2. **Dual-track evaluation protocol** exposing augmentation bias
3. **Empirical evidence**: 45.52pp F1 gap + augmentation ratio curve + two-stage mitigation

---

## PHASE 1: NEW EXPERIMENTS

### Experiment 1: Augmentation Ratio Study ⭐ CRITICAL
**Script:** `reproduce/augmentation_ratio_study.py`
**Purpose:** Show how synthetic data proportion affects real-world performance
**Design:**
- Ratios: 0%, 25%, 50%, 75%, 100% of synthetic data added to manual
- Models: SVM + IndoBERT (+ XLM-R if time permits)
- Evaluation: ALWAYS on manual-only test set (451 samples)
- Creates Figure: "Impact of Augmentation Ratio on Manual Test F1"

**Run:**
```bash
# Quick (SVM only, ~5 min)
python reproduce/augmentation_ratio_study.py --models svm

# Full (SVM + IndoBERT, ~2-3 hours)
python reproduce/augmentation_ratio_study.py --models svm,indobert

# Complete (all models, ~8 hours)
python reproduce/augmentation_ratio_study.py --models svm,indobert,xlmr
```

### Experiment 2: Two-Stage Training ⭐ HIGH VALUE
**Script:** `reproduce/two_stage_training.py`
**Purpose:** Show that curriculum learning (synthetic → manual) mitigates bias
**Design:**
- Stage 1: Fine-tune on synthetic data (3 epochs, lr=2e-5)
- Stage 2: Further fine-tune on manual data (5 epochs, lr=5e-6)
- Compare: single-stage (53.89%) vs two-stage (?)
- If it improves manual F1 → practical solution to augmentation bias

**Run:**
```bash
python reproduce/two_stage_training.py
# ~4-6 hours (IndoBERT + XLM-R)
```

### Experiment 3: Error Analysis ⭐ CRITICAL (addresses "no analysis")
**Script:** `reproduce/error_analysis.py`
**Purpose:** Deep analysis of WHY models fail on certain samples
**Analyzes:**
- Adjacent vs non-adjacent class confusion
- Text length vs prediction quality
- Feature analysis (URLs, mentions, caps ratio)
- Cross-model agreement (ensemble potential)
- Confidence calibration (XLM-R)

**Run:**
```bash
python reproduce/error_analysis.py
# ~10 min (uses existing checkpoints)
```

### Experiment 4: Cross-Validation
**Script:** `reproduce/cross_validation.py`
**Purpose:** More robust performance estimates than single split
**Design:**
- 5-fold stratified CV
- SVM + IndoBERT on full dataset AND manual-only
- Reports mean ± std

**Run:**
```bash
python reproduce/cross_validation.py
# ~3-4 hours
```

### Execution Order (Recommended)
```
1. error_analysis.py          (~10 min, no GPU training needed)
2. augmentation_ratio_study.py --models svm   (~5 min)
3. augmentation_ratio_study.py --models svm,indobert  (~3 hours)
4. two_stage_training.py       (~5 hours)
5. cross_validation.py         (~4 hours)
```

---

## PHASE 2: LITERATURE & GAP ANALYSIS

### References to Add (~15-20 new)

**A. Papers using data augmentation for hate speech (to show GAP):**
- Rizwan et al. (2022) — Hate speech augmentation with GPT-2
- Hartvigsen et al. (2022) — ToxiGen: augmented hate speech dataset
- Vidgen et al. (2021) — Dynamically generated hate speech datasets
- Juuti et al. (2020) — Augmentation for abusive language detection

**B. Evaluation methodology / pitfalls in NLP:**
- Gorman & Bedrick (2019) — Need for multiple test sets in NLP
- Søgaard et al. (2021) — Evaluation pitfalls in NLP
- Bender & Koller (2020) — Climbing towards NLU

**C. Indonesian/Javanese NLP (expand existing):**
- Wibowo et al. (2021) — IndoCollex: Javanese morphology
- Aji et al. (2022) — One Country, 700+ Languages: NLP for Indonesia
- Winata et al. (2021) — Indonesian NLP survey

**D. Cross-lingual transfer:**
- Conneau et al. (2020) — XLM-R: Unsupervised cross-lingual representations
- Pires et al. (2019) — How multilingual is multilingual BERT?

**E. Data quality and annotation:**
- Paun et al. (2018) — Comparing Bayesian models of annotation
- Aroyo & Welty (2015) — Truth is a lie: crowd truth

### Comparison Table (NEW — addresses "no comparison with existing methods")

| Study | Language | Classes | Dataset | Method | F1 (%) | Metric |
|---|---|---|---|---|---|---|
| Davidson et al. (2017) | English | 3 | 24,802 | LR+TF-IDF | 90.00 | weighted |
| Alfina et al. (2017) | Indonesian | 2 | 520 | NB+TF-IDF | 74.00 | macro |
| Ibrohim & Budi (2019) | Indonesian | 3 | 13,169 | LSTM+FT | 71.31 | macro |
| Putri et al. (2021) | Javanese | 2 | ~2,500 | BERT | ~67 | macro |
| **Ours (full test)** | **Javanese** | **4** | **9,775** | **XLM-R** | **80.26** | **macro** |
| **Ours (manual only)** | **Javanese** | **4** | **4,538** | **XLM-R** | **53.89** | **macro** |

Key insight: Our full-test result (80.26%) looks competitive, but manual-only (53.89%) reveals inflation.

### GAP Analysis Statement (for Introduction)

> Despite growing adoption of LLM-generated data for low-resource hate speech 
> detection [refs], no prior study has systematically evaluated whether 
> performance reported on mixed (manual+synthetic) test sets reflects actual 
> capability on naturally-occurring text. This methodological gap is critical: 
> if augmented test data inflates reported metrics, the field may overestimate 
> progress in low-resource hate speech detection.
>
> Furthermore, while binary hate speech classification exists for Indonesian 
> [Alfina, Ibrohim] and preliminary work addresses Javanese [Putri], no 
> severity-level classification system exists for Javanese, limiting the 
> practical utility of detection systems for content moderation.

---

## PHASE 3: PAPER STRUCTURE (REWRITE)

### New Title Options
1. "Unmasking Evaluation Bias: LLM Data Augmentation Inflates Hate Speech Detection Performance in Low-Resource Javanese"
2. "How Reliable is LLM-Augmented Evaluation? A Case Study in Javanese Hate Speech Detection"
3. "Evaluating the Impact of LLM Data Augmentation on Severity-Based Hate Speech Detection in Javanese"

### New Paper Structure

```
1. INTRODUCTION
   1.1 Background (hate speech → Indonesia → regional languages)
   1.2 GAP Analysis (explicitly stated)
   1.3 Research Questions
       RQ1: How effective are transformer models for Javanese severity classification?
       RQ2: Does LLM augmentation inflate reported performance metrics?
       RQ3: Can curriculum learning mitigate augmentation bias?
   1.4 Contributions (3 clear items)

2. RELATED WORK
   2.1 Hate Speech Detection (with comparison table)
   2.2 Low-Resource Language NLP
   2.3 Data Augmentation in NLP
   2.4 Evaluation Methodology and Pitfalls

3. RESEARCH METHOD
   3.1 Dataset Construction
       - Manual annotation process
       - LLM augmentation (DeepSeek-Coder-V2)
       - Quality control (Cohen's κ = 0.72)
   3.2 Dual-Track Evaluation Protocol
       - Full test set evaluation
       - Manual-only test set evaluation
       - Augmentation ratio analysis
   3.3 Models
       - Baselines: SVM, Logistic Regression
       - Transformers: IndoBERT, XLM-R Large, IndoBERT+LS
   3.4 Experimental Setup
       - Split: 80/10/10 stratified, seed=42
       - Hyperparameters
       - Hardware: RTX 4080

4. RESULTS AND DISCUSSION
   4.1 Overall Performance Comparison
       - Table with all models on full test + manual-only test
       - Cross-validation results
   4.2 Comparison with Prior Work
       - Table comparing with published results
       - Discussion of task difficulty (4-class vs binary)
   4.3 Augmentation Bias Analysis (MAIN FINDING)
       - 99.41% vs 53.89% gap
       - Augmentation ratio curve (NEW)
       - Implications for the field
   4.4 Error Analysis (NEW — addresses "no analysis")
       - Confusion patterns (adjacent vs non-adjacent)
       - Text characteristics of errors
       - Cross-model agreement
   4.5 Two-Stage Training as Mitigation (NEW)
       - Curriculum learning results
       - Practical recommendations
   4.6 Label Smoothing Effect
   4.7 Limitations and Future Work

5. CONCLUSION

REFERENCES (target: 30-40 refs)
```

---

## PHASE 4: FORMATTING

### Kinetik Requirements Checklist
- [ ] Use official template (Template Kinetik Mendeley.docx)
- [ ] 6-10 pages
- [ ] Keywords: no overlap with title words
- [ ] IEEE numbered references
- [ ] Abstract: < 250 words
- [ ] All figures/tables referenced in text
- [ ] Past tense in Results section
- [ ] Consistent IAA description

### FATAL/MAJOR Fixes from HANDOFF5 (still apply)
- [ ] F1: Fix per-class analysis (Moderate=40% is worst, not Light)
- [ ] F2: Cite reference [11] Joachims in SVM paragraph
- [ ] F3: Fix label smoothing delta consistency
- [ ] F4: Add disclosure for Table 2 examples
- [ ] F5: Replace keyword "Transformer"
- [ ] M1-M8: All major issues

---

## TIMELINE ESTIMATE

| Phase | Work | GPU Needed |
|---|---|---|
| Phase 1a | Error analysis (quick) | No (uses checkpoints) |
| Phase 1b | Ratio study (SVM) | No |
| Phase 1c | Ratio study (IndoBERT) | Yes (~3h) |
| Phase 1d | Two-stage training | Yes (~5h) |
| Phase 1e | Cross-validation | Yes (~4h) |
| Phase 2 | Literature + GAP | No |
| Phase 3 | Paper rewrite | No |
| Phase 4 | Format + verify | No |

**Total GPU time: ~12 hours**
**Total effort: ~3-5 sessions**
