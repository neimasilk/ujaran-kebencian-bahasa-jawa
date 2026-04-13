# HANDOFF DOCUMENT — Session 3
## Javanese Hate Speech Detection — Pre-Submission Fixes (Kinetik SINTA-2)
**From:** Mata Elang Strategic Review (2026-02-27)
**To:** Next agent (execution of fixes)
**Date:** 2026-02-27

---

## RINGKASAN SITUASI

Paper sudah **selesai secara eksperimen** (semua GPU work done). Paper generator (`paper/generate_kinetik_paper.py`) menghasilkan `paper/paper_kinetik.docx` dari template + results JSON. Namun strategic review menemukan **5 masalah kritikal dan 8 masalah struktural** yang harus diperbaiki sebelum submission ke Kinetik.

**Semua perbaikan adalah di `paper/generate_kinetik_paper.py` — TIDAK ADA eksperimen GPU baru yang diperlukan.**

---

## TARGET JOURNAL: KINETIK (SINTA-2)

Key requirements yang belum terpenuhi:
- **Halaman**: 6–10 pages (paper saat ini kemungkinan >10 karena 12 tabel + 4 figur)
- **Referensi**: Min 20 IEEE-style, **HARUS include referensi dari Kinetik** (saat ini 0)
- **Article Info panel**: Template Kinetik punya 2-column table di bawah author (Keywords, Article History, Citation) — **belum dibuat**
- **Bahasa**: English only
- **Format**: DOCX (bukan PDF/LaTeX), font Arial, single column, A4

Full template: `Template Kinetik Mendeley.docx` di root project.

---

## DAFTAR PERBAIKAN (7 items, urut prioritas)

### FIX 1: Section Numbering Error [5 min]
**File:** `paper/generate_kinetik_paper.py`
**Problem:** Section numbering melompat dari 3.1 langsung ke 3.3 (tidak ada 3.2).
**Location:** Line ~672 onwards in `write_results_discussion()`

Current section structure in Results:
```
3.1 Performance on Manual-Only Test Data
3.3 Performance on Full Test Set          ← ERROR: should be 3.2
3.4 Per-Class Analysis                    ← should be 3.3
3.5 Augmentation Impact Analysis          ← should be 3.4
3.6 Label Smoothing Ablation              ← WILL BE REMOVED (see Fix 4)
3.7 Multi-Seed Statistical Significance   ← renumber
3.8 Comparison with Prior Work            ← renumber
3.9 Limitations                           ← renumber
```

**Action:** Renumber all subsections sequentially: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7.

---

### FIX 2: Add Kinetik Journal References [30 min]
**File:** `paper/generate_kinetik_paper.py`, function `write_references()`
**Problem:** Kinetik requires papers to cite previous Kinetik articles. Current 29 refs have zero Kinetik papers.
**Action:**
1. Search web for relevant Kinetik papers on topics: NLP, text classification, sentiment analysis, hate speech, deep learning for text, Indonesian language processing
2. Add 2-3 genuine Kinetik references (verify they exist at kinetik.umm.ac.id)
3. Cite them appropriately in Introduction or Related Work paragraphs
4. Update reference numbering (currently [1]-[29], will become [1]-[31] or [32])

**IMPORTANT:** References MUST be real. Verify each one exists. Use format:
```
'[30] A. Author et al., "Title," Kinetik: Game Technology, Information System, Computer Network, Computing, Electronics, and Control, vol. X, no. Y, pp. XX-YY, 20XX.'
```

---

### FIX 3: Add Article Info Panel [1 hour]
**File:** `paper/generate_kinetik_paper.py`
**Problem:** Kinetik template has a 2-column table after authors containing Article Info (left) and Abstract (right). Current generator writes abstract as standalone paragraphs.

**Action:** Restructure `write_abstract()` to create a 2-cell table:
- **Left cell (narrow):** "Article Info" heading, then Keywords, then Article history placeholders (Received: [date], Revised: -, Accepted: -, Available online: -), then Citation placeholder
- **Right cell (wide):** Abstract text + Keywords line

Reference the template file `Template Kinetik Mendeley.docx` for exact structure and styles. The template uses styles "Article info heading kinetik", "Article info kinetik" for the left column.

**Note:** Article history dates should be left as placeholders (the journal fills these in). But the STRUCTURE must be present.

---

### FIX 4: Reduce Tables & Figures (12→8 tables, 4→3 figures) [2 hours]
**File:** `paper/generate_kinetik_paper.py`
**Problem:** 12 tables + 4 figures won't fit in 6-10 pages.

**Tables to REMOVE (convert to inline text):**

| Table | Content | Action |
|-------|---------|--------|
| Table 3 | Augmentation method comparison (Manual/Translation/Back-translation/LLM) | Remove table. Keep 1-2 sentences mentioning DeepSeek generated 5,237 samples at $15 cost |
| Table 8 | Per-class full test set (XLM-R) | Remove. Already have Table 6 (per-class manual-only) which is the primary metric |
| Table 10 | Label smoothing ablation (5 epsilon values) | Remove table. Convert to 1 paragraph: "Label smoothing ablation across five ε values (0.0–0.2) shows optimal ε = 0.1 with a modest +0.27 F1 improvement on IndoBERT, suggesting limited benefit for this dataset size." |

**Figure to REMOVE:**

| Figure | Content | Action |
|--------|---------|--------|
| Figure 4 | Label smoothing ablation curve | Remove. The +0.27 improvement is not visually meaningful |

**After removal:** 8 tables, 3 figures. Also renumber all remaining tables and figures sequentially.

**Also:** The label smoothing ablation section (currently 3.6) should be compressed from a full subsection to 1 paragraph within the preceding or following section, or kept as a very short subsection.

**Surviving tables (renumber 1-8):**
1. Dataset Distribution
2. Example texts per severity
3. ~~Augmentation comparison~~ → REMOVED
4. Hyperparameters → becomes Table 3
5. Manual-only results → becomes Table 4
6. Per-class manual-only → becomes Table 5
7. Full test set results → becomes Table 6
8. ~~Per-class full test~~ → REMOVED
9. Augmentation impact → becomes Table 7
10. ~~Ablation~~ → REMOVED
11. Multi-seed → becomes Table 8
12. Comparison with prior work → becomes Table 9... wait, that's 9 tables. Let me recount.

**Final table inventory (9 tables):**
1. Table 1: Dataset Distribution (9,775 samples)
2. Table 2: Example texts per severity level
3. Table 3: Training Hyperparameters
4. Table 4: Manual-Only Test Results (PRIMARY) — 451 samples
5. Table 5: Per-Class Manual-Only (XLM-R Large)
6. Table 6: Full Test Set Results (SECONDARY) — 978 samples
7. Table 7: Augmentation Impact Analysis
8. Table 8: Multi-Seed Evaluation
9. Table 9: Comparison with Prior Work

**Final figure inventory (3 figures):**
1. Figure 1: Dataset distribution (bar chart)
2. Figure 2: Model comparison (baseline vs transformer)
3. Figure 3: Confusion matrix (XLM-R Large) — **change to manual-only test data CM**

---

### FIX 5: Re-frame Paper Narrative [1 hour]
**File:** `paper/generate_kinetik_paper.py`
**Problem:** Paper claims to build a "detection system" but primary F1=53.89% on real data is weak. The 99.41% vs 53.89% gap is the paper's most interesting finding but is currently framed as a limitation.

**Action — rewrite these sections:**

**Title** — Consider changing to emphasize the empirical study aspect:
- Current: "Severity-Based Hate Speech Detection in Javanese Using Transformer Models with LLM Augmentation"
- Better: "Evaluating LLM-Augmented Transformer Models for Severity-Based Hate Speech Detection in Javanese" (keeps under 15 words)

**Abstract** — Re-frame:
- Lead with: "This study empirically evaluates..." not "This study presents a system..."
- Position the 99.41% vs 53.89% gap as a KEY FINDING, not an embarrassment
- Frame contribution as: (1) first Javanese severity dataset, (2) empirical evidence that LLM augmentation creates distributional shift, (3) comparative benchmark

**Introduction, final paragraph (contributions)** — Rewrite contribution list:
- Contribution (1): Dataset (keep)
- Contribution (2): Empirical finding on LLM augmentation bias (elevate from limitation to contribution)
- Contribution (3): Comparative benchmark (keep)
- Contribution (4): Multi-seed stability (keep or compress)

**Conclusion** — Frame honestly:
- "Our results demonstrate that while XLM-RoBERTa Large achieves the best performance, the substantial gap between synthetic (99.41%) and manual (53.89%) test performance reveals that LLM augmentation, though necessary for preventing training collapse, introduces systematic distributional differences that inflate standard evaluation metrics."

---

### FIX 6: Remove/Soften Unverifiable Numbers [30 min]
**File:** `paper/generate_kinetik_paper.py`
**Problem:** Several hardcoded numbers have no backing data in `results/`:

| Number | Location | Issue |
|--------|----------|-------|
| Cohen's κ = 0.72 | Line 284 (abstract), line 552 | No IAA computation script or results file |
| Naturalness 3.8, Cultural Appropriateness 4.1, Register Consistency 3.5, Severity Accuracy 4.2 | Lines 572-574 | No survey results file |
| Table 3: "Manual: 2,500 samples, 3 months" | Lines 561-564 | No source for these numbers |

**Action (choose one per item):**
- **Option A:** If these numbers are from actual work done earlier, create a brief `results/annotation_quality.json` documenting them
- **Option B:** If uncertain, soften the language:
  - κ = 0.72 → "substantial agreement (κ > 0.6)" or remove the specific number
  - Quality scores → "Informal evaluation by native speakers indicated acceptable quality" (no specific numbers)
  - Table 3 → Already being removed in Fix 4

**The safest approach is Option B** — remove specific unverifiable numbers and use qualitative language instead.

---

### FIX 7: Change Confusion Matrix to Manual-Only [30 min]
**File:** `paper/generate_kinetik_paper.py` AND `reproduce/generate_honest_figures.py`
**Problem:** Figure 3 (confusion matrix) is from full test set (978 samples). Since manual-only is the primary metric, the confusion matrix should also be from manual-only data.

**Action:**
1. In `generate_honest_figures.py`, add a function to generate confusion matrix from manual-only test data (data available in `results/manual_only_results.json` → `models.xlmr_large.manual_only.confusion_matrix`)
2. Save as `paper/figures/figure2_confusion_matrix_manual.png` (or overwrite existing)
3. Update caption in `generate_kinetik_paper.py` to specify "manual-only test data (451 samples)"
4. Regenerate the figure

---

## EXECUTION ORDER

```
1. FIX 1: Section numbering        → edit generate_kinetik_paper.py
2. FIX 6: Soften unverifiable nums  → edit generate_kinetik_paper.py
3. FIX 4: Remove tables/figures     → edit generate_kinetik_paper.py (biggest change)
4. FIX 5: Re-frame narrative        → edit generate_kinetik_paper.py
5. FIX 2: Add Kinetik references    → web search + edit generate_kinetik_paper.py
6. FIX 3: Add Article Info panel    → edit generate_kinetik_paper.py (needs template study)
7. FIX 7: Manual-only confusion mat → edit generate_honest_figures.py + regenerate
8. REGENERATE: python paper/generate_kinetik_paper.py
9. VERIFY: Open paper_kinetik.docx, check page count (must be 6-10)
```

---

## KEY FILE LOCATIONS

| File | Purpose |
|------|---------|
| `paper/generate_kinetik_paper.py` | **THE main file to edit** — generates paper DOCX |
| `Template Kinetik Mendeley.docx` | Kinetik template (root dir) — reference for Article Info panel |
| `results/comparative_results.json` | Model comparison results |
| `results/manual_only_results.json` | Manual-only evaluation (all 5 models) |
| `results/augmentation_impact.json` | 99.41% vs 53.89% gap data |
| `results/ablation_results.json` | Label smoothing (will be compressed) |
| `results/multi_seed_results.json` | 5-seed evaluation |
| `results/baseline_results.json` | SVM + LR baselines |
| `data/cleaned/cleaning_report.json` | Dataset cleanup stats |
| `reproduce/generate_honest_figures.py` | Figure generation script |
| `paper/figures/` | Generated figures (PNG) |

---

## NUMBERS CHEAT SHEET (all from results JSON, verified)

| Metric | Value | Source |
|--------|-------|--------|
| Dataset size | 9,775 | cleaning_report.json |
| Manual samples | 4,538 (46.4%) | manual_only_results.json metadata |
| Synthetic samples | 5,237 (53.6%) | manual_only_results.json metadata |
| Train/Val/Test | 7,820 / 977 / 978 | comparative_results.json |
| Manual test samples | 451 | manual_only_results.json |
| XLM-R Large manual-only F1 | 53.89% | manual_only_results.json |
| XLM-R Large full test F1 | 80.26% | comparative_results.json |
| XLM-R Large synthetic test F1 | 99.41% | augmentation_impact.json |
| SVM manual-only F1 | 48.55% | manual_only_results.json |
| SVM full test F1 | 77.77% | baseline_results.json |
| IndoBERT manual-only F1 | 45.27% | manual_only_results.json |
| IndoBERT full test F1 | 76.12% | comparative_results.json |
| IndoBERT+LS manual-only F1 | 49.30% | manual_only_results.json |
| IndoBERT+LS full test F1 | 77.36% | comparative_results.json |
| LR manual-only F1 | 47.18% | manual_only_results.json |
| LR full test F1 | 77.04% | baseline_results.json |
| Label smoothing optimal ε | 0.1 (+0.27 F1) | ablation_results.json |
| Multi-seed mean (4 stable) | 80.83% ± 1.83% | multi_seed_results.json |
| Manual-only model collapsed | F1=14.64% | augmentation_impact.json |
| Seed 1024 collapsed | F1=11.07% | multi_seed_results.json |

---

## ENVIRONMENT

- **OS**: Windows 11
- **Python**: Anaconda (use `python`, not `python3`)
- **Paths**: Use `D:/` for Python, `/d/` or `D:/` for bash
- **Dependencies**: `pip install python-docx` (for paper generation)
- **NO GPU needed** — all fixes are code/text editing + DOCX generation

---

## VERIFICATION CHECKLIST (after all fixes)

```
[ ] Section numbering is sequential (3.1, 3.2, 3.3, ...)
[ ] At least 1-2 Kinetik references added and cited in text
[ ] Article Info panel present (Keywords, Article History, Citation)
[ ] Tables ≤ 9, Figures ≤ 3
[ ] All table/figure numbers sequential
[ ] No hardcoded unverifiable numbers (κ, quality scores)
[ ] Paper narrative frames augmentation bias as finding, not just limitation
[ ] Confusion matrix figure is from manual-only test data
[ ] python paper/generate_kinetik_paper.py runs without error
[ ] paper_kinetik.docx opens in Word
[ ] Page count: 6-10 pages
[ ] All reference numbers in text match reference list
[ ] No reference to removed tables/figures in text
```
