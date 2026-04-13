# HANDOFF DOCUMENT — Session 5
## Javanese Hate Speech Detection — Final Pre-Submission Fixes
**From:** Session 5 Agent (2026-02-27)
**To:** Next agent (fix all FATAL + MAJOR issues, regenerate paper)
**Date:** 2026-02-27

---

## RINGKASAN SITUASI

Paper sudah **hampir siap submit** ke Kinetik (SINTA-2). Session 4 menyelesaikan semua structural fixes. Session 5 memperbaiki footer & figure language issues, lalu melakukan **final critical review** yang menemukan **5 FATAL + 8 MAJOR** issues yang harus diperbaiki.

**Semua perbaikan ada di `paper/generate_kinetik_paper.py` — TIDAK ADA eksperimen baru.**

Paper di-generate oleh script Python → DOCX. Edit script, jalankan `python paper/generate_kinetik_paper.py`, lalu verifikasi `paper/paper_kinetik.docx`.

---

## DAFTAR PERBAIKAN

### 🔴 FATAL ISSUES (5 items — harus fix sebelum submit)

#### F1: Per-Class Analysis Text SALAH (Lines 1084-1091)

**Problem:** Paper klaim "Light Hate class yields the lowest F1" — SALAH.

Data aktual dari `results/manual_only_results.json` (XLM-R Large, manual-only):
- Not Hate: **70.72%**
- Light: **51.13%**
- Moderate: **40.00%** ← TERENDAH
- Severe: **53.73%**

**Current text (line 1084-1091):**
```python
p.add_run(
    "Per-class analysis of XLM-RoBERTa Large on manual-only test data "
    "shows that the Light Hate class yields the lowest F1 "
    f"({xl_pc_manual[light_key]['f1']:.2f}%), consistent with the "
    "expectation that the boundary between neutral speech and light hate "
    "speech is the most subjective. Conversely, Moderate and Severe "
    "classes achieve higher F1 scores as their linguistic markers are more "
    "explicit."
)
```

**Fix:** Rewrite to reflect actual data — Moderate (40.00%) is worst because it's the smallest class (N=61) with ambiguous boundaries on both sides. Light struggles too (51.13%) due to subjectivity. Not Hate is best (70.72%) as it's the majority class and most distinct.

**ALSO fix the fallback branch at lines 1101-1108** — same wrong claim.

---

#### F2: Reference [11] Joachims — Orphaned (Line 1637)

**Problem:** `[11] T. Joachims, "Text categorization with support vector machines..."` is in the reference list but **never cited** in the body text.

**Fix options:**
- **Option A (recommended):** Cite [11] in the SVM methodology paragraph (Section 2.3, around line 786) where SVM baseline is introduced: "Support Vector Machine (SVM) [11] with a linear kernel..."
- **Option B:** Remove ref [11] and renumber [12]→[11], [13]→[12], ... [31]→[30]. This requires updating ALL citations in the body text too — RISKY, not recommended.

---

#### F3: Label Smoothing Delta Contradicts Table 6 (Lines 1235-1253)

**Problem:** Paper reports label smoothing improvement as "+0.27 F1" (from ablation run: 77.09→77.36). But Table 6 shows IndoBERT=76.12 vs IndoBERT+LS=77.36 = **+1.24 difference**. A reviewer will compute 77.36−76.12=1.24 and flag this.

**Root cause:** Ablation and comparative experiments are different training runs (different baseline: 77.09 vs 76.12). Paper never explains this.

**Fix:** Change the label smoothing paragraph to reference Table 6's numbers instead:
- "Label smoothing with ε = 0.1 on IndoBERT improved F1-Macro from 76.12% to 77.36% (+1.24 points) on the full test set (Table 6)."
- Or: Add clarification that the +0.27 comes from a controlled ablation with identical hyperparameters, which isolates the LS effect more precisely than comparing different training runs.

---

#### F4: Table 2 Examples Modified Without Disclosure (Lines 670-700)

**Problem:** Two examples in Table 2 are truncated/edited versions of actual dataset entries, presented as if they're exact samples.

- **Light example:** Paper: "Wah, ibu iki koyo detektif, sak klebatan motor tamu wae wis kudu takon." → Actual: "...wis kudu takon model endi, merk opo." (truncated)
- **Severe example:** Paper: "Kabeh keturunan Cina iku mata duitan, kudu dipeksa lunga." → Actual: "...mata duitan lan bakal nyolong kasugihan Jawa, kudu dipeksa lunga." (middle phrase removed)

**Fix options:**
- **Option A:** Use exact texts from dataset
- **Option B (safer):** Add note to Table 2 caption: "Examples are representative and may be abbreviated for space."

---

#### F5: Keyword "Transformer" Overlaps with Title (Line 417)

**Problem:** Title is "Evaluating LLM-Augmented **Transformer** Models for..." and keyword list includes "Transformer" — violates Kinetik rule.

**Current keywords (line 406-418):**
```
Data Augmentation
Low-Resource NLP
Severity Classification
Text Classification
Transformer          ← OVERLAPS WITH TITLE
```

**Fix:** Replace "Transformer" with "Pre-trained Language Model" or "Deep Learning" or "Cross-lingual Transfer". Keep alphabetical order.

---

### 🟡 MAJOR ISSUES (8 items — should fix, bisa kena reject)

#### M1: Abstract Multi-Seed F1 Misleading (Lines 490-495)

**Problem:** Abstract says "mean F1 of 80.83% ± 1.83%" without clarifying this is on the **full test set** (53.9% synthetic). Reader will think model achieves ~80% on real data.

**Fix:** Add qualifier: "...mean F1 of 80.83% ± 1.83% on the full test set across four stable seeds."

Also fix same issue in Conclusion (line 1530): add "on the full test set".

---

#### M2: Manual Training Count Over-Reported (Line 1208)

**Problem:** Paper says "4087 training samples" computed as 4538−451. But 4538 includes BOTH train AND validation splits. Actual training-only ≈ 3,630 (80% of 4,538).

**Fix:** Change to "4,087 training and validation samples" or compute the actual training-only count.

---

#### M3: Table 7 Never Referenced in Text

**Problem:** Table 7 (Augmentation Impact) exists with caption but the body text around it never says "Table 7" or "as shown in Table 7".

**Fix:** Add "Table 7 presents..." or "as shown in Table 7" in the augmentation impact discussion paragraph (around line 1193).

---

#### M4: Figure 2 Never Referenced in Text

**Problem:** Figure 2 (baseline vs transformer comparison) has caption but is never mentioned in surrounding text.

**Fix:** Add "as illustrated in Figure 2" in the full test set discussion (Section 3.2, around line 1040-1050).

---

#### M5: IAA Description Inconsistent

**Problem:** Two different descriptions of inter-annotator agreement:
- Line 724: "substantial inter-annotator agreement"
- Line 1446: "moderate-to-substantial inter-annotator agreement"

**Fix:** Make consistent. Use "moderate-to-substantial" in both places (safer since we don't have exact κ).

---

#### M6: Davidson et al. F1 Metric Mismatch in Table 9 (Line 1339)

**Problem:** Table 9 reports Davidson et al. F1=90.00%, but original paper reports this as **F1-weighted**, not F1-Macro. Our primary metric is F1-Macro — comparison is apples-to-oranges.

**Fix:** Add footnote/note to Table 9 or change Davidson entry to show "90.00*" with note "* F1-weighted, not directly comparable."

---

#### M7: Results Section Uses Present Tense

**Problem:** Results section uses "achieves", "yields", "shows", "outperforms" while abstract/conclusion use past tense. Inconsistent.

**Fix:** Change results section verbs to past tense: "achieved", "yielded", "showed", "outperformed". Focus on the most prominent instances — don't need to catch every single one but main topic sentences should be past tense.

---

#### M8: Conclusion Present Tense (Line 1555)

**Problem:** "The primary contribution of this work **is providing** empirical evidence..."

**Fix:** Change to "The primary contribution of this work **is** empirical evidence that..." or "This work **provided** empirical evidence that..."

---

## EXECUTION ORDER

```
1. F1: Fix per-class analysis text (lines 1084-1091 + 1101-1108)
2. F2: Add [11] citation in SVM methodology (around line 786)
3. F3: Fix label smoothing delta text (lines 1235-1253)
4. F4: Add disclosure note to Table 2 caption (line 697)
5. F5: Replace keyword "Transformer" (line 417)
6. M1: Qualify multi-seed in abstract (line 490-495) + conclusion (line 1530)
7. M2: Fix "4087 training samples" wording (line 1208)
8. M3: Add "Table 7" reference in augmentation text (around line 1193)
9. M4: Add "Figure 2" reference in Section 3.2 (around line 1040)
10. M5: Harmonize IAA language (lines 724, 1446)
11. M6: Add metric note for Davidson in Table 9 (line 1339)
12. M7: Change key verbs to past tense in Results section
13. M8: Fix conclusion present tense (line 1555)
14. REGENERATE: python paper/generate_kinetik_paper.py
15. VERIFY: Open paper_kinetik.docx in Word, spot-check fixes
```

---

## KEY FILE LOCATIONS

| File | Purpose |
|------|---------|
| `paper/generate_kinetik_paper.py` | **THE main file to edit** (~1,800 lines) |
| `paper/paper_kinetik.docx` | Generated output paper |
| `results/manual_only_results.json` | Manual-only evaluation data (for F1 fix) |
| `results/ablation_results.json` | Label smoothing data (for F3 fix) |
| `results/comparative_results.json` | Full test set results (for F3 fix) |
| `results/augmentation_impact.json` | Augmentation gap data |
| `results/multi_seed_results.json` | Multi-seed evaluation |
| `results/baseline_results.json` | SVM + LR baselines |
| `data/cleaned/cleaning_report.json` | Dataset cleanup stats |

---

## NUMBERS CHEAT SHEET (verified from results JSON)

| Metric | Value | Source |
|--------|-------|--------|
| Dataset size | 9,775 | cleaning_report.json |
| Manual samples | 4,538 (46.4%) | manual_only_results.json |
| Synthetic samples | 5,237 (53.6%) | manual_only_results.json |
| Train/Val/Test | 7,820 / 977 / 978 | comparative_results.json |
| Manual test samples | 451 | manual_only_results.json |
| **XLM-R manual-only F1** | **53.89%** | manual_only_results.json |
| XLM-R full test F1 | 80.26% | comparative_results.json |
| XLM-R synthetic test F1 | 99.41% | augmentation_impact.json |
| SVM manual-only F1 | 48.55% | manual_only_results.json |
| IndoBERT manual-only F1 | 45.27% | manual_only_results.json |
| IndoBERT+LS manual-only F1 | 49.30% | manual_only_results.json |
| **XLM-R per-class manual-only:** | | manual_only_results.json |
| — Not Hate F1 | 70.72% (N=183) | |
| — Light F1 | 51.13% (N=136) | |
| — Moderate F1 | **40.00%** (N=61) ← LOWEST | |
| — Severe F1 | 53.73% (N=71) | |
| IndoBERT full test F1 | 76.12% | comparative_results.json |
| IndoBERT+LS full test F1 | 77.36% | comparative_results.json |
| Ablation baseline (eps=0) | 77.09% | ablation_results.json |
| Ablation best (eps=0.1) | 77.36% | ablation_results.json |
| Ablation delta | +0.27 (isolated) | ablation_results.json |
| Comparative delta | +1.24 (76.12→77.36) | comparative_results.json |
| Multi-seed mean (4 stable) | 80.83% ± 1.83% | multi_seed_results.json |

---

## ENVIRONMENT

- **OS**: Windows 11
- **Python**: Anaconda (use `python`, not `python3`)
- **Dependencies**: `python-docx`, `matplotlib`, `seaborn`, `numpy`
- **Regenerate paper**: `python paper/generate_kinetik_paper.py`
- **NO GPU needed** — all fixes are text edits in Python script

---

## VERIFICATION CHECKLIST (after all fixes)

```
[ ] F1: Per-class text says Moderate is worst (40.00%), NOT Light
[ ] F2: Reference [11] Joachims cited in SVM methodology paragraph
[ ] F3: Label smoothing delta consistent with Table 6 numbers
[ ] F4: Table 2 caption has paraphrase disclosure
[ ] F5: Keyword "Transformer" replaced, no title overlap
[ ] M1: Multi-seed in abstract/conclusion says "on full test set"
[ ] M2: Manual training count accurate (says "training and validation")
[ ] M3: "Table 7" referenced in body text
[ ] M4: "Figure 2" referenced in body text
[ ] M5: IAA language consistent in both locations
[ ] M6: Davidson F1 metric noted in Table 9
[ ] M7: Key verbs in Results are past tense
[ ] M8: Conclusion "is providing" → past tense
[ ] python paper/generate_kinetik_paper.py runs without error
[ ] paper_kinetik.docx opens in Word
[ ] All reference numbers still match [1]-[31]
[ ] No orphaned table/figure references
```

---

## PREVIOUS SESSION HISTORY

- **Session 1-2**: Experiments, data cleaning, model training, figure generation
- **Session 3**: Strategic review → identified 7 fixes
- **Session 4**: Executed all 7 Session 3 fixes + 4 formatting + 7 reviewer fixes (18 total)
- **Session 5**: Fixed footer citations, DOI hyperlink, Indonesian→English figure labels; final critical review found 5 FATAL + 8 MAJOR remaining issues
