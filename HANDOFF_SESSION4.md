# HANDOFF DOCUMENT — Session 4
## Javanese Hate Speech Detection — Post-Fix Status Report
**From:** Session 4 Agent (2026-02-27)
**To:** Next agent (manual verification, further edits, or submission)
**Date:** 2026-02-27T09:58 WIB

---

## RINGKASAN SITUASI

Paper **sudah selesai secara kode dan konten**. Semua perbaikan dari HANDOFF_SESSION3.md telah dieksekusi, plus tambahan reviewer-identified fixes. Paper generator (`paper/generate_kinetik_paper.py`) menghasilkan `paper/paper_kinetik.docx` **tanpa error**.

**Status: SIAP UNTUK VERIFIKASI VISUAL DI WORD.**
Tidak ada lagi perbaikan kode yang diperlukan. Langkah selanjutnya adalah buka DOCX di Microsoft Word dan verifikasi formatting/page count.

---

## APA YANG SUDAH DIKERJAKAN

### Phase 1: HANDOFF_SESSION3 Fixes (7/7 ✅)

| Fix | Deskripsi | Status |
|-----|-----------|--------|
| FIX 1 | Section numbering 3.1→3.3 skip → sequential 3.1–3.7 | ✅ |
| FIX 2 | Tambah 2 referensi Kinetik ([30] Cahyaningtyas 2021, [31] Akbar 2024) | ✅ |
| FIX 3 | Article Info panel (2-column table: kiri=info, kanan=abstract) | ✅ |
| FIX 4 | Hapus Table 3, 8, 10, Figure 4; compress ablation → 1 paragraf; renumber 1–9 / 1–3 | ✅ |
| FIX 5 | Re-frame narrative: title → "Evaluating...", abstract/contributions/conclusion → augmentation bias as finding | ✅ |
| FIX 6 | Softened κ=0.72 → "substantial agreement", hardcoded quality scores → qualitative | ✅ |
| FIX 7 | Confusion matrix → manual-only data (451 samples), generated via matplotlib | ✅ |

### Phase 2: Kinetik Formatting Compliance (4/4 ✅)

| Fix | Deskripsi | Status |
|-----|-----------|--------|
| Keywords alphabetical | → Data Augmentation, Low-Resource NLP, Severity Classification, Text Classification, Transformer | ✅ |
| Keywords no title overlap | Tidak ada kata persis dari title | ✅ |
| Remove duplicate keywords | Dihapus dari abstract body, hanya ada di Article Info panel | ✅ |
| Abstract past tense | "evaluated", "was constructed", "achieved", "confirmed", "provided" | ✅ |

### Phase 3: Reviewer-Identified Fixes (7/7 ✅)

| ID | Severity | Deskripsi | Status |
|----|----------|-----------|--------|
| F1 | 🔴 FATAL | Kontekstualisasi F1=53.89% — tambah paragraf di 3.1 tentang gap XLM-R vs SVM hanya +5.34 pts | ✅ |
| F2 | 🔴 FATAL | Conclusion → past tense (semua verb) | ✅ |
| F3 | 🔴 FATAL | Dataset availability statement + GitHub URL | ✅ |
| M1 | 🟡 MAJOR | Section 3.3 text "full test set" → "manual-only test data" (match Figure 3) | ✅ |
| M2 | 🟡 MAJOR | Multi-seed limitation → diakui di Limitations ("stability on genuine data remains to be verified") | ✅ |
| m1 | 🟢 MINOR | "Three architectures" → "Two architectures + LS variant" | ✅ |
| m2 | 🟢 MINOR | Kinetik refs [30,31] dipindah dari paragraf transformer ke paragraf regional NLP | ✅ |

---

## FINAL PAPER STATS

```
References: 31 (including 2 Kinetik)
Tables: 9 (sequential 1–9)
Figures: 3 (sequential 1–3)
Abstract: 170 words (limit: 100–250)
Keywords: 5, alphabetical, no title overlap
PRIMARY metric: XLM-R Large Manual-only F1 = 53.89%
```

---

## AUTOMATED VERIFICATION (10/10 ✅)

| Check | Result |
|-------|--------|
| Present tense in Conclusion | ✅ None |
| Orphaned refs (Table 10/11/12, Figure 4/5) | ✅ None |
| Refs [30], [31] cited in body | ✅ Both cited |
| Abstract: no citations | ✅ Clean |
| Table numbering 1–9 sequential | ✅ |
| Figure numbering 1–3 sequential | ✅ |
| Dataset availability statement | ✅ Present |
| GitHub URL | ✅ Present |
| Abstract tense (past) | ✅ Consistent |
| Abstract word count (100–250) | ✅ 170 words |

---

## KEY FILE LOCATIONS

| File | Purpose | Status |
|------|---------|--------|
| `paper/generate_kinetik_paper.py` | Main paper generator (1,730 lines) | ✅ Final |
| `paper/paper_kinetik.docx` | Generated paper DOCX | ✅ Latest |
| `Template Kinetik Mendeley.docx` | Kinetik template (root dir) | Unchanged |
| `paper/figures/figure3_confusion_matrix_manual.png` | Manual-only CM figure | ✅ Generated |
| `paper/figures/figure1_dataset_distribution.png` | Dataset distribution | From Session 2 |
| `paper/figures/figure5_baseline_vs_transformer.png` | Model comparison | From Session 2 |
| `HANDOFF_SESSION3.md` | Previous handoff document | Reference only |
| `results/*.json` | All experiment results | Unchanged |

### Results Data Files

| File | Contains |
|------|----------|
| `results/comparative_results.json` | IndoBERT, XLM-R, IndoBERT+LS results |
| `results/manual_only_results.json` | All 5 models on manual-only test (451 samples) |
| `results/augmentation_impact.json` | 99.41% vs 53.89% gap data |
| `results/ablation_results.json` | Label smoothing ε sweep |
| `results/multi_seed_results.json` | 5-seed evaluation |
| `results/baseline_results.json` | SVM + LR baselines |
| `data/cleaned/cleaning_report.json` | Dataset cleanup stats |

---

## NUMBERS CHEAT SHEET (all verified from results JSON)

| Metric | Value | Source |
|--------|-------|--------|
| Dataset size | 9,775 | cleaning_report.json |
| Manual samples | 4,538 (46.4%) | manual_only_results.json |
| Synthetic samples | 5,237 (53.6%) | manual_only_results.json |
| Train/Val/Test | 7,820 / 977 / 978 | comparative_results.json |
| Manual test samples | 451 | manual_only_results.json |
| XLM-R Large manual-only F1 | 53.89% | manual_only_results.json |
| XLM-R Large full test F1 | 80.26% | comparative_results.json |
| XLM-R Large synthetic test F1 | 99.41% | augmentation_impact.json |
| SVM manual-only F1 | 48.55% | manual_only_results.json |
| SVM full test F1 | 77.77% | baseline_results.json |
| XLM-R advantage over SVM (manual) | +5.34 pts | computed |
| IndoBERT manual-only F1 | 45.27% | manual_only_results.json |
| IndoBERT+LS manual-only F1 | 49.30% | manual_only_results.json |
| Label smoothing optimal ε | 0.1 (+0.27 F1) | ablation_results.json |
| Multi-seed mean (4 stable, full test) | 80.83% ± 1.83% | multi_seed_results.json |
| Seeds | 79.32, 79.01, 81.44, 83.55, 11.07(collapsed) | multi_seed_results.json |
| Manual-only model collapsed F1 | 14.43% | augmentation_impact.json |

---

## REMAINING WORK — MANUAL ONLY

### 1. Verifikasi Visual di Microsoft Word
```
[ ] Buka paper/paper_kinetik.docx di Word
[ ] Verifikasi page count: 6–10 halaman
[ ] Verifikasi 2-column Article Info table renders correctly
[ ] Verifikasi font, margin, dan spacing sesuai template
[ ] Verifikasi semua tabel terbaca dengan benar
[ ] Verifikasi ketiga figur tampil (dataset dist, model comparison, confusion matrix)
```

### 2. Jika Page Count > 10
Opsi compress:
- Kurangi teks deskriptif di Section 3 (Results)
- Gabung paragraf kecil
- Kurangi Table 9 (Comparison with Prior Work) — rows bisa dikurangi

### 3. Jika Page Count < 6
Tambah:
- Expand discussion pada Section 3.4 (Augmentation Impact)
- Tambah Acknowledgement section sebelum References

### 4. Submission ke Kinetik
```
[ ] Register di kinetik.umm.ac.id (jika belum)
[ ] Upload paper_kinetik.docx via OJS
[ ] Upload cover letter (jika diminta)
[ ] Pastikan plagiarism check (Turnitin) < 25%
```

---

## PAPER STRUCTURE OVERVIEW

```
Title: "Evaluating LLM-Augmented Transformer Models for Severity-Based
        Hate Speech Detection in Javanese"

Authors: Mukhlis Amien*1, Daniel Rudiaman Sijabat2, Yekti Asmoro Kanthi3
         1,2 Dept Informatics, 3 Dept Information System
         Universitas Bhinneka Nusantara, Malang

[Article Info | Abstract]  <-- 2-column table

1. Introduction (7 paragraphs: context, Indonesia, regional NLP,
                  transformers, augmentation, label smoothing, contributions)

2. Research Method
   2.1 Dataset Construction (Table 1: distribution, Table 2: examples, Figure 1)
   2.2 LLM Data Augmentation
   2.3 Experimental Setup (Table 3: hyperparameters)

3. Results and Discussion
   3.1 Performance on Manual-Only Test Data (Table 4, Table 5) <- PRIMARY
       + paragraph contextualizing narrow XLM-R vs SVM gap
   3.2 Performance on Full Test Set (Table 6, Figure 2) <- SECONDARY
   3.3 Per-Class Analysis (Figure 3: manual-only confusion matrix)
   3.4 Augmentation Impact Analysis (Table 7) <- KEY FINDING
   [Label smoothing compressed paragraph]
   3.5 Multi-Seed Statistical Significance (Table 8)
   3.6 Comparison with Prior Work (Table 9)
   3.7 Limitations (6 points including multi-seed caveat)

4. Conclusion
   + Future Work paragraph
   + Data Availability statement (GitHub link)

References (31 entries, IEEE numbered, includes 2 Kinetik)
```

---

## ENVIRONMENT

- **OS**: Windows 11
- **Python**: Anaconda (use `python`, not `python3`)
- **Dependencies**: `python-docx`, `matplotlib`, `seaborn`, `numpy`
- **Regenerate paper**: `python paper/generate_kinetik_paper.py`
- **NO GPU needed** — paper generation is CPU-only

---

## DESIGN DECISIONS LOG

1. **Primary metric = manual-only F1 (53.89%)**, not full test F1 (80.26%) — because full test is 53.9% synthetic data
2. **99.41% vs 53.89% gap framed as KEY FINDING** — not as embarrassing result
3. **XLM-R vs SVM gap (+5.34 pts) explicitly discussed** — preempts reviewer criticism
4. **Multi-seed evaluated on full test only** — acknowledged as limitation (no GPU re-run possible)
5. **Dataset availability included** — GitHub URL for reproducibility
6. **Kinetik refs placed in regional NLP paragraph** — not transformer paragraph (topical fit)
7. **Keywords avoid title words** — "Severity Classification" instead of "Hate Speech Detection"
