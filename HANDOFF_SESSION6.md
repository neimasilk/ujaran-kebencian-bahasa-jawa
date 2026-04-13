# HANDOFF DOCUMENT — Session 6
## Javanese Hate Speech Detection — Post-Rejection Major Revision
**From:** Session 6 Agent (2026-04-13)
**To:** Next agent (final review, polish, and submit)
**Date:** 2026-04-13

---

## RINGKASAN SITUASI

Paper **DITOLAK** dari Kinetik (SINTA-2) pada 2026-04-13 dengan 6 keluhan: no GAP analysis, no comparison with existing methods, no analysis of proposed methods, no significant contribution, no research problem from literature, dan tidak mengikuti template.

Session 6 melakukan **pivot total**: reframe dari "comparative study" menjadi "evaluation bias study", jalankan 6 eksperimen baru, rewrite paper dari nol, dan sesuaikan struktur ke template Kinetik.

**Status: Paper v3 SELESAI, belum disubmit. User ingin pikir matang dulu.**

---

## APA YANG BERUBAH (Session 6)

### Pivot Strategis
- **Judul lama**: "Evaluating LLM-Augmented Transformer Models for Severity-Based Hate Speech Detection in Javanese"
- **Judul baru**: "Evaluating the Reliability of LLM-Augmented Training Data for Severity-Based Hate Speech Detection in Low-Resource Javanese"
- **Framing lama**: "Kami evaluasi model transformer" (ditolak: tidak novel)
- **Framing baru**: "Kami investigasi apakah augmentasi LLM menciptakan bias evaluasi" (novel)

### 6 Eksperimen Baru Dijalankan
| # | Script | Output | Temuan Kunci |
|---|---|---|---|
| 1 | `reproduce/error_analysis.py` | `results/error_analysis.json` | 73.8% adjacent confusion, overconfidence 0.855, ensemble upper bound 69.18% |
| 2 | `reproduce/augmentation_ratio_study.py` | `results/augmentation_ratio_results.json` | Manual F1 flat ~47% di semua rasio (SVM + IndoBERT) |
| 3 | `reproduce/synthetic_vs_manual_analysis.py` | `results/data_quality_analysis.json` | **97.26% source distinguishable**, 14.6% Jaccard vocab overlap |
| 4 | `reproduce/two_stage_training.py` | `results/two_stage_results.json` | Temuan negatif: XLM-R 52.16% (vs 53.89% single-stage), IndoBERT 45.34% (vs 45.27%) |
| 5 | `reproduce/cross_validation.py` (inline) | `results/cv_baselines.json` | SVM 46.95±0.70 manual, 77.47±1.08 full (konsisten) |
| 6 | Filtered augmentation (inline) | `results/filtered_augmentation.json` | Negatif: filtering tidak membantu, bahkan synthetic "terbaik" punya P(synth)=0.76 |
| 7 | Statistical significance (inline) | `results/statistical_tests.json` | p < 10^-11, Cohen's d > 33 |

### Paper v3 Generator
- **File**: `paper/generate_kinetik_paper_v3.py` (~980 baris)
- **Output**: `paper/paper_kinetik_v3.docx` dan `paper/paper_kinetik_v3.pdf`
- **Regenerate**: `python paper/generate_kinetik_paper_v3.py`

### Struktur Paper Baru (mengikuti template Kinetik)
```
1. Introduction
   - Background, Indonesia, Javanese
   - Literature review (TERINTEGRASI, bukan section terpisah)
   - Table 1: Comparison with Prior Work (6 paper + ours)
   - GAP analysis (eksplisit)
   - 3 Research Questions
   - 3 Contributions

2. Research Method
   2.1. Dataset Construction (Table 2: examples, Table 3: composition)
   2.2. Dual-Track Evaluation Protocol
   2.3. Models (5 models)
   2.4. Experimental Setup

3. Results and Discussion
   3.1. Overall Performance (Table 4 + CV + statistical significance)
   3.2. Comparison with Prior Work
   3.3. Augmentation Bias Analysis (Table 5 + Figure 1: ratio curve)
   3.4. Data Quality Analysis (Table 6 + Figure 2: surface patterns)
   3.5. Error Analysis (Figure 3: confusion matrix + per-class + confidence)
   3.6. Two-Stage Training (Table 7 + honest negative result)
   3.7. Limitations

4. Conclusion (menjawab RQ1, RQ2, RQ3 eksplisit)
Acknowledgement
References (28 refs, IEEE style, "References kinetik" style)
```

### Figures
| Figure | File | Konten |
|---|---|---|
| Figure 1 | `paper/figures/figure_augmentation_ratio.png` | Ratio curve: manual vs full F1 |
| Figure 2 | `paper/figures/figure_data_quality.png` | Bar chart: surface pattern manual vs synthetic |
| Figure 3 | `paper/figures/figure3_confusion_matrix_manual.png` | Confusion matrix XLM-R pada manual test |

---

## CHECKLIST YANG SUDAH SELESAI

### Keluhan Editor → Fix
- [x] "No GAP analysis" → Paragraf 4 Introduction: eksplisit gap statement
- [x] "No comparison with existing methods" → Table 1: 6 paper vs ours
- [x] "No analysis of proposed methods" → Section 3.5: error analysis (3 paragraf)
- [x] "No significant contribution" → 3 kontribusi eksplisit di Introduction
- [x] "No research problem from literature" → 3 RQ di Introduction
- [x] "Follow KINETIK guidelines" → 4-section structure, template style, 28 refs

### Thesis Killers Ditemukan & Diperbaiki
- [x] Ref [8] dan [10] duplikat → [10] diganti Fauzi & Yuniarti 2018
- [x] Section 4.6 salah referensi "Table 5" untuk XLM-R → Dihapus
- [x] Article Info "Resubmitted" → Dikosongkan
- [x] White space halaman 5-6 → Figure diperkecil + reorder
- [x] Triple-dash literal → Em-dash unicode
- [x] Table 1 metric mismatch Ibrohim → Footnote **
- [x] Conclusion tidak jawab RQ → Eksplisit RQ1/RQ2/RQ3
- [x] Error Analysis terlalu tipis → 3 paragraf + per-class
- [x] Ref [10] orphan → Dikutip di literature review
- [x] Struktur 5-section → Merged ke 4-section (template Kinetik)
- [x] References pakai style salah → "References kinetik" style
- [x] Sub-section tanpa titik → Ditambah titik (3.1.)

### Verifikasi Otomatis (SEMUA PASS)
- [x] 28 referensi dikutip, 28 di list, 0 orphan
- [x] 7 tabel + 3 figur semua direferensi
- [x] Past tense di Results
- [x] Keywords tidak overlap title
- [x] RQ1/RQ2/RQ3 dijawab di Conclusion
- [x] Abstract 195 kata (limit 100-250)
- [x] 7 halaman (limit 6-10)
- [x] Statistical significance: p < 0.001, Cohen's d > 33

---

## KEY NUMBERS CHEAT SHEET

| Metrik | Nilai | Sumber |
|---|---|---|
| Dataset total | 9,775 | cleaning_report.json |
| Manual / Synthetic | 4,538 / 5,237 | manual_only_results.json |
| Manual test | 451 samples | manual_only_results.json |
| XLM-R Full F1 | 80.26% | comparative_results.json |
| XLM-R Manual F1 | 53.89% | manual_only_results.json |
| XLM-R Synthetic F1 | 99.41% | augmentation_impact.json |
| SVM Manual F1 | 48.55% | manual_only_results.json |
| Source distinguishability | 97.26% | data_quality_analysis.json |
| Jaccard vocab overlap | 14.6% | data_quality_analysis.json |
| Adjacent error ratio (XLM-R) | 73.82% | error_analysis.json |
| Ensemble upper bound | 69.18% | error_analysis.json |
| Two-stage XLM-R Manual | 52.16% (-1.73pp) | two_stage_results.json |
| CV SVM Manual | 46.95 ± 0.70 | cv_baselines.json |
| CV SVM Full | 77.47 ± 1.08 | cv_baselines.json |
| Statistical significance | p < 10^-11 | statistical_tests.json |

---

## CADANGAN UNTUK RESPON REVIEWER

Jika reviewer bertanya, data sudah tersedia:

1. **"Kenapa tidak filter synthetic data?"** → `results/filtered_augmentation.json`: sudah dicoba, tidak membantu. Bahkan sample "paling mirip manual" masih punya P(synth)=0.76.

2. **"Kenapa tidak test XLM-R di ratio study?"** → XLM-R collapse di 0% (14.64% dari augmentation_impact.json). IndoBERT tidak collapse (48.75%), menunjukkan model kecil lebih robust.

3. **"Apakah multi-seed stabil?"** → `results/multi_seed_results.json`: 80.83% ± 1.83% (4 dari 5 seed stabil). Satu seed collapse (1024: 11.07%).

4. **"Berapa label smoothing improvement?"** → `results/ablation_results.json`: +0.27 pp (77.09→77.36) pada controlled ablation.

---

## APA YANG MUNGKIN MASIH PERLU DIPERBAIKI

### Hal yang user ingin pikirkan matang:
1. **Apakah 54% F1 cukup meyakinkan sebagai kontribusi?** Paper diframe sebagai methodology paper (menunjukkan masalah), bukan performance paper. Tapi reviewer berbeda mungkin punya ekspektasi berbeda.

2. **Ecological validity dataset**: Beberapa teks manual terlihat seperti terjemahan dari hate speech Inggris ("wong coklat", "Mussies"). Ini disebutkan di limitations tapi bisa diperkuat jika perlu.

3. **Apakah perlu tambah referensi artikel Kinetik?** Template prefer self-citation. Belum ada sitasi paper Kinetik.

4. **Apakah tabel terlalu banyak?** 7 tabel + 3 figur di 7 halaman. Mungkin bisa merge beberapa untuk ruang lebih.

5. **Review manual di Word**: python-docx tidak sempurna — pastikan column widths, font consistency, dan page breaks benar di Word.

---

## ENVIRONMENT

- **OS**: Windows 11
- **Python**: Anaconda
- **GPU**: RTX 4080
- **Regenerate paper**: `python paper/generate_kinetik_paper_v3.py`
- **Convert ke PDF**: `python -c "from docx2pdf import convert; convert('paper/paper_kinetik_v3.docx', 'paper/paper_kinetik_v3.pdf')"`
- **Unicode warning**: Jangan pakai emoji/arrow unicode di print() — crash cp1252
