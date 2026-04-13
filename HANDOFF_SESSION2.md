# HANDOFF DOCUMENT — Session 2
## Javanese Hate Speech Detection — Experiment Execution & Paper Completion
**From:** Antigravity (experiment execution, paper writing)
**To:** Next session (final review, submission preparation)
**Date:** 2026-02-25 15:45 WIB

---

## RINGKASAN SESI INI

Semua eksperimen telah dijalankan (Step 1-5), paper sections diupdate dengan angka aktual, dan **paper LaTeX lengkap** sudah dibuat dalam 2 bahasa (Indonesia + English). Semua angka berasal dari eksperimen nyata.

---

## HASIL EKSPERIMEN (SEMUA SELESAI ✓)

### Step 1: Honest Evaluation (`reproduce/honest_evaluation.py`) ✓
- 14 checkpoint existing dievaluasi pada clean test set (978 sampel)
- **Best existing**: IndoBERT Phase5 (exp14, cp1255) — **F1=94.56%**, Acc=94.58%
- Catatan: Model ini dilatih pada dataset lama (sebelum cleanup), performa tinggi mungkin bukan generalisasi sebenarnya
- Output: `results/honest_evaluation_results.json`

### Step 2: Comparative Training (`reproduce/train_comparative.py`) ✓
- **Bug fix**: Line 285, `total_mem` → `total_memory` (PyTorch API)
- 3 model dilatih from scratch:

| Model | Val F1 | Test F1 | Test Acc | Waktu |
|-------|--------|---------|----------|-------|
| IndoBERT base | 75.44% | 76.12% | 76.28% | 152s |
| **XLM-R Large** | **80.54%** | **80.26%** | **80.27%** | 904s |
| IndoBERT+LS (ε=0.1) | 76.59% | 77.36% | 77.51% | 158s |

- Output: `results/comparative_results.json`

### Step 3: Ablation Study (`reproduce/real_ablation_study.py`) ✓
| ε | Test F1 | Test Acc |
|---|---------|----------|
| 0.00 | 77.09% | 77.40% |
| 0.05 | 76.79% | 76.99% |
| **0.10** | **77.36%** | **77.51%** |
| 0.15 | 76.91% | 77.20% |
| 0.20 | 76.54% | 76.58% |

- Output: `results/ablation_results.json`

### Step 4: Multi-Seed Evaluation (`reproduce/multi_seed_eval.py`) ✓
- XLM-R Large dengan 5 seeds [42, 123, 456, 789, 1024]
- **4 seeds stabil**: F1 = 80.83% ± 1.74%
- **Seed 1024 collapse**: F1 = 11.07% (prediksi semua ke 1 kelas)
- Output: `results/multi_seed_results.json`

### Step 5: Generate Figures (`reproduce/generate_honest_figures.py`) ✓
- 5 PNG + 2 LaTeX tables di `paper/figures/`
- Output: `paper/figures/figure1_dataset_distribution.png`, `figure2_confusion_matrix.png`, `figure3_model_comparison.png`, `figure4_label_smoothing_ablation.png`, `figure5_per_class_comparison.png`, `table2_performance.tex`, `table3_per_class.tex`

### Step 6: Paper Sections Updated ✓
- `REVISION_PROGRESS.md` — semua TBD → angka aktual, checklist verified ✓
- `NEXT_ACTIONS.md` — semua step marked DONE ✓
- `SECTION_LIMITATION.md` — multi-seed stats + training collapse note ✓
- `SECTION_LLM_AUGMENTATION.md` — downstream performance XLM-R=80.26% ✓

### Step 7: LaTeX Paper Created ✓
- **Indonesia**: `paper/paper_jitk.tex` → `paper/paper_jitk.pdf` (6 halaman, 1.1MB)
- **English**: `paper/paper_jitk_en.tex` → `paper/paper_jitk_en.pdf` (6 halaman, 1.09MB)
- Compiler: XeLaTeX (MiKTeX) dengan Times New Roman
- Isi: 7 section, 5 figures, 9 tables, 1 pipeline diagram, 11 referensi IEEE

---

## FILE STRUCTURE (UPDATED)

```
d:\documents\ujaran-kebencian-bahasa-jawa\
├── data/
│   ├── improved/phase5_deepseek_relabeled.csv    # Dataset asli (10,019)
│   └── cleaned/
│       ├── final_dataset.csv                     # Dataset bersih (9,775) ← DIGUNAKAN
│       └── cleaning_report.json                  # Laporan cleaning
│
├── models/                                       # Checkpoint existing (14 model)
│   ├── exp3_phase2/checkpoint-...
│   ├── exp14_indobert_phase5/checkpoint-1255     # ← BEST EXISTING (F1=94.56%)
│   └── ...
│
├── reproduce/
│   ├── clean_dataset.py                          # ✓ Sudah jalan
│   ├── honest_evaluation.py                      # ✓ Sudah jalan
│   ├── train_comparative.py                      # ✓ Sudah jalan (BUG FIXED: line 285)
│   ├── real_ablation_study.py                    # ✓ Sudah jalan
│   ├── multi_seed_eval.py                        # ✓ Sudah jalan
│   ├── generate_honest_figures.py                # ✓ Sudah jalan
│   └── verify_references.py                      # ✓ Sudah jalan
│
├── results/
│   ├── honest_evaluation_results.json            # Step 1 output
│   ├── comparative_results.json                  # Step 2 output
│   ├── ablation_results.json                     # Step 3 output
│   └── multi_seed_results.json                   # Step 4 output
│
├── paper/
│   ├── paper_jitk.tex                            # Paper LaTeX (Bahasa Indonesia)
│   ├── paper_jitk.pdf                            # PDF compiled ✓ (6 hal)
│   ├── paper_jitk_en.tex                         # Paper LaTeX (English)
│   ├── paper_jitk_en.pdf                         # PDF compiled ✓ (6 hal)
│   ├── Paper_JITK_NoFigures_Safe.docx            # Paper asli (lama, jangan dipakai)
│   ├── SECTION_RELATED_WORK.md                   # ✓ Updated
│   ├── SECTION_LLM_AUGMENTATION.md               # ✓ Updated
│   ├── SECTION_LIMITATION.md                     # ✓ Updated
│   ├── REVISION_PROGRESS.md                      # ✓ Updated (all done)
│   ├── NEXT_ACTIONS.md                           # ✓ Updated (all done)
│   └── figures/
│       ├── figure1_dataset_distribution.png
│       ├── figure2_confusion_matrix.png
│       ├── figure3_model_comparison.png
│       ├── figure4_label_smoothing_ablation.png
│       ├── figure5_per_class_comparison.png
│       ├── table2_performance.tex
│       └── table3_per_class.tex
│
├── archive/scripts/
│   └── generate_paper_figures.py                 # ARCHIVED — berisi fabricated data
│
└── HANDOFF.md                                    # Handoff session sebelumnya
```

---

## ANGKA-ANGKA KUNCI (DARI EKSPERIMEN NYATA)

| Metrik | Nilai |
|--------|-------|
| Dataset size | 9,775 sampel (setelah cleaning dari 10,019) |
| Data source | 47.7% manual + 52.3% LLM-generated |
| Data split | 80/10/10 stratified (seed=42) |
| Train/Val/Test | 7,820 / 977 / 978 |
| Best from-scratch | **XLM-R Large: F1=80.26%, Acc=80.27%** |
| IndoBERT base | F1=76.12%, Acc=76.28% |
| IndoBERT+LS (ε=0.1) | F1=77.36%, Acc=77.51% |
| Optimal ε | 0.1 (improvement +0.27 F1 points) |
| Multi-seed mean (4 stable) | **F1=80.83% ± 1.74%** |
| Best existing checkpoint | IndoBERT Phase5: F1=94.56% (dataset lama) |
| Training collapse rate | 1/5 seeds (20%) |

---

## ENVIRONMENT

- **OS**: Windows 11
- **Python**: Anaconda (base)
- **GPU**: NVIDIA GeForce RTX 4080 (CUDA enabled)
- **LaTeX**: MiKTeX 24.1 (xelatex-dev.exe)
- **Libraries**: PyTorch, Transformers (Hugging Face), scikit-learn, matplotlib

### Cara compile LaTeX:
```bash
cd d:\documents\ujaran-kebencian-bahasa-jawa\paper
xelatex-dev.exe -interaction=nonstopmode paper_jitk_en.tex
xelatex-dev.exe -interaction=nonstopmode paper_jitk_en.tex   # second pass for refs
```
Note: Exit code 1 dari MiKTeX update warning — BUKAN error kompilasi. PDF tetap tergenerate.

---

## YANG BELUM DILAKUKAN / BISA DILANJUTKAN

### 1. Review & Polish Paper Content
- [ ] Review kualitas bahasa Inggris pada `paper_jitk_en.tex`
- [ ] Verifikasi semua angka di paper konsisten dengan `results/*.json`
- [ ] Update nama penulis, afiliasi, email (saat ini placeholder)
- [ ] Tambahkan abstrak bahasa Indonesia di versi English (jika diperlukan JITK)

### 2. Multi-Seed Analysis
- [ ] Pertimbangkan re-run seed 1024 atau ganti dengan seed lain
- [ ] Atau tambahkan diskusi lebih detail tentang training collapse di paper

### 3. Formatting & Submission
- [ ] Cek format JITK (margin, font size, section numbering)
- [ ] Pastikan gambar resolusi cukup untuk print
- [ ] Buat versi final PDF untuk submission
- [ ] Siapkan cover letter jika diperlukan

### 4. Additional Analysis (Optional)
- [ ] Analisis error qualitative pada misklasifikasi
- [ ] Tambahkan contoh prediksi benar/salah di paper
- [ ] Bandingkan per-class performance antar model lebih detail
- [ ] Evaluasi pada external test set jika tersedia

### 5. Git & Documentation
- [ ] Commit semua perubahan ke Git
- [ ] Update README.md dengan instruksi reproduksi
- [ ] Tag release untuk paper submission

---

## CATATAN PENTING

1. **Jangan gunakan** `archive/scripts/generate_paper_figures.py` — berisi data fabrikasi
2. **Seed 1024 collapse** — fenomena umum pada model besar, sudah didokumentasikan di paper
3. **Exit code 1 pada xelatex** — dari MiKTeX update warning, bukan error. PDF tetap valid.
4. **Bug fix** pada `train_comparative.py` line 285: `total_mem` → `total_memory`
5. **Semua angka** di paper sections (`.md` files) sudah diupdate ke angka aktual
