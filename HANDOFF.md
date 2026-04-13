# HANDOFF DOCUMENT
## Javanese Hate Speech Detection — Honest Comparative Study
**From:** Claude Opus 4.6 (planning & script creation)
**To:** Next model (execution & finalization)
**Date:** 2026-02-25

---

## KONTEKS SINGKAT

Paper ini awalnya mengklaim F1 81.38% tapi angka itu **fabrikasi** (confusion matrix di-hardcode di `generate_paper_figures.py:397`). Kita sudah pivot ke **honest comparative study**: dataset dibersihkan, semua script training/evaluasi sudah dibuat, referensi palsu dihapus, section paper diupdate. Yang tersisa = **menjalankan script di GPU**.

---

## APA YANG SUDAH SELESAI

### 1. Dataset Cleanup (DONE — sudah dijalankan)
- Input: `data/improved/phase5_deepseek_relabeled.csv` (10,019 rows)
- Output: `data/cleaned/final_dataset.csv` (9,775 rows)
- Hapus: 33 duplikat, 186 teks pendek, 25 teks non-Jawa
- Laporan: `data/cleaned/cleaning_report.json`

### 2. Semua Script Sudah Dibuat (DONE — belum dijalankan)
```
reproduce/
├── clean_dataset.py             # Sudah dijalankan ✓
├── honest_evaluation.py         # Belum dijalankan ✗
├── train_comparative.py         # Belum dijalankan ✗
├── real_ablation_study.py       # Belum dijalankan ✗
├── multi_seed_eval.py           # Belum dijalankan ✗
├── generate_honest_figures.py   # Belum dijalankan ✗
└── verify_references.py         # Sudah dijalankan ✓
```

### 3. Paper Sections Diupdate (DONE)
- `paper/SECTION_RELATED_WORK.md` — 4 ref palsu dihapus, 4 ref baru terverifikasi ditambah
- `paper/SECTION_LLM_AUGMENTATION.md` — angka diperbaiki (33%→52.3%, 3319→5240)
- `paper/SECTION_LIMITATION.md` — limitasi data sintetis ditambah
- `paper/NEXT_ACTIONS.md` — ditulis ulang
- `paper/REVISION_PROGRESS.md` — ditulis ulang

### 4. Fabricated Script Di-archive (DONE)
- `reproduce/generate_paper_figures.py` → `archive/scripts/generate_paper_figures.py`

---

## APA YANG HARUS DILAKUKAN (URUTAN EKSEKUSI)

### LANGKAH 1: Jalankan Honest Evaluation (Fase 1)
```bash
cd D:\documents\ujaran-kebencian-bahasa-jawa
python reproduce/honest_evaluation.py
```
- **Apa yang dilakukan**: Evaluasi semua checkpoint model yang ada pada test set bersih
- **Output**: `results/honest_evaluation_results.json`
- **Durasi**: ~1 jam (GPU, inference only)
- **Catatan**: Checkpoint lama di-train pada dataset lama, jadi hasilnya mungkin rendah. Tidak masalah — ini untuk baseline.

### LANGKAH 2: Train Comparative Models (Fase 2)
```bash
python reproduce/train_comparative.py
```
- **Apa yang dilakukan**: Train 3 model dari awal pada dataset bersih:
  1. `indobert` — IndoBERT base (tanpa label smoothing)
  2. `xlmr_large` — XLM-RoBERTa Large
  3. `indobert_ls` — IndoBERT + Label Smoothing (eps=0.1)
- **Output**: `results/comparative_results.json` + model tersimpan di `models/comparative/`
- **Durasi**: ~3-4 jam (GPU)
- **Jika VRAM tidak cukup untuk xlmr_large**: Jalankan satu per satu:
  ```bash
  python reproduce/train_comparative.py --model indobert
  python reproduce/train_comparative.py --model indobert_ls
  python reproduce/train_comparative.py --model xlmr_large
  ```

### LANGKAH 3: Ablation Study (Fase 3)
```bash
python reproduce/real_ablation_study.py
```
- **Apa yang dilakukan**: Train IndoBERT dengan epsilon = [0.0, 0.05, 0.1, 0.15, 0.2]
- **Output**: `results/ablation_results.json`
- **Durasi**: ~2-3 jam (GPU)
- **Bisa parallel** dengan Langkah 2 jika VRAM cukup (tapi hati-hati OOM)

### LANGKAH 4: Multi-Seed Evaluation (Fase 4)
```bash
python reproduce/multi_seed_eval.py
```
- **Apa yang dilakukan**: Train model terbaik (dari Langkah 2) dengan 5 seed berbeda
- **Output**: `results/multi_seed_results.json`
- **Durasi**: ~2-3 jam (GPU)
- **Catatan**: Script otomatis baca `results/comparative_results.json` untuk menentukan model terbaik. Jika file belum ada, default = IndoBERT + LS (eps=0.1)

### LANGKAH 5: Generate Honest Figures (Fase 5)
```bash
python reproduce/generate_honest_figures.py
```
- **Apa yang dilakukan**: Generate semua figure dari data NYATA di `results/`
- **Output**: `paper/figures/*.png` + `paper/figures/table2_performance.tex`
- **Durasi**: ~5 menit (CPU only)
- **PENTING**: Jalankan SETELAH Langkah 2-4 selesai, karena membaca JSON results

### LANGKAH 6: Update Paper dengan Angka Aktual
Setelah semua hasil ada, update angka di paper sections:

1. **Buka `results/comparative_results.json`** → ambil F1-Macro dan Accuracy model terbaik
2. **Update `paper/SECTION_LLM_AUGMENTATION.md`** jika perlu (bagian downstream performance sudah diarahkan ke section Hasil)
3. **Update `paper/SECTION_LIMITATION.md`** — ganti placeholder angka jika ada
4. **Verifikasi semua angka konsisten** antar section

### LANGKAH 7: Final Integration
- Integrate semua section ke `paper/Paper_JITK_NoFigures_Safe.docx`
- Masukkan figure baru dari `paper/figures/`
- Format check sesuai JITK (font Cambria, 2 kolom, IEEE references)

---

## DETAIL TEKNIS PENTING

### Environment
- **OS**: Windows 11
- **Python**: Anaconda (python, bukan python3)
- **Path Python**: Gunakan `D:/` (bukan `/d/` yang hanya untuk bash)
- **GPU**: Ada (cek dengan `python -c "import torch; print(torch.cuda.is_available())"`)

### Dataset Split (konsisten di semua script)
```python
# 80/10/10 stratified, seed=42
Train: 7,820 | Val: 977 | Test: 978
```

### Label Mapping
```
0 = Bukan Ujaran Kebencian (Neutral)
1 = Ujaran Kebencian - Ringan (Light Hate)
2 = Ujaran Kebencian - Sedang (Moderate Hate)
3 = Ujaran Kebencian - Berat (Severe Hate)
```

### Model Checkpoints Yang Ada (untuk Fase 1)
Script `honest_evaluation.py` sudah dikonfigurasi dengan semua path. Yang penting:
- `models/experiment_17_xlmr_large/checkpoint-5010` — XLM-R Large (kemungkinan terbaik)
- `models/experiment_14_indobert_phase5/checkpoint-502` — IndoBERT
- `models/experiment_6a_focal_loss/checkpoint-2505` — BERT + Focal Loss
- `models/improved_model` — Legacy (kemungkinan buruk)

### Jika Script Error
- **CUDA OOM**: Kurangi batch_size di script (edit `BATCH_SIZE` atau `config["batch_size"]`)
- **Model not found**: Cek path di `CHECKPOINTS` dict di `honest_evaluation.py`
- **Import error**: `pip install transformers datasets scikit-learn seaborn`

---

## REFERENSI YANG SUDAH DIVERIFIKASI

| # | Referensi | Status |
|---|-----------|--------|
| 1 | Ibrohim & Budi (2019) | REAL — venue dikoreksi ke ALW3 |
| 2 | Alfina et al. (2017) | REAL — tahun/penulis dikoreksi |
| 3 | Putri et al. (2021) | REAL — penulis/venue dikoreksi |
| 4 | Wilie et al. (2020) | REAL — judul dikoreksi ke "IndoNLU" |
| 5 | Cahyawijaya et al. (2023) | REAL — NusaCrowd, BARU |
| 6 | Müller et al. (2019) | REAL — NeurIPS |
| 7 | Szegedy et al. (2015) | REAL — inisial dikoreksi |
| 8 | Ding et al. (2024) | REAL — LLM augmentation, BARU |
| 9 | Ramos et al. (2024) | REAL — hate speech survey, BARU |
| 10 | Hedderich et al. (2021) | REAL — low-resource NLP, BARU |
| 11 | Dietterich (2000) | REAL — ensemble methods |

**DIHAPUS** (fabrikasi): Khoong 2021, Bryant 2020, Wibowo 2022, Aji & Adriani 2020

---

## CHECKLIST SETELAH SEMUA SELESAI

```
[ ] Setiap angka di paper = dari eksperimen nyata
[ ] F1 yang dilaporkan = test set (bukan validation)
[ ] Confusion matrix = dari model inference nyata
[ ] Referensi = semua terverifikasi (11 referensi)
[ ] Dataset stats = 9,775 sampel, 52.3% sintetis
[ ] Statistical significance = mean ± std dilaporkan
[ ] Tidak ada inkonsistensi antar section
[ ] generate_synthetic_predictions() sudah di-archive
[ ] Figure 1-4 di paper/figures/ = dari data nyata
```

---

## TOTAL ESTIMASI WAKTU

| Langkah | Durasi | Kebutuhan |
|---------|--------|-----------|
| 1. Honest eval | ~1 jam | GPU |
| 2. Train 3 model | ~3-4 jam | GPU |
| 3. Ablation | ~2-3 jam | GPU |
| 4. Multi-seed | ~2-3 jam | GPU |
| 5. Generate figures | ~5 menit | CPU |
| 6-7. Update & integrate | ~1-2 jam | Manual |
| **TOTAL** | **~10-13 jam** | |

Langkah 2 dan 3 bisa dijalankan parallel jika VRAM cukup.
