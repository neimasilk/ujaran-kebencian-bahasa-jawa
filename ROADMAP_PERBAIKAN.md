# ROADMAP PERBAIKAN PROYEK
## Deteksi Ujaran Kebencian Bahasa Jawa

**Tanggal**: 5 Januari 2026

---

## OPSI JALAN KE DEPAN

### Opsi A: Revisi Paper dengan Hasil Reproducible (DIREKOMENDASIKAN)

**Waktu Estimasi**: Langsung bisa dilakukan

**Langkah**:
1. Update paper dengan angka reproducible:
   - F1-Macro: 60.77% (Ensemble) atau 62.55% (Custom BERT v2)
   - Tetap dokumentasikan metodologi sebagai kontribusi

2. Ubah klaim utama:
   - Dari "achieve 94.09% macro-F1"
   - Ke "achieve 60-65% macro-F1 with reproducible pipeline"

3. Tambahkan sebagai limitasi:
   - Gap antara hasil awal dan reproducible
   - Tantangan class imbalance pada bahasa low-resource

**Pro**: Jujur, cepat, masih bisa dipublikasikan
**Con**: Angka lebih rendah dari ekspektasi awal

---

### Opsi B: Investigasi dan Perbaikan Menyeluruh

**Waktu Estimasi**: 2-4 minggu

**Langkah**:

#### Fase 1: Investigasi `improved_model` (3-5 hari)
- [ ] Cari training logs di `logs/` folder
- [ ] Verifikasi dataset yang digunakan
- [ ] Check apakah ada data leak (test data di training)
- [ ] Reproduksi training jika dokumentasi ditemukan

#### Fase 2: Optimasi Model Baru (1-2 minggu)
- [ ] Hyperparameter tuning sistematis dengan Optuna
- [ ] Perbaiki XLM-RoBERTa training (learning rate tuning)
- [ ] Coba arsitektur lain (ELECTRA, DeBERTa multilingual)
- [ ] Implementasi teknik anti-overfitting yang lebih agresif

#### Fase 3: Ensemble Optimization (3-5 hari)
- [ ] Weighted ensemble dengan bobot optimal
- [ ] Stacking dengan berbagai meta-learner
- [ ] Cross-validation untuk semua komponen

**Target Realistis**: 70-75% F1-Macro

**Pro**: Hasil lebih tinggi jika berhasil
**Con**: Memakan waktu, tidak ada jaminan mencapai 86%+

---

## REKOMENDASI PENATAAN FOLDER

### Langkah Cleanup Segera

```bash
# 1. Buat folder arsip untuk dokumentasi lama yang menyesatkan
mkdir -p archive/deprecated_docs
mv REKAP_SEMUA_EKSPERIMEN.md archive/deprecated_docs/
mv REVIEW_EKSPERIMEN_FINAL.md archive/deprecated_docs/
mv DOKUMENTASI_PENCAPAIAN_TARGET.md archive/deprecated_docs/

# 2. Pindahkan hasil historis
mkdir -p results/historical
mv results/ensemble_advanced_results.json results/historical/
mv results/improved_model_evaluation.json results/historical/

# 3. Buat folder untuk hasil reproducible
mkdir -p results/reproducible
mv results/integrated_custom_ensemble_results.json results/reproducible/
mv results/super_ensemble_results.json results/reproducible/

# 4. Konsolidasi model
mkdir -p models/production
cp -r models/custom_javanese_bert_v2 models/production/

mkdir -p models/experimental
mv models/xlm_roberta_fix_* models/experimental/
```

---

## PETA EKSPERIMEN YANG VALID

### Eksperimen yang Hasilnya DAPAT DIPERCAYA

| ID | Eksperimen | Script | Hasil | Status |
|----|------------|--------|-------|--------|
| E1 | Custom BERT v2 | `train_custom_bert_v2.py` | 62.55% | VALID |
| E2 | Integrated Ensemble | `experiment_integrated_custom_ensemble.py` | 60.77% | VALID |
| E3 | Super Ensemble v2 | `super_meta_ensemble_v2.py` | 61.26% | VALID |

### Eksperimen yang PERLU VERIFIKASI ULANG

| ID | Eksperimen | Script | Hasil Klaim | Masalah |
|----|------------|--------|-------------|---------|
| X1 | Improved Model | ? | 86.88% | Tidak ada log training |
| X2 | Self-Ensemble 94% | `improved_meta_ensemble_90_percent.py` | 94.09% val | Overfitting, self-ensemble |
| X3 | Ensemble Advanced | `ensemble_advanced.py` | 86.86% test | Bergantung pada X1 |

---

## PANDUAN EKSPERIMEN LANJUTAN

### Jika Ingin Meningkatkan Performa ke 70%+

1. **Data Augmentation yang Lebih Baik**
   ```python
   # Teknik yang bisa dicoba:
   - Back-translation (Jawa -> Indo -> Jawa)
   - Synonym replacement dengan kamus Jawa
   - Paraphrasing dengan LLM (DeepSeek/GPT)
   ```

2. **Model Architecture**
   ```python
   # Model yang belum dicoba:
   - microsoft/deberta-v3-base (multilingual fine-tune)
   - xlm-roberta-large (lebih besar dari base)
   - Ensemble dengan voting weights yang di-optimize
   ```

3. **Training Techniques**
   ```python
   # Teknik anti-overfitting:
   - Label smoothing (0.1-0.2)
   - Mixup augmentation
   - Adversarial training
   - Early stopping yang lebih ketat
   ```

4. **Ensemble Strategy**
   ```python
   # Meta-learner yang lebih robust:
   - Cross-validated stacking
   - Blending dengan holdout set
   - Neural network meta-learner
   ```

---

## CHECKLIST SEBELUM PUBLIKASI

### Wajib Dilakukan:

- [ ] Semua angka di paper dapat direproduksi dengan script yang ada
- [ ] Training logs tersedia untuk semua model yang diklaim
- [ ] Dataset di-freeze dan di-version
- [ ] Script memiliki random seed yang fixed
- [ ] Validation-test split konsisten dan terdokumentasi

### Opsional tapi Direkomendasikan:

- [ ] Cross-validation results (5-fold)
- [ ] Confidence intervals untuk semua metrics
- [ ] Ablation study yang jelas
- [ ] Statistical significance tests

---

## KONTAK DAN ESKALASI

Jika menemukan masalah selama investigasi:
1. Cek git history untuk training scripts
2. Cari backup di cloud storage jika ada
3. Pertimbangkan untuk re-train dari awal jika dokumentasi tidak ditemukan

---

**Status**: AKTIF
**Update Terakhir**: 5 Januari 2026
