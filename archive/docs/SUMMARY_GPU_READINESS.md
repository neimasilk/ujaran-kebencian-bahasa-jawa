# GPU TRAINING CHECKLIST - Besok di Komputer Kuat

**Date:** 6 Januari 2026
**Status:** ✅ READY - Semua persiapan CPU selesai

---

## 🎯 Target Hari Ini di Komputer Kuat

**Target:** Tembus **82% F1-Macro** (gap hanya 0.62% dari 81.38%)

---

## ✅ Checklist Sebelum Mulai

- [ ] Pull latest dari GitHub: `git pull origin main`
- [ ] Cek GPU tersedia: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Install dependencies jika perlu: `pip install transformers torch scikit-learn`

---

## 📋 Langkah-Langkah Training (Urutan Penting!)

### STEP 1: Training dengan Phase 5 Dataset (1-2 jam)
```bash
cd D:\document\ujaran-kebencian-bahasa-jawa

# Jalankan training dengan Phase 5 dataset (DeepSeek re-labeled)
python experiments/experiment_6c_silent.py \
    --train-data data/improved/phase5_deepseek_relabeled.csv \
    --output models/experiment_13_phase5
```

**Expected Result:** 81.9-82.4% F1-Macro

### STEP 2: Threshold Optimization (30 menit)
```bash
# Jika Step 1 selesai, jalankan threshold optimization
python experiments/experiment_9_threshold_optimization.py \
    --model-path models/experiment_13_phase5/checkpoint-best
```

**Expected Result:** +0.3-0.8% additional

### STEP 3: Test-Time Augmentation (1 jam)
```bash
# Jika masih belum tembus 82%, jalankan TTA
python experiments/experiment_10_tta.py \
    --model-path models/experiment_13_phase5/checkpoint-best \
    --n-aug 5
```

**Expected Result:** +0.5-1% additional

---

## 📊 Dataset Siap Pakai

| File | Location | Samples | Quality |
|------|----------|---------|---------|
| **Phase 5** | `data/improved/phase5_deepseek_relabeled.csv` | 10,019 | 100/100 |

**Class Balance:**
- Neutral: 2,497 (24.92%)
- Light: 2,591 (25.86%)
- Moderate: 2,842 (28.37%)
- Severe: 2,089 (20.85%)
- **Ratio: 1.36:1 (EXCELLENT)**

---

## 🎯 Expected Outcome

| Scenario | F1-Macro | Action |
|----------|----------|--------|
| **Best Case** | 82.4% | ✅ Target achieved! |
| **Expected** | 81.9% | Run Threshold Opt → 82.2% |
| **Worst Case** | 81.5% | Run Threshold Opt + TTA → 82.7% |

---

## 📝 Setelah Training Selesai

1. **Push results ke GitHub:**
```bash
git add models/experiment_13_phase5/ results/
git commit -m "feat: add Phase 5 training results"
git push origin main
```

2. **Update ROADMAP_PENELITIAN_V2.md** dengan hasil baru

3. **Jika target 82% tercapai**, siapkan untuk workshop submission

---

## 🔧 Troubleshooting

### GPU Out of Memory
```bash
# Reduce batch size
python experiments/experiment_6c_silent.py --batch-size 8
```

### Training Terputus
```bash
# Resume from checkpoint
python experiments/experiment_6c_silent.py --resume models/experiment_13_phase5/checkpoint-last
```

### Model Tidak Ada
```bash
# Pull dari GitHub dulu
git pull origin main
```

---

## 📞 Quick Reference

Current Best Result: **81.38% F1-Macro** (Label Smoothing)

Dataset Sebelumnya:
- Phase 3: Original re-labeled (4,779 samples)
- Phase 4: Generated (5,240 samples)
- **Phase 5: DeepSeek re-labeled (10,019 samples) ← GUNAKAN INI**

---

**Good luck! Target 82% sangat achievable dengan Phase 5 dataset! 🚀**
