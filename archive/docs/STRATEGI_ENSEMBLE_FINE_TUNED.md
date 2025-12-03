# Strategi Multi-Architecture Ensemble dengan Fine-Tuning

## 📋 Overview
Dokumentasi lengkap untuk implementasi ensemble method yang benar: **fine-tune setiap model transformer dengan dataset Javanese hate speech**, kemudian melakukan ensemble voting untuk mencapai accuracy >85%.

## 🎯 Konsep Dasar

### ❌ Pendekatan Salah (Sebelumnya)
```
Pre-trained Model → Langsung Ensemble → Hasil Buruk
```

### ✅ Pendekatan Benar (Sekarang)
```
Pre-trained Model → Fine-tune dengan Dataset Jawa → Ensemble → Target >85%
```

## 🏗️ Arsitektur Ensemble

### Model yang Akan Digunakan
1. **IndoBERT-base-p1** (`indobenchmark/indobert-base-p1`)
   - Pre-trained pada bahasa Indonesia
   - Transfer learning ke Javanese
   - Target individual: ~60% accuracy

2. **IndoBERT-base-uncased** (`indolem/indobert-base-uncased`)
   - Variasi uncased untuk handling case sensitivity
   - Target individual: ~45% accuracy

3. **RoBERTa-Indo** (`cahya/roberta-base-indonesian-522M`)
   - Arsitektur RoBERTa untuk bahasa Indonesia
   - Target individual: ~55% accuracy

### Mengapa Fine-Tuning Diperlukan?
1. **Vocabulary Gap**: Kata-kata Javanese tidak ada di pre-trained vocab
2. **Cultural Context**: Pattern hate speech berbeda antar budaya
3. **Language Nuances**: Struktur kalimat dan makna kontekstual
4. **Domain Adaptation**: Dari general Indonesian ke specific Javanese hate speech

## 🔄 Pipeline Eksperimen

### Phase 1: Individual Fine-Tuning
```python
for each model in [IndoBERT, IndoBERT-uncased, RoBERTa-Indo]:
    1. Load pre-trained model
    2. Add classification head (4 classes)
    3. Fine-tune dengan dataset Javanese (3 epochs)
    4. Evaluate pada validation set
    5. Save best checkpoint
```

### Phase 2: Ensemble Optimization
```python
1. Load semua fine-tuned models
2. Get predictions pada validation set
3. Optimize ensemble weights
4. Test ensemble pada test set
```

### Phase 3: Final Evaluation
```python
1. Evaluate individual models
2. Evaluate ensemble (equal weights)
3. Evaluate ensemble (optimized weights)
4. Generate classification report
```

## ⚙️ Konfigurasi Training

### Training Arguments
```python
TrainingArguments(
    output_dir='./models/ensemble_{model_name}',
    num_train_epochs=3,
    per_device_train_batch_size=16,  # Adjust based on GPU memory
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    seed=42,
    fp16=True,  # Untuk efisiensi GPU
    dataloader_num_workers=2
)
```

### Dataset Configuration
```python
# Gunakan 50% dataset untuk eksperimen cepat
X_train_subset, _, y_train_subset, _ = train_test_split(
    X_train, y_train, test_size=0.5, random_state=42, stratify=y_train
)

# Split: Train 50% (9735), Val 20% (3245), Test 20% (3246)
```

## 📊 Target Performance

### Individual Models (Post Fine-Tuning)
- **IndoBERT**: 55-65% accuracy, 55-65% F1-macro
- **IndoBERT-uncased**: 40-50% accuracy, 40-50% F1-macro
- **RoBERTa-Indo**: 50-60% accuracy, 50-60% F1-macro

### Ensemble Performance
- **Equal Weights**: 60-70% accuracy
- **Optimized Weights**: 65-75% accuracy
- **Target**: >85% accuracy (dengan optimasi lanjutan)

## 🔧 Implementation Plan

### Step 1: Perbaiki Script Existing
```bash
# File: multi_architecture_ensemble.py sudah benar
# Pastikan fine-tuning berjalan untuk setiap model
# Pastikan checkpoint tersimpan dengan benar
```

### Step 2: Jalankan Eksperimen
```bash
python multi_architecture_ensemble.py
```

### Step 3: Monitor Progress
```bash
# Cek log training
tail -f logs/multi_architecture_ensemble.log

# Cek checkpoint tersimpan
ls -la models/ensemble_*/
```

### Step 4: Evaluasi Hasil
```bash
# Cek hasil akhir
cat results/multi_architecture_ensemble_results.json
```

## 📈 Expected Timeline

### Training Time Estimation
- **IndoBERT fine-tuning**: ~6 menit (3 epochs)
- **IndoBERT-uncased fine-tuning**: ~6 menit
- **RoBERTa-Indo fine-tuning**: ~9 menit (model lebih besar)
- **Ensemble optimization**: ~2 menit
- **Total**: ~25-30 menit

### Checkpoint Schedule
- **Save every 200 steps** (~2-3 menit)
- **Early stopping** jika tidak ada improvement
- **Best model saved** berdasarkan F1-macro

## 🚀 Optimasi Lanjutan

### Jika Hasil <85%
1. **Increase Training Epochs**: 3 → 5 epochs
2. **Hyperparameter Tuning**: Learning rate, batch size
3. **Data Augmentation**: Lebih banyak synthetic data
4. **Advanced Ensemble**: Stacking, meta-learner
5. **Model Diversity**: Tambah XLM-RoBERTa, mBERT

### Jika GPU Memory Issues
1. **Reduce Batch Size**: 16 → 8
2. **Gradient Accumulation**: steps=2
3. **Sequential Training**: Train satu model per waktu

## 📝 Monitoring Checklist

### During Training
- [ ] Loss menurun secara konsisten
- [ ] Validation F1-score meningkat
- [ ] No overfitting (train vs val gap)
- [ ] Checkpoint tersimpan di `./models/`

### After Training
- [ ] Semua 3 model ter-fine-tune
- [ ] Individual accuracy >40%
- [ ] Ensemble accuracy > individual terbaik
- [ ] Results tersimpan di `./results/`

## 🎯 Success Criteria

### Minimum Viable
- ✅ Semua model berhasil di-fine-tune
- ✅ Ensemble accuracy > individual terbaik
- ✅ F1-macro score > 60%

### Target Optimal
- 🎯 Ensemble accuracy > 75%
- 🎯 F1-macro score > 70%
- 🎯 Balanced performance across all classes

### Stretch Goal
- 🚀 Ensemble accuracy > 85%
- 🚀 F1-macro score > 80%
- 🚀 Production-ready model

## 📚 References

### Papers
- "BERT: Pre-training of Deep Bidirectional Transformers"
- "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- "IndoBERT: A Pre-trained Language Model for Indonesian"

### Code References
- HuggingFace Transformers Documentation
- Ensemble Methods in NLP
- Transfer Learning Best Practices

---

**Next Action**: Jalankan `python multi_architecture_ensemble.py` dengan monitoring ketat untuk memastikan fine-tuning berjalan dengan benar.