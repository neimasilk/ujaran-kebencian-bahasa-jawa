# Roadmap Penelitian: Deteksi Ujaran Kebencian Bahasa Jawa

**Update Terakhir:** 6 Januari 2026
**Status:** FASE 5 SELESAI - MENCAPAI 79.19% F1-MACRO

---

## Ringkasan Hasil Saat Ini

| Model | Dataset | F1-Macro | Status |
|-------|---------|----------|--------|
| IndoBERT Base | improved 10K | **79.19%** | BEST |
| mBERT | improved 10K | 77.93% | Complete |
| XLM-RoBERTa | improved 10K | 78.38% | Complete |
| Ensemble (mBERT+XLMR) | improved 10K | 78.90% | Complete |

**Catatan:** Ensemble memberikan improvement kecil dari single best model.

---

## Target Akurasi Realistik

Berdasarkan state-of-the-art dan baseline saat ini:

| Target | F1-Macro | Keterangan |
|--------|----------|------------|
| **Baseline Saat Ini** | 79.19% | IndoBERT Base + improved dataset |
| **Target Konservatif** | 82-85% | Dataset improvement + ensemble enhancement |
| **Target Moderate** | 85-88% | Semua 3 dimensi (dataset + model + architecture) |
| **Target Optimistik** | 88-90% | Full optimization dengan advanced techniques |

---

## Dimensi 1: Dataset Improvements

### 1.1 Hard Example Mining

**Konsep:** Identifikasi dan fokus pada samples yang sulit diklasifikasikan

**Implementasi:**
```
1. Train model awal
2. Identifikasi samples dengan:
   - Prediction confidence < 0.6
   - False predictions pada validation set
3. Re-label dan/or augment samples tersebut
4. Re-train dengan weighted sampling
```

**Expected Improvement:** +1-2% F1-Macro

**Prioritas:** HIGH - Cost-effective dan cepat implementasinya

---

### 1.2 Active Learning dengan AI Assistant

**Konsep:** Gunakan AI untuk mengidentifikasi samples yang paling informatif

**Implementasi:**
```
1. Gunakan model untuk predict semua unlabeled data
2. Pilih top-K samples dengan:
   - High uncertainty (entropy tinggi)
   - Near decision boundary
3. Human-in-the-loop untuk verifikasi label
4. Iterative training
```

**Expected Improvement:** +2-3% F1-Macro

**Prioritas:** MEDIUM - Memerlukan resources tambahan

---

### 1.3 Contextual Data Augmentation

**Konsep:** Generate variations dari samples existing dengan konteks Jawa/Indonesia

**Implementasi:**
```
1. Back-translation: ID -> EN -> ID (with variations)
2. Synonym replacement dengan kata-kata Jawa
3. Template-based generation dengan nama/nama tempat Jawa
4. Code-mixing variations (Jawa-Indonesia, Jawa-Inggris)
```

**Expected Improvement:** +1-2% F1-Macro

**Prioritas:** HIGH - Relatif murah dan efektif

---

### 1.4 Cross-Domain Transfer Learning

**Konsep:** Leverage data dari domain bahasa Indonesia/Melayu lain

**Sumber Data:**
- Indonesian hate speech dataset (Twitter, Facebook)
- Malay language hate speech dataset
- Code-mixing datasets (ID-EN, MY-EN)

**Implementasi:**
```
1. Collect datasets dari domain lain
2. Adapt label schema
3. Domain adaptation training
4. Fine-tune pada Javanese dataset
```

**Expected Improvement:** +2-4% F1-Macro

**Prioritas:** MEDIUM - Memerlukan data collection

---

### 1.5 Temporal/Dialectal Coverage

**Konsep:** Cover variasi dialek Bahasa Jawa (Jawa Timur, Jawa Tengah, Jawa Barat)

**Implementasi:**
```
1. Collect samples dari berbagai dialek
2. Label dialect-specific patterns
3. Stratified sampling per dialek
4. Dialect-aware fine-tuning
```

**Expected Improvement:** +1-2% F1-Macro

**Prioritas:** LOW - Tambahan untuk robustness

---

## Dimensi 2: Custom BERT Pretraining

### 2.1 Extended Domain-Adaptive Pre-Training (DAPT)

**Konsep:** Lanjutkan pre-training Custom BERT dengan corpus Jawa yang lebih besar

**Corpus Sources:**
- Wikipedia Jawa (full articles)
- Javanese text corpora from CC-100
- Local news sites (Jawapos, Kompas Jawa, etc.)
- Social media scraped data

**Implementasi:**
```
Current: ~5M tokens
Target: 20-50M tokens (4-10x)
Method: Masked Language Modeling (MLM)
Duration: 2-3 hari training
```

**Expected Improvement:** +2-3% F1-Macro

**Prioritas:** HIGH - Signifikan improvement

---

### 2.2 Contrastive Learning for Hate Speech

**Konsep:** Pre-training dengan contrastive objective untuk memisahkan kelas

**Implementasi:**
```
1. Create positive pairs: same label, semantically similar
2. Create negative pairs: different label or semantically different
3. SimCSE or ESM-style contrastive learning
4. Joint training: MLM + Contrastive
```

**Expected Improvement:** +1-2% F1-Macro

**Prioritas:** MEDIUM - Novel approach

---

### 2.3 Adapter-Based Fine-Tuning

**Konsep:** Gunakan adapter layers untuk efficient transfer learning

**Implementasi:**
```
1. Pre-train base model once
2. Train task-specific adapters
3. Composable adapters untuk different scenarios
4. Parameter-efficient tuning (PET)
```

**Expected Improvement:** +0.5-1% F1-Macro

**Prioritas:** LOW - Efisiensi-focused

---

### 2.4 Multi-Task Pre-Training

**Konsep:** Joint training dengan related tasks

**Related Tasks:**
- Sentiment analysis
- Offensive language detection
- Cyberbullying detection
- Emotion detection

**Implementasi:**
```
1. Collect datasets untuk related tasks
2. Multi-task learning architecture
3. Task-specific heads dengan shared encoder
4. Joint optimization
```

**Expected Improvement:** +1-3% F1-Macro

**Prioritas:** MEDIUM - Data-intensive

---

## Dimensi 3: Model Architecture & Methods

### 3.1 Advanced Ensemble Methods

**Konsep:** Leverage multiple models dengan teknik yang lebih sophisticated

**Methods:**
```
A. Weighted Ensemble
   - Weight by validation performance
   - Dynamic weighting per sample difficulty

B. Stacking dengan lebih complex meta-learner
   - XGBoost/LightGBM sebagai meta-learner
   - Neural network meta-learner

C. Cascading Ensemble
   - First stage: fast model untuk easy cases
   - Second stage: complex model untuk hard cases

D. Bayesian Model Averaging
   - Probabilistic model combination
```

**Expected Improvement:** +1-2% F1-Macro

**Prioritas:** HIGH - Relatif mudah implementasinya

---

### 3.2 Loss Function Engineering

**Konsep:** Gunakan loss functions yang address class imbalance dan label noise

**Options:**
```
A. Focal Loss
   - Focus pada hard examples
   - γ parameter untuk modulating factor

B. Label Smoothing
   - Handle label noise
   - Prevent overconfidence

C. Class-Balanced Loss
   - Effective number of samples weighting
   - Better handling imbalance

D. Ordered Loss (for ordinal classification)
   - Recognition: 0<1<2<3 adalah ordinal
   - Cost-sensitive learning
```

**Expected Improvement:** +0.5-1.5% F1-Macro

**Prioritas:** HIGH - Low-hanging fruit

---

### 3.3 Hierarchical Classification

**Konsep:** Treat sebagai binary cascade: Hate vs Non-Hate → Severity Level

**Architecture:**
```
Level 1: Binary (Hate vs Non-Hate)
Level 2: 3-class (Light vs Moderate vs Severe)

Benefits:
- Simpler problems at each level
- Better handle class imbalance
- Interpretable intermediate outputs
```

**Expected Improvement:** +1-2% F1-Macro

**Prioritas:** MEDIUM - Architectural change

---

### 3.4 Attention Mechanism Enhancements

**Konsep:** Enhance model dengan better attention mechanisms

**Methods:**
```
A. Multi-Head AttentionPooling
   - Better sentence representation

B. Cross-Attention between encoder and task embedding
   - Task-aware representation

C. Sparse Attention untuk longer sequences
   - Handle longer texts (>128 tokens)
```

**Expected Improvement:** +0.5-1% F1-Macro

**Prioritas:** LOW - Tambahan improvement

---

### 3.5 Knowledge Distillation

**Konsep:** Train student model dengan knowledge dari teacher ensemble

**Implementasi:**
```
1. Teacher: Ensemble dari best models
2. Student: Smaller architecture (DistilBERT, TinyBERT)
3. Distillation loss: KL divergence pada logits
4. Fine-tune student pada original task
```

**Expected Improvement:** 0% (accuracy trade-off for efficiency)

**Prioritas:** LOW - Untuk deployment

---

## Roadmap Implementasi

### Phase 6: Quick Wins (1-2 minggu)

**Target:** 82-83% F1-Macro

| Task | Method | Expected | Priority |
|------|--------|----------|----------|
| Loss function | Focal + Label Smoothing | +0.5-1% | HIGH |
| Ensemble | Weighted stacking dengan XGBoost | +1% | HIGH |
| Data augmentation | Contextual Javanese augmentation | +1% | HIGH |
| Hyperparameter | Learning rate tuning | +0.5% | MEDIUM |

**Total Expected:** +2.5-3% → **82-83% F1-Macro**

---

### Phase 7: Dataset Expansion (2-3 minggu)

**Target:** 84-86% F1-Macro

| Task | Method | Expected | Priority |
|------|--------|----------|----------|
| Hard example mining | Re-label difficult samples | +1-2% | HIGH |
| Cross-domain transfer | Indonesian hate speech data | +1-2% | HIGH |
| Active learning | Iterative labeling | +1% | MEDIUM |
| Dialectal coverage | Multiple Javanese dialects | +0.5% | LOW |

**Total Expected:** +2-4% → **84-86% F1-Macro**

---

### Phase 8: Model Enhancement (3-4 minggu)

**Target:** 86-88% F1-Macro

| Task | Method | Expected | Priority |
|------|--------|----------|----------|
| Extended DAPT | 20-50M tokens Javanese corpus | +2-3% | HIGH |
| Multi-task learning | Related tasks joint training | +1-2% | MEDIUM |
| Hierarchical classification | Binary → 3-class cascade | +1% | MEDIUM |
| Contrastive learning | SimCSE-style pre-training | +0.5-1% | LOW |

**Total Expected:** +3-5% → **86-88% F1-Macro**

---

## Resource Requirements

### Computational Resources

| Phase | GPU Hours | Cost Estimate (cloud) |
|-------|-----------|----------------------|
| Phase 6 | 50-100 | $50-100 |
| Phase 7 | 100-200 | $100-200 |
| Phase 8 | 500-1000 | $500-1000 |

**Local Setup:** RTX 4080 (16GB) cukup untuk semua phases

### Data Requirements

| Resource | Current | Target |
|----------|---------|--------|
| Labeled data | 10,019 | 15,000-20,000 |
| Pre-training corpus | ~5M tokens | 20-50M tokens |
| Unlabeled pool | 0 | 50,000+ |

---

## Prioritas Implementasi

### HIGH Priority (Quick Wins)

1. **Focal Loss + Label Smoothing** - Mudah, free improvement
2. **Weighted Ensemble dengan XGBoost** - Proven effective
3. **Contextual Data Augmentation** - Cost-effective
4. **Hard Example Mining** - Targeted improvement

### MEDIUM Priority (Significant Improvement)

1. **Extended DAPT** - Major improvement potential
2. **Cross-domain Transfer** - Leverage existing resources
3. **Multi-task Learning** - Comprehensive improvement

### LOW Priority (Optional)

1. **Dialectal Coverage** - Niche improvement
2. **Adapter Learning** - Efficiency-focused
3. **Knowledge Distillation** - Deployment-focused

---

## Risks dan Mitigasi

### Risk 1: Overfitting pada Test Set

**Mitigasi:**
- Strict train/val/test split
- Cross-validation untuk hyperparameter tuning
- Hold-out test set hanya untuk final evaluation

### Risk 2: Data Quality Degradation

**Mitigasi:**
- Quality control untuk augmented data
- Human verification untuk active learning
- Consistency checks untuk label

### Risk 3: Computational Cost

**Mitigasi:**
- Prioritaskan methods dengan best ROI
- Gunakan local GPU (RTX 4080)
- Incremental implementation

---

## Success Criteria

### Minimum Viable Product (MVP)
- F1-Macro >= 82%
- Per-class F1 >= 75%
- Inference time < 100ms per sample

### Production Ready
- F1-Macro >= 85%
- Per-class F1 >= 80%
- Robust terhadap code-mixing
- Handle out-of-domain samples

### State-of-the-Art Competitive
- F1-Macro >= 88%
- Per-class F1 >= 85%
- Published-ready results

---

## Timeline Estimasi

```
Month 1: Phase 6 (Quick Wins)
Week 1-2: Implementation dan evaluation
Week 3-4: Iteration dan refinement

Month 2: Phase 7 (Dataset Expansion)
Week 5-8: Data collection, augmentation, retraining

Month 3: Phase 8 (Model Enhancement)
Week 9-12: Extended DAPT, multi-task learning

Month 4: Integration dan Documentation
Week 13-16: Final ensemble, documentation, paper prep
```

---

## Catatan Penelitian

### Lessons Learned

1. **Quality > Quantity:** Dataset improved 10K mengungguli 39K lama
2. **IndoBERT Domain Knowledge:** Pre-training pada Indonesian sangat membantu
3. **Ensemble Limitations:** Stacking sederhana tidak memberikan improvement signifikan
4. **Context Matters:** Indonesian/Javanese context sangat penting

### Future Work

1. Multimodal hate speech detection (text + image)
2. Real-time detection system
3. Explainable AI untuk interpretability
4. Cross-lingual transfer untuk bahasa daerah lain

---

**Status:** READY FOR PHASE 6 IMPLEMENTATION
**Last Update:** 6 Januari 2026
