# Rencana Eksperimen Selanjutnya - Peningkatan Akurasi Model

**Researcher:** AI Assistant  
**Date:** 3 Juli 2025  
**Project Phase:** Advanced Model Optimization  
**Current Baseline:** F1-Score Macro 80.36%, Accuracy 80.37%  
**Target:** F1-Score Macro >85%, Accuracy >85%  

---

## 📊 Status Proyek Saat Ini

### Pencapaian Terkini
- ✅ **Model Baseline**: IndoBERT dengan F1-Score Macro 80.36%
- ✅ **Threshold Optimization**: Peningkatan dari 73.7% ke 80.36%
- ✅ **Class Imbalance Handling**: Focal Loss + Class Weighting
- ✅ **Production Ready**: Deployment pipeline tersedia
- ✅ **Comprehensive Evaluation**: Balanced evaluation methodology

### Analisis Kelemahan Model Saat Ini
1. **Ujaran Kebencian - Ringan**: F1-Score 78.52% (terendah)
2. **Ujaran Kebencian - Sedang**: Precision 69.55% (perlu perbaikan)
3. **False Negative Rate**: 22.7% ujaran kebencian tidak terdeteksi
4. **Inference Speed**: ~50ms (target <30ms)

---

## 🎯 Eksperimen yang Direncanakan

### Eksperimen 1: Advanced Model Architecture
**Tujuan**: Meningkatkan kapasitas model dengan arsitektur yang lebih powerful

#### 1.1 IndoBERT Large Experiment ✅ **IMPLEMENTED**
```yaml
Model: indobenchmark/indobert-large-p1
Parameters: 340M (vs 110M baseline)
Expected Improvement: +3-5% F1-Score
Computational Cost: High
Priority: High
Status: READY FOR EXECUTION
Implementation: /experiments/experiment_1_indobert_large.py

Configuration:
  learning_rate: 1e-5  # Lower for larger model
  batch_size: 8        # Reduced due to memory
  gradient_accumulation: 2
  max_length: 256      # Increased context
  epochs: 5            # More epochs for convergence
  
Implemented Features:
  ✅ WeightedFocalLoss with class weights
  ✅ Custom trainer with advanced loss function
  ✅ Comprehensive evaluation metrics
  ✅ Early stopping with patience
  ✅ Mixed precision training (FP16)
  ✅ Detailed logging and result saving
  ✅ Confusion matrix visualization
```

#### 1.2 XLM-RoBERTa Cross-lingual Experiment
```yaml
Model: xlm-roberta-base
Rationale: Better multilingual representation
Expected Improvement: +2-4% F1-Score
Priority: High

Configuration:
  learning_rate: 2e-5
  batch_size: 16
  max_length: 256
  warmup_ratio: 0.1
```

### Eksperimen 2: Advanced Training Techniques
**Tujuan**: Optimasi training process untuk performa maksimal

#### 2.1 Multi-Stage Fine-tuning
```python
# Stage 1: General Indonesian language adaptation
stage_1_config = {
    "model": "xlm-roberta-base",
    "data": "indonesian_general_corpus",
    "epochs": 2,
    "learning_rate": 3e-5
}

# Stage 2: Javanese language adaptation
stage_2_config = {
    "model": "stage_1_checkpoint",
    "data": "javanese_unlabeled_corpus",
    "epochs": 3,
    "learning_rate": 2e-5
}

# Stage 3: Hate speech classification
stage_3_config = {
    "model": "stage_2_checkpoint",
    "data": "labeled_hate_speech_dataset",
    "epochs": 5,
    "learning_rate": 1e-5
}
```

#### 2.2 Advanced Loss Functions
```python
# Experiment 2.2.1: Label Smoothing + Focal Loss
class LabelSmoothingFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, smoothing=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        # Combine label smoothing with focal loss
        pass

# Experiment 2.2.2: Class-Balanced Focal Loss
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, beta=0.9999, gamma=2.0):
        # Effective number of samples approach
        pass
```

#### 2.3 Data Augmentation Strategies
```python
# Augmentation techniques for minority classes
augmentation_strategies = {
    "back_translation": {
        "source_lang": "jv",  # Javanese
        "intermediate_langs": ["id", "en", "ms"],
        "target_multiplier": 2
    },
    "paraphrasing": {
        "model": "t5-base-indonesian",
        "num_paraphrases": 3
    },
    "synonym_replacement": {
        "javanese_wordnet": True,
        "replacement_rate": 0.15
    },
    "contextual_word_embedding": {
        "model": "word2vec-javanese",
        "similarity_threshold": 0.8
    }
}
```

### Eksperimen 3: Ensemble Methods
**Tujuan**: Kombinasi multiple models untuk performa optimal

#### 3.1 Heterogeneous Ensemble
```python
ensemble_config = {
    "models": [
        {
            "name": "indobert_large",
            "weight": 0.4,
            "specialty": "general_javanese"
        },
        {
            "name": "xlm_roberta",
            "weight": 0.3,
            "specialty": "cross_lingual"
        },
        {
            "name": "indobert_base_optimized",
            "weight": 0.3,
            "specialty": "hate_speech_specific"
        }
    ],
    "combination_method": "weighted_voting",
    "calibration": "platt_scaling"
}
```

#### 3.2 Stacking Ensemble
```python
# Meta-learner approach
meta_learner_config = {
    "base_models": ["indobert", "xlm_roberta", "mbert"],
    "meta_model": "lightgbm",
    "features": [
        "prediction_probabilities",
        "confidence_scores",
        "text_length",
        "dialect_indicators"
    ]
}
```

### Eksperimen 4: Specialized Architectures
**Tujuan**: Custom architectures untuk hate speech detection

#### 4.1 Hierarchical Classification
```python
# Two-stage classification
stage_1 = {
    "task": "binary_classification",
    "classes": ["hate_speech", "non_hate_speech"],
    "model": "indobert_base",
    "threshold": 0.3  # Low threshold for high recall
}

stage_2 = {
    "task": "severity_classification",
    "classes": ["ringan", "sedang", "berat"],
    "model": "indobert_large",
    "input": "stage_1_positive_predictions"
}
```

#### 4.2 Multi-Head Attention Enhancement
```python
class EnhancedBERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes=4):
        super().__init__()
        self.bert = bert_model
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(768, 8) for _ in range(3)
        ])
        self.classifier = nn.Linear(768 * 3, num_classes)
    
    def forward(self, input_ids, attention_mask):
        # Enhanced attention mechanism
        pass
```

---

## 📋 Eksperimen Schedule

### Week 1-2: Model Architecture Experiments
- [x] **Day 1-3**: IndoBERT Large training dan evaluasi ✅ **READY FOR EXECUTION**
- [ ] **Day 4-6**: XLM-RoBERTa training dan evaluasi
- [ ] **Day 7**: Perbandingan hasil dan analisis

### Week 3-4: Advanced Training Techniques
- [ ] **Day 8-10**: Multi-stage fine-tuning implementation
- [ ] **Day 11-12**: Advanced loss functions testing
- [ ] **Day 13-14**: Data augmentation experiments

### Week 5-6: Ensemble Methods
- [ ] **Day 15-17**: Heterogeneous ensemble development
- [ ] **Day 18-20**: Stacking ensemble implementation
- [ ] **Day 21**: Ensemble optimization dan tuning

### Week 7-8: Specialized Architectures
- [ ] **Day 22-24**: Hierarchical classification
- [ ] **Day 25-27**: Multi-head attention enhancement
- [ ] **Day 28**: Final evaluation dan comparison

---

## 🔬 Evaluation Protocol

### Metrics untuk Setiap Eksperimen
```python
evaluation_metrics = {
    "primary_metrics": [
        "f1_score_macro",
        "accuracy",
        "precision_macro",
        "recall_macro"
    ],
    "secondary_metrics": [
        "f1_score_per_class",
        "confusion_matrix",
        "roc_auc_per_class",
        "precision_recall_curves"
    ],
    "efficiency_metrics": [
        "inference_time",
        "model_size",
        "training_time",
        "memory_usage"
    ]
}
```

### Statistical Significance Testing
```python
significance_tests = {
    "mcnemar_test": {
        "purpose": "Compare paired predictions",
        "alpha": 0.05
    },
    "bootstrap_confidence_intervals": {
        "purpose": "Estimate metric uncertainty",
        "n_bootstrap": 1000,
        "confidence_level": 0.95
    },
    "cross_validation": {
        "method": "stratified_k_fold",
        "k": 5,
        "repeats": 3
    }
}
```

---

## 💾 Documentation Requirements

### Untuk Setiap Eksperimen
1. **Experiment Log**: Detailed configuration dan hyperparameters
2. **Results Report**: Comprehensive metrics dan analysis
3. **Error Analysis**: Failure cases dan improvement suggestions
4. **Computational Cost**: Training time, memory usage, inference speed
5. **Reproducibility**: Seeds, environment, dependencies

### Academic Documentation
1. **Methodology Section**: Detailed experimental setup
2. **Results Section**: Statistical analysis dan comparisons
3. **Discussion**: Insights dan implications
4. **Future Work**: Next steps berdasarkan findings

---

## 🎯 Success Criteria

### Primary Goals
- **F1-Score Macro**: >85% (current: 80.36%)
- **Accuracy**: >85% (current: 80.37%)
- **Balanced Performance**: All classes F1-Score >80%

### Secondary Goals
- **Inference Speed**: <30ms per sample
- **Model Size**: <500MB
- **Robustness**: Consistent performance across dialects
- **Explainability**: Interpretable predictions

### Stretch Goals
- **F1-Score Macro**: >90%
- **Real-time Performance**: <10ms inference
- **Multi-dialect Support**: Separate models per dialect
- **Active Learning**: Continuous improvement pipeline

---

## 📊 Expected Outcomes

### Conservative Estimates
- **IndoBERT Large**: +3% F1-Score → 83.36%
- **Advanced Training**: +2% F1-Score → 85.36%
- **Ensemble Methods**: +1% F1-Score → 86.36%

### Optimistic Estimates
- **Combined Improvements**: +8-10% F1-Score → 88-90%
- **Production Deployment**: Ready for real-world application
- **Academic Publication**: High-impact conference submission

---

## 🔄 Iterative Improvement Process

1. **Experiment Execution**: Run planned experiments systematically
2. **Results Analysis**: Comprehensive evaluation dan comparison
3. **Insight Generation**: Identify patterns dan improvement opportunities
4. **Hypothesis Formation**: Develop new experimental hypotheses
5. **Next Iteration**: Plan follow-up experiments based on findings

---

**Next Action**: Mulai dengan Eksperimen 1.1 (IndoBERT Large) untuk baseline improvement yang signifikan.