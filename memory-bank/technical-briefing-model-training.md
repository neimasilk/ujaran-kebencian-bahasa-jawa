# Technical Briefing: Model Training Implementation

**From:** Arsitek Proyek  
**To:** Developer Backend Team  
**Date:** 2025-01-01  
**Priority:** HIGH - Critical Path  
**Phase:** Model Training & Evaluation  

---

## 🎯 Mission Briefing

**OBJECTIVE:** Implement production-ready IndoBERT fine-tuning pipeline for Javanese hate speech detection using our labeled dataset (41,887 samples).

**SUCCESS CRITERIA:**
- ✅ Model achieves >85% accuracy on test set
- ✅ F1-macro score >0.80 across all 4 classes
- ✅ Training pipeline is reproducible and well-documented
- ✅ Model artifacts ready for API deployment

---

## 📋 Technical Specifications

### Dataset Configuration
```python
# File: src/data_collection/hasil-labeling.csv
# Columns: text, original_label, final_label, confidence_score, response_time, labeling_method, error
# Total samples: 41,887
# Classes: 4 (Bukan/Ringan/Sedang/Berat)
```

### Model Architecture
```python
# Base Model: indobenchmark/indobert-base-p1
# Task: SequenceClassification
# Num Labels: 4
# Max Length: 128 tokens
# Output: Logits for 4 classes + softmax probabilities
```

### Training Configuration (Recommended)
```python
TRAINING_CONFIG = {
    "model_name": "indobenchmark/indobert-base-p1",
    "num_labels": 4,
    "max_length": 128,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_f1_macro",
    "greater_is_better": True,
    "save_total_limit": 2
}
```

---

## 🔧 Implementation Tasks Breakdown

### T1: Data Preprocessing Pipeline
**File:** `src/modelling/train_model.py` (function: `prepare_training_data`)

**Requirements:**
1. **Label Mapping Function**
   ```python
   def map_labels_to_numeric(df):
       label_map = {
           "Bukan Ujaran Kebencian": 0,
           "Ujaran Kebencian - Ringan": 1,
           "Ujaran Kebencian - Sedang": 2,
           "Ujaran Kebencian - Berat": 3
       }
       df['numeric_label'] = df['final_label'].map(label_map)
       return df
   ```

2. **Quality Filtering**
   ```python
   # Filter by confidence score
   df_filtered = df[df['confidence_score'] >= 0.7]
   # Remove error samples
   df_clean = df_filtered[df_filtered['error'].isna()]
   ```

3. **Data Split Strategy**
   ```python
   # Stratified split to maintain class distribution
   from sklearn.model_selection import train_test_split
   train_df, test_df = train_test_split(
       df_clean, 
       test_size=0.2, 
       stratify=df_clean['numeric_label'],
       random_state=42
   )
   ```

**Validation Criteria:**
- [ ] All 41,887 samples loaded successfully
- [ ] Label mapping produces valid integers 0-3
- [ ] Class distribution maintained in train/test split
- [ ] No missing values in critical columns

### T2: Training Pipeline Implementation
**File:** `src/modelling/train_model.py` (function: `train_hate_speech_model`)

**Core Components:**

1. **Dataset Class**
   ```python
   class JavaneseHateSpeechDataset(torch.utils.data.Dataset):
       def __init__(self, texts, labels, tokenizer, max_length=128):
           self.texts = texts
           self.labels = labels
           self.tokenizer = tokenizer
           self.max_length = max_length
       
       def __getitem__(self, idx):
           text = str(self.texts[idx])
           encoding = self.tokenizer(
               text,
               truncation=True,
               padding='max_length',
               max_length=self.max_length,
               return_tensors='pt'
           )
           return {
               'input_ids': encoding['input_ids'].flatten(),
               'attention_mask': encoding['attention_mask'].flatten(),
               'labels': torch.tensor(self.labels[idx], dtype=torch.long)
           }
   ```

2. **Class Weight Calculation**
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   
   def calculate_class_weights(labels):
       class_weights = compute_class_weight(
           'balanced',
           classes=np.unique(labels),
           y=labels
       )
       return torch.tensor(class_weights, dtype=torch.float)
   ```

3. **Custom Trainer with Weighted Loss**
   ```python
   class WeightedTrainer(Trainer):
       def __init__(self, class_weights, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.class_weights = class_weights
       
       def compute_loss(self, model, inputs, return_outputs=False):
           labels = inputs.get("labels")
           outputs = model(**inputs)
           logits = outputs.get("logits")
           
           loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
           loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
           
           return (loss, outputs) if return_outputs else loss
   ```

**Validation Criteria:**
- [ ] Model loads IndoBERT successfully
- [ ] Training loop completes without errors
- [ ] Model checkpoints saved correctly
- [ ] Training metrics logged properly
- [ ] Final model saved with tokenizer

### T3: Evaluation Pipeline
**File:** `src/modelling/evaluate_model.py`

**Required Functions:**

1. **Comprehensive Metrics**
   ```python
   def evaluate_model_comprehensive(model, test_dataset, tokenizer):
       predictions = model.predict(test_dataset)
       y_pred = np.argmax(predictions.predictions, axis=1)
       y_true = predictions.label_ids
       
       # Calculate metrics
       accuracy = accuracy_score(y_true, y_pred)
       precision, recall, f1, _ = precision_recall_fscore_support(
           y_true, y_pred, average=None
       )
       f1_macro = f1_score(y_true, y_pred, average='macro')
       
       # Confusion matrix
       cm = confusion_matrix(y_true, y_pred)
       
       return {
           'accuracy': accuracy,
           'f1_macro': f1_macro,
           'precision_per_class': precision,
           'recall_per_class': recall,
           'f1_per_class': f1,
           'confusion_matrix': cm
       }
   ```

2. **Visualization Functions**
   ```python
   def plot_confusion_matrix(cm, class_names, save_path):
       plt.figure(figsize=(10, 8))
       sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
       plt.title('Confusion Matrix - Javanese Hate Speech Detection')
       plt.ylabel('True Label')
       plt.xlabel('Predicted Label')
       plt.tight_layout()
       plt.savefig(save_path, dpi=300, bbox_inches='tight')
       plt.close()
   ```

**Validation Criteria:**
- [ ] All evaluation metrics calculated correctly
- [ ] Confusion matrix visualization generated
- [ ] Classification report saved
- [ ] Performance comparison with baseline
- [ ] Error analysis on misclassified samples

### T4: Training Utilities & Optimization
**File:** `src/modelling/train_utils.py`

**Utility Functions:**

1. **Hyperparameter Search**
   ```python
   def hyperparameter_search(train_dataset, val_dataset):
       param_grid = {
           'learning_rate': [1e-5, 2e-5, 3e-5],
           'batch_size': [8, 16, 32],
           'warmup_steps': [100, 500, 1000]
       }
       # Implement grid search or random search
   ```

2. **Early Stopping Callback**
   ```python
   class EarlyStoppingCallback(TrainerCallback):
       def __init__(self, early_stopping_patience=3):
           self.early_stopping_patience = early_stopping_patience
           self.best_metric = None
           self.patience_counter = 0
   ```

**Validation Criteria:**
- [ ] Hyperparameter tuning implemented
- [ ] Early stopping prevents overfitting
- [ ] Model checkpointing works correctly
- [ ] Training utilities well-documented

---

## 🚨 Critical Success Factors

### 1. Data Quality Assurance
- **MUST:** Verify label distribution before training
- **MUST:** Handle class imbalance with appropriate weights
- **MUST:** Filter low-confidence samples (< 0.7)

### 2. Training Stability
- **MUST:** Implement gradient clipping (max_grad_norm=1.0)
- **MUST:** Use learning rate scheduler
- **MUST:** Monitor training/validation loss curves

### 3. Reproducibility
- **MUST:** Set random seeds (torch, numpy, transformers)
- **MUST:** Log all hyperparameters
- **MUST:** Save training configuration with model

### 4. Performance Validation
- **MUST:** Achieve minimum F1-macro > 0.80
- **MUST:** Validate on held-out test set
- **MUST:** Perform error analysis on failures

---

## 📊 Expected Outcomes

### Model Performance Targets
```
Minimum Acceptable Performance:
- Overall Accuracy: >85%
- F1-Macro Score: >0.80
- Per-class F1: >0.70 for all classes

Optimal Performance Goals:
- Overall Accuracy: >90%
- F1-Macro Score: >0.85
- Per-class F1: >0.80 for all classes
```

### Deliverables
1. **Model Artifacts**
   - `models/bert_jawa_hate_speech/pytorch_model.bin`
   - `models/bert_jawa_hate_speech/config.json`
   - `models/bert_jawa_hate_speech/tokenizer.json`

2. **Evaluation Reports**
   - `reports/training_metrics.json`
   - `reports/evaluation_report.json`
   - `reports/confusion_matrix.png`
   - `reports/classification_report.txt`

3. **Training Logs**
   - `logs/training.log`
   - `logs/tensorboard_logs/`

---

## 🔄 Quality Gates

### Gate 1: Data Preprocessing ✅
- [ ] Dataset loaded and validated
- [ ] Labels mapped correctly
- [ ] Train/test split completed
- [ ] Class distribution analyzed

### Gate 2: Training Pipeline ✅
- [ ] Model training completes successfully
- [ ] No memory/GPU issues
- [ ] Training metrics show convergence
- [ ] Model checkpoints saved

### Gate 3: Evaluation ✅
- [ ] Test set evaluation completed
- [ ] Performance meets minimum thresholds
- [ ] Confusion matrix analyzed
- [ ] Error analysis documented

### Gate 4: Production Readiness ✅
- [ ] Model artifacts properly saved
- [ ] Inference pipeline tested
- [ ] Documentation completed
- [ ] Code reviewed and approved

---

## 🆘 Escalation Points

**Contact Arsitek immediately if:**
- Training accuracy plateaus below 80%
- Severe class imbalance cannot be resolved
- GPU memory issues prevent training
- Model fails to converge after 5 epochs
- Any technical blocker lasting >4 hours

**Emergency Fallback Plan:**
- Reduce model complexity (DistilBERT)
- Implement data augmentation
- Use transfer learning from similar domain
- Consider ensemble methods

---

**Status:** Ready for implementation  
**Next Review:** Daily standup until completion  
**Expected Completion:** 1-2 weeks  

---

*Good luck, team! This is the critical phase that will determine our project success. Focus on quality over speed, and don't hesitate to ask for architectural guidance.*

**- Arsitek Proyek**