# Model Architecture - Transformer-based Hate Speech Detection

## 1. Overview Arsitektur

### 1.1 Paradigma Model
- **Approach:** Transfer Learning dengan Pre-trained Transformers
- **Architecture Family:** BERT-based dan RoBERTa-based models
- **Task Type:** Multi-class Text Classification (4 classes)
- **Fine-tuning Strategy:** Full model fine-tuning dengan task-specific head

### 1.2 Model Selection Rationale
- **Language Focus:** Model yang mendukung bahasa Indonesia/Melayu untuk transfer ke Jawa
- **Performance:** Model dengan track record baik pada NLP tasks
- **Computational Efficiency:** Balance antara performa dan resource requirements
- **Availability:** Model yang tersedia di Hugging Face Hub

## 2. Model Variants Evaluated

### 2.1 IndoBERT Family

#### 2.1.1 IndoBERT Base
- **Model ID:** `indobenchmark/indobert-base-p1`
- **Parameters:** 110M
- **Architecture:** BERT-base dengan Indonesian pre-training
- **Vocabulary:** 30,000 WordPiece tokens
- **Max Sequence Length:** 512 tokens
- **Pre-training Data:** Indonesian Wikipedia, news, web text

**Configuration:**
```python
model_config = {
    "model_name": "indobenchmark/indobert-base-p1",
    "num_labels": 4,
    "max_length": 128,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1
}
```

#### 2.1.2 IndoBERT Large
- **Model ID:** `indobenchmark/indobert-large-p1`
- **Parameters:** 340M
- **Architecture:** BERT-large dengan Indonesian pre-training
- **Advantages:** Lebih banyak parameters untuk representasi yang lebih kaya
- **Challenges:** Memerlukan GPU memory yang lebih besar

**Configuration:**
```python
model_config = {
    "model_name": "indobenchmark/indobert-large-p1",
    "num_labels": 4,
    "max_length": 256,  # Increased for larger model
    "batch_size": 4,    # Reduced due to memory constraints
    "gradient_accumulation_steps": 4
}
```

### 2.2 Multilingual Models

#### 2.2.1 mBERT (Multilingual BERT)
- **Model ID:** `bert-base-multilingual-cased`
- **Parameters:** 110M
- **Languages:** 104 languages including Indonesian
- **Vocabulary:** 119,547 WordPiece tokens
- **Advantage:** Cross-lingual capabilities

**Configuration:**
```python
model_config = {
    "model_name": "bert-base-multilingual-cased",
    "num_labels": 4,
    "max_length": 256,
    "batch_size": 16,
    "learning_rate": 2e-5
}
```

#### 2.2.2 XLM-RoBERTa
- **Model ID:** `xlm-roberta-base`
- **Parameters:** 125M
- **Architecture:** RoBERTa dengan multilingual pre-training
- **Languages:** 100 languages
- **Advantage:** RoBERTa improvements + multilingual

**Configuration:**
```python
model_config = {
    "model_name": "xlm-roberta-base",
    "num_labels": 4,
    "max_length": 256,
    "batch_size": 8,
    "learning_rate": 1e-5
}
```

## 3. Architecture Components

### 3.1 Input Processing

#### 3.1.1 Tokenization Pipeline
```python
class TextProcessor:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def process(self, text):
        # Text cleaning
        text = self.clean_text(text)
        
        # Tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
```

#### 3.1.2 Special Tokens
- **[CLS]:** Classification token untuk representasi sequence
- **[SEP]:** Separator token (tidak digunakan untuk single sentence)
- **[PAD]:** Padding token untuk batch processing
- **[UNK]:** Unknown token untuk out-of-vocabulary words

### 3.2 Transformer Encoder

#### 3.2.1 Multi-Head Self-Attention
- **Attention Heads:** 12 (base) / 16 (large)
- **Hidden Size:** 768 (base) / 1024 (large)
- **Intermediate Size:** 3072 (base) / 4096 (large)
- **Attention Dropout:** 0.1

#### 3.2.2 Feed-Forward Networks
- **Activation:** GELU
- **Hidden Dropout:** 0.1
- **Layer Normalization:** Applied before each sub-layer

### 3.3 Classification Head

#### 3.3.1 Standard Classification Head
```python
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, pooled_output):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
```

#### 3.3.2 Enhanced Classification Head (untuk model improved)
```python
class EnhancedClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_prob=0.1):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.intermediate = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size // 2, num_labels)
    
    def forward(self, pooled_output):
        x = self.dropout1(pooled_output)
        x = self.intermediate(x)
        x = self.activation(x)
        x = self.dropout2(x)
        logits = self.classifier(x)
        return logits
```

## 4. Training Strategy

### 4.1 Loss Functions

#### 4.1.1 Standard Cross-Entropy Loss
```python
loss_fn = nn.CrossEntropyLoss()
```

#### 4.1.2 Weighted Cross-Entropy Loss
```python
# Class weights berdasarkan inverse frequency
class_weights = {
    0: 1.0,    # Bukan Ujaran Kebencian
    1: 11.3,   # Ujaran Kebencian - Ringan
    2: 17.0,   # Ujaran Kebencian - Sedang
    3: 34.0    # Ujaran Kebencian - Berat
}

weight_tensor = torch.tensor([class_weights[i] for i in range(4)])
loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
```

#### 4.1.3 Focal Loss (untuk model advanced)
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
```

### 4.2 Optimization Strategy

#### 4.2.1 Optimizer Configuration
```python
optimizer_config = {
    "optimizer": "AdamW",
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "eps": 1e-8,
    "betas": (0.9, 0.999)
}
```

#### 4.2.2 Learning Rate Scheduling
```python
scheduler_config = {
    "scheduler_type": "linear",
    "warmup_ratio": 0.1,
    "num_training_steps": total_steps
}
```

### 4.3 Regularization Techniques

#### 4.3.1 Dropout
- **Hidden Dropout:** 0.1 (standard)
- **Attention Dropout:** 0.1
- **Classifier Dropout:** 0.1-0.3 (tergantung model)

#### 4.3.2 Early Stopping
```python
early_stopping_config = {
    "patience": 3,
    "min_delta": 0.001,
    "metric": "eval_f1_macro",
    "mode": "max"
}
```

#### 4.3.3 Gradient Clipping
```python
training_args.max_grad_norm = 1.0
```

## 5. Model Improvements

### 5.1 Configuration Optimization

#### 5.1.1 XLM-RoBERTa Improvements
**Baseline Configuration:**
```python
baseline_config = {
    "learning_rate": 1e-5,
    "batch_size": 8,
    "max_length": 256,
    "epochs": 3,
    "warmup_ratio": 0.1
}
```

**Improved Configuration:**
```python
improved_config = {
    "learning_rate": 2e-5,     # Increased
    "batch_size": 16,         # Increased
    "max_length": 128,        # Optimized
    "epochs": 5,              # Increased
    "warmup_ratio": 0.2,      # Increased
    "weight_decay": 0.01,     # Added
    "gradient_accumulation_steps": 2
}
```

**Result:** F1-Macro improvement dari 36.39% → 61.86% (+25.47%)

### 5.2 Architecture Enhancements

#### 5.2.1 Multi-layer Classification Head
- **Standard:** Single linear layer
- **Enhanced:** Multi-layer dengan intermediate representations
- **Benefit:** Better feature extraction untuk classification

#### 5.2.2 Attention Mechanism Modifications
- **Standard:** Default attention patterns
- **Enhanced:** Task-specific attention weights
- **Implementation:** Custom attention layers (future work)

## 6. Computational Requirements

### 6.1 Hardware Specifications

| Model | GPU Memory | Training Time | Inference Speed |
|-------|------------|---------------|----------------|
| IndoBERT Base | 6-8 GB | 2-3 hours | ~50ms/sample |
| IndoBERT Large | 12-16 GB | 4-6 hours | ~80ms/sample |
| mBERT | 6-8 GB | 2-3 hours | ~50ms/sample |
| XLM-RoBERTa | 8-10 GB | 3-4 hours | ~60ms/sample |

### 6.2 Optimization Strategies

#### 6.2.1 Memory Optimization
- **Gradient Accumulation:** Untuk batch size yang lebih besar
- **Mixed Precision:** FP16 training untuk memory efficiency
- **Gradient Checkpointing:** Trade-off memory vs computation

#### 6.2.2 Speed Optimization
- **DataLoader Workers:** Parallel data loading
- **Batch Size Tuning:** Optimal batch size untuk hardware
- **Model Compilation:** TorchScript untuk inference

## 7. Model Evaluation Framework

### 7.1 Evaluation Metrics
- **Primary:** F1-Score Macro (untuk class imbalance)
- **Secondary:** Accuracy, Precision, Recall per class
- **Additional:** Confusion Matrix, Classification Report

### 7.2 Validation Strategy
- **Cross-Validation:** Stratified K-fold (future work)
- **Hold-out:** 80/20 split dengan stratified sampling
- **Balanced Evaluation:** 200 samples per class

### 7.3 Statistical Significance
- **Multiple Runs:** 3-5 runs dengan random seeds berbeda
- **Confidence Intervals:** 95% CI untuk metrics
- **Significance Tests:** Paired t-test untuk model comparison

## 8. Limitations dan Future Work

### 8.1 Current Limitations
- **Language Mismatch:** Pre-trained pada Indonesian, applied ke Javanese
- **Domain Gap:** Pre-training data vs target domain
- **Class Imbalance:** Masih challenging meskipun sudah ada mitigation
- **Computational Cost:** Large models memerlukan resources signifikan

### 8.2 Future Improvements
- **Javanese Pre-training:** Model yang di-pre-train khusus untuk bahasa Jawa
- **Domain Adaptation:** Techniques untuk bridging domain gap
- **Ensemble Methods:** Kombinasi multiple models
- **Active Learning:** Untuk efficient data labeling

---

**Technical References:**
- Model implementations: `experiments/`
- Training utilities: `src/modelling/`
- Evaluation framework: `src/evaluation/`

**Performance References:**
- [Experiment Results Summary](../02-experiments/experiment-summary.md)
- [Comparative Analysis](../02-experiments/comparative-analysis.md)

**Metadata:**
- Created: 2025-01-06
- Version: 1.0
- Status: Production Ready
- Contact: Research Team