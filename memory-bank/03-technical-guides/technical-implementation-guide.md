# Technical Implementation Guide
# BERT-based Javanese Hate Speech Detection System

**Target Audience:** Development Team, Researchers, Contributors  
**Last Updated:** 2025-01-02  
**Version:** 2.0 (Post-Experiment)  
**Status:** Production Ready  

---

## 🏗️ System Architecture Overview

### High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Model Layer    │    │ Application     │
│                 │    │                 │    │ Layer           │
│ • Raw Dataset   │───▶│ • IndoBERT      │───▶│ • API Endpoints │
│ • Preprocessed  │    │ • Fine-tuned    │    │ • Web Interface │
│ • Labeled Data  │    │ • Optimized     │    │ • Batch Process │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Storage Layer   │    │ Training Layer  │    │ Monitoring      │
│                 │    │                 │    │ Layer           │
│ • Model Files   │    │ • Training Loop │    │ • Metrics       │
│ • Checkpoints   │    │ • Validation    │    │ • Logging       │
│ • Configs       │    │ • Evaluation    │    │ • Alerts        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack
```yaml
Core Framework:
  - PyTorch: 2.0+
  - Transformers: 4.30+
  - Datasets: 2.12+
  - Scikit-learn: 1.3+

Model Infrastructure:
  - IndoBERT: indobenchmark/indobert-base-p1
  - Tokenizer: WordPiece
  - Max Length: 128 tokens
  - Vocabulary: 30,000 tokens

Data Processing:
  - Pandas: 2.0+
  - NumPy: 1.24+
  - JSON: Built-in
  - CSV: Built-in

Evaluation & Metrics:
  - Scikit-learn metrics
  - Custom threshold optimization
  - Confusion matrix analysis
  - Per-class performance tracking
```

---

## 📁 Project Structure

```
ujaran-kebencian-bahasa-jawa/
├── src/
│   ├── data_collection/
│   │   ├── hasil-labeling.csv          # Main dataset (41,887 samples)
│   │   ├── collect_data.py             # Data collection script
│   │   └── preprocess_data.py          # Data preprocessing
│   ├── model_training/
│   │   ├── train_model.py              # Original training script
│   │   ├── improved_training_strategy.py # Enhanced training
│   │   ├── balanced_evaluation.py      # Balanced evaluation
│   │   └── threshold_tuning.py         # Threshold optimization
│   └── evaluation/
│       ├── evaluate_model.py           # Model evaluation
│       └── metrics_calculator.py       # Custom metrics
├── models/
│   ├── trained_model/                  # Original model (Experiment 1)
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── evaluation_results.json
│   └── improved_model/                 # Enhanced model (Experiment 2)
│       ├── pytorch_model.bin
│       ├── config.json
│       ├── tokenizer.json
│       └── balanced_evaluation_results.json
├── vibe-guide/                         # Project guidelines
│   └── roles/
│       └── arsitek.md                  # Architect role definition
├── memory-bank/                        # Project context
│   ├── project-summary.md              # Current status
│   └── papan-proyek.md                 # Project board
├── docs/                               # Documentation
│   ├── ACADEMIC_PAPER_DOCUMENTATION.md
│   ├── FINAL_MODEL_IMPROVEMENT_REPORT.md
│   ├── IMPROVED_MODEL_COMPARISON_REPORT.md
│   └── ARCHITECTURAL_DOCUMENTATION_FOR_PAPER.md
└── configs/
    ├── training_config.json            # Training configurations
    └── model_config.json               # Model configurations
```

---

## 🔧 Implementation Details

### 1. Data Pipeline Implementation

#### Dataset Structure
```python
# Dataset Schema
Dataset Columns:
- 'text': str           # Raw Javanese text
- 'final_label': str    # Label category
- 'confidence': float   # Labeling confidence (0.0-1.0)
- 'reasoning': str      # Labeling reasoning

# Label Mapping
LABEL_MAPPING = {
    'Bukan Ujaran Kebencian': 0,
    'Ujaran Kebencian - Ringan': 1,
    'Ujaran Kebencian - Sedang': 2,
    'Ujaran Kebencian - Berat': 3
}
```

#### Preprocessing Pipeline
```python
def preprocess_text(text):
    """
    Preprocessing pipeline for Javanese text
    """
    # 1. Basic cleaning
    text = text.strip().lower()
    
    # 2. Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # 3. Handle special characters (preserve Javanese)
    text = re.sub(r'[^\w\s\u0900-\u097F]', '', text)
    
    # 4. Tokenization ready
    return text

def create_dataset(df, tokenizer, max_length=128):
    """
    Create PyTorch dataset from DataFrame
    """
    texts = df['text'].apply(preprocess_text).tolist()
    labels = df['final_label'].map(LABEL_MAPPING).tolist()
    
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return HateSpeechDataset(encodings, labels)
```

### 2. Model Architecture Implementation

#### Base Model Configuration
```python
from transformers import AutoModel, AutoTokenizer, AutoConfig

class JavaneseHateSpeechClassifier(nn.Module):
    def __init__(self, model_name='indobenchmark/indobert-base-p1', num_classes=4):
        super().__init__()
        
        # Load pre-trained IndoBERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        
        # Classification head
        self.classifier = nn.Linear(
            self.bert.config.hidden_size, 
            num_classes
        )
        
        # Initialize weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits
```

#### Training Configuration
```python
# Experiment 1: Baseline Configuration
BASELINE_CONFIG = {
    'model_name': 'indobenchmark/indobert-base-p1',
    'num_classes': 4,
    'max_length': 128,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'weight_decay': 0.01,
    'warmup_steps': 0,
    'save_steps': 500,
    'eval_steps': 500,
    'logging_steps': 100
}

# Experiment 2: Improved Configuration
IMPROVED_CONFIG = {
    'model_name': 'indobenchmark/indobert-base-p1',
    'num_classes': 4,
    'max_length': 128,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 5,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    'save_steps': 500,
    'eval_steps': 500,
    'logging_steps': 100,
    
    # Enhanced features
    'use_class_weights': True,
    'use_focal_loss': True,
    'focal_alpha': 1.0,
    'focal_gamma': 2.0,
    'stratified_split': True,
    'balanced_evaluation': True
}
```

### 3. Advanced Training Strategies

#### Class Weighting Implementation
```python
def compute_class_weights(labels):
    """
    Compute class weights based on inverse frequency
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=labels
    )
    
    return {i: weights[i] for i in range(len(classes))}

# Computed weights for our dataset
CLASS_WEIGHTS = {
    0: 0.2537,  # Bukan Ujaran Kebencian (majority)
    1: 2.2857,  # Ujaran Kebencian - Ringan
    2: 3.4286,  # Ujaran Kebencian - Sedang
    3: 6.8571   # Ujaran Kebencian - Berat (minority)
}
```

#### Focal Loss Implementation
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
```

#### Stratified Sampling
```python
def create_stratified_split(df, test_size=0.2, random_state=42):
    """
    Create stratified train/validation split
    """
    from sklearn.model_selection import train_test_split
    
    X = df['text']
    y = df['final_label']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    train_df = pd.DataFrame({'text': X_train, 'final_label': y_train})
    val_df = pd.DataFrame({'text': X_val, 'final_label': y_val})
    
    return train_df, val_df
```

### 4. Evaluation Framework

#### Balanced Evaluation Implementation
```python
def create_balanced_evaluation_set(df, samples_per_class=200, random_state=42):
    """
    Create balanced evaluation set with equal samples per class
    """
    balanced_dfs = []
    
    for label in df['final_label'].unique():
        class_df = df[df['final_label'] == label]
        
        if len(class_df) >= samples_per_class:
            sampled_df = class_df.sample(
                n=samples_per_class, 
                random_state=random_state
            )
        else:
            # Use all available samples if less than required
            sampled_df = class_df
        
        balanced_dfs.append(sampled_df)
    
    return pd.concat(balanced_dfs, ignore_index=True)

def comprehensive_evaluation(model, eval_dataloader, device):
    """
    Comprehensive model evaluation with multiple metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report
    )
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )
    
    macro_precision = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro'
    )[0]
    macro_recall = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro'
    )[1]
    macro_f1 = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro'
    )[2]
    
    results = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class_metrics': {
            f'class_{i}': {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            } for i in range(len(precision))
        },
        'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),
        'classification_report': classification_report(
            all_labels, all_predictions, output_dict=True
        )
    }
    
    return results, all_probabilities
```

### 5. Threshold Optimization

#### Implementation
```python
class ThresholdOptimizer:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.optimal_thresholds = None
    
    def optimize_thresholds(self, y_true, y_proba, metric='f1'):
        """
        Optimize thresholds for each class using grid search
        """
        from sklearn.metrics import f1_score
        
        best_thresholds = []
        best_scores = []
        
        for class_idx in range(self.num_classes):
            best_threshold = 0.5
            best_score = 0.0
            
            # Grid search over threshold values
            for threshold in np.arange(0.1, 0.9, 0.01):
                # Create binary predictions for current class
                y_pred_binary = (y_proba[:, class_idx] >= threshold).astype(int)
                y_true_binary = (y_true == class_idx).astype(int)
                
                # Calculate F1 score for current threshold
                if len(np.unique(y_true_binary)) > 1:  # Avoid division by zero
                    score = f1_score(y_true_binary, y_pred_binary)
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
            
            best_thresholds.append(best_threshold)
            best_scores.append(best_score)
        
        self.optimal_thresholds = best_thresholds
        return best_thresholds, best_scores
    
    def predict_with_thresholds(self, y_proba, thresholds=None):
        """
        Make predictions using optimized thresholds
        """
        if thresholds is None:
            thresholds = self.optimal_thresholds
        
        if thresholds is None:
            raise ValueError("Thresholds not set. Run optimize_thresholds first.")
        
        predictions = []
        for i, proba in enumerate(y_proba):
            # Find class with highest probability above threshold
            valid_classes = []
            for class_idx in range(self.num_classes):
                if proba[class_idx] >= thresholds[class_idx]:
                    valid_classes.append((class_idx, proba[class_idx]))
            
            if valid_classes:
                # Choose class with highest probability among valid ones
                predicted_class = max(valid_classes, key=lambda x: x[1])[0]
            else:
                # Fallback to class with highest probability
                predicted_class = np.argmax(proba)
            
            predictions.append(predicted_class)
        
        return np.array(predictions)

# Optimal thresholds from our experiments
OPTIMAL_THRESHOLDS = {
    0: 0.7128,  # Bukan Ujaran Kebencian
    1: 0.2332,  # Ujaran Kebencian - Ringan
    2: 0.2023,  # Ujaran Kebencian - Sedang
    3: 0.3395   # Ujaran Kebencian - Berat
}
```

---

## 🚀 Deployment Guidelines

### 1. Model Serving

#### FastAPI Implementation
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer

app = FastAPI(title="Javanese Hate Speech Detection API")

# Load model and tokenizer
model = JavaneseHateSpeechClassifier.from_pretrained('models/improved_model')
tokenizer = AutoTokenizer.from_pretrained('models/improved_model')
model.eval()

class PredictionRequest(BaseModel):
    text: str
    use_optimal_thresholds: bool = True

class PredictionResponse(BaseModel):
    text: str
    predicted_class: str
    confidence: float
    probabilities: dict
    processing_time: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_hate_speech(request: PredictionRequest):
    import time
    start_time = time.time()
    
    try:
        # Preprocess text
        processed_text = preprocess_text(request.text)
        
        # Tokenize
        inputs = tokenizer(
            processed_text,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding=True
        )
        
        # Predict
        with torch.no_grad():
            logits = model(**inputs)
            probabilities = F.softmax(logits, dim=-1).squeeze().numpy()
        
        # Apply optimal thresholds if requested
        if request.use_optimal_thresholds:
            optimizer = ThresholdOptimizer()
            optimizer.optimal_thresholds = list(OPTIMAL_THRESHOLDS.values())
            predicted_class_idx = optimizer.predict_with_thresholds(
                probabilities.reshape(1, -1)
            )[0]
        else:
            predicted_class_idx = np.argmax(probabilities)
        
        # Map to class name
        class_names = [
            'Bukan Ujaran Kebencian',
            'Ujaran Kebencian - Ringan',
            'Ujaran Kebencian - Sedang',
            'Ujaran Kebencian - Berat'
        ]
        predicted_class = class_names[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        # Prepare response
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            text=request.text,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities={
                class_names[i]: float(probabilities[i]) 
                for i in range(len(class_names))
            },
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "loaded"}
```

### 2. Performance Optimization

#### Model Quantization
```python
def quantize_model(model_path, output_path):
    """
    Quantize model for faster inference
    """
    import torch.quantization as quantization
    
    # Load model
    model = JavaneseHateSpeechClassifier.from_pretrained(model_path)
    model.eval()
    
    # Prepare for quantization
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    quantization.prepare(model, inplace=True)
    
    # Calibrate with sample data (you need to provide this)
    # calibrate_model(model, calibration_dataloader)
    
    # Convert to quantized model
    quantized_model = quantization.convert(model, inplace=False)
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), output_path)
    
    return quantized_model
```

#### Batch Processing
```python
def batch_predict(texts, model, tokenizer, batch_size=32):
    """
    Efficient batch prediction for multiple texts
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding=True
        )
        
        # Predict
        with torch.no_grad():
            logits = model(**inputs)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
    
    return all_predictions, all_probabilities
```

---

## 📊 Monitoring & Maintenance

### 1. Performance Monitoring

```python
class ModelMonitor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.metrics_history = []
        self.prediction_counts = {i: 0 for i in range(4)}
    
    def log_prediction(self, text, prediction, confidence, processing_time):
        """
        Log prediction for monitoring
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'prediction': prediction,
            'confidence': confidence,
            'processing_time': processing_time
        }
        
        self.metrics_history.append(log_entry)
        self.prediction_counts[prediction] += 1
    
    def get_performance_stats(self, window_hours=24):
        """
        Get performance statistics for the last N hours
        """
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        if not recent_metrics:
            return None
        
        avg_processing_time = np.mean([m['processing_time'] for m in recent_metrics])
        avg_confidence = np.mean([m['confidence'] for m in recent_metrics])
        total_predictions = len(recent_metrics)
        
        return {
            'total_predictions': total_predictions,
            'avg_processing_time': avg_processing_time,
            'avg_confidence': avg_confidence,
            'prediction_distribution': {
                i: sum(1 for m in recent_metrics if m['prediction'] == i)
                for i in range(4)
            }
        }
```

### 2. Model Versioning

```python
class ModelVersionManager:
    def __init__(self, base_path='models/'):
        self.base_path = base_path
        self.version_info = self.load_version_info()
    
    def register_model(self, model_path, version, metrics, description):
        """
        Register a new model version
        """
        version_entry = {
            'version': version,
            'path': model_path,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'description': description,
            'status': 'registered'
        }
        
        self.version_info[version] = version_entry
        self.save_version_info()
    
    def deploy_model(self, version):
        """
        Deploy a specific model version
        """
        if version not in self.version_info:
            raise ValueError(f"Version {version} not found")
        
        # Update status
        for v in self.version_info:
            self.version_info[v]['status'] = 'archived'
        
        self.version_info[version]['status'] = 'deployed'
        self.save_version_info()
    
    def get_current_model(self):
        """
        Get currently deployed model
        """
        for version, info in self.version_info.items():
            if info['status'] == 'deployed':
                return version, info
        
        return None, None
```

---

## 🔒 Security & Best Practices

### 1. Input Validation

```python
def validate_input_text(text):
    """
    Validate and sanitize input text
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    if len(text.strip()) == 0:
        raise ValueError("Input text cannot be empty")
    
    if len(text) > 1000:  # Reasonable limit
        raise ValueError("Input text too long (max 1000 characters)")
    
    # Remove potentially harmful characters
    sanitized_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return sanitized_text
```

### 2. Rate Limiting

```python
from collections import defaultdict
from time import time

class RateLimiter:
    def __init__(self, max_requests=100, window_seconds=3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id):
        now = time()
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window_seconds
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(now)
            return True
        
        return False
```

### 3. Error Handling

```python
class ModelError(Exception):
    """Base exception for model-related errors"""
    pass

class ModelLoadError(ModelError):
    """Raised when model fails to load"""
    pass

class PredictionError(ModelError):
    """Raised when prediction fails"""
    pass

def safe_predict(model, tokenizer, text):
    """
    Safe prediction with comprehensive error handling
    """
    try:
        # Validate input
        text = validate_input_text(text)
        
        # Preprocess
        processed_text = preprocess_text(text)
        
        # Tokenize
        inputs = tokenizer(
            processed_text,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding=True
        )
        
        # Predict
        with torch.no_grad():
            logits = model(**inputs)
            probabilities = F.softmax(logits, dim=-1)
        
        return probabilities.squeeze().numpy()
    
    except Exception as e:
        raise PredictionError(f"Prediction failed: {str(e)}")
```

---

## 📈 Performance Benchmarks

### Current Model Performance

| Metric | Experiment 1 | Experiment 2 | Experiment 2 + Tuning |
|--------|--------------|--------------|------------------------|
| **Accuracy** | 73.8% | 73.75% | **80.37%** |
| **F1-Score Macro** | 40.0% | 73.7% | **80.36%** |
| **Precision Macro** | 45.7% | 77.6% | **80.62%** |
| **Recall Macro** | 49.1% | 73.75% | **80.38%** |
| **Inference Time** | ~50ms | ~50ms | ~50ms |
| **Memory Usage** | ~1.2GB | ~1.2GB | ~1.2GB |

### Hardware Requirements

```yaml
Minimum Requirements:
  CPU: 4 cores, 2.0 GHz
  RAM: 8 GB
  Storage: 5 GB
  GPU: Optional (CUDA-compatible)

Recommended for Production:
  CPU: 8 cores, 3.0 GHz
  RAM: 16 GB
  Storage: 20 GB SSD
  GPU: NVIDIA Tesla T4 or better

Scaling Considerations:
  - 1 GPU can handle ~100 requests/second
  - CPU-only: ~10 requests/second
  - Memory scales linearly with batch size
```

---

## 🔄 Continuous Improvement

### 1. Model Retraining Pipeline

```python
def automated_retraining_pipeline():
    """
    Automated pipeline for model retraining
    """
    # 1. Check for new data
    new_data = check_for_new_labeled_data()
    
    if len(new_data) < MIN_NEW_SAMPLES:
        return "Insufficient new data for retraining"
    
    # 2. Validate new data quality
    quality_score = validate_data_quality(new_data)
    
    if quality_score < QUALITY_THRESHOLD:
        return "New data quality below threshold"
    
    # 3. Retrain model
    new_model = retrain_model_with_new_data(new_data)
    
    # 4. Evaluate new model
    evaluation_results = evaluate_model_comprehensive(new_model)
    
    # 5. Compare with current model
    current_performance = get_current_model_performance()
    
    if evaluation_results['f1_macro'] > current_performance['f1_macro']:
        # Deploy new model
        deploy_new_model(new_model, evaluation_results)
        return "New model deployed successfully"
    else:
        return "New model performance not better than current"
```

### 2. A/B Testing Framework

```python
class ABTestManager:
    def __init__(self):
        self.experiments = {}
        self.traffic_split = 0.5  # 50/50 split
    
    def create_experiment(self, name, model_a_path, model_b_path):
        """
        Create new A/B test experiment
        """
        self.experiments[name] = {
            'model_a': load_model(model_a_path),
            'model_b': load_model(model_b_path),
            'results_a': [],
            'results_b': [],
            'start_time': datetime.now()
        }
    
    def get_prediction(self, experiment_name, text, user_id):
        """
        Get prediction from A/B test
        """
        # Determine which model to use based on user_id
        use_model_a = hash(user_id) % 2 == 0
        
        experiment = self.experiments[experiment_name]
        
        if use_model_a:
            model = experiment['model_a']
            result_list = experiment['results_a']
        else:
            model = experiment['model_b']
            result_list = experiment['results_b']
        
        # Get prediction
        prediction = model.predict(text)
        
        # Log result
        result_list.append({
            'timestamp': datetime.now(),
            'user_id': user_id,
            'prediction': prediction,
            'model': 'A' if use_model_a else 'B'
        })
        
        return prediction
```

---

## 📚 Documentation & Resources

### API Documentation
- **Swagger UI**: Available at `/docs` when running FastAPI server
- **ReDoc**: Available at `/redoc`
- **OpenAPI Schema**: Available at `/openapi.json`

### Model Documentation
- **Model Card**: `models/improved_model/MODEL_CARD.md`
- **Training Logs**: `models/improved_model/training_logs/`
- **Evaluation Reports**: `models/improved_model/evaluation/`

### Development Resources
- **Setup Guide**: `docs/SETUP.md`
- **Contributing Guidelines**: `docs/CONTRIBUTING.md`
- **Code Style Guide**: `docs/CODE_STYLE.md`
- **Testing Guide**: `docs/TESTING.md`

---

**Status**: ✅ **PRODUCTION READY**  
**Next Review**: Q2 2025  
**Maintainer**: Development Team  
**Contact**: [team-email@domain.com]

---

*This document serves as the comprehensive technical guide for the Javanese Hate Speech Detection system. For questions or contributions, please refer to the contributing guidelines or contact the development team.*