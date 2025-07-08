# Technical Implementation Details

## 1. System Architecture Overview

### 1.1 High-Level Architecture
```
Javanese Hate Speech Detection System
├── Data Pipeline
│   ├── Raw Data Collection
│   ├── Preprocessing & Cleaning
│   ├── Labeling & Annotation
│   └── Dataset Standardization
├── Model Training Pipeline
│   ├── Data Loading & Tokenization
│   ├── Model Initialization
│   ├── Training Loop dengan Optimization
│   └── Checkpoint Management
├── Evaluation Framework
│   ├── Balanced Evaluation Sets
│   ├── Metrics Computation
│   ├── Statistical Analysis
│   └── Error Analysis
└── Deployment Infrastructure
    ├── Model Serving
    ├── API Endpoints
    ├── Monitoring & Logging
    └── Performance Optimization
```

### 1.2 Technology Stack
- **Framework:** PyTorch 1.13+, Transformers 4.21+
- **Hardware:** NVIDIA GPU dengan CUDA support
- **Languages:** Python 3.8+
- **Dependencies:** NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Model Hub:** Hugging Face Transformers

## 2. Data Pipeline Implementation

### 2.1 Dataset Structure
```python
class JavaneseHateSpeechDataset:
    """
    Dataset structure untuk Javanese hate speech detection
    """
    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path)
        self.texts = self.data['text'].values
        self.labels = self.data['label_numeric'].values
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': self.labels[idx]
        }
```

### 2.2 Preprocessing Pipeline
```python
def preprocess_javanese_text(text: str) -> str:
    """
    Preprocessing pipeline untuk teks bahasa Jawa
    """
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    # Preserve Javanese linguistic features
    # No aggressive normalization to maintain cultural context
    
    return text

def create_balanced_evaluation_set(df: pd.DataFrame, samples_per_class: int = 200) -> pd.DataFrame:
    """
    Create balanced evaluation set untuk fair comparison
    """
    balanced_samples = []
    for label in [0, 1, 2, 3]:
        class_samples = df[df['label_numeric'] == label].sample(
            n=samples_per_class, 
            random_state=42
        )
        balanced_samples.append(class_samples)
    
    return pd.concat(balanced_samples, ignore_index=True)
```

### 2.3 Tokenization Strategy
```python
class JavaneseTokenizer:
    """
    Wrapper untuk handling Javanese text tokenization
    """
    def __init__(self, model_name: str, max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
```

## 3. Model Implementation Details

### 3.1 Base Model Architecture
```python
class JavaneseHateSpeechClassifier(nn.Module):
    """
    Base classifier untuk hate speech detection
    """
    def __init__(self, model_name: str, num_labels: int = 4, dropout_prob: float = 0.1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
```

### 3.2 Enhanced Model dengan Custom Head
```python
class EnhancedJavaneseClassifier(nn.Module):
    """
    Enhanced classifier dengan multi-layer head
    """
    def __init__(self, model_name: str, num_labels: int = 4, dropout_prob: float = 0.1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        
        # Multi-layer classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
```

### 3.3 Loss Function Implementation
```python
class WeightedFocalLoss(nn.Module):
    """
    Focal Loss dengan class weights untuk imbalanced data
    """
    def __init__(self, alpha: Dict[int, float], gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Apply class weights
        alpha_weights = torch.tensor(
            [self.alpha[i] for i in targets.cpu().numpy()]
        ).to(inputs.device)
        
        focal_loss = alpha_weights * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

## 4. Training Pipeline Implementation

### 4.1 Training Configuration
```python
@dataclass
class TrainingConfig:
    """
    Training configuration untuk reproducibility
    """
    model_name: str
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    max_length: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Class weights untuk imbalanced data
    class_weights: Dict[int, float] = field(default_factory=lambda: {
        0: 1.0, 1: 11.3, 2: 17.0, 3: 34.0
    })
    
    # Paths
    output_dir: str = "experiments/results"
    model_save_path: str = "models/javanese_hate_speech"
    log_file: str = "training.log"
```

### 4.2 Custom Trainer Implementation
```python
class JavaneseHateSpeechTrainer(Trainer):
    """
    Custom trainer dengan weighted focal loss
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights:
            self.loss_fn = WeightedFocalLoss(alpha=class_weights, gamma=2.0)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if hasattr(self, 'loss_fn'):
            loss = self.loss_fn(logits, labels)
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def log_metrics(self, logs, step):
        """Enhanced logging untuk monitoring"""
        super().log_metrics(logs, step)
        
        # Custom logging
        if 'eval_f1_macro' in logs:
            logger.info(f"Step {step}: F1-Macro = {logs['eval_f1_macro']:.4f}")
```

### 4.3 Training Loop Implementation
```python
def train_model(config: TrainingConfig, train_dataset, eval_dataset):
    """
    Main training function
    """
    # Set seeds untuk reproducibility
    set_seed(42)
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=4
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=50,
        eval_steps=100,
        save_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        report_to=None  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = JavaneseHateSpeechTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold
        )],
        class_weights=config.class_weights
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model(config.model_save_path)
    
    return trainer
```

## 5. Evaluation Implementation

### 5.1 Metrics Computation
```python
def compute_metrics(eval_pred):
    """
    Compute comprehensive evaluation metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Primary metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    accuracy = accuracy_score(labels, predictions)
    
    # Per-class metrics
    per_class_f1 = f1_score(labels, predictions, average=None)
    per_class_precision = precision_score(labels, predictions, average=None)
    per_class_recall = recall_score(labels, predictions, average=None)
    
    # Additional metrics
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_weighted': weighted_f1,
        'f1_class_0': per_class_f1[0],
        'f1_class_1': per_class_f1[1],
        'f1_class_2': per_class_f1[2],
        'f1_class_3': per_class_f1[3],
    }
```

### 5.2 Detailed Evaluation Function
```python
def detailed_evaluation(model, tokenizer, test_texts, test_labels, device):
    """
    Perform detailed evaluation dengan error analysis
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for text in test_texts:
            # Tokenize
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Predict
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            
            all_predictions.append(prediction.cpu().item())
            all_probabilities.append(probabilities.cpu().numpy())
    
    # Compute metrics
    metrics = compute_detailed_metrics(test_labels, all_predictions)
    
    # Error analysis
    error_analysis = analyze_errors(test_texts, test_labels, all_predictions)
    
    return {
        'metrics': metrics,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'error_analysis': error_analysis
    }
```

## 6. Device Management Solution

### 6.1 Device Mismatch Fix
```python
class DeviceManager:
    """
    Centralized device management untuk avoid mismatch errors
    """
    def __init__(self, force_cpu: bool = False):
        if force_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
    
    def to_device(self, tensor_or_model):
        """Move tensor atau model ke device yang benar"""
        return tensor_or_model.to(self.device)
    
    def ensure_device_consistency(self, model, batch):
        """Ensure model dan batch pada device yang sama"""
        # Move model to device
        model = model.to(self.device)
        
        # Move batch to device
        if isinstance(batch, dict):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
        elif torch.is_tensor(batch):
            batch = batch.to(self.device)
        
        return model, batch
```

### 6.2 Safe Evaluation Function
```python
def safe_evaluation(model, test_loader, device_manager):
    """
    Evaluation function dengan proper device management
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Ensure device consistency
            model, batch = device_manager.ensure_device_consistency(model, batch)
            
            # Extract labels before moving to device
            labels = batch.pop('labels')
            
            # Forward pass
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            # Move back to CPU untuk collection
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return compute_metrics((np.array(all_predictions), np.array(all_labels)))
```

## 7. Optimization Strategies

### 7.1 Memory Optimization
```python
def optimize_memory_usage():
    """
    Memory optimization strategies
    """
    # Clear cache
    torch.cuda.empty_cache()
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Use mixed precision training
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    
    return scaler

def memory_efficient_training(model, dataloader, optimizer, scaler):
    """
    Memory efficient training loop
    """
    for batch in dataloader:
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 7.2 Speed Optimization
```python
def optimize_inference_speed(model):
    """
    Optimize model untuk faster inference
    """
    # Convert to TorchScript
    model.eval()
    traced_model = torch.jit.trace(model, example_input)
    
    # Optimize untuk inference
    traced_model = torch.jit.optimize_for_inference(traced_model)
    
    return traced_model

def batch_inference(model, texts, tokenizer, batch_size=32):
    """
    Efficient batch inference
    """
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**encodings)
            batch_predictions = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(batch_predictions.cpu().numpy())
    
    return predictions
```

## 8. Deployment Infrastructure

### 8.1 Model Serving API
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Javanese Hate Speech Detection API")

class PredictionRequest(BaseModel):
    text: str
    return_probabilities: bool = False

class PredictionResponse(BaseModel):
    text: str
    predicted_label: int
    predicted_category: str
    confidence: float
    probabilities: Optional[List[float]] = None

@app.post("/predict", response_model=PredictionResponse)
async def predict_hate_speech(request: PredictionRequest):
    try:
        # Preprocess text
        cleaned_text = preprocess_javanese_text(request.text)
        
        # Tokenize
        encoding = tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**encoding)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]
        
        # Format response
        label_mapping = {
            0: "Bukan Ujaran Kebencian",
            1: "Ujaran Kebencian - Ringan",
            2: "Ujaran Kebencian - Sedang",
            3: "Ujaran Kebencian - Berat"
        }
        
        response = PredictionResponse(
            text=request.text,
            predicted_label=prediction.item(),
            predicted_category=label_mapping[prediction.item()],
            confidence=confidence.item()
        )
        
        if request.return_probabilities:
            response.probabilities = probabilities.squeeze().tolist()
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 8.2 Monitoring dan Logging
```python
import logging
from datetime import datetime

class ModelMonitor:
    """
    Monitor model performance dan usage
    """
    def __init__(self, log_file: str = "model_monitor.log"):
        self.logger = logging.getLogger("ModelMonitor")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.prediction_count = 0
        self.error_count = 0
        
    def log_prediction(self, text: str, prediction: int, confidence: float):
        """Log successful prediction"""
        self.prediction_count += 1
        self.logger.info(
            f"Prediction #{self.prediction_count}: "
            f"Label={prediction}, Confidence={confidence:.4f}, "
            f"TextLength={len(text)}"
        )
    
    def log_error(self, error: Exception, text: str = None):
        """Log prediction error"""
        self.error_count += 1
        self.logger.error(
            f"Error #{self.error_count}: {str(error)}"
            f"{f', Text: {text[:100]}...' if text else ''}"
        )
    
    def get_stats(self):
        """Get monitoring statistics"""
        return {
            "total_predictions": self.prediction_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(self.prediction_count, 1),
            "uptime": datetime.now().isoformat()
        }
```

## 9. Configuration Management

### 9.1 Environment Configuration
```python
import os
from typing import Optional

class EnvironmentConfig:
    """
    Environment-specific configuration
    """
    def __init__(self):
        self.model_path = os.getenv("MODEL_PATH", "models/best_model")
        self.device = os.getenv("DEVICE", "auto")
        self.batch_size = int(os.getenv("BATCH_SIZE", "16"))
        self.max_length = int(os.getenv("MAX_LENGTH", "128"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
    def get_device(self):
        if self.device == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(self.device)
```

### 9.2 Model Configuration Registry
```python
MODEL_CONFIGS = {
    "indobert_base": {
        "model_name": "indobenchmark/indobert-base-p1",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "max_length": 128,
        "epochs": 3
    },
    "xlm_roberta_improved": {
        "model_name": "xlm-roberta-base",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "max_length": 128,
        "epochs": 5,
        "warmup_ratio": 0.2,
        "weight_decay": 0.01
    },
    "mbert": {
        "model_name": "bert-base-multilingual-cased",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "max_length": 256,
        "epochs": 3
    }
}

def get_model_config(model_type: str) -> dict:
    """Get configuration untuk specific model type"""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_CONFIGS[model_type].copy()
```

## 10. Testing Framework

### 10.1 Unit Tests
```python
import unittest
from unittest.mock import Mock, patch

class TestJavaneseHateSpeechModel(unittest.TestCase):
    """
    Unit tests untuk model components
    """
    def setUp(self):
        self.sample_text = "Iki conto teks bahasa Jawa"
        self.sample_labels = [0, 1, 2, 3]
        
    def test_text_preprocessing(self):
        """Test text preprocessing function"""
        dirty_text = "  Iki   teks  kotor  \n\n  "
        clean_text = preprocess_javanese_text(dirty_text)
        self.assertEqual(clean_text, "Iki teks kotor")
    
    def test_tokenization(self):
        """Test tokenization process"""
        tokenizer = JavaneseTokenizer("bert-base-multilingual-cased")
        result = tokenizer.tokenize(self.sample_text)
        
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertEqual(result['input_ids'].shape[0], 128)
    
    def test_model_forward_pass(self):
        """Test model forward pass"""
        model = JavaneseHateSpeechClassifier(
            "bert-base-multilingual-cased", 
            num_labels=4
        )
        
        # Mock input
        input_ids = torch.randint(0, 1000, (1, 128))
        attention_mask = torch.ones(1, 128)
        
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        
        self.assertEqual(logits.shape, (1, 4))
```

### 10.2 Integration Tests
```python
class TestIntegration(unittest.TestCase):
    """
    Integration tests untuk end-to-end pipeline
    """
    def test_full_pipeline(self):
        """Test complete prediction pipeline"""
        # Load model (mock)
        with patch('torch.load') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # Test prediction
            text = "Sample Javanese text"
            result = predict_hate_speech(text, mock_model)
            
            self.assertIn('prediction', result)
            self.assertIn('confidence', result)
```

---

**Implementation Files:**
- Core implementation: `src/`
- Experiment scripts: `experiments/`
- Configuration files: `configs/`
- Tests: `tests/`

**Dependencies:**
```
torch>=1.13.0
transformers>=4.21.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
fastapi>=0.68.0
uvicorn>=0.15.0
```

**Hardware Requirements:**
- GPU: NVIDIA GPU dengan 8GB+ VRAM (untuk training)
- CPU: Multi-core processor (untuk inference)
- RAM: 16GB+ (untuk large models)
- Storage: 10GB+ untuk models dan data

**Metadata:**
- Implementation Date: 2025-01-06
- Framework Version: PyTorch 1.13+
- Model Hub: Hugging Face Transformers
- License: Academic Research Only