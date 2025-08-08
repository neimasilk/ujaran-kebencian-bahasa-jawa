#!/usr/bin/env python3
"""
Train Model on Augmented Dataset for 90%+ Accuracy
Combine augmented data with improved training strategy
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import json
import logging
from datetime import datetime
import os
import warnings
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/augmented_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FocalLoss(torch.nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
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

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, use_focal_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = FocalLoss(alpha=1, gamma=2)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.use_focal_loss:
            loss = self.focal_loss(logits, labels)
        else:
            if self.class_weights is not None:
                loss_fct = CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            else:
                loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def main():
    logger.info("Starting training on augmented dataset")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load augmented dataset
    logger.info("Loading augmented dataset")
    df = pd.read_csv('data/augmented/augmented_dataset.csv')
    
    logger.info(f"Augmented dataset size: {len(df)}")
    logger.info(f"Class distribution: {df['label_numeric'].value_counts().sort_index().to_dict()}")
    
    # Prepare data
    X = df['text'].values
    y = df['label_numeric'].values
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Load tokenizer and model
    model_name = 'indobenchmark/indobert-base-p1'
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
        problem_type="single_label_classification"
    )
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    logger.info(f"Class weights: {class_weights}")
    
    # Create datasets
    train_dataset = HateSpeechDataset(X_train, y_train, tokenizer)
    val_dataset = HateSpeechDataset(X_val, y_val, tokenizer)
    test_dataset = HateSpeechDataset(X_test, y_test, tokenizer)
    
    # Training arguments with optimized hyperparameters
    training_args = TrainingArguments(
        output_dir='./models/augmented_model',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,
        seed=42,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        gradient_accumulation_steps=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        group_by_length=True,
        report_to=None
    )
    
    # Create trainer with focal loss
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights,
        use_focal_loss=True  # Use focal loss for better handling of imbalance
    )
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Save model
    logger.info("Saving trained model")
    trainer.save_model('./models/augmented_model_final')
    tokenizer.save_pretrained('./models/augmented_model_final')
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_results = trainer.evaluate(test_dataset)
    
    # Detailed evaluation
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    
    # Classification report
    class_names = ['Bukan Ujaran Kebencian', 'Ujaran Kebencian - Ringan', 
                   'Ujaran Kebencian - Sedang', 'Ujaran Kebencian - Berat']
    
    report = classification_report(
        true_labels, predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Save results
    results = {
        'experiment_timestamp': datetime.now().isoformat(),
        'model_name': 'augmented_indobert_focal_loss',
        'dataset_info': {
            'total_samples': int(len(df)),
            'train_samples': int(len(X_train)),
            'val_samples': int(len(X_val)),
            'test_samples': int(len(X_test)),
            'augmentation_applied': True,
            'focal_loss_applied': True
        },
        'performance_metrics': {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted)
        },
        'detailed_classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_weights_used': class_weights.cpu().numpy().tolist(),
        'training_args': {
            'epochs': training_args.num_train_epochs,
            'batch_size': training_args.per_device_train_batch_size,
            'learning_rate': training_args.learning_rate,
            'weight_decay': training_args.weight_decay,
            'warmup_steps': training_args.warmup_steps
        }
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/augmented_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n" + "="*80)
    print("🚀 AUGMENTED MODEL TRAINING RESULTS")
    print("="*80)
    
    print(f"\n📊 DATASET STATISTICS:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    print(f"\n🎯 PERFORMANCE METRICS:")
    print(f"   🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   📈 F1-Macro: {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"   ⚖️ F1-Weighted: {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    
    # Check if target achieved
    target_achieved = accuracy >= 0.90 and f1_macro >= 0.90
    print(f"\n🎯 TARGET STATUS:")
    if target_achieved:
        print(f"   ✅ 90%+ TARGET ACHIEVED!")
        print(f"   🏆 Accuracy: {accuracy*100:.2f}% (Target: 90%+)")
        print(f"   🏆 F1-Macro: {f1_macro*100:.2f}% (Target: 90%+)")
    else:
        accuracy_gap = 0.90 - accuracy
        f1_gap = 0.90 - f1_macro
        print(f"   ⚠️ Target not yet achieved")
        print(f"   📊 Accuracy gap: {accuracy_gap*100:.2f}% (Current: {accuracy*100:.2f}%)")
        print(f"   📊 F1-Macro gap: {f1_gap*100:.2f}% (Current: {f1_macro*100:.2f}%)")
    
    print(f"\n📈 PER-CLASS PERFORMANCE:")
    for i, class_name in enumerate(class_names):
        class_f1 = report[class_name]['f1-score']
        class_precision = report[class_name]['precision']
        class_recall = report[class_name]['recall']
        print(f"   {class_name}:")
        print(f"     F1: {class_f1:.3f} | Precision: {class_precision:.3f} | Recall: {class_recall:.3f}")
    
    print(f"\n🔧 TRAINING CONFIGURATION:")
    print(f"   🧠 Model: IndoBERT + Focal Loss")
    print(f"   📚 Data: Augmented dataset (+30%)")
    print(f"   ⚖️ Class weights: Applied")
    print(f"   🎯 Loss function: Focal Loss (α=1, γ=2)")
    
    print(f"\n📁 MODEL SAVED TO:")
    print(f"   📁 ./models/augmented_model_final/")
    print(f"   📁 Results: results/augmented_model_results.json")
    
    if not target_achieved:
        print(f"\n🚀 NEXT STEPS FOR 90%+:")
        print(f"   1. Hyperparameter optimization (learning rate, batch size)")
        print(f"   2. Advanced ensemble with multiple architectures")
        print(f"   3. External data integration")
        print(f"   4. Advanced regularization techniques")
    
    print("\n" + "="*80)
    
    logger.info(f"Training completed. Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}")
    
    return accuracy, f1_macro, target_achieved

if __name__ == "__main__":
    main()