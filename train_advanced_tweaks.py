#!/usr/bin/env python3
"""
Advanced Training Script dengan Quick Tweaks untuk Meningkatkan Performa
- Focal Loss untuk mengatasi class imbalance
- Label Smoothing untuk regularization
- Early Stopping yang lebih ketat
- Learning Rate Scheduling yang lebih optimal
- Gradient Clipping
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import argparse
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss untuk mengatasi class imbalance"""
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

class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        log_prob = F.log_softmax(inputs, dim=-1)
        weight = inputs.new_ones(inputs.size()) * self.smoothing / (inputs.size(-1) - 1.)
        weight.scatter_(-1, targets.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

class AdvancedDataset(Dataset):
    """Dataset dengan preprocessing yang lebih baik"""
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
        
        # Tokenize dengan attention mask
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

class AdvancedTrainer(Trainer):
    """Custom Trainer dengan Focal Loss dan Label Smoothing"""
    def __init__(self, loss_type='focal', focal_alpha=1.0, focal_gamma=2.0, 
                 label_smoothing=0.1, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.class_weights = class_weights
        
        if loss_type == 'focal':
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif loss_type == 'label_smoothing':
            self.loss_fn = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        elif loss_type == 'weighted':
            if class_weights is not None:
                self.loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
            else:
                self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Move class weights to same device as logits if using weighted loss
        if self.loss_type == 'weighted' and hasattr(self.loss_fn, 'weight') and self.loss_fn.weight is not None:
            self.loss_fn.weight = self.loss_fn.weight.to(logits.device)
        
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def load_and_prepare_data(data_path, test_size=0.2, random_state=42):
    """Load dan prepare data dengan stratified split"""
    logger.info(f"Loading data from: {data_path}")
    
    # Load dataset
    df = pd.read_csv(data_path, header=None)
    df.columns = ['text', 'sentiment', 'final_label', 'confidence_score', 'cost', 'method', 'extra1', 'extra2']
    
    # Filter by confidence
    df['confidence_score'] = pd.to_numeric(df['confidence_score'], errors='coerce')
    df_filtered = df[df['confidence_score'] >= 0.7].copy()
    
    logger.info(f"Total samples after filtering: {len(df_filtered)}")
    
    # Label mapping
    label_mapping = {
        "Bukan Ujaran Kebencian": 0,
        "Ujaran Kebencian - Ringan": 1,
        "Ujaran Kebencian - Sedang": 2,
        "Ujaran Kebencian - Berat": 3
    }
    
    df_filtered['label_numeric'] = df_filtered['final_label'].map(label_mapping)
    df_filtered = df_filtered.dropna(subset=['label_numeric'])
    df_filtered['label_numeric'] = df_filtered['label_numeric'].astype(int)
    
    # Print distribution
    logger.info("Label distribution:")
    for label, count in df_filtered['final_label'].value_counts().items():
        logger.info(f"  {label}: {count} ({count/len(df_filtered)*100:.2f}%)")
    
    # Stratified split
    X = df_filtered['text'].tolist()
    y = df_filtered['label_numeric'].tolist()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    return X_train, X_val, y_train, y_val, label_mapping

def compute_class_weights(y_train):
    """Compute class weights untuk balanced training"""
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    logger.info(f"Class weights: {dict(zip(classes, class_weights))}")
    return class_weights

def compute_metrics(eval_pred):
    """Compute metrics untuk evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'f1_weighted': f1_w,
        'precision_macro': precision,
        'recall_macro': recall,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w
    }

def main():
    parser = argparse.ArgumentParser(description='Advanced Training dengan Quick Tweaks')
    parser.add_argument('--data_path', type=str, default='src/data_collection/hasil-labeling.csv',
                       help='Path to training data')
    parser.add_argument('--model_name', type=str, default='indolem/indobert-base-uncased',
                       help='Model name or path')
    parser.add_argument('--output_dir', type=str, default='models/advanced_tweaks',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--loss_type', type=str, default='focal', 
                       choices=['focal', 'label_smoothing', 'weighted', 'standard'],
                       help='Loss function type')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping')
    
    args = parser.parse_args()
    
    logger.info("=== ADVANCED TRAINING WITH QUICK TWEAKS ===")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Loss type: {args.loss_type}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    # Load data
    X_train, X_val, y_train, y_val, label_mapping = load_and_prepare_data(args.data_path)
    
    # Compute class weights
    class_weights = compute_class_weights(y_train)
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=4,
        problem_type="single_label_classification"
    )
    
    # Create datasets
    train_dataset = AdvancedDataset(X_train, y_train, tokenizer, args.max_length)
    val_dataset = AdvancedDataset(X_val, y_val, tokenizer, args.max_length)
    
    # Training arguments dengan early stopping yang ketat
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=50,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,
        dataloader_drop_last=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        max_grad_norm=args.gradient_clip,
        lr_scheduler_type="cosine",
        report_to=None
    )
    
    # Early stopping callback yang lebih ketat
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,  # Lebih ketat
        early_stopping_threshold=0.001  # Threshold lebih kecil
    )
    
    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        class_weights=class_weights
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Final evaluation
    logger.info("Final evaluation...")
    eval_results = trainer.evaluate()
    
    # Save training summary
    summary = {
        "model_name": args.model_name,
        "training_args": vars(args),
        "final_metrics": eval_results,
        "class_weights": class_weights.tolist(),
        "label_mapping": label_mapping,
        "training_date": datetime.now().isoformat()
    }
    
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Training completed! Model saved to: {args.output_dir}")
    logger.info(f"Final F1-Macro: {eval_results['eval_f1_macro']:.4f}")
    logger.info(f"Final Accuracy: {eval_results['eval_accuracy']:.4f}")
    
if __name__ == "__main__":
    main()