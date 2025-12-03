#!/usr/bin/env python3
"""
Retrain model with best hyperparameters from optimization
"""

import argparse
import json
import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from torch.utils.data import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss implementation"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CustomTrainer(Trainer):
    """Custom trainer with focal loss"""
    def __init__(self, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        loss = self.focal_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

class HateSpeechDataset(Dataset):
    """Dataset class for hate speech detection"""
    def __init__(self, texts, labels, tokenizer, max_length=256):
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

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro
    }

def main():
    parser = argparse.ArgumentParser(description='Retrain model with best hyperparameters')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--base_model', type=str, default='indolem/indobert-base-uncased', help='Base model name')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Best hyperparameters from optimization (Trial 13)
    best_params = {
        'learning_rate': 3.7909412079529265e-05,
        'batch_size': 32,
        'max_length': 256,
        'weight_decay': 0.028129935174160545,
        'warmup_ratio': 0.06092632100996669,
        'focal_gamma': 1.7352101403669877,
        'dropout_rate': 0.24433669071285388
    }
    
    logger.info(f"Using best hyperparameters: {best_params}")
    
    # Load and prepare data
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    # Clean data - remove rows with missing values
    logger.info(f"Original data shape: {df.shape}")
    
    # Check if we have the correct columns
    if 'final_label' in df.columns:
        label_col = 'final_label'
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        raise ValueError("Neither 'final_label' nor 'label' column found in data")
    
    logger.info(f"Using label column: {label_col}")
    logger.info(f"Unique labels: {df[label_col].unique()}")
    
    # Filter data with confidence >= 0.7 if confidence_score column exists
    if 'confidence_score' in df.columns:
        df['confidence_score'] = pd.to_numeric(df['confidence_score'], errors='coerce')
        logger.info(f"Before confidence filtering: {len(df)} samples")
        df = df[df['confidence_score'] >= 0.7]
        logger.info(f"After confidence filtering (>= 0.7): {len(df)} samples")
    
    # Remove rows with missing values
    df = df.dropna(subset=['text', label_col])
    df = df[df['text'].str.strip() != '']
    logger.info(f"After removing NaN and empty text: {df.shape}")
    
    # Map labels to integers - handle both formats
    if 'Bukan Ujaran Kebencian' in df[label_col].values:
        # Format from hasil-labeling.csv
        label_map = {
            'Bukan Ujaran Kebencian': 0,
            'Ujaran Kebencian - Ringan': 1,
            'Ujaran Kebencian - Sedang': 1,
            'Ujaran Kebencian - Berat': 1
        }
    else:
        # Format from other sources
        label_map = {'NOT': 0, 'HS': 1, 'HATE': 1, 'NON_HATE': 0}
    
    df['label_int'] = df[label_col].map(label_map)
    
    # Remove any rows where label mapping failed
    df = df.dropna(subset=['label_int'])
    logger.info(f"After label mapping: {df.shape}")
    logger.info(f"Label distribution: {df['label_int'].value_counts().to_dict()}")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label_int'].tolist(),
        test_size=0.2,
        random_state=args.seed,
        stratify=df['label_int']
    )
    
    logger.info(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    
    # Load tokenizer and model
    logger.info(f"Loading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        hidden_dropout_prob=best_params['dropout_rate'],
        attention_probs_dropout_prob=best_params['dropout_rate']
    )
    
    # Create datasets
    train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer, best_params['max_length'])
    val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer, best_params['max_length'])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=best_params['batch_size'],
        per_device_eval_batch_size=best_params['batch_size'],
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        warmup_ratio=best_params['warmup_ratio'],
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=None,
        seed=args.seed
    )
    
    # Create trainer
    trainer = CustomTrainer(
        focal_gamma=best_params['focal_gamma'],
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Final evaluation
    logger.info("Final evaluation...")
    eval_results = trainer.evaluate()
    
    # Save results
    results = {
        'best_hyperparameters': best_params,
        'final_metrics': eval_results,
        'training_args': training_args.to_dict()
    }
    
    with open(f"{args.output_dir}/training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training completed!")
    logger.info(f"Final F1-Macro: {eval_results['eval_f1_macro']:.4f}")
    logger.info(f"Final Accuracy: {eval_results['eval_accuracy']:.4f}")
    
if __name__ == "__main__":
    main()