#!/usr/bin/env python3
"""
Script training yang diperbaiki untuk model hate speech classification
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import argparse
import logging
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            # Apply class weights - ensure same device as model
            device_weights = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=device_weights)
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
    precision_macro = precision_recall_fscore_support(labels, predictions, average='macro')[0]
    recall_macro = precision_recall_fscore_support(labels, predictions, average='macro')[1]
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to balanced_dataset.csv')
    parser.add_argument('--output_dir', default='models/improved_baseline', help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--model_name', default='indolem/indobert-base-uncased', help='Model name')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for imbalanced data')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    
    args = parser.parse_args()
    
    logger.info(f"Loading data from: {args.data_path}")
    
    # Load data
    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Check label distribution
    label_counts = df['final_label'].value_counts()
    logger.info(f"Label distribution:\n{label_counts}")
    
    # Prepare data
    texts = df['text'].tolist()
    labels = df['label_numeric'].tolist()
    
    # Check label distribution for class weights
    label_counter = Counter(labels)
    logger.info(f"Numeric label distribution: {label_counter}")
    
    # Split data with stratification
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Train samples: {len(train_texts)}")
    logger.info(f"Val samples: {len(val_texts)}")
    
    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=4,  # 0: Bukan, 1: Ringan, 2: Sedang, 3: Berat
        problem_type="single_label_classification"
    )
    
    # Compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        unique_labels = np.unique(train_labels)
        class_weights_array = compute_class_weight(
            'balanced', 
            classes=unique_labels, 
            y=train_labels
        )
        class_weights = torch.FloatTensor(class_weights_array)
        logger.info(f"Class weights: {class_weights}")
    
    # Create datasets
    train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer, args.max_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=50,
        eval_strategy='steps',
        eval_steps=200,
        save_strategy='steps',
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        save_total_limit=3,
        report_to=None,
        seed=42,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False
    )
    
    # Create trainer
    if args.use_class_weights and class_weights is not None:
        trainer = WeightedTrainer(
            class_weights=class_weights.to(model.device) if torch.cuda.is_available() else class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Final evaluation
    logger.info("Final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Final results: {eval_results}")
    
    # Save training summary
    summary = {
        'model_name': args.model_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'use_class_weights': args.use_class_weights,
        'warmup_steps': args.warmup_steps,
        'weight_decay': args.weight_decay,
        'train_samples': len(train_texts),
        'val_samples': len(val_texts),
        'final_results': eval_results
    }
    
    import json
    with open(f'{args.output_dir}/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Best model saved to: {args.output_dir}")
    logger.info(f"Training summary saved to: {args.output_dir}/training_summary.json")

if __name__ == '__main__':
    main()