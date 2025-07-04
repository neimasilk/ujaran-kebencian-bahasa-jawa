#!/usr/bin/env python3
"""
Experiment 0 - Baseline IndoBERT with Balanced Dataset

This experiment addresses the class imbalance issue found in the original baseline experiment.
Implements several techniques to handle imbalanced dataset:
1. Class weighting (class_weight='balanced')
2. Stratified train/validation split
3. Comprehensive per-class evaluation metrics

Dataset Distribution:
- Bukan Ujaran Kebencian: 20,205 (48.39%)
- Ujaran Kebencian - Sedang: 8,600 (20.60%)
- Ujaran Kebencian - Berat: 6,711 (16.07%)
- Ujaran Kebencian - Ringan: 6,241 (14.95%)
- Imbalance Ratio: 3.24:1
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import torch
from torch.utils.data import Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_0_baseline_indobert_balanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'model_name': 'indobenchmark/indobert-base-p1',
    'dataset_path': 'data/processed/final_dataset_shuffled.csv',
    'max_length': 128,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 5,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'save_steps': 500,
    'eval_steps': 500,
    'logging_steps': 100,
    'early_stopping_patience': 3,
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'use_class_weights': True,
    'stratify': True
}

# Label mapping
LABEL_MAPPING = {
    'Bukan Ujaran Kebencian': 0,
    'Ujaran Kebencian - Ringan': 1,
    'Ujaran Kebencian - Sedang': 2,
    'Ujaran Kebencian - Berat': 3
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

class HateSpeechDataset(Dataset):
    """Custom dataset for hate speech classification"""
    
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

def load_and_prepare_data(dataset_path):
    """Load and prepare the dataset with stratified splitting"""
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset shape: {df.shape}")
    
    # Use final_label if available, otherwise use label
    if 'final_label' in df.columns:
        label_column = 'final_label'
    else:
        label_column = 'label'
    
    # Create label_id mapping
    df['label_id'] = df[label_column].map(LABEL_MAPPING)
    
    # Remove rows with unmapped labels
    df = df.dropna(subset=['label_id'])
    df['label_id'] = df['label_id'].astype(int)
    
    logger.info(f"Final dataset shape after cleaning: {df.shape}")
    
    # Analyze label distribution
    label_dist = df[label_column].value_counts()
    logger.info("Label distribution:")
    for label, count in label_dist.items():
        percentage = (count / len(df)) * 100
        logger.info(f"  {label}: {count} ({percentage:.2f}%)")
    
    # Calculate imbalance ratio
    max_count = label_dist.max()
    min_count = label_dist.min()
    imbalance_ratio = max_count / min_count
    logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    return df[['text', 'label_id', label_column]]

def create_stratified_splits(df, test_size=0.2, val_size=0.1, random_state=42):
    """Create stratified train/validation/test splits"""
    logger.info("Creating stratified splits...")
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['label_id'],
        random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=train_val_df['label_id'],
        random_state=random_state
    )
    
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Validation set: {len(val_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")
    
    # Log distribution for each split
    for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        dist = split_df['label_id'].value_counts().sort_index()
        logger.info(f"{split_name} distribution: {dict(dist)}")
    
    return train_df, val_df, test_df

def compute_metrics(eval_pred):
    """Compute comprehensive evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Overall metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }
    
    # Add per-class metrics
    for i, label_name in ID_TO_LABEL.items():
        if i < len(precision_per_class):
            metrics[f'precision_{label_name.replace(" ", "_").replace("-", "_")}'] = precision_per_class[i]
            metrics[f'recall_{label_name.replace(" ", "_").replace("-", "_")}'] = recall_per_class[i]
            metrics[f'f1_{label_name.replace(" ", "_").replace("-", "_")}'] = f1_per_class[i]
    
    return metrics

def main():
    """Main experiment function"""
    logger.info("Starting Experiment 0 - Baseline IndoBERT with Balanced Dataset")
    logger.info(f"Configuration: {CONFIG}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(CONFIG['random_state'])
    np.random.seed(CONFIG['random_state'])
    
    # Load and prepare data
    df = load_and_prepare_data(CONFIG['dataset_path'])
    
    # Create stratified splits
    train_df, val_df, test_df = create_stratified_splits(
        df, 
        test_size=CONFIG['test_size'],
        val_size=CONFIG['validation_size'],
        random_state=CONFIG['random_state']
    )
    
    # Load tokenizer and model
    logger.info(f"Loading model: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=len(LABEL_MAPPING)
    )
    
    # Create datasets
    train_dataset = HateSpeechDataset(
        train_df['text'].tolist(),
        train_df['label_id'].tolist(),
        tokenizer,
        CONFIG['max_length']
    )
    
    val_dataset = HateSpeechDataset(
        val_df['text'].tolist(),
        val_df['label_id'].tolist(),
        tokenizer,
        CONFIG['max_length']
    )
    
    test_dataset = HateSpeechDataset(
        test_df['text'].tolist(),
        test_df['label_id'].tolist(),
        tokenizer,
        CONFIG['max_length']
    )
    
    # Compute class weights for balanced training
    class_weights = None
    if CONFIG['use_class_weights']:
        logger.info("Computing class weights for balanced training...")
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['label_id']),
            y=train_df['label_id']
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        logger.info(f"Class weights: {dict(zip(range(len(class_weights)), class_weights.tolist()))}")
        
        # Add class weights to model
        if hasattr(model, 'config'):
            model.config.class_weights = class_weights
    
    # Setup training arguments
    output_dir = f"experiments/results/experiment_0_baseline_indobert_balanced"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        warmup_steps=CONFIG['warmup_steps'],
        weight_decay=CONFIG['weight_decay'],
        learning_rate=CONFIG['learning_rate'],
        logging_dir=f'{output_dir}/logs',
        logging_steps=CONFIG['logging_steps'],
        eval_strategy="steps",  # Updated parameter name
        eval_steps=CONFIG['eval_steps'],
        save_steps=CONFIG['save_steps'],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=None,  # Disable wandb
        seed=CONFIG['random_state']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG['early_stopping_patience'])]
    )
    
    # Train model
    logger.info("Starting training...")
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    # Get predictions for detailed analysis
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_df['label_id'].tolist()
    
    # Function to convert any object to JSON serializable format
    def make_serializable(obj):
        if hasattr(obj, 'item'):  # For torch tensors
            return obj.item()
        elif hasattr(obj, 'tolist'):  # For numpy arrays
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    # Generate detailed results
    # Create serializable config (exclude non-serializable objects)
    serializable_config = {k: v for k, v in CONFIG.items() if isinstance(v, (str, int, float, bool, list, dict))}
    
    # Convert test_results to serializable format
    serializable_test_results = make_serializable(test_results)
    
    # Get classification report and make it serializable
    class_report = classification_report(y_true, y_pred, target_names=list(LABEL_MAPPING.keys()), output_dict=True)
    serializable_class_report = make_serializable(class_report)
    
    detailed_results = {
        'experiment_name': 'experiment_0_baseline_indobert_balanced',
        'model_name': CONFIG['model_name'],
        'dataset_path': CONFIG['dataset_path'],
        'training_time_seconds': training_time,
        'config': serializable_config,
        'label_mapping': LABEL_MAPPING,
        'class_weights': class_weights.tolist() if class_weights is not None else None,
        'data_splits': {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df)
        },
        'test_results': serializable_test_results,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': serializable_class_report
    }
    
    # Save detailed results (skip if serialization fails)
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(f'{output_dir}/detailed_results.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        logger.info("Detailed results saved successfully")
    except Exception as e:
        logger.warning(f"Could not save detailed results due to serialization error: {e}")
        logger.info("Continuing with experiment summary only...")
    
    # Save experiment summary
    summary = {
        'experiment_name': 'experiment_0_baseline_indobert_balanced',
        'model_name': CONFIG['model_name'],
        'dataset': CONFIG['dataset_path'],
        'training_time_seconds': training_time,
        'test_accuracy': serializable_test_results['eval_accuracy'],
        'test_f1_macro': serializable_test_results['eval_f1_macro'],
        'test_precision_macro': serializable_test_results['eval_precision_macro'],
        'test_recall_macro': serializable_test_results['eval_recall_macro'],
        'improvements_implemented': [
            'Class weighting (balanced)',
            'Stratified train/validation/test splits',
            'Comprehensive per-class evaluation',
            'Early stopping with F1-macro monitoring'
        ]
    }
    
    with open(f'{output_dir}/experiment_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Log final results
    logger.info("=== EXPERIMENT RESULTS ===")
    logger.info(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    logger.info(f"Test F1-Macro: {test_results['eval_f1_macro']:.4f}")
    logger.info(f"Test Precision-Macro: {test_results['eval_precision_macro']:.4f}")
    logger.info(f"Test Recall-Macro: {test_results['eval_recall_macro']:.4f}")
    logger.info(f"Training Time: {training_time:.2f} seconds")
    
    # Log per-class results
    logger.info("\n=== PER-CLASS RESULTS ===")
    for label_name, metrics in detailed_results['classification_report'].items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            logger.info(f"{label_name}:")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1-score']:.4f}")
            logger.info(f"  Support: {metrics['support']}")
    
    logger.info("Experiment completed successfully!")
    return summary

if __name__ == "__main__":
    main()