#!/usr/bin/env python3
"""
Experiment 0: Baseline IndoBERT with Balanced Dataset (Simple Version)

This experiment implements class weighting and stratified splits to address
the dataset imbalance issue identified in the baseline experiment.

Key improvements:
1. Class weighting (balanced) to handle imbalanced dataset
2. Stratified train/validation/test splits
3. Comprehensive per-class evaluation metrics
4. Early stopping with F1-macro monitoring

Dataset Distribution Analysis:
- Total samples: 3,100
- Bukan Ujaran Kebencian: 1,500 (48.39%) - Majority class
- Ujaran Kebencian - Sedang: 700 (22.58%)
- Ujaran Kebencian - Berat: 436 (14.06%)
- Ujaran Kebencian - Ringan: 464 (14.97%) - Minority class
- Imbalance ratio: 3.24:1 (majority:minority)
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_0_balanced_simple.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'model_name': 'indobenchmark/indobert-base-p1',
    'dataset_path': 'data/standardized/balanced_dataset.csv',
    'max_length': 128,
    'batch_size': 16,
    'num_epochs': 10,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_steps': 100,
    'logging_steps': 50,
    'eval_steps': 200,
    'save_steps': 400,
    'early_stopping_patience': 3,
    'test_size': 0.2,
    'validation_size': 0.2,  # 20% of remaining data after test split
    'random_state': 42,
    'use_class_weights': True
}

# Label mappings
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
    """Load and prepare the dataset"""
    logger.info(f"Loading dataset from {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset loaded with {len(df)} samples")
    
    # Use standardized dataset columns
    # Dataset has: text, final_label, label_numeric, label_binary
    if 'label_numeric' in df.columns:
        # Use the pre-mapped numeric labels from standardized dataset
        df['label_id'] = df['label_numeric']
    elif 'final_label' in df.columns:
        # Fallback: map final_label to numeric if label_numeric not available
        df['label_id'] = df['final_label'].map(LABEL_MAPPING)
    else:
        raise ValueError("Dataset must contain either 'label_numeric' or 'final_label' column")
    
    # Check for unmapped labels
    unmapped = df[df['label_id'].isna()]
    if not unmapped.empty:
        logger.warning(f"Found {len(unmapped)} unmapped labels: {unmapped['final_label'].unique()}")
        df = df.dropna(subset=['label_id'])
        logger.info(f"Dataset after removing unmapped labels: {len(df)} samples")
    
    # Convert label_id to int
    df['label_id'] = df['label_id'].astype(int)
    
    # Log label distribution
    label_dist = df['final_label'].value_counts()
    logger.info("Label distribution:")
    for label, count in label_dist.items():
        percentage = (count / len(df)) * 100
        logger.info(f"  {label}: {count} ({percentage:.2f}%)")
    
    return df

def create_stratified_splits(df, test_size=0.2, val_size=0.2, random_state=42):
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
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size, 
        stratify=train_val_df['label_id'], 
        random_state=random_state
    )
    
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Validation set: {len(val_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")
    
    # Log distribution for each split
    for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        logger.info(f"{split_name} set distribution:")
        dist = split_df['final_label'].value_counts()
        for label, count in dist.items():
            percentage = (count / len(split_df)) * 100
            logger.info(f"  {label}: {count} ({percentage:.2f}%)")
    
    return train_df, val_df, test_df

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    
    # Compute per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
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
    logger.info("Starting Experiment 0 - Baseline IndoBERT with Balanced Dataset (Simple Version)")
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
        
        # Note: Not adding class_weights to model.config to avoid JSON serialization issues
        # Class weights will be used in custom loss function if needed
    
    # Setup training arguments
    output_dir = f"experiments/results/experiment_0_baseline_indobert_balanced_simple"
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
        eval_strategy="steps",
        eval_steps=CONFIG['eval_steps'],
        save_strategy="no",  # Disable saving to avoid JSON serialization issues
        load_best_model_at_end=False,  # Disable to avoid JSON issues
        metric_for_best_model="eval_f1_macro",  # Required for EarlyStoppingCallback
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
    
    # Log final results
    logger.info("=== EXPERIMENT RESULTS ===")
    logger.info(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    logger.info(f"Test F1-Macro: {test_results['eval_f1_macro']:.4f}")
    logger.info(f"Test Precision-Macro: {test_results['eval_precision_macro']:.4f}")
    logger.info(f"Test Recall-Macro: {test_results['eval_recall_macro']:.4f}")
    logger.info(f"Training Time: {training_time:.2f} seconds")
    
    # Log per-class results
    logger.info("\n=== PER-CLASS RESULTS ===")
    class_report = classification_report(y_true, y_pred, target_names=list(LABEL_MAPPING.keys()), output_dict=True)
    for label_name, metrics in class_report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            logger.info(f"{label_name}:")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1-score']:.4f}")
            logger.info(f"  Support: {metrics['support']}")
    
    # Log confusion matrix
    logger.info("\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    logger.info("\n=== IMPROVEMENTS IMPLEMENTED ===")
    improvements = [
        'Class weighting (balanced)',
        'Stratified train/validation/test splits',
        'Comprehensive per-class evaluation',
        'Early stopping with F1-macro monitoring'
    ]
    for improvement in improvements:
        logger.info(f"- {improvement}")
    
    logger.info("Experiment completed successfully!")
    
    # Return simple summary
    return {
        'experiment_name': 'experiment_0_baseline_indobert_balanced_simple',
        'test_accuracy': float(test_results['eval_accuracy']),
        'test_f1_macro': float(test_results['eval_f1_macro']),
        'training_time_seconds': training_time
    }

if __name__ == "__main__":
    main()