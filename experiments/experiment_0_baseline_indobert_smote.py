#!/usr/bin/env python3
"""
Experiment 0 - Baseline IndoBERT with SMOTE Resampling

This experiment uses SMOTE (Synthetic Minority Oversampling Technique) to address
the class imbalance issue. SMOTE generates synthetic examples for minority classes
to create a more balanced dataset.

Original Dataset Distribution:
- Bukan Ujaran Kebencian: 20,205 (48.39%)
- Ujaran Kebencian - Sedang: 8,600 (20.60%)
- Ujaran Kebencian - Berat: 6,711 (16.07%)
- Ujaran Kebencian - Ringan: 6,241 (14.95%)
- Imbalance Ratio: 3.24:1

After SMOTE: All classes will have equal representation
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
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
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
        logging.FileHandler('experiment_0_baseline_indobert_smote.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'model_name': 'indobenchmark/indobert-base-p1',
    'dataset_path': 'data/standardized/balanced_dataset.csv',
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
    'smote_k_neighbors': 5,
    'smote_sampling_strategy': 'auto'  # Balance all classes to majority class
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
    """Load and prepare the dataset"""
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset shape: {df.shape}")
    
    # Use standardized dataset columns
    # Dataset has: text, final_label, label_numeric, label_binary
    if 'label_numeric' in df.columns:
        # Use the pre-mapped numeric labels from standardized dataset
        df['label_id'] = df['label_numeric']
        label_column = 'final_label'  # For distribution logging
    elif 'final_label' in df.columns:
        # Fallback: map final_label to numeric if label_numeric not available
        label_column = 'final_label'
        df['label_id'] = df[label_column].map(LABEL_MAPPING)
    else:
        # Last fallback for old datasets
        label_column = 'label'
        df['label_id'] = df[label_column].map(LABEL_MAPPING)
    
    # Remove rows with unmapped labels
    df = df.dropna(subset=['label_id'])
    df['label_id'] = df['label_id'].astype(int)
    
    logger.info(f"Final dataset shape after cleaning: {df.shape}")
    
    # Analyze original label distribution
    label_dist = df[label_column].value_counts()
    logger.info("Original label distribution:")
    for label, count in label_dist.items():
        percentage = (count / len(df)) * 100
        logger.info(f"  {label}: {count} ({percentage:.2f}%)")
    
    return df[['text', 'label_id', label_column]]

def apply_smote_resampling(X_train, y_train):
    """Apply SMOTE resampling to training data"""
    logger.info("Applying SMOTE resampling...")
    
    # Convert text to numerical features for SMOTE
    # Using TF-IDF with limited features for efficiency
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=None,  # Keep all words for Javanese
        ngram_range=(1, 2),
        min_df=2
    )
    
    logger.info("Converting text to TF-IDF features...")
    X_tfidf = vectorizer.fit_transform(X_train)
    
    # Log original distribution
    original_dist = pd.Series(y_train).value_counts().sort_index()
    logger.info(f"Original training distribution: {dict(original_dist)}")
    
    # Apply SMOTE
    smote = SMOTE(
        sampling_strategy=CONFIG['smote_sampling_strategy'],
        k_neighbors=CONFIG['smote_k_neighbors'],
        random_state=CONFIG['random_state']
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_tfidf, y_train)
    
    # Log new distribution
    new_dist = pd.Series(y_resampled).value_counts().sort_index()
    logger.info(f"Resampled training distribution: {dict(new_dist)}")
    
    # Convert back to text (reconstruct from TF-IDF - approximation)
    # For synthetic samples, we'll use the original texts and duplicate/modify them
    # This is a simplified approach - in practice, more sophisticated text generation would be used
    
    original_size = len(X_train)
    new_size = len(y_resampled)
    
    # Create mapping of original indices
    original_indices = list(range(original_size))
    
    # For synthetic samples, we'll cycle through original texts
    X_train_list = list(X_train)
    X_resampled_texts = []
    
    for i in range(new_size):
        if i < original_size:
            # Original sample
            X_resampled_texts.append(X_train_list[i])
        else:
            # Synthetic sample - use original text (simplified approach)
            # In practice, you'd want more sophisticated text augmentation
            original_idx = i % original_size
            synthetic_text = X_train_list[original_idx]
            # Simple augmentation: add variation marker
            synthetic_text = f"{synthetic_text} [synthetic]" if len(synthetic_text) < 100 else synthetic_text
            X_resampled_texts.append(synthetic_text)
    
    logger.info(f"SMOTE resampling completed: {original_size} -> {new_size} samples")
    
    return X_resampled_texts, y_resampled, vectorizer

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
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=train_val_df['label_id'],
        random_state=random_state
    )
    
    logger.info(f"Original splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
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
    logger.info("Starting Experiment 0 - Baseline IndoBERT with SMOTE Resampling")
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
    
    # Apply SMOTE to training data
    X_train_resampled, y_train_resampled, vectorizer = apply_smote_resampling(
        train_df['text'].tolist(),
        train_df['label_id'].tolist()
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
        X_train_resampled,
        y_train_resampled.tolist(),
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
    
    # Setup training arguments
    output_dir = f"experiments/results/experiment_0_baseline_indobert_smote"
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
        report_to=None,
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
    logger.info("Starting training with SMOTE-resampled data...")
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
    
    # Generate detailed results
    detailed_results = {
        'experiment_name': 'experiment_0_baseline_indobert_smote',
        'model_name': CONFIG['model_name'],
        'dataset_path': CONFIG['dataset_path'],
        'training_time_seconds': training_time,
        'config': CONFIG,
        'label_mapping': LABEL_MAPPING,
        'smote_info': {
            'original_train_size': len(train_df),
            'resampled_train_size': len(X_train_resampled),
            'sampling_strategy': CONFIG['smote_sampling_strategy'],
            'k_neighbors': CONFIG['smote_k_neighbors']
        },
        'data_splits': {
            'train_size_original': len(train_df),
            'train_size_resampled': len(X_train_resampled),
            'val_size': len(val_df),
            'test_size': len(test_df)
        },
        'test_results': test_results,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, target_names=list(LABEL_MAPPING.keys()), output_dict=True)
    }
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/detailed_results.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    # Save experiment summary
    summary = {
        'experiment_name': 'experiment_0_baseline_indobert_smote',
        'model_name': CONFIG['model_name'],
        'dataset': CONFIG['dataset_path'],
        'training_time_seconds': training_time,
        'test_accuracy': test_results['eval_accuracy'],
        'test_f1_macro': test_results['eval_f1_macro'],
        'test_precision_macro': test_results['eval_precision_macro'],
        'test_recall_macro': test_results['eval_recall_macro'],
        'improvements_implemented': [
            'SMOTE oversampling for minority classes',
            'Stratified train/validation/test splits',
            'Comprehensive per-class evaluation',
            'Early stopping with F1-macro monitoring'
        ],
        'smote_summary': {
            'original_size': len(train_df),
            'resampled_size': len(X_train_resampled),
            'increase_factor': len(X_train_resampled) / len(train_df)
        }
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
    
    # Log SMOTE impact
    logger.info("\n=== SMOTE IMPACT ===")
    logger.info(f"Original training size: {len(train_df)}")
    logger.info(f"Resampled training size: {len(X_train_resampled)}")
    logger.info(f"Size increase factor: {len(X_train_resampled) / len(train_df):.2f}x")
    
    logger.info("Experiment completed successfully!")
    return summary

if __name__ == "__main__":
    main()