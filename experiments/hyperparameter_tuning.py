#!/usr/bin/env python3
"""
Hyperparameter Tuning for IndoBERT Hate Speech Detection
Systematically searches for optimal hyperparameters

Author: AI Assistant
Date: 2025-07-24
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hyperparameter_tuning.log'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Ensure immediate output
import sys
sys.stdout.flush()
sys.stderr.flush()

class HyperparameterConfig:
    """Configuration for hyperparameter tuning"""
    
    # Model configuration
    MODEL_NAME = "indobenchmark/indobert-base-p1"
    MAX_LENGTH = 128
    NUM_LABELS = 4
    
    # Hyperparameter search space
    LEARNING_RATES = [1e-5, 2e-5, 3e-5, 5e-5]
    BATCH_SIZES = [8, 16, 32]
    NUM_EPOCHS_OPTIONS = [3, 5]
    WARMUP_RATIOS = [0.05, 0.1, 0.15]
    
    # Paths
    DATA_PATH = "data/standardized/balanced_dataset.csv"
    OUTPUT_DIR = "experiments/results/hyperparameter_tuning"
    
    # Label mapping
    LABEL_MAPPING = {
        0: "Bukan Ujaran Kebencian",
        1: "Ujaran Kebencian - Ringan",
        2: "Ujaran Kebencian - Sedang",
        3: "Ujaran Kebencian - Berat"
    }

class HateSpeechDataset(Dataset):
    """Dataset for hate speech classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
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

def load_and_preprocess_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the dataset"""
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Use standardized dataset columns
    if 'label_numeric' in df.columns:
        df = df[['text', 'label_numeric']].copy()
        df = df.rename(columns={'label_numeric': 'label'})
    elif 'final_label' in df.columns:
        label_mapping = {
            'Bukan Ujaran Kebencian': 0,
            'Ujaran Kebencian - Ringan': 1,
            'Ujaran Kebencian - Sedang': 2,
            'Ujaran Kebencian - Berat': 3
        }
        df['label_id'] = df['final_label'].map(label_mapping)
        df = df[['text', 'label_id']].copy()
        df = df.rename(columns={'label_id': 'label'})
    else:
        raise ValueError("Dataset must contain either 'label_numeric' or 'final_label' column")
    
    # Clean data
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)
    
    return df['text'].values, df['label'].values
def create_stratified_split(texts: np.ndarray, labels: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """Create stratified train-test split"""
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }

def run_experiment(learning_rate: float, batch_size: int, num_epochs: int, 
                   warmup_ratio: float, config: HyperparameterConfig) -> Dict:
    """Run a single experiment with given hyperparameters"""
    
    try:
        # Load and preprocess data
        texts, labels = load_and_preprocess_data(config.DATA_PATH)
        X_train, X_test, y_train, y_test = create_stratified_split(texts, labels)
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=config.NUM_LABELS
        )
        
        # Create datasets
        train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, config.MAX_LENGTH)
        val_dataset = HateSpeechDataset(X_test, y_test, tokenizer, config.MAX_LENGTH)
        
        # Unique experiment identifier
        exp_id = f"lr{learning_rate}_bs{batch_size}_ep{num_epochs}_wr{warmup_ratio}".replace(".", "_")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{config.OUTPUT_DIR}/{exp_id}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=warmup_ratio,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            report_to=None,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train model
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        # Evaluate model
        eval_results = trainer.evaluate()
        
        # Return results
        return {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'warmup_ratio': warmup_ratio,
            'accuracy': eval_results['eval_accuracy'],
            'f1_macro': eval_results['eval_f1_macro'],
            'precision_macro': eval_results['eval_precision_macro'],
            'recall_macro': eval_results['eval_recall_macro'],
            'training_time': training_time,
            'exp_id': exp_id
        }
        
    except Exception as e:
        logger.error(f"Experiment failed with params LR={learning_rate}, BS={batch_size}, EP={num_epochs}, WR={warmup_ratio}: {str(e)}")
        return {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'warmup_ratio': warmup_ratio,
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'training_time': 0.0,
            'exp_id': f"failed_lr{learning_rate}_bs{batch_size}_ep{num_epochs}_wr{warmup_ratio}".replace(".", "_"),
            'error': str(e)
        }

def main():
    """Main hyperparameter tuning function"""
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING FOR INDOBERT HATE SPEECH DETECTION")
    logger.info("=" * 60)
    
    config = HyperparameterConfig()
    
    # Create output directory
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # Grid search over hyperparameters
    total_experiments = (
        len(config.LEARNING_RATES) * 
        len(config.BATCH_SIZES) * 
        len(config.NUM_EPOCHS_OPTIONS) * 
        len(config.WARMUP_RATIOS)
    )
    
    logger.info(f"Starting grid search with {total_experiments} experiments")
    
    experiment_count = 0
    best_f1 = 0.0
    best_config = None
    
    for lr in config.LEARNING_RATES:
        for bs in config.BATCH_SIZES:
            for ep in config.NUM_EPOCHS_OPTIONS:
                for wr in config.WARMUP_RATIOS:
                    experiment_count += 1
                    logger.info(f"Running experiment {experiment_count}/{total_experiments}")
                    logger.info(f"Params: LR={lr}, BS={bs}, EP={ep}, WR={wr}")
                    
                    # Run experiment
                    result = run_experiment(lr, bs, ep, wr, config)
                    all_results.append(result)
                    
                    # Track best result
                    if result['f1_macro'] > best_f1:
                        best_f1 = result['f1_macro']
                        best_config = {
                            'learning_rate': lr,
                            'batch_size': bs,
                            'num_epochs': ep,
                            'warmup_ratio': wr
                        }
                    
                    # Save intermediate results
                    if experiment_count % 5 == 0:
                        results_file = Path(config.OUTPUT_DIR) / 'intermediate_results.json'
                        with open(results_file, 'w') as f:
                            json.dump(all_results, f, indent=2)
                        logger.info(f"Saved intermediate results after {experiment_count} experiments")
    
    # Save final results
    results_file = Path(config.OUTPUT_DIR) / 'final_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Sort results by F1-macro
    sorted_results = sorted(all_results, key=lambda x: x['f1_macro'], reverse=True)
    
    # Print top 5 configurations
    logger.info("=" * 60)
    logger.info("TOP 5 HYPERPARAMETER CONFIGURATIONS")
    logger.info("=" * 60)
    
    for i, result in enumerate(sorted_results[:5]):
        logger.info(f"\n{i+1}. F1-Macro: {result['f1_macro']:.4f}")
        logger.info(f"   Accuracy: {result['accuracy']:.4f}")
        logger.info(f"   Config: LR={result['learning_rate']}, BS={result['batch_size']}, "
                   f"EP={result['num_epochs']}, WR={result['warmup_ratio']}")
        logger.info(f"   Training Time: {result['training_time']:.2f}s")
    
    # Print best configuration
    logger.info("=" * 60)
    logger.info("BEST CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"F1-Macro: {best_f1:.4f}")
    logger.info(f"Learning Rate: {best_config['learning_rate']}")
    logger.info(f"Batch Size: {best_config['batch_size']}")
    logger.info(f"Num Epochs: {best_config['num_epochs']}")
    logger.info(f"Warmup Ratio: {best_config['warmup_ratio']}")
    
    # Save best configuration
    best_config_file = Path(config.OUTPUT_DIR) / 'best_configuration.json'
    with open(best_config_file, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING COMPLETED")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()