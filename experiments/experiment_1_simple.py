#!/usr/bin/env python3
"""
Eksperimen 1: IndoBERT Large Fine-tuning (Simplified Version)
Tujuan: Meningkatkan baseline performance dengan model yang lebih besar
Target: F1-Score Macro >83% (peningkatan 3% dari baseline 80.36%)

Author: AI Research Assistant
Date: 3 Juli 2025
Project: Javanese Hate Speech Detection
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_1_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExperimentConfig:
    """Configuration for IndoBERT Large experiment"""
    
    # Model configuration
    MODEL_NAME = "indobenchmark/indobert-large-p1"
    MAX_LENGTH = 256
    NUM_LABELS = 4
    
    # Training configuration
    BATCH_SIZE = 16  # Increased for GPU
    GRADIENT_ACCUMULATION_STEPS = 1  # Reduced since we have larger batch size
    LEARNING_RATE = 2e-5  # Slightly higher for GPU training
    NUM_EPOCHS = 5  # Full epochs for better results
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 2
    EARLY_STOPPING_THRESHOLD = 0.001
    
    # Paths
    DATA_PATH = "../src/data_collection/hasil-labeling.csv"
    OUTPUT_DIR = "results/experiment_1_simple"
    MODEL_SAVE_PATH = "models/indobert_large_simple"
    
    # Label mapping
    LABEL_MAPPING = {
        0: "Bukan Ujaran Kebencian",
        1: "Ujaran Kebencian - Ringan",
        2: "Ujaran Kebencian - Sedang",
        3: "Ujaran Kebencian - Berat"
    }

class HateSpeechDataset(Dataset):
    """Custom dataset for hate speech classification"""
    
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

def load_and_preprocess_data(data_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and preprocess the dataset"""
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Dataset shape: {df.shape}")
    
    # Clean and prepare data - use final_label column
    df = df.dropna(subset=['text', 'final_label'])
    df['text'] = df['text'].astype(str)
    
    # Convert labels to numeric
    label_map = {
        'Bukan Ujaran Kebencian': 0,
        'Ujaran Kebencian - Ringan': 1,
        'Ujaran Kebencian - Sedang': 2,
        'Ujaran Kebencian - Berat': 3
    }
    df['label_numeric'] = df['final_label'].map(label_map)
    
    # Remove rows with unmapped labels
    df = df.dropna(subset=['label_numeric'])
    df['label_numeric'] = df['label_numeric'].astype(int)
    
    # Log class distribution
    class_counts = df['label_numeric'].value_counts().sort_index()
    logger.info("Class distribution:")
    for label, count in class_counts.items():
        logger.info(f"  {ExperimentConfig.LABEL_MAPPING[label]}: {count} ({count/len(df)*100:.2f}%)")
    
    return df, df['text'].values, df['label_numeric'].values

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

def detailed_evaluation(model, tokenizer, test_texts, test_labels, config):
    """Perform detailed evaluation"""
    logger.info("Starting detailed evaluation")
    
    # Create test dataset
    test_dataset = HateSpeechDataset(test_texts, test_labels, tokenizer, config.MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
    
    # Per-class metrics
    class_report = classification_report(all_labels, all_predictions, 
                                       target_names=list(config.LABEL_MAPPING.values()),
                                       output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    report = {
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall,
        'classification_report': class_report,
        'confusion_matrix': cm.tolist()
    }
    
    logger.info(f"Final Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1-Score (Macro): {f1:.4f}")
    logger.info(f"  Precision (Macro): {precision:.4f}")
    logger.info(f"  Recall (Macro): {recall:.4f}")
    
    return report

def save_results(report: Dict, config: ExperimentConfig, training_time: float):
    """Save experiment results"""
    logger.info("Saving experiment results")
    
    # Create output directory
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    report['experiment_metadata'] = {
        'model_name': config.MODEL_NAME,
        'max_length': config.MAX_LENGTH,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'num_epochs': config.NUM_EPOCHS,
        'training_time_seconds': training_time,
        'timestamp': datetime.now().isoformat(),
        'baseline_comparison': {
            'baseline_f1_macro': 0.8036,
            'current_f1_macro': report['f1_macro'],
            'improvement': report['f1_macro'] - 0.8036
        }
    }
    
    # Save results
    results_file = output_dir / 'experiment_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")

def main():
    """Main experiment execution"""
    logger.info("Starting Experiment 1: IndoBERT Large Fine-tuning (Simple)")
    start_time = time.time()
    
    config = ExperimentConfig()
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info("Mixed precision training: ENABLED")
    else:
        logger.warning("No GPU detected, using CPU (training will be slower)")
    
    try:
        # Load and preprocess data
        df, texts, labels = load_and_preprocess_data(config.DATA_PATH)
        
        # Create train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        # Load tokenizer and model
        logger.info(f"Loading tokenizer and model: {config.MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=config.NUM_LABELS
        )
        
        # Move model to GPU if available
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")
        
        # Create datasets
        train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, config.MAX_LENGTH)
        eval_dataset = HateSpeechDataset(X_test, y_test, tokenizer, config.MAX_LENGTH)
        
        # Training arguments with GPU optimization
        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            num_train_epochs=config.NUM_EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            warmup_ratio=config.WARMUP_RATIO,
            weight_decay=config.WEIGHT_DECAY,
            learning_rate=config.LEARNING_RATE,
            logging_dir=f'{config.OUTPUT_DIR}/logs',
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=2,
            report_to=None,  # Disable wandb/tensorboard
            dataloader_pin_memory=True,  # Enable for GPU
            fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
            dataloader_num_workers=2 if torch.cuda.is_available() else 0,  # Parallel data loading
            remove_unused_columns=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=config.EARLY_STOPPING_THRESHOLD
            )]
        )
        
        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Save model and tokenizer
        logger.info(f"Saving model to {config.MODEL_SAVE_PATH}")
        model_save_dir = Path(config.MODEL_SAVE_PATH)
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model(config.MODEL_SAVE_PATH)
        tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
        
        # Detailed evaluation
        report = detailed_evaluation(model, tokenizer, X_test, y_test, config)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Save results
        save_results(report, config, training_time)
        
        # Log final results
        logger.info(f"Experiment completed in {training_time:.2f} seconds")
        logger.info(f"Final F1-Score (Macro): {report['f1_macro']:.4f}")
        logger.info(f"Baseline comparison: {report['f1_macro'] - 0.8036:.4f} improvement")
        
        # Log per-class metrics
        logger.info("Per-class metrics:")
        for class_name, metrics in report['classification_report'].items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                logger.info(f"  {class_name}: F1={metrics['f1-score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()