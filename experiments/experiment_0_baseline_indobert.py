#!/usr/bin/env python3
"""
Experiment 0: IndoBERT Base Baseline
Tujuan: Membuat baseline dengan IndoBERT Base menggunakan dataset yang sama dengan Experiment 1.2
Target: Reproduksi hasil baseline 80.36% F1-Score Macro

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
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_0_baseline_indobert.log'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Ensure immediate output
import sys
sys.stdout.flush()
sys.stderr.flush()

class BaselineConfig:
    """Configuration for IndoBERT Base baseline experiment"""
    
    # Model configuration
    MODEL_NAME = "indobenchmark/indobert-base-p1"
    MAX_LENGTH = 128
    NUM_LABELS = 4
    
    # Training configuration
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 1
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_THRESHOLD = 0.001
    
    # Paths
    DATA_PATH = "data/standardized/balanced_dataset.csv"
    OUTPUT_DIR = "experiments/results/experiment_0_baseline_indobert"
    MODEL_SAVE_PATH = "models/indobert_baseline_hate_speech"
    
    # Class weights for baseline (balanced approach)
    CLASS_WEIGHTS = {
        0: 1.0,    # Bukan Ujaran Kebencian
        1: 11.3,   # Ujaran Kebencian - Ringan
        2: 17.0,   # Ujaran Kebencian - Sedang
        3: 34.0    # Ujaran Kebencian - Berat
    }
    
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

class WeightedFocalLoss(nn.Module):
    """Focal Loss with class weights for handling imbalanced data"""
    
    def __init__(self, alpha: Dict[int, float], gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Apply class weights
        alpha_weights = torch.tensor([self.alpha[i] for i in targets.cpu().numpy()]).to(inputs.device)
        focal_loss = alpha_weights * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class CustomTrainer(Trainer):
    """Custom trainer with weighted focal loss"""
    
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

def load_and_preprocess_data(data_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and preprocess the dataset"""
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Log first few rows for debugging
    logger.info(f"First 3 rows of dataframe:")
    logger.info(df.head(3).to_string())
    
    # Use standardized dataset columns
    # Dataset has: text, final_label, label_numeric, label_binary
    # We can directly use label_numeric which is already mapped correctly
    if 'label_numeric' in df.columns:
        # Use the pre-mapped numeric labels from standardized dataset
        df = df[['text', 'label_numeric']].copy()
        df = df.rename(columns={'label_numeric': 'label'})
    elif 'final_label' in df.columns:
        # Fallback: map final_label to numeric if label_numeric not available
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
    
    logger.info(f"After cleaning - Dataset shape: {df.shape}")
    
    # Log class distribution
    class_counts = df['label'].value_counts().sort_index()
    logger.info("Class distribution:")
    for label, count in class_counts.items():
        logger.info(f"  {BaselineConfig.LABEL_MAPPING[label]}: {count} ({count/len(df)*100:.2f}%)")
    
    return df, df['text'].values, df['label'].values

def create_stratified_split(texts: np.ndarray, labels: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """Create stratified train-test split"""
    logger.info("Creating stratified train-test split")
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    logger.info(f"Train set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Log train distribution
    train_counts = pd.Series(y_train).value_counts().sort_index()
    logger.info("Training set distribution:")
    for label, count in train_counts.items():
        logger.info(f"  {BaselineConfig.LABEL_MAPPING[label]}: {count} ({count/len(y_train)*100:.2f}%)")
    
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

def detailed_evaluation(model, tokenizer, X_test, y_test, config: BaselineConfig):
    """Perform detailed evaluation of the model"""
    logger.info("Performing detailed evaluation")
    
    # Create test dataset
    test_dataset = HateSpeechDataset(X_test, y_test, tokenizer, config.MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Get predictions
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            all_predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    # Per-class metrics
    class_report = classification_report(all_labels, all_predictions, 
                                       target_names=list(config.LABEL_MAPPING.values()),
                                       output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'f1_macro': f1,
        'f1_weighted': f1_w,
        'precision_macro': precision,
        'recall_macro': recall,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'predictions': all_predictions,
        'true_labels': all_labels,
        'probabilities': all_probabilities
    }
    
    return results

def save_results(report: Dict, config: BaselineConfig, training_time: float):
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
        'experiment_type': 'baseline_indobert_base',
        'target_comparison': {
            'target_f1_macro': 0.8036,
            'current_f1_macro': report['f1_macro'],
            'difference': report['f1_macro'] - 0.8036
        }
    }
    
    # Save detailed results
    results_file = output_dir / 'detailed_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    # Save summary
    summary = {
        'experiment': 'Baseline IndoBERT Base',
        'model': config.MODEL_NAME,
        'dataset': config.DATA_PATH,
        'results': {
            'accuracy': f"{report['accuracy']:.4f}",
            'f1_macro': f"{report['f1_macro']:.4f}",
            'precision_macro': f"{report['precision_macro']:.4f}",
            'recall_macro': f"{report['recall_macro']:.4f}"
        },
        'training_time': f"{training_time:.2f} seconds",
        'timestamp': datetime.now().isoformat()
    }
    
    summary_file = output_dir / 'experiment_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_dir}")
    return output_dir

def main():
    """Main experiment function"""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 0: BASELINE INDOBERT BASE")
    logger.info("=" * 60)
    
    config = BaselineConfig()
    
    try:
        # Load and preprocess data
        df, texts, labels = load_and_preprocess_data(config.DATA_PATH)
        
        # Create train-test split
        X_train, X_test, y_train, y_test = create_stratified_split(texts, labels)
        
        # Load tokenizer and model
        logger.info(f"Loading tokenizer and model: {config.MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=config.NUM_LABELS
        )
        
        # Create datasets
        logger.info("Creating datasets")
        train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, config.MAX_LENGTH)
        val_dataset = HateSpeechDataset(X_test, y_test, tokenizer, config.MAX_LENGTH)
        
        # Training arguments
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
            save_strategy="no",  # Disable checkpoint saving during training
            load_best_model_at_end=False,  # Disable to avoid checkpoint issues
            report_to=None,
            dataloader_pin_memory=False,
        )
        
        # Create trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            class_weights=config.CLASS_WEIGHTS,
            # Remove EarlyStoppingCallback to avoid checkpoint issues
        )
        
        # Train model
        logger.info("Starting training...")
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save model
        logger.info(f"Saving model to {config.MODEL_SAVE_PATH}")
        model_save_path = Path(config.MODEL_SAVE_PATH)
        model_save_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(config.MODEL_SAVE_PATH)
        tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
        
        # Detailed evaluation
        logger.info("Performing detailed evaluation")
        evaluation_results = detailed_evaluation(model, tokenizer, X_test, y_test, config)
        
        # Save results
        output_dir = save_results(evaluation_results, config, training_time)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"F1-Score Macro: {evaluation_results['f1_macro']:.4f}")
        logger.info(f"Precision Macro: {evaluation_results['precision_macro']:.4f}")
        logger.info(f"Recall Macro: {evaluation_results['recall_macro']:.4f}")
        logger.info(f"Training Time: {training_time:.2f} seconds")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Model saved to: {config.MODEL_SAVE_PATH}")
        
        # Comparison with target
        target_f1 = 0.8036
        current_f1 = evaluation_results['f1_macro']
        difference = current_f1 - target_f1
        
        logger.info("=" * 60)
        logger.info("COMPARISON WITH TARGET BASELINE")
        logger.info("=" * 60)
        logger.info(f"Target F1-Score Macro: {target_f1:.4f}")
        logger.info(f"Current F1-Score Macro: {current_f1:.4f}")
        logger.info(f"Difference: {difference:+.4f}")
        
        if difference >= 0:
            logger.info("[SUCCESS] BASELINE TARGET ACHIEVED OR EXCEEDED!")
        else:
            logger.info("[INFO] Baseline target not reached")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()