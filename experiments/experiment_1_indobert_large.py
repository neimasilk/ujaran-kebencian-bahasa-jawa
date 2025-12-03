#!/usr/bin/env python3
"""
Eksperimen 1: IndoBERT Large Fine-tuning
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
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_1_indobert_large.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExperimentConfig:
    """Configuration for IndoBERT Large experiment"""
    
    # Model configuration
    MODEL_NAME = "indobenchmark/indobert-large-p1"
    MAX_LENGTH = 256  # Increased from 128
    NUM_LABELS = 4
    
    # Training configuration
    BATCH_SIZE = 4  # Further reduced for IndoBERT Large
    GRADIENT_ACCUMULATION_STEPS = 4  # Increased to maintain effective batch size
    LEARNING_RATE = 2e-5  # Increased from 1e-5 for better convergence
    NUM_EPOCHS = 5
    WARMUP_RATIO = 0.2  # Increased from 0.1 for better warmup
    WEIGHT_DECAY = 0.01
    
    # Early stopping (FIXED: More lenient settings)
    EARLY_STOPPING_PATIENCE = 5  # Increased from 2 to 5
    EARLY_STOPPING_THRESHOLD = 0.01  # Increased from 0.001 to 0.01
    
    # Paths
    DATA_PATH = "data/standardized/balanced_dataset.csv"
    OUTPUT_DIR = "experiments/results/experiment_1_indobert_large"
    MODEL_SAVE_PATH = "models/indobert_large_hate_speech"
    
    # Class weights (FIXED: More balanced weights)
    CLASS_WEIGHTS = {
        0: 1.0,    # Bukan Ujaran Kebencian
        1: 3.0,    # Ujaran Kebencian - Ringan (Reduced from 8.5)
        2: 2.5,    # Ujaran Kebencian - Sedang (Reduced from 15.2)
        3: 3.5     # Ujaran Kebencian - Berat (Reduced from 25.8)
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

def detailed_evaluation(model, tokenizer, X_test, y_test, config: ExperimentConfig):
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
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )
    macro_f1 = precision_recall_fscore_support(all_labels, all_predictions, average='macro')[2]
    
    # Create detailed report
    report = {
        'accuracy': accuracy,
        'f1_macro': macro_f1,
        'per_class_metrics': {},
        'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),
        'classification_report': classification_report(all_labels, all_predictions, target_names=list(config.LABEL_MAPPING.values()))
    }
    
    for i, label_name in config.LABEL_MAPPING.items():
        if i < len(precision):
            report['per_class_metrics'][label_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
    
    return report, all_predictions, all_probabilities

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
    
    # Save detailed results
    with open(output_dir / 'experiment_1_results.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    cm = np.array(report['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(config.LABEL_MAPPING.values()),
                yticklabels=list(config.LABEL_MAPPING.values()))
    plt.title('Confusion Matrix - IndoBERT Large')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Results saved to {output_dir}")

def main():
    """Main experiment execution"""
    logger.info("Starting Experiment 1: IndoBERT Large Fine-tuning")
    start_time = time.time()
    
    config = ExperimentConfig()
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Load and preprocess data
        df, texts, labels = load_and_preprocess_data(config.DATA_PATH)
        
        # Create train-test split
        X_train, X_test, y_train, y_test = create_stratified_split(texts, labels)
        
        # Initialize tokenizer and model
        logger.info(f"Loading tokenizer and model: {config.MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=config.NUM_LABELS
        )
        
        # Create datasets
        train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, config.MAX_LENGTH)
        eval_dataset = HateSpeechDataset(X_test, y_test, tokenizer, config.MAX_LENGTH)
        
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
            logging_dir=f"{config.OUTPUT_DIR}/logs",
            logging_steps=50,
            eval_steps=50,  # Reduced from 100 for more frequent monitoring
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=3,
            report_to=None,  # Disable wandb
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        )
        
        # Initialize trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            class_weights=config.CLASS_WEIGHTS,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=config.EARLY_STOPPING_THRESHOLD
            )]
        )
        
        # Train the model
        logger.info("Starting training...")
        train_start = time.time()
        trainer.train()
        training_time = time.time() - train_start
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save the model
        logger.info(f"Saving model to {config.MODEL_SAVE_PATH}")
        trainer.save_model(config.MODEL_SAVE_PATH)
        tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
        
        # Detailed evaluation
        report, predictions, probabilities = detailed_evaluation(
            model, tokenizer, X_test, y_test, config
        )
        
        # Save results
        save_results(report, config, training_time)
        
        # Log final results
        logger.info("=" * 50)
        logger.info("EXPERIMENT 1 RESULTS")
        logger.info("=" * 50)
        logger.info(f"Accuracy: {report['accuracy']:.4f}")
        logger.info(f"F1-Score Macro: {report['f1_macro']:.4f}")
        logger.info(f"Baseline F1-Score: 0.8036")
        logger.info(f"Improvement: {report['f1_macro'] - 0.8036:.4f}")
        logger.info(f"Training Time: {training_time:.2f} seconds")
        
        logger.info("\nPer-class Results:")
        for class_name, metrics in report['per_class_metrics'].items():
            logger.info(f"  {class_name}:")
            logger.info(f"    F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall: {metrics['recall']:.4f}")
        
        total_time = time.time() - start_time
        logger.info(f"\nTotal experiment time: {total_time:.2f} seconds")
        logger.info("Experiment 1 completed successfully!")
        
        return report
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()