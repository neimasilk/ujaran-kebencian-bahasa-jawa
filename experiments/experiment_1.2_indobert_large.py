#!/usr/bin/env python3
"""
Experiment 1.2: IndoBERT Large Fine-tuning - Optimized Configuration
Objective: Achieve F1-Score Macro >50% (intermediate target towards 83%)

Key Improvements from Experiment 1.1:
1. Increased learning rate (2e-5 → 3e-5)
2. Gradient accumulation for larger effective batch size
3. Extended early stopping patience (5 → 8)
4. Optimized warmup and scheduling
5. Enhanced monitoring and logging

Author: AI Assistant
Date: July 3, 2025
"""

import os
import sys
import logging
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_1.2_indobert_large.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExperimentConfig:
    """Enhanced configuration for Experiment 1.2"""
    
    # Model parameters
    MODEL_NAME = "indobenchmark/indobert-large-p1"
    MAX_LENGTH = 128
    NUM_LABELS = 4
    
    # Optimized training parameters
    BATCH_SIZE = 8  # Per device batch size
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 8 * 4 = 32
    LEARNING_RATE = 3e-5  # Increased from 2e-5
    NUM_EPOCHS = 5
    WARMUP_STEPS = 500  # Increased from ratio-based
    WEIGHT_DECAY = 0.01  # Added regularization
    
    # Enhanced early stopping
    EARLY_STOPPING_PATIENCE = 8  # Increased from 5
    EARLY_STOPPING_THRESHOLD = 0.005  # Reduced from 0.01
    
    # Evaluation and logging
    EVAL_STEPS = 50
    LOGGING_STEPS = 25  # More frequent logging
    SAVE_STEPS = 100
    
    # File paths
    DATA_PATH = "data/processed/final_dataset.csv"
    RESULTS_DIR = "experiments/results/experiment_1.2_indobert_large"
    MODEL_SAVE_PATH = "models/experiment_1.2_indobert_large"
    
    # Optimized class weights (less aggressive than 1.1)
    CLASS_WEIGHTS = {
        0: 1.0,    # Bukan Ujaran Kebencian (baseline)
        1: 2.5,    # Ujaran Kebencian - Ringan (reduced from 3.0)
        2: 2.0,    # Ujaran Kebencian - Sedang (reduced from 2.5)
        3: 2.8     # Ujaran Kebencian - Berat (reduced from 3.5)
    }
    
    # Label mapping
    LABEL_MAPPING = {
        0: "Bukan Ujaran Kebencian",
        1: "Ujaran Kebencian - Ringan",
        2: "Ujaran Kebencian - Sedang",
        3: "Ujaran Kebencian - Berat"
    }

class HateSpeechDataset(Dataset):
    """Enhanced dataset class with better error handling"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        # Ensure labels are integers
        self.labels = np.array(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Dataset created with {len(texts)} samples")
        logger.info(f"Label distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize with proper error handling
        try:
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
        except Exception as e:
            logger.error(f"Error tokenizing text at index {idx}: {e}")
            # Return a safe default
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.tensor(0, dtype=torch.long)
            }

class IndoBERTDataset(Dataset):
    """Optimized dataset for IndoBERT with enhanced preprocessing"""
    
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Enhanced text preprocessing
        self.data['text'] = self.data['text'].apply(self._preprocess_text)
        
        logger.info(f"IndoBERTDataset initialized with {len(self.data)} samples")
    
    def _preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Ensure minimum length
        if len(text) < 3:
            text = "teks kosong"
        
        return text
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        label = row['label']
        
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
    """Enhanced Focal Loss with optimized parameters"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma  # Reduced from 2.5 to 2.0 for less aggressive focusing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CustomTrainer(Trainer):
    """Enhanced trainer with improved loss computation"""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        if class_weights is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.class_weights = torch.tensor(list(class_weights.values()), 
                                            dtype=torch.float32).to(device)
            self.loss_fn = WeightedFocalLoss(alpha=self.class_weights, gamma=2.0)
            logger.info(f"Custom loss function initialized with weights: {class_weights}")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss = self.loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

def load_and_prepare_data(data_path):
    """Enhanced data loading with better error handling"""
    try:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Enhanced data validation
        logger.info(f"Original dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"First few rows:")
        logger.info(df.head())
        
        # Use the correct columns from the CSV structure
        # CSV has: text, label, label_id, confidence_score, labeling_method, response_time
        # We need 'text' and 'label_id' (which is already numeric)
        if 'text' not in df.columns or 'label_id' not in df.columns:
            raise ValueError(f"Required columns 'text' and 'label_id' not found. Available columns: {df.columns.tolist()}")
        
        logger.info(f"Using columns: text and label_id")
        logger.info(f"Available columns: {df.columns.tolist()}")
        logger.info(f"Sample label values: {df['label'].head()}")
        logger.info(f"Sample label_id values: {df['label_id'].head()}")
        
        # Clean data
        initial_size = len(df)
        
        # Keep only text and label_id columns, rename label_id to label for consistency
        df = df[['text', 'label_id']].copy()
        df = df.rename(columns={'label_id': 'label'})
        
        # Clean missing data
        df = df.dropna(subset=['text', 'label'])
        df = df[df['text'].str.strip() != '']
        
        # Ensure labels are integers (label_id should already be numeric)
        df['label'] = df['label'].astype(int)
        
        logger.info(f"Cleaned dataset: {len(df)} samples (removed {initial_size - len(df)} invalid entries)")
        
        # Validate labels
        unique_labels = sorted(df['label'].unique())
        expected_labels = [0, 1, 2, 3]
        if not set(unique_labels).issubset(set(expected_labels)):
            logger.warning(f"Unexpected labels found: {unique_labels}, expected subset of: {expected_labels}")
        
        # Log class distribution
        class_dist = df['label'].value_counts().sort_index()
        logger.info(f"Class distribution: {class_dist.to_dict()}")
        
        return df[['text', 'label']]  # Return only needed columns
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_stratified_split(df, test_size=0.2, random_state=42):
    """Create stratified train-test split with enhanced logging"""
    try:
        X = df['text'].values
        y = df['label'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Log class distribution for both sets
        train_dist = pd.Series(y_train).value_counts().sort_index()
        test_dist = pd.Series(y_test).value_counts().sort_index()
        
        logger.info(f"Train class distribution: {train_dist.to_dict()}")
        logger.info(f"Test class distribution: {test_dist.to_dict()}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error creating stratified split: {e}")
        raise

def compute_metrics(eval_pred):
    """Enhanced metrics computation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }
    
    # Add per-class metrics
    for i, label_name in ExperimentConfig.LABEL_MAPPING.items():
        if i < len(f1_per_class):
            metrics[f'f1_{label_name.lower().replace(" ", "_").replace("-", "_")}'] = f1_per_class[i]
            metrics[f'precision_{label_name.lower().replace(" ", "_").replace("-", "_")}'] = precision_per_class[i]
            metrics[f'recall_{label_name.lower().replace(" ", "_").replace("-", "_")}'] = recall_per_class[i]
    
    return metrics

def save_confusion_matrix(y_true, y_pred, save_path):
    """Enhanced confusion matrix visualization"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=list(ExperimentConfig.LABEL_MAPPING.values()),
            yticklabels=list(ExperimentConfig.LABEL_MAPPING.values())
        )
        plt.title('Confusion Matrix - Experiment 1.2: IndoBERT Large', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Error saving confusion matrix: {e}")

def main():
    """Enhanced main experiment execution"""
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("EXPERIMENT 1.2: IndoBERT Large Fine-tuning - Optimized Configuration")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now()}")
    
    # Create results directory
    os.makedirs(ExperimentConfig.RESULTS_DIR, exist_ok=True)
    os.makedirs(ExperimentConfig.MODEL_SAVE_PATH, exist_ok=True)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Load and prepare data
        df = load_and_prepare_data(ExperimentConfig.DATA_PATH)
        X_train, X_test, y_train, y_test = create_stratified_split(df)
        
        # Initialize tokenizer and model
        logger.info(f"Loading tokenizer and model: {ExperimentConfig.MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(ExperimentConfig.MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            ExperimentConfig.MODEL_NAME,
            num_labels=ExperimentConfig.NUM_LABELS
        )
        
        # Create datasets
        train_texts = [str(text) for text in X_train]
        test_texts = [str(text) for text in X_test]
        
        # Ensure labels are proper integers
        y_train_int = np.array(y_train, dtype=np.int64)
        y_test_int = np.array(y_test, dtype=np.int64)
        
        train_dataset = HateSpeechDataset(
            train_texts, y_train_int, tokenizer, ExperimentConfig.MAX_LENGTH
        )
        test_dataset = HateSpeechDataset(
            test_texts, y_test_int, tokenizer, ExperimentConfig.MAX_LENGTH
        )
        
        # Enhanced training arguments
        training_args = TrainingArguments(
            output_dir=ExperimentConfig.RESULTS_DIR,
            num_train_epochs=ExperimentConfig.NUM_EPOCHS,
            per_device_train_batch_size=ExperimentConfig.BATCH_SIZE,
            per_device_eval_batch_size=ExperimentConfig.BATCH_SIZE,
            gradient_accumulation_steps=ExperimentConfig.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=ExperimentConfig.LEARNING_RATE,
            weight_decay=ExperimentConfig.WEIGHT_DECAY,
            warmup_steps=ExperimentConfig.WARMUP_STEPS,
            logging_steps=ExperimentConfig.LOGGING_STEPS,
            eval_steps=ExperimentConfig.EVAL_STEPS,
            save_steps=ExperimentConfig.SAVE_STEPS,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=3,
            report_to=None,
            seed=42,
            fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
            dataloader_num_workers=2,
            remove_unused_columns=False
        )
        
        # Initialize trainer with enhanced configuration
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            class_weights=ExperimentConfig.CLASS_WEIGHTS,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=ExperimentConfig.EARLY_STOPPING_PATIENCE,
                    early_stopping_threshold=ExperimentConfig.EARLY_STOPPING_THRESHOLD
                )
            ]
        )
        
        # Log training configuration
        logger.info("Training Configuration:")
        logger.info(f"  Model: {ExperimentConfig.MODEL_NAME}")
        logger.info(f"  Learning Rate: {ExperimentConfig.LEARNING_RATE}")
        logger.info(f"  Batch Size: {ExperimentConfig.BATCH_SIZE}")
        logger.info(f"  Gradient Accumulation: {ExperimentConfig.GRADIENT_ACCUMULATION_STEPS}")
        logger.info(f"  Effective Batch Size: {ExperimentConfig.BATCH_SIZE * ExperimentConfig.GRADIENT_ACCUMULATION_STEPS}")
        logger.info(f"  Warmup Steps: {ExperimentConfig.WARMUP_STEPS}")
        logger.info(f"  Early Stopping: {ExperimentConfig.EARLY_STOPPING_PATIENCE}/{ExperimentConfig.EARLY_STOPPING_THRESHOLD}")
        logger.info(f"  Class Weights: {ExperimentConfig.CLASS_WEIGHTS}")
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        logger.info(f"Saving model to {ExperimentConfig.MODEL_SAVE_PATH}")
        trainer.save_model(ExperimentConfig.MODEL_SAVE_PATH)
        tokenizer.save_pretrained(ExperimentConfig.MODEL_SAVE_PATH)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # Compute detailed metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
        
        # Create detailed results
        results = {
            "experiment_name": "Experiment 1.2: IndoBERT Large - Optimized",
            "model_name": ExperimentConfig.MODEL_NAME,
            "timestamp": datetime.now().isoformat(),
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "per_class_metrics": {},
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, 
                                                         target_names=list(ExperimentConfig.LABEL_MAPPING.values()),
                                                         output_dict=True),
            "training_config": {
                "learning_rate": ExperimentConfig.LEARNING_RATE,
                "batch_size": ExperimentConfig.BATCH_SIZE,
                "gradient_accumulation_steps": ExperimentConfig.GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size": ExperimentConfig.BATCH_SIZE * ExperimentConfig.GRADIENT_ACCUMULATION_STEPS,
                "warmup_steps": ExperimentConfig.WARMUP_STEPS,
                "early_stopping_patience": ExperimentConfig.EARLY_STOPPING_PATIENCE,
                "early_stopping_threshold": ExperimentConfig.EARLY_STOPPING_THRESHOLD,
                "class_weights": ExperimentConfig.CLASS_WEIGHTS
            }
        }
        
        # Add per-class metrics
        for i, (label_idx, label_name) in enumerate(ExperimentConfig.LABEL_MAPPING.items()):
            if i < len(f1_per_class):
                results["per_class_metrics"][label_name] = {
                    "precision": float(precision_per_class[i]),
                    "recall": float(recall_per_class[i]),
                    "f1_score": float(f1_per_class[i]),
                    "support": int(support_per_class[i])
                }
        
        # Save results
        results_path = os.path.join(ExperimentConfig.RESULTS_DIR, "experiment_1.2_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save confusion matrix
        cm_path = os.path.join(ExperimentConfig.RESULTS_DIR, "confusion_matrix.png")
        save_confusion_matrix(y_test, y_pred, cm_path)
        
        # Log final results
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT 1.2 RESULTS")
        logger.info("="*80)
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1-Score Macro: {f1_macro:.4f}")
        logger.info(f"Precision Macro: {precision:.4f}")
        logger.info(f"Recall Macro: {recall:.4f}")
        logger.info("\nPer-class metrics:")
        
        for label_name, metrics in results["per_class_metrics"].items():
            logger.info(f"  {label_name}:")
            logger.info(f"    F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall: {metrics['recall']:.4f}")
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"\nTotal experiment time: {total_time:.2f} seconds")
        logger.info("Experiment 1.2 completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        raise

if __name__ == "__main__":
    main()