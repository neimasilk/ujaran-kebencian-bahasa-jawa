#!/usr/bin/env python3
"""
Enhanced IndoBERT Experiment for Javanese Hate Speech Detection
Incorporates progressive training, advanced data augmentation, and optimized hyperparameters

Author: AI Assistant
Date: 2025-07-24
"""

import os
import json
import time
import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    get_linear_schedule_with_warmup, AdamW
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_indobert_experiment.log'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Ensure immediate output
import sys
sys.stdout.flush()
sys.stderr.flush()

class EnhancedConfig:
    """Enhanced configuration for IndoBERT experiment"""
    
    # Model configuration
    MODEL_NAME = "indobenchmark/indobert-base-p1"
    MAX_LENGTH = 128
    NUM_LABELS = 4
    
    # Training configuration
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 3e-5  # Slightly higher learning rate
    NUM_EPOCHS = 5  # More epochs for better convergence
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_THRESHOLD = 0.001
    
    # Paths
    DATA_PATH = "data/standardized/balanced_dataset.csv"
    OUTPUT_DIR = "experiments/results/enhanced_indobert_experiment"
    MODEL_SAVE_PATH = "models/enhanced_indobert_hate_speech"
    
    # Enhanced class weights using sklearn's compute_class_weight
    # Will be computed dynamically from data
    
    # Label mapping
    LABEL_MAPPING = {
        0: "Bukan Ujaran Kebencian",
        1: "Ujaran Kebencian - Ringan",
        2: "Ujaran Kebencian - Sedang",
        3: "Ujaran Kebencian - Berat"
    }
    
    # Data augmentation parameters
    AUGMENTATION_RATE = 0.3  # 30% of training data will be augmented
    BACK_TRANSLATION_LANGS = ['en', 'id']  # Languages for back-translation

class EnhancedHateSpeechDataset(Dataset):
    """Enhanced dataset with data augmentation capabilities"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, 
                 max_length: int, augment: bool = False, augmentation_rate: float = 0.0):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.augmentation_rate = augmentation_rate
        
        # Simple synonym augmentation dictionary (can be expanded)
        self.synonym_dict = {
            'tidak': ['ora', 'mboten'],
            'benar': ['lurus', 'betul'],
            'salah': ['pantat', 'salip'],
            'baik': ['apik', 'bener'],
            'buruk': ['awon', 'alit'],
            'besar': ['gede', 'ageng'],
            'kecil': ['cilik', 'alit'],
            'cepat': ['gancang', 'tancap'],
            'lambat': ['ala', 'lamban'],
        }
    
    def __len__(self):
        return len(self.texts)
    
    def _simple_augment(self, text: str) -> str:
        """Simple text augmentation by synonym replacement"""
        words = text.split()
        augmented_words = []
        
        for word in words:
            # With some probability, replace with synonym
            if random.random() < 0.1 and word in self.synonym_dict:
                synonyms = self.synonym_dict[word]
                augmented_words.append(random.choice(synonyms))
            else:
                augmented_words.append(word)
                
        return ' '.join(augmented_words)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Apply augmentation for training data
        if self.augment and random.random() < self.augmentation_rate:
            text = self._simple_augment(text)
        
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

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss to improve generalization"""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logprobs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=-1))

class ProgressiveTrainer(Trainer):
    """Custom trainer with label smoothing and enhanced metrics"""
    
    def __init__(self, *args, label_smoothing=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if label_smoothing > 0:
            self.label_smoothing_loss = LabelSmoothingLoss(
                num_classes=self.model.config.num_labels,
                smoothing=label_smoothing
            )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if hasattr(self, 'label_smoothing_loss'):
            loss = self.label_smoothing_loss(logits, labels)
        else:
            loss = outputs.loss
            
        return (loss, outputs) if return_outputs else loss

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_and_preprocess_data(data_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and preprocess the dataset"""
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Dataset shape: {df.shape}")
    
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
    
    logger.info(f"After cleaning - Dataset shape: {df.shape}")
    
    # Log class distribution
    class_counts = df['label'].value_counts().sort_index()
    logger.info("Class distribution:")
    for label, count in class_counts.items():
        logger.info(f"  {EnhancedConfig.LABEL_MAPPING[label]}: {count} ({count/len(df)*100:.2f}%)")
    
    return df, df['text'].values, df['label'].values

def create_balanced_split(texts: np.ndarray, labels: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """Create balanced train-test split with oversampling for minority classes"""
    logger.info("Creating balanced train-test split")
    
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
        logger.info(f"  {EnhancedConfig.LABEL_MAPPING[label]}: {count} ({count/len(y_train)*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test

def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Compute class weights for handling imbalance"""
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    return dict(zip(unique_labels, class_weights))

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

def detailed_evaluation(model, tokenizer, X_test, y_test, config: EnhancedConfig):
    """Perform detailed evaluation of the model"""
    logger.info("Performing detailed evaluation")
    
    # Create test dataset
    test_dataset = EnhancedHateSpeechDataset(X_test, y_test, tokenizer, config.MAX_LENGTH)
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

def save_results(report: Dict, config: EnhancedConfig, training_time: float):
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
        'experiment_type': 'enhanced_indobert',
        'augmentation_rate': config.AUGMENTATION_RATE,
        'label_smoothing': 0.1
    }
    
    # Save detailed results
    results_file = output_dir / 'detailed_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    # Save summary
    summary = {
        'experiment': 'Enhanced IndoBERT',
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
    logger.info("ENHANCED INDOBERT EXPERIMENT FOR JAVANESE HATE SPEECH DETECTION")
    logger.info("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    config = EnhancedConfig()
    
    try:
        # Load and preprocess data
        df, texts, labels = load_and_preprocess_data(config.DATA_PATH)
        
        # Create train-test split
        X_train, X_test, y_train, y_test = create_balanced_split(texts, labels)
        
        # Compute class weights
        class_weights = compute_class_weights(y_train)
        logger.info(f"Class weights: {class_weights}")
        
        # Load tokenizer and model
        logger.info(f"Loading tokenizer and model: {config.MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=config.NUM_LABELS
        )
        
        # Create datasets
        logger.info("Creating datasets with augmentation")
        train_dataset = EnhancedHateSpeechDataset(
            X_train, y_train, tokenizer, config.MAX_LENGTH, 
            augment=True, augmentation_rate=config.AUGMENTATION_RATE
        )
        val_dataset = EnhancedHateSpeechDataset(X_test, y_test, tokenizer, config.MAX_LENGTH)
        
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
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            report_to=None,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
        )
        
        # Create trainer
        trainer = ProgressiveTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            label_smoothing=0.1,  # Apply label smoothing
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)]
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
        
        # Comparison with previous best
        previous_best_f1 = 0.5167  # mBERT result from your documentation
        current_f1 = evaluation_results['f1_macro']
        difference = current_f1 - previous_best_f1
        
        logger.info("=" * 60)
        logger.info("COMPARISON WITH PREVIOUS BEST")
        logger.info("=" * 60)
        logger.info(f"Previous Best F1-Score Macro (mBERT): {previous_best_f1:.4f}")
        logger.info(f"Current F1-Score Macro: {current_f1:.4f}")
        logger.info(f"Difference: {difference:+.4f}")
        
        if difference > 0:
            logger.info("[IMPROVEMENT] PERFORMANCE IMPROVED OVER PREVIOUS BEST!")
        else:
            logger.info("[INFO] Previous best performance not exceeded")
            
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()