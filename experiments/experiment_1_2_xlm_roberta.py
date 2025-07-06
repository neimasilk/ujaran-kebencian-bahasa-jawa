#!/usr/bin/env python3
"""
Experiment 1.2: XLM-RoBERTa Cross-Lingual Analysis
Javanese Hate Speech Detection using XLM-RoBERTa Base

Objective: Evaluate multilingual pre-training for low-resource Javanese
Target: >76% accuracy (+2.2% improvement from baseline)
Focus: Cross-dialect robustness and multilingual capability

Author: Research Team
Date: 2025-01-06
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from torch.utils.data import Dataset
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 256
BATCH_SIZE = 8  # Reduced for XLM-RoBERTa (270M parameters)
LEARNING_RATE = 1e-5
NUM_EPOCHS = 5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
SEED = 42

# Paths
DATA_PATH = "src/data_collection/hasil-labeling.csv"
OUTPUT_DIR = "experiments/results/experiment_1_2_xlm_roberta"
MODEL_SAVE_PATH = "experiments/models/xlm_roberta_javanese_hate_speech"
LOG_FILE = "experiments/experiment_1_2_xlm_roberta.log"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

class HateSpeechDataset(Dataset):
    """Custom dataset for Javanese hate speech classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
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

class WeightedFocalLoss(torch.nn.Module):
    """Focal Loss with class weights for imbalanced dataset"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CustomTrainer(Trainer):
    """Custom trainer with weighted focal loss"""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Use weighted focal loss
        loss_fct = WeightedFocalLoss(
            alpha=self.class_weights,
            gamma=2.0
        )
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load and preprocess the dataset"""
    logger.info(f"Loading data from {file_path}")
    
    # Load data
    df = pd.read_csv(file_path)
    logger.info(f"Dataset shape: {df.shape}")
    
    # Use final_label column and clean data
    df = df.dropna(subset=['text', 'final_label'])
    
    # Create label mapping
    label_mapping = {
        'Bukan Ujaran Kebencian': 0,
        'Ujaran Kebencian - Ringan': 1,
        'Ujaran Kebencian - Sedang': 2,
        'Ujaran Kebencian - Berat': 3
    }
    
    # Convert labels to numeric
    df['numeric_label'] = df['final_label'].map(label_mapping)
    
    # Remove unmapped labels
    df = df.dropna(subset=['numeric_label'])
    df['numeric_label'] = df['numeric_label'].astype(int)
    
    # Log class distribution
    logger.info("Class distribution:")
    class_counts = df['final_label'].value_counts()
    total_samples = len(df)
    for label, count in class_counts.items():
        percentage = (count / total_samples) * 100
        logger.info(f"  {label}: {count} ({percentage:.2f}%)")
    
    return df, label_mapping

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }

def detailed_evaluation(model, tokenizer, test_texts: List[str], test_labels: List[int], 
                      label_mapping: Dict[str, int]) -> Dict:
    """Perform detailed evaluation with per-class metrics"""
    logger.info("Performing detailed evaluation...")
    
    # Create test dataset
    test_dataset = HateSpeechDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    
    # Get predictions
    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_labels
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    macro_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
    
    # Create reverse label mapping
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Per-class metrics
    per_class_metrics = {}
    for i, (label_name) in enumerate([reverse_mapping[j] for j in range(len(label_mapping))]):
        per_class_metrics[label_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    class_names = [reverse_mapping[i] for i in range(len(label_mapping))]
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    results = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': y_pred.tolist(),
        'true_labels': y_true
    }
    
    return results

def save_results(results: Dict, output_dir: str, baseline_f1: float = 0.8036):
    """Save evaluation results and generate visualizations"""
    logger.info(f"Saving results to {output_dir}")
    
    # Save detailed results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    
    # Add metadata
    results['metadata'] = {
        'model_name': MODEL_NAME,
        'experiment': 'XLM-RoBERTa Cross-Lingual Analysis',
        'timestamp': datetime.now().isoformat(),
        'baseline_f1': baseline_f1,
        'improvement': results['macro_f1'] - baseline_f1,
        'target_achieved': results['accuracy'] > 0.76
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Generate confusion matrix plot
    plt.figure(figsize=(10, 8))
    cm = np.array(results['confusion_matrix'])
    
    # Class names for display
    class_names = ['Non-Hate', 'Mild', 'Moderate', 'Severe']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('XLM-RoBERTa - Confusion Matrix\nJavanese Hate Speech Detection')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    cm_file = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Results saved: {results_file}")
    logger.info(f"Confusion matrix saved: {cm_file}")

def main():
    """Main experiment execution"""
    logger.info("Starting Experiment 1.2: XLM-RoBERTa Cross-Lingual Analysis")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    df, label_mapping = load_and_preprocess_data(DATA_PATH)
    
    # Prepare data
    texts = df['text'].tolist()
    labels = df['numeric_label'].tolist()
    
    # Create stratified train-test split
    logger.info("Creating stratified train-test split")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    
    logger.info(f"Train set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_mapping),
        problem_type="single_label_classification"
    )
    
    # Create datasets
    train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    eval_dataset = HateSpeechDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    # Calculate class weights for focal loss
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) for count in class_counts])
    
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,
        seed=SEED,
        fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
        dataloader_num_workers=0,
        remove_unused_columns=False
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save model and tokenizer
    logger.info(f"Saving model to {MODEL_SAVE_PATH}")
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    
    # Detailed evaluation
    results = detailed_evaluation(model, tokenizer, X_test, y_test, label_mapping)
    
    # Save results
    save_results(results, OUTPUT_DIR)
    
    # Log final results
    logger.info("=== EXPERIMENT 1.2 RESULTS ===")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Macro F1-Score: {results['macro_f1']:.4f}")
    logger.info(f"Target (>76% accuracy): {'✓ ACHIEVED' if results['accuracy'] > 0.76 else '✗ NOT ACHIEVED'}")
    
    # Per-class results
    logger.info("\nPer-class metrics:")
    for class_name, metrics in results['per_class_metrics'].items():
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {metrics['precision']:.4f}")
        logger.info(f"    Recall: {metrics['recall']:.4f}")
        logger.info(f"    F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"    Support: {metrics['support']}")
    
    logger.info("Experiment 1.2 completed successfully!")

if __name__ == "__main__":
    main()