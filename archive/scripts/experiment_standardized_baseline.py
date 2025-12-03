#!/usr/bin/env python3
"""
Experiment: Baseline IndoBERT dengan Dataset Terstandarisasi
Tujuan: Menguji performa baseline IndoBERT pada dataset yang sudah dibalance
Target: Mencapai F1-Score Macro >80% dengan dataset seimbang

Dataset: Menggunakan dataset terstandarisasi yang sudah dibalance
- Train: 19,971 sampel (seimbang per kelas)
- Test: 4,993 sampel (seimbang per kelas)
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
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

def load_standardized_data():
    """
    Load dataset terstandarisasi yang sudah dibalance
    """
    print("Loading standardized dataset...")
    
    # Load training dan test dataset
    train_df = pd.read_csv('data/standardized/train_dataset.csv')
    test_df = pd.read_csv('data/standardized/test_dataset.csv')
    
    print(f"Train dataset: {len(train_df)} samples")
    print(f"Test dataset: {len(test_df)} samples")
    
    # Analisis distribusi
    print("\nTrain distribution:")
    train_dist = train_df['final_label'].value_counts()
    for label, count in train_dist.items():
        print(f"  {label}: {count} ({count/len(train_df)*100:.1f}%)")
    
    print("\nTest distribution:")
    test_dist = test_df['final_label'].value_counts()
    for label, count in test_dist.items():
        print(f"  {label}: {count} ({count/len(test_df)*100:.1f}%)")
    
    return (
        train_df['text'].values, train_df['label_numeric'].values,
        test_df['text'].values, test_df['label_numeric'].values
    )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }

def create_detailed_report(y_true, y_pred, label_names):
    """
    Buat laporan evaluasi yang detail
    """
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=label_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return report, cm

def save_results(results, model_name, experiment_name):
    """
    Simpan hasil eksperimen
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"experiments/results/{experiment_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    with open(f"{results_dir}/metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_dir}")
    return results_dir

def plot_confusion_matrix(cm, label_names, save_path):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_names, yticklabels=label_names
    )
    plt.title('Confusion Matrix - Standardized Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=== EXPERIMENT: BASELINE INDOBERT - STANDARDIZED DATASET ===")
    print(f"Start time: {datetime.now()}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    X_train, y_train, X_test, y_test = load_standardized_data()
    
    # Label names
    label_names = [
        'Bukan Ujaran Kebencian',
        'Ujaran Kebencian - Ringan', 
        'Ujaran Kebencian - Sedang',
        'Ujaran Kebencian - Berat'
    ]
    
    # Load tokenizer and model
    model_name = "indobenchmark/indobert-base-p1"
    print(f"\nLoading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=4,
        problem_type="single_label_classification"
    )
    
    # Create datasets
    train_dataset = HateSpeechDataset(X_train, y_train, tokenizer)
    test_dataset = HateSpeechDataset(X_test, y_test, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results_standardized_baseline',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=None,
        seed=42
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train model
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating model...")
    eval_results = trainer.evaluate()
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # Create detailed report
    report, cm = create_detailed_report(y_test, y_pred, label_names)
    
    # Compile results
    results = {
        'experiment_name': 'baseline_indobert_standardized',
        'model_name': model_name,
        'dataset_info': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'balanced': True,
            'train_distribution': dict(Counter(y_train)),
            'test_distribution': dict(Counter(y_test))
        },
        'training_args': {
            'epochs': training_args.num_train_epochs,
            'batch_size': training_args.per_device_train_batch_size,
            'learning_rate': training_args.learning_rate,
            'weight_decay': training_args.weight_decay
        },
        'metrics': {
            'accuracy': eval_results['eval_accuracy'],
            'f1_macro': eval_results['eval_f1_macro'],
            'precision_macro': eval_results['eval_precision_macro'],
            'recall_macro': eval_results['eval_recall_macro']
        },
        'detailed_report': report,
        'confusion_matrix': cm.tolist(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    results_dir = save_results(results, model_name, 'baseline_standardized')
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, label_names, results_dir)
    
    # Print summary
    print("\n=== EXPERIMENT RESULTS ===")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1-Score Macro: {eval_results['eval_f1_macro']:.4f}")
    print(f"Precision Macro: {eval_results['eval_precision_macro']:.4f}")
    print(f"Recall Macro: {eval_results['eval_recall_macro']:.4f}")
    
    print("\n=== PER-CLASS RESULTS ===")
    for i, label in enumerate(label_names):
        if str(i) in report:
            metrics = report[str(i)]
            print(f"{label}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
    
    print(f"\nResults saved to: {results_dir}")
    print(f"End time: {datetime.now()}")
    
    return results

if __name__ == "__main__":
    results = main()