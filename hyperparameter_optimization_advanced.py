#!/usr/bin/env python3
"""
Advanced Hyperparameter Optimization for Hate Speech Detection
Menggunakan Optuna untuk mencari hyperparameter optimal untuk mencapai 90% performance
"""

import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple

import optuna
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from torch.utils.data import Dataset
import torch.nn as nn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss untuk mengatasi class imbalance"""
    def __init__(self, alpha=1, gamma=2, num_classes=4):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss

class HateSpeechDataset(Dataset):
    """Dataset class untuk hate speech detection"""
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
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

class CustomTrainer(Trainer):
    """Custom Trainer dengan Focal Loss"""
    def __init__(self, focal_loss=None, **kwargs):
        super().__init__(**kwargs)
        self.focal_loss = focal_loss
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.focal_loss:
            loss = self.focal_loss(logits, labels)
        else:
            loss = outputs.loss
            
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    """Compute metrics untuk evaluasi"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

class HyperparameterOptimizer:
    """Hyperparameter optimizer menggunakan Optuna"""
    
    def __init__(self, data_path: str, base_model: str, output_dir: str, n_trials: int = 50):
        self.data_path = data_path
        self.base_model = base_model
        self.output_dir = output_dir
        self.n_trials = n_trials
        
        # Load dan prepare data
        self.train_texts, self.train_labels, self.val_texts, self.val_labels = self._load_data()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _load_data(self) -> Tuple[List[str], List[int], List[str], List[int]]:
        """Load dan prepare data untuk training"""
        logger.info(f"Loading data from: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Filter data yang valid
        df = df.dropna(subset=['text', 'final_label'])
        df = df[df['final_label'].isin([
            'Bukan Ujaran Kebencian',
            'Ujaran Kebencian - Ringan', 
            'Ujaran Kebencian - Sedang',
            'Ujaran Kebencian - Berat'
        ])]
        
        # Mapping labels
        label_mapping = {
            'Bukan Ujaran Kebencian': 0,
            'Ujaran Kebencian - Ringan': 1,
            'Ujaran Kebencian - Sedang': 2,
            'Ujaran Kebencian - Berat': 3
        }
        
        texts = df['text'].tolist()
        labels = [label_mapping[label] for label in df['final_label']]
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"Training samples: {len(train_texts)}")
        logger.info(f"Validation samples: {len(val_texts)}")
        
        return train_texts, train_labels, val_texts, val_labels
    
    def objective(self, trial):
        """Objective function untuk Optuna optimization"""
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        max_length = trial.suggest_categorical('max_length', [256, 384, 512])
        weight_decay = trial.suggest_float('weight_decay', 0.001, 0.1, log=True)
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.2)
        focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        
        logger.info(f"Trial {trial.number}: lr={learning_rate:.2e}, bs={batch_size}, ml={max_length}, wd={weight_decay:.3f}, wr={warmup_ratio:.2f}, fg={focal_gamma:.1f}, dr={dropout_rate:.2f}")
        
        try:
            # Load tokenizer dan model
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model, 
                num_labels=4,
                hidden_dropout_prob=dropout_rate,
                attention_probs_dropout_prob=dropout_rate
            )
            
            # Prepare datasets
            train_dataset = HateSpeechDataset(
                self.train_texts, self.train_labels, tokenizer, max_length
            )
            val_dataset = HateSpeechDataset(
                self.val_texts, self.val_labels, tokenizer, max_length
            )
            
            # Setup focal loss
            focal_loss = FocalLoss(gamma=focal_gamma, num_classes=4)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"{self.output_dir}/trial_{trial.number}",
                num_train_epochs=3,  # Reduced untuk speed
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                logging_steps=50,
                eval_strategy="steps",
                eval_steps=200,
                save_strategy="steps",
                save_steps=200,
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1_macro",
                greater_is_better=True,
                save_total_limit=1,
                report_to=None,
                dataloader_pin_memory=False,
                remove_unused_columns=False,
            )
            
            # Setup trainer
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                focal_loss=focal_loss,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            
            # Train model
            trainer.train()
            
            # Evaluate
            eval_results = trainer.evaluate()
            f1_macro = eval_results['eval_f1_macro']
            
            logger.info(f"Trial {trial.number} F1-Macro: {f1_macro:.4f}")
            
            # Cleanup
            del model, trainer, train_dataset, val_dataset
            torch.cuda.empty_cache()
            
            return f1_macro
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            return 0.0
    
    def optimize(self):
        """Run hyperparameter optimization"""
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            storage=f'sqlite:///{self.output_dir}/optuna_study.db',
            study_name='hate_speech_hpo',
            load_if_exists=True
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Save results
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best F1-Macro: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Save results
        results = {
            'best_value': best_value,
            'best_params': best_params,
            'n_trials': len(study.trials),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{self.output_dir}/best_hyperparameters.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return best_params, best_value

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization untuk Hate Speech Detection')
    parser.add_argument('--data_path', type=str, required=True, help='Path ke dataset CSV')
    parser.add_argument('--base_model', type=str, default='indolem/indobert-base-uncased', help='Base model untuk fine-tuning')
    parser.add_argument('--output_dir', type=str, default='results/hyperparameter_optimization', help='Output directory')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of optimization trials')
    
    args = parser.parse_args()
    
    logger.info("=== HYPERPARAMETER OPTIMIZATION ===") 
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Number of trials: {args.n_trials}")
    
    # Run optimization
    optimizer = HyperparameterOptimizer(
        data_path=args.data_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
        n_trials=args.n_trials
    )
    
    best_params, best_value = optimizer.optimize()
    
    logger.info("=== OPTIMIZATION COMPLETED ===")
    logger.info(f"Best F1-Macro achieved: {best_value:.4f}")
    logger.info(f"Best hyperparameters saved to: {args.output_dir}/best_hyperparameters.json")

if __name__ == "__main__":
    main()