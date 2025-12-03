
#!/usr/bin/env python3
"""
Extended Multi-Architecture Ensemble (5 epochs)
Based on multi_architecture_ensemble.py with extended training
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import torch
from torch.utils.data import Dataset
from scipy.optimize import minimize

# Setup logging with proper Unicode support
class UnicodeStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        if hasattr(self.stream, 'reconfigure'):
            try:
                self.stream.reconfigure(encoding='utf-8')
            except:
                pass
    
    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            # Fallback: remove emojis and special characters
            record.msg = str(record.msg).encode('ascii', 'ignore').decode('ascii')
            super().emit(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_architecture_ensemble.log', encoding='utf-8'),
        UnicodeStreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_data():
    """Load and prepare dataset"""
    logger.info("Loading augmented dataset")
    df = pd.read_csv('data/augmented/augmented_dataset.csv')
    
    # Use label_numeric column
    X = df['text'].values
    y = df['label_numeric'].values
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(model_name, model_path, tokenizer, X_train, X_val, y_train, y_val):
    """Train individual model with 5 epochs"""
    logger.info(f"Training {model_name} with 5 epochs")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=4
    )
    
    train_dataset = HateSpeechDataset(X_train, y_train, tokenizer)
    val_dataset = HateSpeechDataset(X_val, y_val, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=f'./models/extended_ensemble_{model_name}',
        num_train_epochs=5,  # Extended to 5 epochs
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=3,  # Keep more checkpoints
        seed=42,
        fp16=True,
        dataloader_num_workers=2,
        report_to=None
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1_macro': f1_score(labels, predictions, average='macro')
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()
    
    # Get final evaluation
    eval_results = trainer.evaluate()
    logger.info(f"{model_name} validation results: {eval_results}")
    
    return model, tokenizer, eval_results

def main():
    logger.info("Starting Extended Multi-Architecture Ensemble Experiment (5 epochs)")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Model configurations
    models_config = {
        'indobert': 'indobenchmark/indobert-base-p1',
        'indobert_uncased': 'indolem/indobert-base-uncased',
        'roberta_indo': 'cahya/roberta-base-indonesian-522M'
    }
    
    trained_models = {}
    tokenizers = {}
    
    # Train each model
    for model_name, model_path in models_config.items():
        try:
            logger.info(f"Loading {model_name}: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            model, tokenizer, eval_results = train_model(
                model_name, model_path, tokenizer, X_train, X_val, y_train, y_val
            )
            
            trained_models[model_name] = model
            tokenizers[model_name] = tokenizer
            
            logger.info(f"Successfully trained {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {str(e)}")
            continue
    
    logger.info(f"Successfully trained {len(trained_models)} models")
    
    # Save results
    results = {
        'experiment_type': 'extended_ensemble_5_epochs',
        'timestamp': datetime.now().isoformat(),
        'models_trained': list(trained_models.keys()),
        'total_models': len(trained_models),
        'epochs': 5,
        'dataset_size': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        }
    }
    
    with open('results/extended_ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Extended ensemble experiment completed successfully")

if __name__ == "__main__":
    main()
