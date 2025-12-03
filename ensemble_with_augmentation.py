
#!/usr/bin/env python3
"""
Ensemble with Enhanced Data Augmentation
Combines ensemble method with advanced data augmentation techniques
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import torch
from torch.utils.data import Dataset
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/augmentation_ensemble.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.texts)
    
    def augment_text(self, text):
        """Simple text augmentation"""
        if not self.augment or random.random() > 0.3:
            return text
            
        # Random word order change (simple)
        words = text.split()
        if len(words) > 3:
            # Swap two random adjacent words
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
            return ' '.join(words)
        return text
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.augment:
            text = self.augment_text(text)
            
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
    """Load and prepare dataset with augmentation"""
    logger.info("Loading dataset with augmentation support")
    df = pd.read_csv('data/augmented/augmented_dataset.csv')
    
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

def train_model_with_augmentation(model_name, model_path, tokenizer, X_train, X_val, y_train, y_val):
    """Train model with data augmentation"""
    logger.info(f"Training {model_name} with data augmentation")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=4
    )
    
    # Use augmentation for training data
    train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, augment=True)
    val_dataset = HateSpeechDataset(X_val, y_val, tokenizer, augment=False)
    
    training_args = TrainingArguments(
        output_dir=f'./models/augmented_ensemble_{model_name}',
        num_train_epochs=3,
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
        save_total_limit=2,
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
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    eval_results = trainer.evaluate()
    logger.info(f"{model_name} validation results: {eval_results}")
    
    return model, tokenizer, eval_results

def main():
    logger.info("Starting Ensemble with Data Augmentation Experiment")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Model configurations
    models_config = {
        'indobert': 'indobenchmark/indobert-base-p1',
        'indobert_uncased': 'indolem/indobert-base-uncased',
        'roberta_indo': 'cahya/roberta-base-indonesian-522M'
    }
    
    trained_models = {}
    
    # Train each model with augmentation
    for model_name, model_path in models_config.items():
        try:
            logger.info(f"Loading {model_name}: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            model, tokenizer, eval_results = train_model_with_augmentation(
                model_name, model_path, tokenizer, X_train, X_val, y_train, y_val
            )
            
            trained_models[model_name] = model
            logger.info(f"Successfully trained {model_name} with augmentation")
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {str(e)}")
            continue
    
    # Save results
    results = {
        'experiment_type': 'ensemble_with_augmentation',
        'timestamp': datetime.now().isoformat(),
        'models_trained': list(trained_models.keys()),
        'total_models': len(trained_models),
        'augmentation_enabled': True,
        'dataset_size': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        }
    }
    
    with open('results/augmentation_ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Augmentation ensemble experiment completed")

if __name__ == "__main__":
    main()
