#!/usr/bin/env python3
"""
Fix XLM-RoBERTa Training
Script khusus untuk mencari hyperparameter yang stabil untuk XLM-RoBERTa
Update: Menggunakan Weighted Loss untuk mengatasi Collapse to Majority Class
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import torch
from torch import nn
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class JavaneseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
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

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure weights are on the same device as the model
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): # Added **kwargs
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Move weights to device
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
            
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro
    }

def train_xlm_roberta(learning_rate, warmup_ratio, output_dir, X_train, y_train, X_val, y_val, class_weights):
    print(f"\n=== Training XLM-R (LR={learning_rate}, Weighted Loss) ===")
    
    model_name = 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Sanity check tokenizer
    print("Tokenization check:")
    print(f"Sample: {X_train[0][:50]}...")
    tokens = tokenizer.tokenize(str(X_train[0]))
    print(f"Tokens: {tokens[:10]}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=4
    )
    
    train_dataset = JavaneseDataset(X_train, y_train, tokenizer)
    val_dataset = JavaneseDataset(X_val, y_val, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4, # Keep accumulation
        per_device_eval_batch_size=32,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        eval_strategy="steps",
        eval_steps=50, 
        save_strategy="steps",
        save_steps=50,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        seed=42,
        fp16=True, # Try FP16 again with stable loss
        report_to=None
    )
    
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )
    
    trainer.train()
    return trainer.evaluate()

def main():
    print("=== XLM-RoBERTa Tuning with Weighted Loss ===")
    
    # Load dataset
    df = pd.read_csv('data/standardized/balanced_dataset.csv')
    X = df['text'].values
    y = df['label'].values
    
    # Calculate Class Weights
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    print(f"Computed Class Weights: {weights}")
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Experiment 3: LR 2e-5 + Weighted Loss + Grad Acc
    results_3 = train_xlm_roberta(
        learning_rate=2e-5,
        warmup_ratio=0.1,
        output_dir='./models/xlm_roberta_fix_weighted',
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        class_weights=weights
    )
    
    print("\n=== Tuning Results ===")
    print(f"Config 3 (LR=2e-5, Weighted): F1-Macro = {results_3['eval_f1_macro']:.4f}")
    
    if results_3['eval_f1_macro'] > 0.5:
        print("SUCCESS: Found stable configuration!")
    else:
        print("FAILED: Model still not learning.")

if __name__ == "__main__":
    main()