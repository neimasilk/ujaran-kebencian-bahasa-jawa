import os
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import torch.nn as nn
import argparse
import json
from datetime import datetime

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CustomTrainer(Trainer):
    def __init__(self, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        loss = self.focal_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=None)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_class_0': precision[0],
        'precision_class_1': precision[1],
        'recall_class_0': recall[0],
        'recall_class_1': recall[1],
        'f1_class_0': f1[0],
        'f1_class_1': f1[1]
    }

def load_and_preprocess_data(file_path, confidence_threshold=0.7):
    """Load and preprocess the dataset"""
    df = pd.read_csv(file_path)
    print(f"Original dataset size: {len(df)}")
    
    # Filter by confidence if available
    if 'confidence_score' in df.columns:
        df = df[df['confidence_score'] >= confidence_threshold]
        print(f"After confidence filtering (>={confidence_threshold}): {len(df)}")
    
    # Remove rows with missing text or labels
    df = df.dropna(subset=['text', 'final_label'])
    
    # Remove empty text
    df = df[df['text'].str.strip() != '']
    
    # Map labels to integers
    label_mapping = {
        'Bukan Ujaran Kebencian': 0,
        'Ujaran Kebencian - Ringan': 1,
        'Ujaran Kebencian - Sedang': 1,
        'Ujaran Kebencian - Berat': 1
    }
    
    df['label'] = df['final_label'].map(label_mapping)
    df = df.dropna(subset=['label'])  # Remove unmapped labels
    df['label'] = df['label'].astype(int)
    
    print(f"Final dataset size: {len(df)}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df

class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
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

def train_model(model_name, output_dir, train_texts, train_labels, val_texts, val_labels, 
                hyperparams, seed=42):
    """Train a single model with given hyperparameters"""
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\nTraining {model_name} with seed {seed}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        hidden_dropout_prob=hyperparams['dropout_rate']
    )
    
    # Create datasets
    train_dataset = Dataset(train_texts, train_labels, tokenizer, hyperparams['max_length'])
    val_dataset = Dataset(val_texts, val_labels, tokenizer, hyperparams['max_length'])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=hyperparams['batch_size'],
        per_device_eval_batch_size=hyperparams['batch_size'],
        warmup_ratio=hyperparams['warmup_ratio'],
        weight_decay=hyperparams['weight_decay'],
        learning_rate=hyperparams['learning_rate'],
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=seed,
        fp16=True if torch.cuda.is_available() else False
    )
    
    # Create trainer
    trainer = CustomTrainer(
        focal_gamma=hyperparams['focal_gamma'],
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Get final metrics
    eval_results = trainer.evaluate()
    
    # Save training results
    results = {
        'model_name': model_name,
        'seed': seed,
        'hyperparameters': hyperparams,
        'final_metrics': eval_results,
        'training_completed': datetime.now().isoformat()
    }
    
    with open(f'{output_dir}/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Model saved to {output_dir}")
    print(f"Final F1-Macro: {eval_results['eval_f1_macro']:.4f}")
    
    return eval_results['eval_f1_macro']

def main():
    # Best hyperparameters from previous optimization
    best_hyperparams = {
        'learning_rate': 3.7909412079529265e-05,
        'batch_size': 32,
        'max_length': 256,
        'weight_decay': 0.028129935174160545,
        'warmup_ratio': 0.06092632100996669,
        'focal_gamma': 1.7352101403669877,
        'dropout_rate': 0.24433669071285388
    }
    
    # Different model architectures to try
    models_to_train = [
        'indolem/indobert-base-uncased',
        'cahya/bert-base-indonesian-522M',
        'indobenchmark/indobert-base-p1'
    ]
    
    # Different seeds for diversity
    seeds = [42, 123, 456]
    
    # Load data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('src/data_collection/hasil-labeling.csv')
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    val_texts = val_df['text'].tolist()
    val_labels = val_df['label'].tolist()
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    # Train ensemble models
    ensemble_results = []
    
    for i, model_name in enumerate(models_to_train):
        for j, seed in enumerate(seeds):
            model_id = f"model_{i}_{j}"
            output_dir = f"models/ensemble/{model_id}"
            
            try:
                f1_score = train_model(
                    model_name=model_name,
                    output_dir=output_dir,
                    train_texts=train_texts,
                    train_labels=train_labels,
                    val_texts=val_texts,
                    val_labels=val_labels,
                    hyperparams=best_hyperparams,
                    seed=seed
                )
                
                ensemble_results.append({
                    'model_id': model_id,
                    'model_name': model_name,
                    'seed': seed,
                    'f1_macro': f1_score,
                    'output_dir': output_dir
                })
                
            except Exception as e:
                print(f"Error training {model_id}: {str(e)}")
                continue
    
    # Save ensemble results
    with open('models/ensemble/ensemble_results.json', 'w') as f:
        json.dump(ensemble_results, f, indent=2)
    
    print("\n=== ENSEMBLE TRAINING COMPLETED ===")
    print(f"Successfully trained {len(ensemble_results)} models")
    
    # Sort by F1 score
    ensemble_results.sort(key=lambda x: x['f1_macro'], reverse=True)
    
    print("\nTop 5 models:")
    for i, result in enumerate(ensemble_results[:5]):
        print(f"{i+1}. {result['model_id']} ({result['model_name']}) - F1: {result['f1_macro']:.4f}")

if __name__ == "__main__":
    main()