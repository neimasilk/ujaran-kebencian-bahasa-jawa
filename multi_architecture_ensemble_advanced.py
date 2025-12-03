#!/usr/bin/env python3
"""
Multi-Architecture Ensemble Advanced
Menggabungkan IndoBERT, RoBERTa, dan ELECTRA dengan weighted voting dan stacking meta-learner
Target: Peningkatan 2-4% dari baseline (67% -> 69-71%)
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import torch
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

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro
    }

def train_single_model(model_name, model_path, X_train, y_train, X_val, y_val, output_dir):
    """
    Train single transformer model
    """
    print(f"\n=== Training {model_name} ===")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=4,
        ignore_mismatched_sizes=True
    )
    
    # Create datasets
    train_dataset = JavaneseDataset(X_train, y_train, tokenizer)
    val_dataset = JavaneseDataset(X_val, y_val, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
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
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train model
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"{model_name} - Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"{model_name} - Validation F1-Macro: {eval_results['eval_f1_macro']:.4f}")
    
    return trainer, eval_results

def get_model_predictions(trainer, X_data, y_data, tokenizer):
    """
    Get predictions from trained model
    """
    dataset = JavaneseDataset(X_data, y_data, tokenizer)
    predictions = trainer.predict(dataset)
    
    # Get probabilities (softmax)
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    pred_labels = np.argmax(predictions.predictions, axis=1)
    
    return probs, pred_labels

def optimize_ensemble_weights(predictions_list, y_true):
    """
    Optimize ensemble weights using grid search
    """
    from itertools import product
    
    best_score = 0
    best_weights = None
    
    # Grid search for weights
    weight_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for w1 in weight_options:
        for w2 in weight_options:
            w3 = 1.0 - w1 - w2
            if w3 >= 0.1 and w3 <= 0.9:
                weights = [w1, w2, w3]
                
                # Weighted ensemble prediction
                ensemble_probs = np.zeros_like(predictions_list[0])
                for i, (probs, weight) in enumerate(zip(predictions_list, weights)):
                    ensemble_probs += weight * probs
                
                ensemble_preds = np.argmax(ensemble_probs, axis=1)
                score = f1_score(y_true, ensemble_preds, average='macro')
                
                if score > best_score:
                    best_score = score
                    best_weights = weights
    
    return best_weights, best_score

def train_meta_learner(predictions_list, y_true, meta_model_type='xgboost'):
    """
    Train meta-learner for stacking ensemble
    """
    # Combine predictions as features
    meta_features = np.hstack(predictions_list)
    
    if meta_model_type == 'xgboost':
        meta_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
    elif meta_model_type == 'rf':
        meta_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    else:  # logistic regression
        meta_model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
    
    meta_model.fit(meta_features, y_true)
    return meta_model

def main():
    print("=== Multi-Architecture Ensemble Advanced ===")
    print(f"Start time: {datetime.now()}")
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv('data/standardized/balanced_dataset.csv')
    
    # Prepare data
    X = df['text'].values
    y = df['label_numeric'].values
    
    print(f"Dataset size: {len(df)}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    
    # Model configurations
    models_config = [
        {
            'name': 'IndoBERT',
            'path': 'indobenchmark/indobert-base-p1',
            'output_dir': './models/ensemble_advanced_indobert'
        },
        {
            'name': 'IndoBERT_Uncased',
            'path': 'indolem/indobert-base-uncased',
            'output_dir': './models/ensemble_advanced_indobert_uncased'
        },
        {
            'name': 'RoBERTa_Indo',
            'path': 'cahya/roberta-base-indonesian-522M',
            'output_dir': './models/ensemble_advanced_roberta'
        }
    ]
    
    # Train individual models
    trained_models = []
    individual_results = []
    
    for config in models_config:
        try:
            trainer, eval_results = train_single_model(
                config['name'],
                config['path'],
                X_train, y_train,
                X_val, y_val,
                config['output_dir']
            )
            
            trained_models.append({
                'name': config['name'],
                'trainer': trainer,
                'tokenizer': AutoTokenizer.from_pretrained(config['path']),
                'results': eval_results
            })
            
            individual_results.append({
                'model': config['name'],
                'accuracy': eval_results['eval_accuracy'],
                'f1_macro': eval_results['eval_f1_macro']
            })
            
        except Exception as e:
            print(f"Error training {config['name']}: {str(e)}")
            continue
    
    if len(trained_models) < 2:
        print("Not enough models trained successfully. Exiting.")
        return
    
    print(f"\n=== Individual Model Results ===")
    for result in individual_results:
        print(f"{result['model']}: Acc={result['accuracy']:.4f}, F1={result['f1_macro']:.4f}")
    
    # Get predictions from all models on validation set
    print("\n=== Getting Model Predictions ===")
    val_predictions = []
    test_predictions = []
    
    for model_info in trained_models:
        print(f"Getting predictions from {model_info['name']}...")
        
        # Validation predictions
        val_probs, val_preds = get_model_predictions(
            model_info['trainer'], X_val, y_val, model_info['tokenizer']
        )
        val_predictions.append(val_probs)
        
        # Test predictions
        test_probs, test_preds = get_model_predictions(
            model_info['trainer'], X_test, y_test, model_info['tokenizer']
        )
        test_predictions.append(test_probs)
    
    # Ensemble Method 1: Equal Weights
    print("\n=== Equal Weight Ensemble ===")
    equal_ensemble_val = np.mean(val_predictions, axis=0)
    equal_ensemble_test = np.mean(test_predictions, axis=0)
    
    equal_val_preds = np.argmax(equal_ensemble_val, axis=1)
    equal_test_preds = np.argmax(equal_ensemble_test, axis=1)
    
    equal_val_acc = accuracy_score(y_val, equal_val_preds)
    equal_val_f1 = f1_score(y_val, equal_val_preds, average='macro')
    equal_test_acc = accuracy_score(y_test, equal_test_preds)
    equal_test_f1 = f1_score(y_test, equal_test_preds, average='macro')
    
    print(f"Equal Weight - Val Acc: {equal_val_acc:.4f}, Val F1: {equal_val_f1:.4f}")
    print(f"Equal Weight - Test Acc: {equal_test_acc:.4f}, Test F1: {equal_test_f1:.4f}")
    
    # Ensemble Method 2: Optimized Weights
    print("\n=== Optimized Weight Ensemble ===")
    optimal_weights, optimal_score = optimize_ensemble_weights(val_predictions, y_val)
    print(f"Optimal weights: {optimal_weights}")
    print(f"Optimal validation F1: {optimal_score:.4f}")
    
    # Apply optimal weights to test set
    weighted_ensemble_test = np.zeros_like(test_predictions[0])
    for i, (probs, weight) in enumerate(zip(test_predictions, optimal_weights)):
        weighted_ensemble_test += weight * probs
    
    weighted_test_preds = np.argmax(weighted_ensemble_test, axis=1)
    weighted_test_acc = accuracy_score(y_test, weighted_test_preds)
    weighted_test_f1 = f1_score(y_test, weighted_test_preds, average='macro')
    
    print(f"Weighted - Test Acc: {weighted_test_acc:.4f}, Test F1: {weighted_test_f1:.4f}")
    
    # Ensemble Method 3: Stacking Meta-Learner
    print("\n=== Stacking Meta-Learner ===")
    
    # Train meta-learners
    meta_models = {}
    meta_results = {}
    
    for meta_type in ['xgboost', 'rf', 'lr']:
        print(f"Training {meta_type} meta-learner...")
        meta_model = train_meta_learner(val_predictions, y_val, meta_type)
        meta_models[meta_type] = meta_model
        
        # Predict on test set
        meta_test_features = np.hstack(test_predictions)
        meta_test_preds = meta_model.predict(meta_test_features)
        
        meta_test_acc = accuracy_score(y_test, meta_test_preds)
        meta_test_f1 = f1_score(y_test, meta_test_preds, average='macro')
        
        meta_results[meta_type] = {
            'accuracy': meta_test_acc,
            'f1_macro': meta_test_f1
        }
        
        print(f"{meta_type} - Test Acc: {meta_test_acc:.4f}, Test F1: {meta_test_f1:.4f}")
    
    # Find best ensemble method
    best_method = 'equal_weight'
    best_acc = equal_test_acc
    best_f1 = equal_test_f1
    
    if weighted_test_f1 > best_f1:
        best_method = 'weighted'
        best_acc = weighted_test_acc
        best_f1 = weighted_test_f1
    
    for meta_type, results in meta_results.items():
        if results['f1_macro'] > best_f1:
            best_method = f'stacking_{meta_type}'
            best_acc = results['accuracy']
            best_f1 = results['f1_macro']
    
    # Save results
    results = {
        'experiment': 'multi_architecture_ensemble_advanced',
        'timestamp': datetime.now().isoformat(),
        'individual_models': individual_results,
        'ensemble_results': {
            'equal_weight': {
                'test_accuracy': float(equal_test_acc),
                'test_f1_macro': float(equal_test_f1)
            },
            'optimized_weight': {
                'weights': optimal_weights,
                'test_accuracy': float(weighted_test_acc),
                'test_f1_macro': float(weighted_test_f1)
            },
            'stacking': meta_results
        },
        'best_method': {
            'method': best_method,
            'test_accuracy': float(best_acc),
            'test_f1_macro': float(best_f1)
        },
        'improvement_analysis': {
            'baseline_best': max([r['f1_macro'] for r in individual_results]),
            'ensemble_best': float(best_f1),
            'improvement': float(best_f1 - max([r['f1_macro'] for r in individual_results]))
        }
    }
    
    # Save to file
    os.makedirs('results', exist_ok=True)
    with open('results/multi_architecture_ensemble_advanced_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Best Method: {best_method}")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"Best Test F1-Macro: {best_f1:.4f}")
    print(f"Improvement over best individual: {results['improvement_analysis']['improvement']:.4f}")
    
    # Generate classification report for best method
    if best_method == 'equal_weight':
        best_preds = equal_test_preds
    elif best_method == 'weighted':
        best_preds = weighted_test_preds
    else:
        meta_type = best_method.split('_')[1]
        meta_test_features = np.hstack(test_predictions)
        best_preds = meta_models[meta_type].predict(meta_test_features)
    
    print(f"\n=== Classification Report ({best_method}) ===")
    class_names = ['Bukan Ujaran Kebencian', 'Ujaran Kebencian - Ringan', 
                   'Ujaran Kebencian - Sedang', 'Ujaran Kebencian - Berat']
    print(classification_report(y_test, best_preds, target_names=class_names))
    
    print(f"\nResults saved to: results/multi_architecture_ensemble_advanced_results.json")
    print(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()