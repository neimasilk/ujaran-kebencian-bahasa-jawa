#!/usr/bin/env python3
"""
Multi-Architecture Ensemble for 90%+ Accuracy
Combine different transformer architectures for superior performance
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize
import json
import logging
from datetime import datetime
import os
import warnings
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from collections import defaultdict
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/multi_architecture_ensemble.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

class MultiArchitectureEnsemble:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.tokenizers = {}
        self.model_configs = {
            'indobert': {
                'name': 'indobenchmark/indobert-base-p1',
                'max_length': 512,
                'batch_size': 16
            },
            'indobert_uncased': {
                'name': 'indolem/indobert-base-uncased',
                'max_length': 512,
                'batch_size': 16
            },
            'roberta_indo': {
                'name': 'cahya/roberta-base-indonesian-522M',
                'max_length': 512,
                'batch_size': 8  # Smaller batch for larger model
            }
        }
        
    def load_models(self):
        """Load all transformer models"""
        logger.info("Loading multiple transformer architectures")
        
        # Create a copy of model_configs to avoid dictionary size change during iteration
        model_configs_copy = dict(self.model_configs)
        failed_models = []
        
        for model_key, config in model_configs_copy.items():
            try:
                logger.info(f"Loading {model_key}: {config['name']}")
                
                tokenizer = AutoTokenizer.from_pretrained(config['name'])
                model = AutoModelForSequenceClassification.from_pretrained(
                    config['name'],
                    num_labels=4,
                    problem_type="single_label_classification"
                )
                model.to(self.device)
                
                self.tokenizers[model_key] = tokenizer
                self.models[model_key] = model
                
                logger.info(f"Successfully loaded {model_key}")
                
            except Exception as e:
                logger.error(f"Failed to load {model_key}: {e}")
                failed_models.append(model_key)
        
        # Remove failed models from configs after iteration
        for model_key in failed_models:
            if model_key in self.model_configs:
                del self.model_configs[model_key]
        
        logger.info(f"Successfully loaded {len(self.models)} models")
    
    def train_individual_models(self, X_train, y_train, X_val, y_val, epochs=3):
        """Train each model individually"""
        logger.info("Training individual models")
        
        trained_models = {}
        
        for model_key in self.models.keys():
            logger.info(f"Training {model_key}")
            
            try:
                config = self.model_configs[model_key]
                tokenizer = self.tokenizers[model_key]
                model = self.models[model_key]
                
                # Create datasets
                train_dataset = HateSpeechDataset(
                    X_train, y_train, tokenizer, config['max_length']
                )
                val_dataset = HateSpeechDataset(
                    X_val, y_val, tokenizer, config['max_length']
                )
                
                # Training arguments optimized for GPU
                training_args = TrainingArguments(
                    output_dir=f'./models/ensemble_{model_key}',
                    num_train_epochs=epochs,
                    per_device_train_batch_size=32,  # Increased for GPU
                    per_device_eval_batch_size=64,   # Increased for GPU
                    learning_rate=2e-5,
                    weight_decay=0.01,
                    warmup_ratio=0.1,
                    eval_strategy="steps",
                    eval_steps=500,  # Less frequent evaluation
                    save_strategy="steps",
                    save_steps=500,  # Less frequent saving
                    logging_steps=100,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_f1_macro",
                    greater_is_better=True,
                    save_total_limit=2,
                    seed=42,
                    fp16=torch.cuda.is_available(),
                    dataloader_num_workers=4,  # Increased for better performance
                    report_to=None,
                    disable_tqdm=True
                )
                
                def compute_metrics(eval_pred):
                    predictions, labels = eval_pred
                    predictions = np.argmax(predictions, axis=1)
                    
                    accuracy = accuracy_score(labels, predictions)
                    f1_macro = f1_score(labels, predictions, average='macro')
                    
                    return {
                        'accuracy': accuracy,
                        'f1_macro': f1_macro
                    }
                
                # Create trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=compute_metrics,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
                )
                
                # Train
                trainer.train()
                
                # Save trained model
                trainer.save_model(f'./models/ensemble_{model_key}_final')
                tokenizer.save_pretrained(f'./models/ensemble_{model_key}_final')
                
                # Evaluate
                eval_results = trainer.evaluate()
                logger.info(f"{model_key} validation results: {eval_results}")
                
                trained_models[model_key] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'config': config,
                    'eval_results': eval_results
                }
                
                # Clear memory
                del trainer
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Failed to train {model_key}: {e}")
                continue
        
        logger.info(f"Successfully trained {len(trained_models)} models")
        return trained_models
    
    def get_model_predictions(self, model_info, texts, batch_size=32):
        """Get predictions from a single model"""
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        max_length = model_info['config']['max_length']
        
        model.eval()
        all_predictions = []
        all_probabilities = []
        
        # Create dataset
        dataset = HateSpeechDataset(
            texts, [0] * len(texts), tokenizer, max_length
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Get predictions and probabilities
                predictions = torch.argmax(logits, dim=-1)
                probabilities = F.softmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def optimize_ensemble_weights(self, models_info, X_val, y_val):
        """Optimize ensemble weights using validation data"""
        logger.info("Optimizing ensemble weights")
        
        # Get predictions from all models
        model_predictions = {}
        model_probabilities = {}
        
        for model_key, model_info in models_info.items():
            logger.info(f"Getting predictions from {model_key}")
            preds, probs = self.get_model_predictions(model_info, X_val)
            model_predictions[model_key] = preds
            model_probabilities[model_key] = probs
        
        # Optimize weights
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Weighted average of probabilities
            ensemble_probs = np.zeros_like(list(model_probabilities.values())[0])
            for i, (model_key, probs) in enumerate(model_probabilities.items()):
                ensemble_probs += weights[i] * probs
            
            # Get predictions
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            
            # Calculate F1-macro (negative for minimization)
            f1 = f1_score(y_val, ensemble_preds, average='macro')
            return -f1
        
        # Initial weights (equal)
        n_models = len(models_info)
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(
            objective, initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        
        optimal_weights = result.x / np.sum(result.x)  # Normalize
        
        # Create weight mapping
        weight_mapping = {}
        for i, model_key in enumerate(models_info.keys()):
            weight_mapping[model_key] = float(optimal_weights[i])
        
        logger.info(f"Optimal weights: {weight_mapping}")
        
        return weight_mapping
    
    def ensemble_predict(self, models_info, texts, weights=None):
        """Make ensemble predictions"""
        if weights is None:
            # Equal weights
            weights = {key: 1.0/len(models_info) for key in models_info.keys()}
        
        # Get predictions from all models
        ensemble_probs = None
        
        for model_key, model_info in models_info.items():
            _, probs = self.get_model_predictions(model_info, texts)
            
            if ensemble_probs is None:
                ensemble_probs = weights[model_key] * probs
            else:
                ensemble_probs += weights[model_key] * probs
        
        # Get final predictions
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        return ensemble_preds, ensemble_probs
    
    def evaluate_ensemble(self, models_info, X_test, y_test, weights=None):
        """Evaluate ensemble performance"""
        logger.info("Evaluating ensemble performance")
        
        # Get ensemble predictions
        predictions, probabilities = self.ensemble_predict(models_info, X_test, weights)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average='macro')
        f1_weighted = f1_score(y_test, predictions, average='weighted')
        
        # Classification report
        class_names = ['Bukan Ujaran Kebencian', 'Ujaran Kebencian - Ringan', 
                       'Ujaran Kebencian - Sedang', 'Ujaran Kebencian - Berat']
        
        report = classification_report(
            y_test, predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        results = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'ensemble_weights': weights if weights else 'equal'
        }
        
        return results

def main():
    logger.info("Starting Multi-Architecture Ensemble Experiment")
    
    # Load data
    logger.info("Loading augmented dataset")
    df = pd.read_csv('data/augmented/augmented_dataset.csv')
    
    # Sample subset for faster training (use 50% of data)
    # Use train_test_split for stratified sampling instead of df.sample with stratify
    X_full = df['text'].values
    y_full = df['label_numeric'].values
    
    X, _, y, _ = train_test_split(
        X_full, y_full, test_size=0.5, random_state=42, stratify=y_full
    )
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Initialize ensemble
    ensemble = MultiArchitectureEnsemble()
    
    # Load models
    ensemble.load_models()
    
    if len(ensemble.models) == 0:
        logger.error("No models loaded successfully. Exiting.")
        return
    
    # Train individual models
    trained_models = ensemble.train_individual_models(
        X_train, y_train, X_val, y_val, epochs=2  # Reduced for faster experimentation
    )
    
    if len(trained_models) == 0:
        logger.error("No models trained successfully. Exiting.")
        return
    
    # Optimize ensemble weights
    optimal_weights = ensemble.optimize_ensemble_weights(trained_models, X_val, y_val)
    
    # Evaluate individual models on test set
    individual_results = {}
    for model_key, model_info in trained_models.items():
        logger.info(f"Evaluating individual model: {model_key}")
        preds, _ = ensemble.get_model_predictions(model_info, X_test)
        
        accuracy = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average='macro')
        
        individual_results[model_key] = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro)
        }
        
        logger.info(f"{model_key} - Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}")
    
    # Evaluate ensemble with equal weights
    logger.info("Evaluating ensemble with equal weights")
    equal_weights_results = ensemble.evaluate_ensemble(trained_models, X_test, y_test)
    
    # Evaluate ensemble with optimized weights
    logger.info("Evaluating ensemble with optimized weights")
    optimized_weights_results = ensemble.evaluate_ensemble(trained_models, X_test, y_test, optimal_weights)
    
    # Save results
    results = {
        'experiment_timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': int(len(df)),
            'train_samples': int(len(X_train)),
            'val_samples': int(len(X_val)),
            'test_samples': int(len(X_test))
        },
        'models_used': list(trained_models.keys()),
        'individual_model_results': individual_results,
        'ensemble_equal_weights': equal_weights_results,
        'ensemble_optimized_weights': optimized_weights_results,
        'optimal_weights': optimal_weights
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/multi_architecture_ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n" + "="*80)
    print("🚀 MULTI-ARCHITECTURE ENSEMBLE RESULTS")
    print("="*80)
    
    print(f"\n📊 DATASET STATISTICS:")
    print(f"   Total samples: {len(df_sample):,}")
    print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    print(f"\n🤖 MODELS USED:")
    for model_key in trained_models.keys():
        config = ensemble.model_configs[model_key]
        print(f"   ✅ {model_key}: {config['name']}")
    
    print(f"\n📈 INDIVIDUAL MODEL PERFORMANCE:")
    for model_key, results in individual_results.items():
        print(f"   {model_key}:")
        print(f"     Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"     F1-Macro: {results['f1_macro']:.4f} ({results['f1_macro']*100:.2f}%)")
    
    print(f"\n🎯 ENSEMBLE PERFORMANCE:")
    
    print(f"\n   Equal Weights:")
    print(f"     Accuracy: {equal_weights_results['accuracy']:.4f} ({equal_weights_results['accuracy']*100:.2f}%)")
    print(f"     F1-Macro: {equal_weights_results['f1_macro']:.4f} ({equal_weights_results['f1_macro']*100:.2f}%)")
    
    print(f"\n   Optimized Weights:")
    print(f"     Accuracy: {optimized_weights_results['accuracy']:.4f} ({optimized_weights_results['accuracy']*100:.2f}%)")
    print(f"     F1-Macro: {optimized_weights_results['f1_macro']:.4f} ({optimized_weights_results['f1_macro']*100:.2f}%)")
    
    print(f"\n⚖️ OPTIMAL WEIGHTS:")
    for model_key, weight in optimal_weights.items():
        print(f"   {model_key}: {weight:.3f}")
    
    # Check if 90% target achieved
    best_accuracy = max(equal_weights_results['accuracy'], optimized_weights_results['accuracy'])
    best_f1 = max(equal_weights_results['f1_macro'], optimized_weights_results['f1_macro'])
    
    print(f"\n🎯 TARGET STATUS:")
    if best_accuracy >= 0.90 and best_f1 >= 0.90:
        print(f"   🎉 90%+ TARGET ACHIEVED!")
        print(f"   ✅ Best Accuracy: {best_accuracy*100:.2f}%")
        print(f"   ✅ Best F1-Macro: {best_f1*100:.2f}%")
    else:
        accuracy_gap = 0.90 - best_accuracy
        f1_gap = 0.90 - best_f1
        print(f"   ⚠️ Target not yet achieved")
        print(f"   📊 Accuracy gap: {accuracy_gap*100:.2f}% (Best: {best_accuracy*100:.2f}%)")
        print(f"   📊 F1-Macro gap: {f1_gap*100:.2f}% (Best: {best_f1*100:.2f}%)")
    
    print(f"\n📁 Results saved to: results/multi_architecture_ensemble_results.json")
    print("="*80)
    
    logger.info("Multi-architecture ensemble experiment completed")
    
    return best_accuracy, best_f1

if __name__ == "__main__":
    main()