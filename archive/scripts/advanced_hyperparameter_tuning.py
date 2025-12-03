#!/usr/bin/env python3
"""
Advanced Hyperparameter Tuning with Optuna
Comprehensive optimization for 90%+ accuracy
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import json
import logging
from datetime import datetime
import os
import warnings
from collections import defaultdict
import gc
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/advanced_hyperparameter_tuning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing for regularization"""
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class AdvancedTrainer(Trainer):
    """Custom trainer with advanced loss functions"""
    def __init__(self, loss_type='focal', focal_alpha=1.0, focal_gamma=2.0, 
                 label_smoothing=0.1, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.class_weights = class_weights
        
        if loss_type == 'focal':
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif loss_type == 'label_smoothing':
            self.loss_fn = LabelSmoothingLoss(num_classes=4, smoothing=label_smoothing)
        elif loss_type == 'weighted_ce':
            if class_weights is not None:
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
                self.loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
            else:
                self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Move loss function to same device as logits
        if hasattr(self.loss_fn, 'to'):
            self.loss_fn = self.loss_fn.to(logits.device)
        
        loss = self.loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

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

class AdvancedHyperparameterTuner:
    def __init__(self, model_name='indobenchmark/indobert-base-p1', device='cuda'):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.best_score = 0.0
        self.best_params = None
        
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'combined_score': (accuracy + f1_macro) / 2  # Combined metric for optimization
        }
    
    def objective(self, trial, X_train, y_train, X_val, y_val, class_weights):
        """Optuna objective function"""
        
        # Suggest hyperparameters
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True),
            'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.3),
            'max_length': trial.suggest_categorical('max_length', [256, 384, 512]),
            'gradient_accumulation_steps': trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 4]),
            'lr_scheduler_type': trial.suggest_categorical('lr_scheduler_type', 
                ['linear', 'cosine', 'cosine_with_restarts', 'polynomial']),
            'adam_epsilon': trial.suggest_float('adam_epsilon', 1e-9, 1e-6, log=True),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.5, 2.0),
            
            # Loss function parameters
            'loss_type': trial.suggest_categorical('loss_type', ['focal', 'label_smoothing', 'weighted_ce']),
        }
        
        # Loss-specific parameters
        if params['loss_type'] == 'focal':
            params['focal_alpha'] = trial.suggest_float('focal_alpha', 0.25, 2.0)
            params['focal_gamma'] = trial.suggest_float('focal_gamma', 1.0, 3.0)
        elif params['loss_type'] == 'label_smoothing':
            params['label_smoothing'] = trial.suggest_float('label_smoothing', 0.05, 0.3)
        
        # Dropout parameters
        params['hidden_dropout_prob'] = trial.suggest_float('hidden_dropout_prob', 0.1, 0.5)
        params['attention_probs_dropout_prob'] = trial.suggest_float('attention_probs_dropout_prob', 0.1, 0.5)
        
        try:
            # Load model with custom dropout
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=4,
                hidden_dropout_prob=params['hidden_dropout_prob'],
                attention_probs_dropout_prob=params['attention_probs_dropout_prob'],
                problem_type="single_label_classification"
            )
            model.to(self.device)
            
            # Create datasets
            train_dataset = HateSpeechDataset(
                X_train, y_train, self.tokenizer, params['max_length']
            )
            val_dataset = HateSpeechDataset(
                X_val, y_val, self.tokenizer, params['max_length']
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f'./tmp/trial_{trial.number}',
                num_train_epochs=2,  # Reduced for faster optimization
                per_device_train_batch_size=params['batch_size'],
                per_device_eval_batch_size=params['batch_size'] * 2,
                gradient_accumulation_steps=params['gradient_accumulation_steps'],
                learning_rate=params['learning_rate'],
                weight_decay=params['weight_decay'],
                warmup_ratio=params['warmup_ratio'],
                lr_scheduler_type=params['lr_scheduler_type'],
                adam_epsilon=params['adam_epsilon'],
                max_grad_norm=params['max_grad_norm'],
                eval_strategy="steps",
                eval_steps=100,
                save_strategy="no",
                logging_steps=50,
                load_best_model_at_end=True,
                metric_for_best_model="eval_combined_score",
                greater_is_better=True,
                seed=42,
                fp16=torch.cuda.is_available(),
                dataloader_num_workers=0,  # Avoid multiprocessing issues
                report_to=None,
                disable_tqdm=True,
                remove_unused_columns=False
            )
            
            # Create trainer with advanced loss
            trainer_kwargs = {
                'model': model,
                'args': training_args,
                'train_dataset': train_dataset,
                'eval_dataset': val_dataset,
                'compute_metrics': self.compute_metrics,
                'callbacks': [EarlyStoppingCallback(early_stopping_patience=2)]
            }
            
            # Add loss-specific parameters
            if params['loss_type'] == 'focal':
                trainer_kwargs.update({
                    'loss_type': 'focal',
                    'focal_alpha': params['focal_alpha'],
                    'focal_gamma': params['focal_gamma']
                })
            elif params['loss_type'] == 'label_smoothing':
                trainer_kwargs.update({
                    'loss_type': 'label_smoothing',
                    'label_smoothing': params['label_smoothing']
                })
            elif params['loss_type'] == 'weighted_ce':
                trainer_kwargs.update({
                    'loss_type': 'weighted_ce',
                    'class_weights': class_weights
                })
            
            trainer = AdvancedTrainer(**trainer_kwargs)
            
            # Train
            trainer.train()
            
            # Evaluate
            eval_results = trainer.evaluate()
            
            # Get the combined score
            score = eval_results['eval_combined_score']
            
            # Clean up
            del model, trainer, train_dataset, val_dataset
            torch.cuda.empty_cache()
            gc.collect()
            
            # Report intermediate value for pruning
            trial.report(score, step=0)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            return score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Clean up on error
            torch.cuda.empty_cache()
            gc.collect()
            return 0.0
    
    def optimize(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Run hyperparameter optimization"""
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = class_weights.tolist()
        
        logger.info(f"Class weights: {class_weights}")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val, class_weights),
            n_trials=n_trials,
            timeout=None,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return study
    
    def train_final_model(self, X_train, y_train, X_val, y_val, X_test, y_test, epochs=3):
        """Train final model with best parameters"""
        if self.best_params is None:
            raise ValueError("No optimization has been run. Call optimize() first.")
        
        logger.info("Training final model with best parameters")
        
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = class_weights.tolist()
        
        # Load model with best parameters
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=4,
            hidden_dropout_prob=self.best_params.get('hidden_dropout_prob', 0.1),
            attention_probs_dropout_prob=self.best_params.get('attention_probs_dropout_prob', 0.1),
            problem_type="single_label_classification"
        )
        model.to(self.device)
        
        # Create datasets
        max_length = self.best_params.get('max_length', 512)
        train_dataset = HateSpeechDataset(X_train, y_train, self.tokenizer, max_length)
        val_dataset = HateSpeechDataset(X_val, y_val, self.tokenizer, max_length)
        test_dataset = HateSpeechDataset(X_test, y_test, self.tokenizer, max_length)
        
        # Training arguments with best parameters
        training_args = TrainingArguments(
            output_dir='./models/optimized_model',
            num_train_epochs=epochs,
            per_device_train_batch_size=self.best_params.get('batch_size', 16),
            per_device_eval_batch_size=self.best_params.get('batch_size', 16) * 2,
            gradient_accumulation_steps=self.best_params.get('gradient_accumulation_steps', 1),
            learning_rate=self.best_params.get('learning_rate', 2e-5),
            weight_decay=self.best_params.get('weight_decay', 0.01),
            warmup_ratio=self.best_params.get('warmup_ratio', 0.1),
            lr_scheduler_type=self.best_params.get('lr_scheduler_type', 'linear'),
            adam_epsilon=self.best_params.get('adam_epsilon', 1e-8),
            max_grad_norm=self.best_params.get('max_grad_norm', 1.0),
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_combined_score",
            greater_is_better=True,
            save_total_limit=2,
            seed=42,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=2,
            report_to=None
        )
        
        # Create trainer with best loss function
        trainer_kwargs = {
            'model': model,
            'args': training_args,
            'train_dataset': train_dataset,
            'eval_dataset': val_dataset,
            'compute_metrics': self.compute_metrics,
            'callbacks': [EarlyStoppingCallback(early_stopping_patience=3)]
        }
        
        # Add loss-specific parameters
        loss_type = self.best_params.get('loss_type', 'focal')
        if loss_type == 'focal':
            trainer_kwargs.update({
                'loss_type': 'focal',
                'focal_alpha': self.best_params.get('focal_alpha', 1.0),
                'focal_gamma': self.best_params.get('focal_gamma', 2.0)
            })
        elif loss_type == 'label_smoothing':
            trainer_kwargs.update({
                'loss_type': 'label_smoothing',
                'label_smoothing': self.best_params.get('label_smoothing', 0.1)
            })
        elif loss_type == 'weighted_ce':
            trainer_kwargs.update({
                'loss_type': 'weighted_ce',
                'class_weights': class_weights
            })
        
        trainer = AdvancedTrainer(**trainer_kwargs)
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model('./models/optimized_model_final')
        self.tokenizer.save_pretrained('./models/optimized_model_final')
        
        # Evaluate on validation set
        val_results = trainer.evaluate(eval_dataset=val_dataset)
        
        # Evaluate on test set
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        
        # Get detailed predictions for test set
        test_predictions = trainer.predict(test_dataset)
        test_preds = np.argmax(test_predictions.predictions, axis=1)
        
        # Classification report
        class_names = ['Bukan Ujaran Kebencian', 'Ujaran Kebencian - Ringan', 
                       'Ujaran Kebencian - Sedang', 'Ujaran Kebencian - Berat']
        
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(
            y_test, test_preds, 
            target_names=class_names, 
            output_dict=True
        )
        
        cm = confusion_matrix(y_test, test_preds)
        
        results = {
            'best_hyperparameters': self.best_params,
            'optimization_score': float(self.best_score),
            'validation_results': {k: float(v) for k, v in val_results.items()},
            'test_results': {k: float(v) for k, v in test_results.items()},
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        return results

def main():
    logger.info("Starting Advanced Hyperparameter Tuning")
    
    # Load data
    logger.info("Loading augmented dataset")
    df = pd.read_csv('data/augmented/augmented_dataset.csv')
    
    # Use subset for faster optimization (30% of data)
    df_sample = df.sample(frac=0.3, random_state=42, stratify=df['label_numeric'])
    
    X = df_sample['text'].values
    y = df_sample['label_numeric'].values
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Initialize tuner
    tuner = AdvancedHyperparameterTuner()
    
    # Run optimization
    study = tuner.optimize(X_train, y_train, X_val, y_val, n_trials=30)
    
    # Train final model
    final_results = tuner.train_final_model(
        X_train, y_train, X_val, y_val, X_test, y_test, epochs=3
    )
    
    # Add study information
    final_results['optimization_study'] = {
        'n_trials': len(study.trials),
        'best_trial_number': study.best_trial.number,
        'optimization_history': [
            {'trial': i, 'value': trial.value} 
            for i, trial in enumerate(study.trials) 
            if trial.value is not None
        ]
    }
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/advanced_hyperparameter_tuning_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print results
    print("\n" + "="*80)
    print("🔧 ADVANCED HYPERPARAMETER TUNING RESULTS")
    print("="*80)
    
    print(f"\n📊 DATASET STATISTICS:")
    print(f"   Total samples: {len(df_sample):,}")
    print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    print(f"\n🎯 OPTIMIZATION RESULTS:")
    print(f"   Trials completed: {len(study.trials)}")
    print(f"   Best trial: #{study.best_trial.number}")
    print(f"   Best optimization score: {tuner.best_score:.4f} ({tuner.best_score*100:.2f}%)")
    
    print(f"\n⚙️ BEST HYPERPARAMETERS:")
    for param, value in tuner.best_params.items():
        if isinstance(value, float):
            print(f"   {param}: {value:.6f}")
        else:
            print(f"   {param}: {value}")
    
    print(f"\n📈 FINAL MODEL PERFORMANCE:")
    test_results = final_results['test_results']
    print(f"   Test Accuracy: {test_results['eval_accuracy']:.4f} ({test_results['eval_accuracy']*100:.2f}%)")
    print(f"   Test F1-Macro: {test_results['eval_f1_macro']:.4f} ({test_results['eval_f1_macro']*100:.2f}%)")
    print(f"   Test F1-Weighted: {test_results['eval_f1_weighted']:.4f} ({test_results['eval_f1_weighted']*100:.2f}%)")
    
    # Check if 90% target achieved
    test_accuracy = test_results['eval_accuracy']
    test_f1 = test_results['eval_f1_macro']
    
    print(f"\n🎯 TARGET STATUS:")
    if test_accuracy >= 0.90 and test_f1 >= 0.90:
        print(f"   🎉 90%+ TARGET ACHIEVED!")
        print(f"   ✅ Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"   ✅ Test F1-Macro: {test_f1*100:.2f}%")
    else:
        accuracy_gap = 0.90 - test_accuracy
        f1_gap = 0.90 - test_f1
        print(f"   ⚠️ Target not yet achieved")
        print(f"   📊 Accuracy gap: {accuracy_gap*100:.2f}% (Current: {test_accuracy*100:.2f}%)")
        print(f"   📊 F1-Macro gap: {f1_gap*100:.2f}% (Current: {test_f1*100:.2f}%)")
    
    print(f"\n📁 Results saved to: results/advanced_hyperparameter_tuning_results.json")
    print(f"📁 Model saved to: models/optimized_model_final/")
    print("="*80)
    
    logger.info("Advanced hyperparameter tuning completed")
    
    return test_accuracy, test_f1

if __name__ == "__main__":
    main()