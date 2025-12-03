#!/usr/bin/env python3
"""
Hyperparameter Optimization for 90%+ Accuracy
Using Optuna for Bayesian optimization of training parameters
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import optuna
import json
import logging
from datetime import datetime
import os
import warnings
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hyperparameter_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FocalLoss(torch.nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
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

class OptimizedTrainer(Trainer):
    def __init__(self, *args, focal_alpha=1, focal_gamma=2, use_focal_loss=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.use_focal_loss:
            loss = self.focal_loss(logits, labels)
        else:
            loss = outputs.loss
        
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

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    weight_decay = trial.suggest_float('weight_decay', 0.001, 0.1, log=True)
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.05, 0.2)
    focal_alpha = trial.suggest_float('focal_alpha', 0.5, 2.0)
    focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0)
    gradient_accumulation_steps = trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 4])
    max_length = trial.suggest_categorical('max_length', [256, 512])
    
    logger.info(f"Trial {trial.number}: lr={learning_rate:.2e}, bs={batch_size}, wd={weight_decay:.3f}")
    
    try:
        # Load data
        df = pd.read_csv('data/augmented/augmented_dataset.csv')
        
        # Sample subset for faster optimization (use 20% of data)
        # Use train_test_split for stratified sampling instead of df.sample with stratify
        X_full = df['text'].values
        y_full = df['label_numeric'].values
        
        X, _, y, _ = train_test_split(
            X_full, y_full, test_size=0.8, random_state=42, stratify=y_full
        )
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Load model and tokenizer
        model_name = 'indobenchmark/indobert-base-p1'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=4,
            problem_type="single_label_classification"
        )
        
        # Create datasets
        train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, max_length)
        val_dataset = HateSpeechDataset(X_val, y_val, tokenizer, max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'./models/optuna_trial_{trial.number}',
            num_train_epochs=2,  # Reduced for faster optimization
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            seed=42,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=2,
            report_to=None,
            disable_tqdm=True
        )
        
        # Create trainer
        trainer = OptimizedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            use_focal_loss=True
        )
        
        # Train
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        
        # Clean up
        del model
        del trainer
        torch.cuda.empty_cache()
        
        # Return F1-macro as objective (to maximize)
        return eval_results['eval_f1_macro']
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return 0.0  # Return low score for failed trials

def run_hyperparameter_optimization():
    """Run Optuna hyperparameter optimization"""
    logger.info("Starting hyperparameter optimization")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='javanese_hate_speech_optimization',
        storage='sqlite:///optuna_study.db',
        load_if_exists=True
    )
    
    # Optimize
    n_trials = 50  # Adjust based on computational resources
    logger.info(f"Running {n_trials} optimization trials")
    
    study.optimize(objective, n_trials=n_trials, timeout=3600*4)  # 4 hours timeout
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best F1-macro: {best_value:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Save results
    results = {
        'optimization_timestamp': datetime.now().isoformat(),
        'best_f1_macro': float(best_value),
        'best_parameters': best_params,
        'n_trials': len(study.trials),
        'study_summary': {
            'best_trial_number': study.best_trial.number,
            'best_trial_value': float(study.best_value),
            'optimization_direction': 'maximize'
        },
        'all_trials': [
            {
                'number': trial.number,
                'value': float(trial.value) if trial.value is not None else None,
                'params': trial.params,
                'state': trial.state.name
            }
            for trial in study.trials
        ]
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/hyperparameter_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n" + "="*80)
    print("🔧 HYPERPARAMETER OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"\n🎯 BEST PERFORMANCE:")
    print(f"   F1-Macro: {best_value:.4f} ({best_value*100:.2f}%)")
    print(f"   Trial Number: {study.best_trial.number}")
    
    print(f"\n⚙️ OPTIMAL HYPERPARAMETERS:")
    for param, value in best_params.items():
        if isinstance(value, float):
            if param == 'learning_rate':
                print(f"   {param}: {value:.2e}")
            else:
                print(f"   {param}: {value:.4f}")
        else:
            print(f"   {param}: {value}")
    
    print(f"\n📊 OPTIMIZATION STATISTICS:")
    print(f"   Total trials: {len(study.trials)}")
    print(f"   Completed trials: {len([t for t in study.trials if t.state.name == 'COMPLETE'])}")
    print(f"   Failed trials: {len([t for t in study.trials if t.state.name == 'FAIL'])}")
    
    # Top 5 trials
    completed_trials = [t for t in study.trials if t.value is not None]
    top_trials = sorted(completed_trials, key=lambda x: x.value, reverse=True)[:5]
    
    print(f"\n🏆 TOP 5 TRIALS:")
    for i, trial in enumerate(top_trials, 1):
        print(f"   {i}. Trial {trial.number}: F1={trial.value:.4f}")
        print(f"      LR: {trial.params['learning_rate']:.2e}, BS: {trial.params['batch_size']}")
    
    print(f"\n📁 Results saved to: results/hyperparameter_optimization_results.json")
    print("="*80)
    
    return best_params, best_value

def train_with_optimal_params(best_params):
    """Train final model with optimal hyperparameters"""
    logger.info("Training final model with optimal hyperparameters")
    
    # Load full augmented dataset
    df = pd.read_csv('data/augmented/augmented_dataset.csv')
    
    X = df['text'].values
    y = df['label_numeric'].values
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Load model and tokenizer
    model_name = 'indobenchmark/indobert-base-p1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
        problem_type="single_label_classification"
    )
    
    # Create datasets
    max_length = best_params.get('max_length', 512)
    train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = HateSpeechDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = HateSpeechDataset(X_test, y_test, tokenizer, max_length)
    
    # Training arguments with optimal parameters
    training_args = TrainingArguments(
        output_dir='./models/optimized_model',
        num_train_epochs=5,
        per_device_train_batch_size=best_params['batch_size'],
        per_device_eval_batch_size=best_params['batch_size'] * 2,
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        warmup_ratio=best_params['warmup_ratio'],
        gradient_accumulation_steps=best_params['gradient_accumulation_steps'],
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,
        seed=42,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to=None
    )
    
    # Create trainer with optimal focal loss parameters
    trainer = OptimizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        focal_alpha=best_params['focal_alpha'],
        focal_gamma=best_params['focal_gamma'],
        use_focal_loss=True
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model('./models/optimized_model_final')
    tokenizer.save_pretrained('./models/optimized_model_final')
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    
    logger.info(f"Optimized model test results: {test_results}")
    
    return test_results

def main():
    """Main function"""
    logger.info("Starting hyperparameter optimization experiment")
    
    # Run optimization
    best_params, best_value = run_hyperparameter_optimization()
    
    # Train final model with optimal parameters
    if best_value > 0.88:  # Only train final model if optimization shows promise
        logger.info("Optimization successful, training final model")
        test_results = train_with_optimal_params(best_params)
        
        print(f"\n🎯 FINAL OPTIMIZED MODEL RESULTS:")
        print(f"   Test Accuracy: {test_results['eval_accuracy']:.4f} ({test_results['eval_accuracy']*100:.2f}%)")
        print(f"   Test F1-Macro: {test_results['eval_f1_macro']:.4f} ({test_results['eval_f1_macro']*100:.2f}%)")
        
        # Check if 90% target achieved
        if test_results['eval_accuracy'] >= 0.90 and test_results['eval_f1_macro'] >= 0.90:
            print(f"\n🎉 90%+ TARGET ACHIEVED!")
            print(f"   ✅ Accuracy: {test_results['eval_accuracy']*100:.2f}%")
            print(f"   ✅ F1-Macro: {test_results['eval_f1_macro']*100:.2f}%")
        else:
            print(f"\n⚠️ 90% target not yet achieved")
            print(f"   Gap - Accuracy: {(0.90 - test_results['eval_accuracy'])*100:.2f}%")
            print(f"   Gap - F1-Macro: {(0.90 - test_results['eval_f1_macro'])*100:.2f}%")
    else:
        logger.warning(f"Optimization did not show significant improvement (best: {best_value:.4f})")
        print(f"\n⚠️ Hyperparameter optimization did not achieve expected improvement")
        print(f"   Best F1-Macro: {best_value:.4f} ({best_value*100:.2f}%)")
        print(f"   Consider trying different optimization strategies")

if __name__ == "__main__":
    main()