import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import json
import os
from datetime import datetime
import logging
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

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

def compute_metrics(eval_pred):
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

class OptimizationObjective:
    def __init__(self, model_name, X_train, y_train, X_val, y_val, tokenizer):
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.tokenizer = tokenizer
        self.trial_count = 0
    
    def __call__(self, trial):
        self.trial_count += 1
        logger.info(f"Starting trial {self.trial_count}: {trial.number}")
        
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        num_epochs = trial.suggest_int('num_epochs', 3, 8)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.3)
        label_smoothing = trial.suggest_float('label_smoothing_factor', 0.0, 0.2)
        gradient_accumulation_steps = trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 4])
        lr_scheduler_type = trial.suggest_categorical('lr_scheduler_type', ['linear', 'cosine', 'polynomial'])
        
        # Advanced hyperparameters
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        attention_dropout = trial.suggest_float('attention_dropout', 0.1, 0.3)
        
        try:
            # Load model with suggested dropout rates
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=4,
                problem_type="single_label_classification",
                hidden_dropout_prob=dropout_rate,
                attention_probs_dropout_prob=attention_dropout
            )
            
            # Create datasets
            train_dataset = JavaneseDataset(self.X_train, self.y_train, self.tokenizer)
            val_dataset = JavaneseDataset(self.X_val, self.y_val, self.tokenizer)
            
            # Training arguments
            model_safe_name = self.model_name.replace('/', '_')
            output_dir = f"tmp/optuna_trial_{trial.number}_{model_safe_name}"
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size * 2,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                logging_steps=50,
                eval_strategy="steps",
        eval_steps=200,
                save_strategy="steps",
                 save_steps=200,
                 load_best_model_at_end=True,
                metric_for_best_model="f1_macro",
                greater_is_better=True,
                seed=42,
                fp16=torch.cuda.is_available(),
                dataloader_num_workers=0,
                remove_unused_columns=False,
                label_smoothing_factor=label_smoothing,
                learning_rate=learning_rate,
                lr_scheduler_type=lr_scheduler_type,
                gradient_accumulation_steps=gradient_accumulation_steps,
                report_to=None,  # Disable wandb/tensorboard
                logging_dir=None,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            
            # Train the model
            trainer.train()
            
            # Evaluate
            eval_results = trainer.evaluate()
            f1_macro = eval_results['eval_f1_macro']
            
            logger.info(f"Trial {trial.number} completed - F1 Macro: {f1_macro:.4f}")
            
            # Clean up
            del model
            del trainer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Remove temporary directory
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            
            return f1_macro
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            # Clean up on error
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return 0.0  # Return low score for failed trials

def optimize_hyperparameters(model_name, data_path, n_trials=100, timeout=None):
    """Optimize hyperparameters using Optuna"""
    
    logger.info(f"Starting hyperparameter optimization for {model_name}")
    logger.info(f"Number of trials: {n_trials}")
    
    # Load data
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    
    # Use a subset for faster optimization if dataset is large
    if len(df) > 15000:
        df = df.sample(n=15000, random_state=42)
        logger.info(f"Using subset of {len(df)} samples for optimization")
    
    texts = df['text'].tolist()
    labels = df['label_numeric'].tolist()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create study
    study_name = f"hyperopt_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Use SQLite storage for persistence
    storage_url = f"sqlite:///optuna_studies/{study_name}.db"
    os.makedirs('optuna_studies', exist_ok=True)
    
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Create objective
    objective = OptimizationObjective(model_name, X_train, y_train, X_val, y_val, tokenizer)
    
    # Optimize
    logger.info("Starting optimization...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Optimization completed!")
    logger.info(f"Best F1 Macro: {best_value:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Save results
    results = {
        'model_name': model_name,
        'study_name': study_name,
        'n_trials': len(study.trials),
        'best_value': best_value,
        'best_params': best_params,
        'optimization_history': [
            {
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            }
            for trial in study.trials
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to file
    model_safe_name = model_name.replace('/', '_')
    results_file = f"results/hyperopt_{model_safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")
    
    return results, study

def train_with_best_params(model_name, best_params, data_path, test_size=0.2):
    """Train final model with best hyperparameters"""
    
    logger.info(f"Training final model with best parameters")
    
    # Load data
    df = pd.read_csv(data_path)
    texts = df['text'].tolist()
    labels = df['label_numeric'].tolist()
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
        problem_type="single_label_classification",
        hidden_dropout_prob=best_params.get('dropout_rate', 0.1),
        attention_probs_dropout_prob=best_params.get('attention_dropout', 0.1)
    )
    
    # Create datasets
    train_dataset = JavaneseDataset(X_train, y_train, tokenizer)
    val_dataset = JavaneseDataset(X_val, y_val, tokenizer)
    test_dataset = JavaneseDataset(X_test, y_test, tokenizer)
    
    # Training arguments with best parameters
    model_safe_name = model_name.replace('/', '_')
    output_dir = f"models/{model_safe_name}_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=best_params['num_epochs'],
        per_device_train_batch_size=best_params['batch_size'],
        per_device_eval_batch_size=best_params['batch_size'] * 2,
        warmup_ratio=best_params['warmup_ratio'],
        weight_decay=best_params['weight_decay'],
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,
        seed=42,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_smoothing_factor=best_params['label_smoothing_factor'],
        learning_rate=best_params['learning_rate'],
        lr_scheduler_type=best_params['lr_scheduler_type'],
        gradient_accumulation_steps=best_params['gradient_accumulation_steps'],
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
    
    # Train
    trainer.train()
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1_macro = f1_score(y_test, y_pred, average='macro')
    test_f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    results = {
        'model_name': model_name,
        'output_dir': output_dir,
        'best_params': best_params,
        'test_metrics': {
            'accuracy': float(test_accuracy),
            'f1_macro': float(test_f1_macro),
            'f1_weighted': float(test_f1_weighted)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Final Results - Accuracy: {test_accuracy:.4f}, F1-Macro: {test_f1_macro:.4f}")
    
    return results, trainer

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    models_to_optimize = [
        'indolem/indobert-base-uncased',
        'indobenchmark/indobert-base-p1',
        'flax-community/indonesian-roberta-base'
    ]
    
    # Use augmented dataset if available, otherwise use original
    data_paths = [
        'data/augmented/augmented_dataset.csv',
        'balanced_dataset.csv'
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        logger.error("No dataset found!")
        return
    
    logger.info(f"Using dataset: {data_path}")
    
    # Optimization settings
    n_trials = 50  # Reduced for faster execution
    timeout = 3600 * 4  # 4 hours timeout
    
    all_results = []
    
    for model_name in models_to_optimize:
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"Optimizing hyperparameters for {model_name}")
            logger.info(f"{'='*80}")
            
            # Optimize hyperparameters
            opt_results, study = optimize_hyperparameters(
                model_name, data_path, n_trials=n_trials, timeout=timeout
            )
            
            # Train final model with best parameters
            logger.info(f"Training final model with best parameters...")
            final_results, trainer = train_with_best_params(
                model_name, opt_results['best_params'], data_path
            )
            
            # Combine results
            combined_results = {
                'model_name': model_name,
                'optimization_results': opt_results,
                'final_model_results': final_results
            }
            
            all_results.append(combined_results)
            
            # Save individual results
            model_safe_name = model_name.replace('/', '_')
            result_file = f"results/{model_safe_name}_hyperopt_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(combined_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {result_file}")
            
        except Exception as e:
            logger.error(f"Error optimizing {model_name}: {str(e)}")
            continue
    
    # Save combined results
    final_summary = {
        'experiment_name': 'Advanced Hyperparameter Optimization',
        'timestamp': datetime.now().isoformat(),
        'models_optimized': len(all_results),
        'results': all_results,
        'best_overall': None
    }
    
    # Find best overall model
    if all_results:
        best_model = max(all_results, 
                        key=lambda x: x['final_model_results']['test_metrics']['f1_macro'])
        final_summary['best_overall'] = {
            'model_name': best_model['model_name'],
            'accuracy': best_model['final_model_results']['test_metrics']['accuracy'],
            'f1_macro': best_model['final_model_results']['test_metrics']['f1_macro'],
            'best_params': best_model['optimization_results']['best_params']
        }
    
    # Save final summary
    summary_file = f"results/hyperopt_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("HYPERPARAMETER OPTIMIZATION - SUMMARY")
    logger.info(f"{'='*80}")
    
    if all_results:
        logger.info(f"Models optimized: {len(all_results)}")
        logger.info(f"Best model: {final_summary['best_overall']['model_name']}")
        logger.info(f"Best accuracy: {final_summary['best_overall']['accuracy']:.4f}")
        logger.info(f"Best F1-Macro: {final_summary['best_overall']['f1_macro']:.4f}")
        
        logger.info("\nAll Results:")
        for result in all_results:
            metrics = result['final_model_results']['test_metrics']
            logger.info(f"  {result['model_name']}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
    
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info(f"{'='*80}")

if __name__ == "__main__":
    main()