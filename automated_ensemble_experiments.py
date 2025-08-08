#!/usr/bin/env python3
"""
Automated Multi-Experiment Ensemble Pipeline
Menjalankan beberapa eksperimen ensemble secara otomatis tanpa intervensi manual

Author: AI Assistant
Date: 2025-08-07
"""

import os
import sys
import time
import json
import logging
import argparse
import traceback
from datetime import datetime
from pathlib import Path
import subprocess
import psutil
import torch
from experiment_state_manager import ExperimentStateManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/automated_experiments.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AutomatedExperimentRunner:
    def __init__(self, state_manager=None):
        self.start_time = datetime.now()
        self.results = {}
        self.state_manager = state_manager or ExperimentStateManager()
        self.experiment_configs = [
            {
                'name': 'multi_architecture_ensemble_full',
                'script': 'multi_architecture_ensemble.py',
                'description': 'Full 3-epoch ensemble training',
                'estimated_time_minutes': 30,
                'epochs': 3
            },
            {
                'name': 'multi_architecture_ensemble_extended',
                'script': 'multi_architecture_ensemble_extended.py',
                'description': '5-epoch ensemble with advanced optimization',
                'estimated_time_minutes': 50,
                'epochs': 5
            },
            {
                'name': 'ensemble_with_data_augmentation',
                'script': 'ensemble_with_augmentation.py',
                'description': 'Ensemble with enhanced data augmentation',
                'estimated_time_minutes': 40,
                'epochs': 3
            }
        ]
        
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
    def check_system_resources(self):
        """Check system resources before starting experiments"""
        logger.info("=== System Resource Check ===")
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.warning("No GPU available - experiments will run on CPU (very slow)")
            
        # Check RAM
        ram_gb = psutil.virtual_memory().total / 1024**3
        logger.info(f"RAM: {ram_gb:.1f} GB")
        
        # Check disk space
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / 1024**3
        logger.info(f"Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 5:
            logger.warning("Low disk space - may cause issues")
            
        return True
        
    def create_extended_ensemble_script(self):
        """Create extended ensemble script with 5 epochs"""
        script_content = '''
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
'''
        
        with open('multi_architecture_ensemble_extended.py', 'w', encoding='utf-8') as f:
            f.write(script_content)
        logger.info("Created extended ensemble script")
        
    def create_augmentation_ensemble_script(self):
        """Create ensemble script with enhanced data augmentation"""
        script_content = '''
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
'''
        
        with open('ensemble_with_augmentation.py', 'w', encoding='utf-8') as f:
            f.write(script_content)
        logger.info("Created augmentation ensemble script")
        
    def run_experiment(self, config):
        """Run a single experiment"""
        experiment_name = config['name']
        script_path = config['script']
        
        # Check if experiment should be skipped
        if self.state_manager.should_skip_experiment(experiment_name):
            logger.info(f"Skipping {experiment_name} - already completed")
            return self.state_manager.get_experiment_result(experiment_name)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Experiment: {experiment_name}")
        logger.info(f"Description: {config['description']}")
        logger.info(f"Estimated time: {config['estimated_time_minutes']} minutes")
        logger.info(f"{'='*60}")
        
        # Mark experiment as running
        self.state_manager.start_experiment(experiment_name)
        
        start_time = time.time()
        
        try:
            # Run the experiment
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=config['estimated_time_minutes'] * 60 * 2  # 2x safety margin
            )
            
            end_time = time.time()
            duration_minutes = (end_time - start_time) / 60
            
            if result.returncode == 0:
                logger.info(f"✅ {experiment_name} completed successfully in {duration_minutes:.1f} minutes")
                experiment_result = {
                    'status': 'success',
                    'duration_minutes': duration_minutes,
                    'stdout': result.stdout[-1000:],  # Last 1000 chars
                    'stderr': result.stderr[-500:] if result.stderr else None
                }
                self.results[experiment_name] = experiment_result
                self.state_manager.complete_experiment(experiment_name, experiment_result)
            else:
                logger.error(f"{experiment_name} failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                experiment_result = {
                    'status': 'failed',
                    'duration_minutes': duration_minutes,
                    'return_code': result.returncode,
                    'stdout': result.stdout[-1000:],
                    'stderr': result.stderr[-1000:] if result.stderr else None
                }
                self.results[experiment_name] = experiment_result
                self.state_manager.fail_experiment(experiment_name, f"Return code {result.returncode}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"{experiment_name} timed out after {config['estimated_time_minutes'] * 2} minutes")
            experiment_result = {
                'status': 'timeout',
                'duration_minutes': config['estimated_time_minutes'] * 2
            }
            self.results[experiment_name] = experiment_result
            self.state_manager.fail_experiment(experiment_name, "Timeout")
            
        except Exception as e:
            logger.error(f"💥 {experiment_name} crashed: {str(e)}")
            experiment_result = {
                'status': 'crashed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.results[experiment_name] = experiment_result
            self.state_manager.fail_experiment(experiment_name, str(e))
            
    def save_final_results(self):
        """Save comprehensive results"""
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        
        final_results = {
            'experiment_suite': 'automated_ensemble_experiments',
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration_hours': total_duration.total_seconds() / 3600,
            'experiments': self.results,
            'summary': {
                'total_experiments': len(self.experiment_configs),
                'successful': len([r for r in self.results.values() if r.get('status') == 'success']),
                'failed': len([r for r in self.results.values() if r.get('status') == 'failed']),
                'timeout': len([r for r in self.results.values() if r.get('status') == 'timeout']),
                'crashed': len([r for r in self.results.values() if r.get('status') == 'crashed'])
            }
        }
        
        # Save to file
        results_file = f"results/automated_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
            
        logger.info(f"\n{'='*60}")
        logger.info("FINAL RESULTS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total duration: {total_duration.total_seconds()/3600:.1f} hours")
        logger.info(f"Successful experiments: {final_results['summary']['successful']}/{len(self.experiment_configs)}")
        logger.info(f"Results saved to: {results_file}")
        
        return final_results
        
    def run_all_experiments(self):
        """Run all experiments sequentially"""
        logger.info("Starting Automated Ensemble Experiment Suite")
        logger.info(f"Total experiments planned: {len(self.experiment_configs)}")
        
        # Register experiments in state manager
        for config in self.experiment_configs:
            self.state_manager.register_experiment(config['name'], config)
        
        # Check system resources
        self.check_system_resources()
        
        # Create additional scripts
        self.create_extended_ensemble_script()
        self.create_augmentation_ensemble_script()
        
        # Get pending experiments
        pending_experiments = self.state_manager.get_pending_experiments()
        if not pending_experiments:
            logger.info("All experiments already completed!")
            return self.save_final_results()
        
        logger.info(f"Found {len(pending_experiments)} pending experiments")
        
        # Filter experiments to only run pending ones
        experiments_to_run = [exp for exp in self.experiment_configs if exp['name'] in pending_experiments]
        
        # Estimate total time
        total_estimated_minutes = sum(config['estimated_time_minutes'] for config in experiments_to_run)
        logger.info(f"Estimated total time: {total_estimated_minutes/60:.1f} hours")
        
        # Run each experiment
        for i, config in enumerate(experiments_to_run, 1):
            logger.info(f"\nProgress: {i}/{len(experiments_to_run)}")
            
            # Check if script exists
            if not os.path.exists(config['script']):
                logger.warning(f"Script {config['script']} not found, skipping...")
                continue
                
            self.run_experiment(config)
            
            # Brief pause between experiments
            if i < len(experiments_to_run):
                logger.info("Pausing 30 seconds before next experiment...")
                time.sleep(30)
        
        # Save final results
        return self.save_final_results()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run automated ensemble experiments')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')
    args = parser.parse_args()
    
    try:
        # Initialize state manager
        state_manager = ExperimentStateManager()
        
        # Initialize runner with state manager
        runner = AutomatedExperimentRunner(state_manager=state_manager)
        
        # Handle resume mode
        if args.resume:
            logger.info("\n" + "="*80)
            logger.info("RESUME MODE ACTIVATED")
            logger.info("="*80)
            
            # Analyze previous run
            analysis = state_manager.analyze_previous_run()
            
            if not analysis["has_previous_run"]:
                logger.warning("No previous experiment run detected")
                logger.info("Starting fresh experiment suite instead...")
                args.resume = False
            else:
                # Show resume summary
                logger.info("Previous Run Analysis:")
                logger.info(f"   Completed: {len(analysis['completed_experiments'])}")
                logger.info(f"   Failed: {len(analysis['failed_experiments'])}")
                logger.info(f"   Result files: {len(analysis['result_files'])}")
                
                if analysis["completed_experiments"]:
                    logger.info(f"   Completed experiments: {', '.join(analysis['completed_experiments'])}")
                
                if analysis["failed_experiments"]:
                    logger.info(f"   Failed experiments to retry: {', '.join(analysis['failed_experiments'])}")
                
                # Show recommendations
                if analysis["recommendations"]:
                    logger.info("\nRecommendations:")
                    for rec in analysis["recommendations"]:
                        logger.info(f"   - {rec}")
        
        # Run experiments
        final_results = runner.run_all_experiments()
        
        logger.info("\n" + "="*80)
        logger.info("ALL EXPERIMENTS COMPLETED!")
        logger.info("="*80)
        
        # Show final summary
        successful = final_results['summary']['successful']
        total = final_results['summary']['total_experiments']
        
        logger.info(f"Final Summary:")
        logger.info(f"   Completed: {successful}")
        logger.info(f"   Failed: {total - successful}")
        logger.info(f"   Results saved in: results/")
        
        if successful == total:
            logger.info("All experiments completed successfully!")
        elif successful > 0:
            logger.info(f"{successful}/{total} experiments completed successfully")
        else:
            logger.error("No experiments completed successfully")
            
        return 0 if successful > 0 else 1
        
    except KeyboardInterrupt:
        logger.info("\nExperiments interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())