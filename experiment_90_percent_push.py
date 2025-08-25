#!/usr/bin/env python3
"""
Eksperimen untuk mencapai target 90% F1-Macro
Berdasarkan dataset asli yang menghasilkan 86.88% F1-Macro

Strategi:
1. Gunakan dataset balanced_dataset.csv (24,964 samples)
2. Implementasi advanced training techniques
3. Hyperparameter optimization
4. Ensemble method dengan multiple models
5. Advanced data preprocessing
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

class AdvancedTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Model configurations for ensemble (using available models)
        self.model_configs = [
            {
                'name': 'indobert-base-uncased',
                'model_name': 'indolem/indobert-base-uncased',
                'learning_rate': 1e-5,
                'warmup_steps': 500
            }
        ]
        
        self.label_map = {
            'Bukan Ujaran Kebencian': 0,
            'Ujaran Kebencian - Ringan': 1, 
            'Ujaran Kebencian - Sedang': 2,
            'Ujaran Kebencian - Berat': 3
        }
        
        self.results = {
            'experiment_name': 'experiment_90_percent_push',
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'ensemble_results': {},
            'target_achieved': False
        }
    
    def load_and_prepare_data(self):
        """Load dataset asli yang menghasilkan 86.88% F1-Macro"""
        print("📊 Loading original balanced dataset...")
        
        # Load dataset yang terbukti bagus
        df = pd.read_csv('data/standardized/balanced_dataset.csv')
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check data quality
        print(f"NaN values before cleaning: {df.isnull().sum().sum()}")
        print(f"Label distribution before cleaning:\n{df['final_label'].value_counts()}")
        
        # Clean NaN values
        df = df.dropna(subset=['text', 'final_label'])
        print(f"Dataset shape after removing NaN: {df.shape}")
        print(f"NaN values after cleaning: {df.isnull().sum().sum()}")
        
        # Advanced preprocessing
        df = self.advanced_preprocessing(df)
        
        # Prepare features and labels
        X = df['text'].values
        y = df['final_label'].map(self.label_map).values
        
        # Check for any remaining NaN values in labels
        nan_mask = pd.isna(y)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} unmapped labels")
            print(f"Unique labels in dataset: {df['final_label'].unique()}")
            # Remove samples with unmapped labels
            X = X[~nan_mask]
            y = y[~nan_mask]
            print(f"Final dataset shape after cleaning: {len(X)} samples")
        
        # Stratified split with larger training set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Final label distribution:\n{pd.Series(y_train).value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def advanced_preprocessing(self, df):
        """Advanced text preprocessing untuk meningkatkan kualitas data"""
        print("🔧 Applying advanced preprocessing...")
        
        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates(subset=['text'])
        print(f"Removed {initial_size - len(df)} duplicates")
        
        # Text cleaning
        df['text'] = df['text'].str.strip()
        df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
        
        # Remove very short or very long texts
        df['text_length'] = df['text'].str.len()
        df = df[(df['text_length'] >= 10) & (df['text_length'] <= 512)]
        print(f"After length filtering: {len(df)} samples")
        
        return df.drop('text_length', axis=1)
    
    def create_dataset(self, texts, labels, tokenizer, max_length=128):
        """Create dataset untuk training"""
        encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return HateSpeechDataset(encodings, labels)

    def train_single_model(self, config, X_train, X_test, y_train, y_test):
        """Train single model dengan advanced techniques"""
        print(f"\n🤖 Training {config['name']}...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'],
            num_labels=4
        ).to(self.device)
        
        # Create datasets
        train_dataset = self.create_dataset(X_train, y_train, tokenizer)
        test_dataset = self.create_dataset(X_test, y_test, tokenizer)
        
        # Compute class weights for focal loss
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        # Advanced training arguments
        training_args = TrainingArguments(
            output_dir=f'tmp_90_percent_{config["name"]}',
            num_train_epochs=6,  # Increased epochs
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=config['warmup_steps'],
            weight_decay=0.01,
            logging_dir=f'logs_90_percent_{config["name"]}',
            logging_steps=100,
            eval_strategy='steps',
            eval_steps=200,
            save_strategy='steps',
            save_steps=400,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            learning_rate=config['learning_rate'],
            lr_scheduler_type='cosine',  # Cosine annealing
            fp16=True,  # Mixed precision
            dataloader_num_workers=0,
            gradient_accumulation_steps=2,  # Effective batch size = 32
            max_grad_norm=1.0,  # Gradient clipping
            seed=42
        )
        
        # Custom trainer with focal loss
        class FocalLossTrainer(Trainer):
            def __init__(self, class_weights, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights
            
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                
                # Focal loss implementation
                ce_loss = torch.nn.functional.cross_entropy(
                    logits, labels, weight=self.class_weights, reduction='none'
                )
                pt = torch.exp(-ce_loss)
                focal_loss = (1 - pt) ** 2 * ce_loss
                loss = focal_loss.mean()
                
                return (loss, outputs) if return_outputs else loss
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            f1_macro = f1_score(labels, predictions, average='macro')
            accuracy = accuracy_score(labels, predictions)
            return {'f1': f1_macro, 'accuracy': accuracy}
        
        # Initialize trainer
        trainer = FocalLossTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        print(f"Starting training for {config['name']}...")
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        print(f"Evaluation results for {config['name']}: {eval_results}")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Classification report
        target_names = ['Bukan Ujaran Kebencian', 'Ujaran Kebencian - Ringan', 'Ujaran Kebencian - Sedang', 'Ujaran Kebencian - Berat']
        class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        
        model_results = {
            'model_name': config['name'],
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': class_report,
            'predictions': y_pred.tolist()
        }
        
        print(f"✅ {config['name']} - Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}")
        
        return model_results, predictions.predictions
    
    def ensemble_prediction(self, all_predictions, y_test):
        """Ensemble prediction dengan weighted voting"""
        print("\n🎯 Creating ensemble prediction...")
        
        # Weighted average of predictions (soft voting)
        ensemble_probs = np.mean(all_predictions, axis=0)
        ensemble_pred = np.argmax(ensemble_probs, axis=1)
        
        # Calculate ensemble metrics
        accuracy = accuracy_score(y_test, ensemble_pred)
        f1_macro = f1_score(y_test, ensemble_pred, average='macro')
        f1_weighted = f1_score(y_test, ensemble_pred, average='weighted')
        
        # Classification report
        target_names = ['Bukan Ujaran Kebencian', 'Ujaran Kebencian - Ringan', 'Ujaran Kebencian - Sedang', 'Ujaran Kebencian - Berat']
        class_report = classification_report(y_test, ensemble_pred, target_names=target_names, output_dict=True)
        
        ensemble_results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': class_report,
            'predictions': ensemble_pred.tolist()
        }
        
        print(f"🎯 Ensemble Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Macro: {f1_macro:.4f}")
        print(f"   F1-Weighted: {f1_weighted:.4f}")
        
        return ensemble_results
    
    def run_experiment(self):
        """Run complete experiment untuk mencapai 90% F1-Macro"""
        print("🚀 Starting 90% F1-Macro Push Experiment")
        print("="*60)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()
        
        # Train multiple models
        all_predictions = []
        
        for config in self.model_configs:
            try:
                model_results, predictions = self.train_single_model(
                    config, X_train, X_test, y_train, y_test
                )
                self.results['models'][config['name']] = model_results
                all_predictions.append(predictions)
                
                # Check if single model achieves 90%
                if model_results['f1_macro'] >= 0.90:
                    print(f"🎉 TARGET ACHIEVED! {config['name']} reached {model_results['f1_macro']:.4f} F1-Macro")
                    self.results['target_achieved'] = True
                    
            except Exception as e:
                print(f"❌ Error training {config['name']}: {str(e)}")
                continue
        
        # Ensemble prediction if we have multiple models
        if len(all_predictions) >= 2:
            ensemble_results = self.ensemble_prediction(all_predictions, y_test)
            self.results['ensemble_results'] = ensemble_results
            
            # Check if ensemble achieves 90%
            if ensemble_results['f1_macro'] >= 0.90:
                print(f"🎉 TARGET ACHIEVED! Ensemble reached {ensemble_results['f1_macro']:.4f} F1-Macro")
                self.results['target_achieved'] = True
        else:
            print("⚠️  Not enough models for ensemble (need at least 2)")
            self.results['ensemble_results'] = {}
        
        # Save results
        os.makedirs('results', exist_ok=True)
        with open('results/experiment_90_percent_push_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\n" + "="*60)
        print("📊 EXPERIMENT SUMMARY")
        print("="*60)
        
        for model_name, results in self.results['models'].items():
            print(f"{model_name}: F1-Macro = {results['f1_macro']:.4f}")
        
        if 'ensemble_results' in self.results and 'f1_macro' in self.results['ensemble_results']:
            print(f"Ensemble: F1-Macro = {self.results['ensemble_results']['f1_macro']:.4f}")
        elif len(self.results['models']) < 2:
            print("Ensemble: Not available (insufficient models)")
        
        if self.results['target_achieved']:
            print("\n🎉 TARGET 90% F1-MACRO ACHIEVED!")
        else:
            print("\n⚠️  Target 90% F1-Macro not yet achieved")
            if self.results['models']:
                best_score = max([r['f1_macro'] for r in self.results['models'].values()])
                if 'ensemble_results' in self.results and 'f1_macro' in self.results['ensemble_results']:
                    best_score = max(best_score, self.results['ensemble_results']['f1_macro'])
                print(f"   Best score: {best_score:.4f}")
                print(f"   Gap to target: {0.90 - best_score:.4f}")
            else:
                print("   No models trained successfully")
        
        print("\n✅ Results saved to: results/experiment_90_percent_push_results.json")
        return self.results

if __name__ == "__main__":
    trainer = AdvancedTrainer()
    results = trainer.run_experiment()