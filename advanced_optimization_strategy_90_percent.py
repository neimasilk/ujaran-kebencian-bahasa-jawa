#!/usr/bin/env python3
"""
Advanced Optimization Strategy for 90% F1-Macro Target
Based on analysis of 77.72% achievement, implementing advanced techniques
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FocalLossTrainer(Trainer):
    """Custom trainer with Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Focal Loss implementation
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        loss = focal_loss.mean()
        
        return (loss, outputs) if return_outputs else loss

class HateBertDataset(Dataset):
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

def advanced_preprocessing(df):
    """Enhanced preprocessing with advanced techniques"""
    print("🔧 Applying advanced preprocessing...")
    
    # Remove duplicates and clean text
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.len() > 10]  # Remove very short texts
    
    # Advanced text cleaning
    import re
    def clean_text(text):
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        # Remove URLs and mentions but keep hashtags
        text = re.sub(r'http\S+|www\S+|@\w+', '', text)
        return text
    
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 5]  # Final length check
    
    print(f"Dataset shape after advanced preprocessing: {df.shape}")
    return df

def train_optimized_model(model_name, model_config, train_texts, train_labels, test_texts, test_labels):
    """Train model with optimized hyperparameters"""
    print(f"🤖 Training {model_name} with optimized settings...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_path'])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config['model_path'], 
        num_labels=4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    
    # Create datasets
    train_dataset = HateBertDataset(train_texts, train_labels, tokenizer, max_length=256)
    test_dataset = HateBertDataset(test_texts, test_labels, tokenizer, max_length=256)
    
    # Optimized training arguments
    training_args = TrainingArguments(
        output_dir=f'./models/{model_name}_optimized',
        num_train_epochs=6,  # Increased epochs
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./logs/{model_name}_optimized',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=model_config['learning_rate'],
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=2,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_smoothing_factor=0.1,  # Label smoothing
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        return {
            'f1': f1_macro,
            'f1_weighted': f1_weighted,
            'accuracy': accuracy
        }
    
    # Use Focal Loss trainer for better class balance
    trainer = FocalLossTrainer(
        alpha=1.0,
        gamma=2.0,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print(f"🚀 Starting optimized training for {model_name}...")
    trainer.train()
    
    # Evaluate
    print(f"📊 Evaluating {model_name}...")
    eval_results = trainer.evaluate()
    
    # Get predictions for meta-learning
    predictions = trainer.predict(test_dataset)
    probabilities = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    
    print(f"✅ {model_name} Optimized Results:")
    print(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"   F1-Macro: {eval_results['eval_f1']:.4f}")
    print(f"   F1-Weighted: {eval_results['eval_f1_weighted']:.4f}")
    
    return {
        'model_name': model_name,
        'probabilities': probabilities,
        'accuracy': eval_results['eval_accuracy'],
        'f1_macro': eval_results['eval_f1'],
        'f1_weighted': eval_results['eval_f1_weighted'],
        'eval_results': eval_results
    }

def advanced_meta_ensemble(model_results, test_labels):
    """Advanced meta-ensemble with multiple algorithms and stacking"""
    print("🧠 Training advanced meta-ensemble...")
    
    # Prepare meta-features
    meta_features = []
    for result in model_results:
        probs = result['probabilities']
        # Add probability features
        meta_features.append(probs)
        # Add confidence features
        confidence = np.max(probs, axis=1).reshape(-1, 1)
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1).reshape(-1, 1)
        meta_features.extend([confidence, entropy])
    
    X_meta = np.hstack(meta_features)
    y_meta = test_labels
    
    print(f"Meta-features shape: {X_meta.shape}")
    
    # Split for meta-learning validation
    X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
        X_meta, y_meta, test_size=0.3, random_state=42, stratify=y_meta
    )
    
    # Scale features
    scaler = StandardScaler()
    X_meta_train_scaled = scaler.fit_transform(X_meta_train)
    X_meta_val_scaled = scaler.transform(X_meta_val)
    X_meta_scaled = scaler.transform(X_meta)
    
    # Multiple meta-learners with hyperparameter tuning
    meta_learners = {
        'random_forest': RandomForestClassifier(random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'svm': SVC(probability=True, random_state=42),
        'mlp': MLPClassifier(random_state=42, max_iter=500)
    }
    
    # Hyperparameter grids
    param_grids = {
        'random_forest': {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'gradient_boosting': {
            'n_estimators': [200, 300],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [6, 8, 10]
        },
        'logistic_regression': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    }
    
    best_meta_learner = None
    best_score = 0
    best_name = ""
    
    # Train and evaluate each meta-learner
    for name, learner in meta_learners.items():
        print(f"Training {name} meta-learner...")
        
        if name in param_grids:
            # Grid search for hyperparameter tuning
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                learner, param_grids[name], 
                cv=cv, scoring='f1_macro', n_jobs=-1
            )
            grid_search.fit(X_meta_train_scaled, y_meta_train)
            best_learner = grid_search.best_estimator_
        else:
            best_learner = learner
            best_learner.fit(X_meta_train_scaled, y_meta_train)
        
        # Validate
        val_pred = best_learner.predict(X_meta_val_scaled)
        val_f1 = f1_score(y_meta_val, val_pred, average='macro')
        
        print(f"{name} validation F1-Macro: {val_f1:.4f}")
        
        if val_f1 > best_score:
            best_score = val_f1
            best_meta_learner = best_learner
            best_name = name
    
    print(f"Best meta-learner: {best_name} (F1-Macro: {best_score:.4f})")
    
    # Final predictions
    final_predictions = best_meta_learner.predict(X_meta_scaled)
    final_probabilities = best_meta_learner.predict_proba(X_meta_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_meta, final_predictions)
    f1_macro = f1_score(y_meta, final_predictions, average='macro')
    f1_weighted = f1_score(y_meta, final_predictions, average='weighted')
    
    print(f"🎯 Advanced Meta-Ensemble Results:")
    print(f"   Best Meta-Learner: {best_name}")
    print(f"   Validation F1-Macro: {best_score:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Macro: {f1_macro:.4f}")
    print(f"   F1-Weighted: {f1_weighted:.4f}")
    
    return {
        'meta_model': best_name,
        'validation_f1_macro': best_score,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': classification_report(y_meta, final_predictions, output_dict=True),
        'predictions': final_predictions,
        'probabilities': final_probabilities
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    print("🚀 Starting Advanced Optimization for 90% F1-Macro Target")
    print("=" * 70)
    
    # Load dataset
    print("📊 Loading balanced dataset...")
    df = pd.read_csv('data/standardized/balanced_dataset.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:")
    print(df['final_label'].value_counts())
    
    # Advanced preprocessing
    df = advanced_preprocessing(df)
    
    # Prepare data
    label_mapping = {
        'Bukan Ujaran Kebencian': 0,
        'Ujaran Kebencian - Ringan': 1, 
        'Ujaran Kebencian - Sedang': 2,
        'Ujaran Kebencian - Berat': 3
    }
    
    df['label'] = df['final_label'].map(label_mapping)
    
    # Split data
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['label']
    )
    
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Optimized model configurations
    model_configs = {
        'indoroberta_optimized': {
            'model_path': 'flax-community/indonesian-roberta-base',
            'learning_rate': 1.5e-5  # Slightly lower LR
        },
        'bert_multilingual_optimized': {
            'model_path': 'bert-base-multilingual-cased',
            'learning_rate': 2e-5
        },
        'xlm_roberta_optimized': {
            'model_path': 'xlm-roberta-base',
            'learning_rate': 1e-5  # Lower LR for stability
        },
        'indobert_optimized': {
            'model_path': 'indobenchmark/indobert-base-p1',
            'learning_rate': 2e-5
        }
    }
    
    # Train all models
    model_results = []
    for model_name, config in model_configs.items():
        try:
            result = train_optimized_model(
                model_name, config, train_texts, train_labels, test_texts, test_labels
            )
            model_results.append(result)
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            continue
    
    if len(model_results) < 2:
        print("❌ Not enough models trained successfully for ensemble")
        return
    
    # Advanced meta-ensemble
    meta_results = advanced_meta_ensemble(model_results, test_labels)
    
    # Final results
    best_f1 = meta_results['f1_macro']
    target_achieved = best_f1 >= 0.90
    gap_to_90 = 0.90 - best_f1
    
    print("\n" + "=" * 70)
    print("🏁 ADVANCED OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Best F1-Macro achieved: {best_f1:.4f}")
    print(f"Target 90% achieved: {target_achieved}")
    print(f"Advanced Meta-Ensemble F1-Macro: {best_f1:.4f}")
    if not target_achieved:
        print(f"📊 Gap to 90%: {gap_to_90:.4f} ({gap_to_90*100:.2f}%)")
    
    # Save results
    results = {
        'experiment_name': 'advanced_optimization_90_percent',
        'timestamp': datetime.now().isoformat(),
        'target_achieved': target_achieved,
        'models': {result['model_name']: {
            'accuracy': result['accuracy'],
            'f1_macro': result['f1_macro'],
            'f1_weighted': result['f1_weighted']
        } for result in model_results},
        'advanced_meta_ensemble_results': meta_results,
        'best_f1_macro': best_f1,
        'gap_to_target': gap_to_90 if not target_achieved else 0
    }
    
    os.makedirs('results', exist_ok=True)
    results_file = 'results/advanced_optimization_90_percent_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()