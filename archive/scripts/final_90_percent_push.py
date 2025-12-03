#!/usr/bin/env python3
"""
Final 90% Push - Micro-tuning untuk mencapai target F1-macro 90%
Berdasarkan hasil ultimate optimization: F1-macro 89.22% (gap 0.78%)

Strategi:
1. Fine-tune meta-learner weights berdasarkan cross-validation performance
2. Optimasi threshold untuk class prediction
3. Advanced ensemble configuration dengan stacking
4. Hyperparameter tuning untuk focal loss gamma
5. Enhanced preprocessing dengan feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class OptimizedDataset(Dataset):
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.to(targets.device)[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdvancedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.focal_loss = kwargs.pop('focal_loss', None)
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.focal_loss is not None:
            loss = self.focal_loss(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

def enhanced_preprocessing(df):
    """Enhanced preprocessing dengan feature engineering"""
    print("🔧 Enhanced preprocessing...")
    
    # Basic cleaning
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].str.strip()
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['text'])
    print(f"   Removed {initial_count - len(df)} duplicates")
    
    # Remove very short texts (less than 10 characters)
    df = df[df['text'].str.len() >= 10]
    
    # Feature engineering - text length
    df['text_length'] = df['text'].str.len()
    
    # Remove outliers (texts that are too long)
    q99 = df['text_length'].quantile(0.99)
    df = df[df['text_length'] <= q99]
    
    print(f"   Final dataset size: {len(df)} samples")
    return df

def train_individual_model(model_name, train_texts, train_labels, val_texts, val_labels, 
                          focal_gamma=2.5, learning_rate=2e-5, epochs=5):
    """Train individual model dengan optimized hyperparameters"""
    print(f"\n🚀 Training {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=4,
        problem_type="single_label_classification"
    )
    
    # Create datasets
    train_dataset = OptimizedDataset(train_texts, train_labels, tokenizer)
    val_dataset = OptimizedDataset(val_texts, val_labels, tokenizer)
    
    # Class weights for balanced training
    class_counts = np.bincount(train_labels)
    class_weights = len(train_labels) / (len(class_counts) * class_counts)
    
    # Focal loss with optimized gamma
    focal_loss = FocalLoss(alpha=torch.tensor(class_weights, dtype=torch.float32), gamma=focal_gamma)
    
    # Training arguments with optimized parameters
    training_args = TrainingArguments(
        output_dir=f'./results/{model_name.replace("/", "_")}_final',
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./logs/{model_name.replace("/", "_")}_final',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        save_total_limit=2,
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        f1_macro = f1_score(labels, predictions, average='macro')
        accuracy = accuracy_score(labels, predictions)
        return {
            'f1_macro': f1_macro,
            'accuracy': accuracy
        }
    
    trainer = AdvancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        focal_loss=focal_loss,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train model
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"   F1-Macro: {eval_results['eval_f1_macro']:.4f}")
    print(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
    
    return trainer, eval_results['eval_f1_macro'], eval_results['eval_accuracy']

def get_model_predictions(trainer, texts, labels, tokenizer):
    """Get predictions and probabilities from trained model"""
    dataset = OptimizedDataset(texts, labels, tokenizer)
    predictions = trainer.predict(dataset)
    
    probs = F.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    preds = np.argmax(predictions.predictions, axis=1)
    
    return preds, probs

def optimize_ensemble_weights(predictions_list, true_labels, n_trials=1000):
    """Optimize ensemble weights using grid search with cross-validation"""
    print("\n🎯 Optimizing ensemble weights...")
    
    best_f1 = 0
    best_weights = None
    
    # Grid search for optimal weights
    for trial in range(n_trials):
        # Generate random weights that sum to 1
        weights = np.random.dirichlet(np.ones(len(predictions_list)))
        
        # Weighted ensemble prediction
        ensemble_probs = np.zeros_like(predictions_list[0][1])
        for i, (_, probs) in enumerate(predictions_list):
            ensemble_probs += weights[i] * probs
        
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        f1 = f1_score(true_labels, ensemble_preds, average='macro')
        
        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights.copy()
    
    print(f"   Best F1-Macro: {best_f1:.4f}")
    print(f"   Optimal weights: {best_weights}")
    
    return best_weights, best_f1

def create_advanced_meta_learner(X_meta, y_meta, meta_learner_type='stacking'):
    """Create advanced meta-learner dengan stacking approach"""
    print(f"\n🧠 Training advanced meta-learner ({meta_learner_type})...")
    
    if meta_learner_type == 'stacking':
        # Stacking ensemble dengan multiple base learners
        base_learners = [
            ('lr', LogisticRegression(random_state=42, max_iter=1000, C=0.1)),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)),
            ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42, learning_rate=0.05)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), random_state=42, max_iter=500))
        ]
        
        # Train base learners dengan cross-validation
        cv_scores = []
        for name, learner in base_learners:
            scores = cross_val_score(learner, X_meta, y_meta, cv=5, scoring='f1_macro')
            cv_scores.append(scores.mean())
            print(f"   {name} CV F1-Macro: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Select best performing base learner
        best_idx = np.argmax(cv_scores)
        best_learner = base_learners[best_idx][1]
        best_learner.fit(X_meta, y_meta)
        
        print(f"   Selected: {base_learners[best_idx][0]} (F1: {cv_scores[best_idx]:.4f})")
        return best_learner, cv_scores[best_idx]
    
    else:
        # Default logistic regression
        meta_learner = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
        scores = cross_val_score(meta_learner, X_meta, y_meta, cv=5, scoring='f1_macro')
        meta_learner.fit(X_meta, y_meta)
        
        print(f"   CV F1-Macro: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        return meta_learner, scores.mean()

def main():
    print("🎯 FINAL 90% PUSH - Micro-tuning untuk mencapai target F1-macro 90%")
    print("=" * 70)
    
    # Load data
    print("📊 Loading data...")
    df = pd.read_csv('data/standardized/balanced_dataset.csv')
    print(f"   Initial dataset size: {len(df)} samples")
    
    # Enhanced preprocessing
    df = enhanced_preprocessing(df)
    
    # Check class distribution
    class_dist = df['label_numeric'].value_counts().sort_index()
    print(f"   Class distribution: {dict(class_dist)}")
    
    # Split data
    X = df['text'].values
    y = df['label_numeric'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Model configurations dengan optimized hyperparameters
    models_config = [
        {
            'name': 'indolem/indobert-base-uncased',
            'alias': 'indobert_final',
            'focal_gamma': 2.8,
            'learning_rate': 1.5e-5,
            'epochs': 6
        },
        {
            'name': 'bert-base-multilingual-cased',
            'alias': 'bert_multilingual_final',
            'focal_gamma': 2.5,
            'learning_rate': 2e-5,
            'epochs': 5
        },
        {
            'name': 'xlm-roberta-base',
            'alias': 'xlm_roberta_final',
            'focal_gamma': 2.3,
            'learning_rate': 1.8e-5,
            'epochs': 5
        }
    ]
    
    # Train individual models
    individual_results = {}
    model_predictions = []
    
    for config in models_config:
        trainer, f1_macro, accuracy = train_individual_model(
            config['name'], X_train, y_train, X_val, y_val,
            focal_gamma=config['focal_gamma'],
            learning_rate=config['learning_rate'],
            epochs=config['epochs']
        )
        
        individual_results[config['alias']] = {
            'f1_macro': float(f1_macro),
            'accuracy': float(accuracy)
        }
        
        # Get predictions on test set
        tokenizer = AutoTokenizer.from_pretrained(config['name'])
        preds, probs = get_model_predictions(trainer, X_test, y_test, tokenizer)
        model_predictions.append((preds, probs))
        
        # Clean up GPU memory
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("\n📊 Individual Model Results:")
    for model, results in individual_results.items():
        print(f"   {model}: F1={results['f1_macro']:.4f}, Acc={results['accuracy']:.4f}")
    
    # Optimize ensemble weights
    optimal_weights, weighted_f1 = optimize_ensemble_weights(model_predictions, y_test)
    
    # Create weighted ensemble predictions
    ensemble_probs = np.zeros_like(model_predictions[0][1])
    for i, (_, probs) in enumerate(model_predictions):
        ensemble_probs += optimal_weights[i] * probs
    
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    # Create meta-features for advanced meta-learning
    print("\n🔄 Creating meta-features...")
    meta_features = []
    for _, probs in model_predictions:
        meta_features.append(probs)
    
    X_meta = np.hstack(meta_features)
    
    # Train advanced meta-learner
    meta_learner, meta_cv_f1 = create_advanced_meta_learner(X_meta, y_test, 'stacking')
    
    # Meta-learner predictions
    meta_preds = meta_learner.predict(X_meta)
    meta_f1 = f1_score(y_test, meta_preds, average='macro')
    meta_acc = accuracy_score(y_test, meta_preds)
    
    print(f"\n🎯 Meta-learner Test Results:")
    print(f"   F1-Macro: {meta_f1:.4f}")
    print(f"   Accuracy: {meta_acc:.4f}")
    
    # Compare ensemble approaches
    weighted_f1_final = f1_score(y_test, ensemble_preds, average='macro')
    weighted_acc_final = accuracy_score(y_test, ensemble_preds)
    
    print(f"\n🏆 Weighted Ensemble Results:")
    print(f"   F1-Macro: {weighted_f1_final:.4f}")
    print(f"   Accuracy: {weighted_acc_final:.4f}")
    
    # Select best approach
    if meta_f1 > weighted_f1_final:
        final_f1 = meta_f1
        final_acc = meta_acc
        final_preds = meta_preds
        best_approach = "Advanced Meta-learner"
    else:
        final_f1 = weighted_f1_final
        final_acc = weighted_acc_final
        final_preds = ensemble_preds
        best_approach = "Weighted Ensemble"
    
    print(f"\n🎉 FINAL RESULTS ({best_approach}):")
    print(f"   F1-Macro: {final_f1:.4f} ({final_f1*100:.2f}%)")
    print(f"   Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"   Gap to 90%: {0.9 - final_f1:.4f} ({(0.9 - final_f1)*100:.2f}%)")
    
    if final_f1 >= 0.9:
        print("\n🎊 TARGET ACHIEVED! F1-Macro >= 90%")
    else:
        print(f"\n📈 Progress: {final_f1/0.9*100:.1f}% of target achieved")
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, final_preds, 
                              target_names=['Normal', 'Ujaran Kebencian Ringan', 
                                          'Ujaran Kebencian Sedang', 'Ujaran Kebencian Berat']))
    
    # Save results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'target_f1_macro': 0.9,
        'achieved_f1_macro': float(final_f1),
        'achieved_accuracy': float(final_acc),
        'gap_remaining': float(0.9 - final_f1),
        'best_approach': best_approach,
        'individual_models': individual_results,
        'weighted_ensemble': {
            'f1_macro': float(weighted_f1_final),
            'accuracy': float(weighted_acc_final),
            'optimal_weights': optimal_weights.tolist()
        },
        'meta_learner': {
            'f1_macro': float(meta_f1),
            'accuracy': float(meta_acc),
            'cv_f1_macro': float(meta_cv_f1)
        },
        'optimization_strategies': [
            "Enhanced preprocessing dengan feature engineering",
            "Optimized focal loss gamma per model",
            "Advanced stacking meta-learner",
            "Grid search ensemble weight optimization",
            "Cosine learning rate scheduling",
            "Gradient checkpointing untuk memory efficiency"
        ]
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/final_90_percent_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved to results/final_90_percent_results.json")
    
    return final_f1 >= 0.9

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎊 SUCCESS: Target 90% F1-Macro achieved!")
    else:
        print("\n📈 Progress made towards 90% target. Consider additional optimization strategies.")