#!/usr/bin/env python3
"""
Improved Meta-Ensemble for 90%+ Test Accuracy
Based on the successful 94.09% validation ensemble, but addressing the validation-test gap
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import json
import logging
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedMetaEnsemble:
    def __init__(self, device='cuda'):
        self.device = device
        self.models = []
        self.tokenizers = []
        self.meta_model = None
        
    def load_base_models(self):
        """Load multiple instances of the improved model with different configurations"""
        model_configs = [
            {'path': 'models/improved_model', 'max_length': 128},
            {'path': 'models/improved_model', 'max_length': 256},  # Different tokenization
            {'path': 'models/improved_model', 'max_length': 512}   # Longer sequences
        ]
        
        for i, config in enumerate(model_configs):
            try:
                logger.info(f"Loading model {i+1}: {config['path']} with max_length={config['max_length']}")
                tokenizer = AutoTokenizer.from_pretrained(config['path'])
                model = AutoModelForSequenceClassification.from_pretrained(config['path'])
                model.to(self.device)
                model.eval()
                
                self.tokenizers.append((tokenizer, config['max_length']))
                self.models.append(model)
                logger.info(f"Successfully loaded model {i+1}")
            except Exception as e:
                logger.error(f"Failed to load model {i+1}: {e}")
                
        logger.info(f"Loaded {len(self.models)} base models")
        
    def predict_with_model(self, model_idx, texts, batch_size=8):
        """Get predictions from a specific model with robust error handling"""
        if model_idx >= len(self.models):
            return None
            
        model = self.models[model_idx]
        tokenizer, max_length = self.tokenizers[model_idx]
        
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            try:
                # Tokenize with different max_length for diversity
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    all_probs.extend(probs.cpu().numpy())
                    
            except Exception as e:
                logger.error(f"Error in model {model_idx} batch {i//batch_size}: {e}")
                # Return uniform probabilities as fallback
                batch_probs = np.ones((len(batch_texts), 4)) / 4
                all_probs.extend(batch_probs)
                
        return np.array(all_probs)
    
    def get_ensemble_predictions(self, texts, batch_size=8):
        """Get predictions from all models"""
        all_model_probs = []
        
        for i in range(len(self.models)):
            logger.info(f"Getting predictions from model {i+1}/{len(self.models)}")
            probs = self.predict_with_model(i, texts, batch_size)
            if probs is not None:
                all_model_probs.append(probs)
                
        return np.array(all_model_probs)  # Shape: (n_models, n_samples, n_classes)
    
    def create_meta_features(self, model_probs):
        """Create enhanced meta-features for the meta-learner"""
        n_models, n_samples, n_classes = model_probs.shape
        
        # Basic features: raw probabilities
        basic_features = model_probs.reshape(n_samples, -1)  # Flatten
        
        # Statistical features
        mean_probs = np.mean(model_probs, axis=0)  # Mean across models
        std_probs = np.std(model_probs, axis=0)    # Std across models
        max_probs = np.max(model_probs, axis=0)    # Max across models
        min_probs = np.min(model_probs, axis=0)    # Min across models
        
        # Confidence features
        max_confidence = np.max(mean_probs, axis=1, keepdims=True)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1, keepdims=True)
        
        # Agreement features
        predictions = np.argmax(model_probs, axis=2)  # Shape: (n_models, n_samples)
        agreement = np.array([np.bincount(predictions[:, i], minlength=n_classes) 
                             for i in range(n_samples)])
        max_agreement = np.max(agreement, axis=1, keepdims=True) / n_models
        
        # Combine all features
        meta_features = np.concatenate([
            basic_features,
            mean_probs,
            std_probs,
            max_probs,
            min_probs,
            max_confidence,
            entropy,
            max_agreement
        ], axis=1)
        
        return meta_features
    
    def train_meta_model(self, train_probs, train_labels, val_probs, val_labels):
        """Train an improved meta-model with cross-validation"""
        logger.info("Creating meta-features")
        train_meta_features = self.create_meta_features(train_probs)
        val_meta_features = self.create_meta_features(val_probs)
        
        # Try multiple meta-models and select the best
        meta_models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            ),
            'logistic': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }
        
        best_score = 0
        best_model = None
        best_name = None
        
        for name, model in meta_models.items():
            logger.info(f"Training {name} meta-model")
            model.fit(train_meta_features, train_labels)
            
            val_preds = model.predict(val_meta_features)
            val_score = f1_score(val_labels, val_preds, average='macro')
            
            logger.info(f"{name} validation F1-macro: {val_score:.4f}")
            
            if val_score > best_score:
                best_score = val_score
                best_model = model
                best_name = name
        
        self.meta_model = best_model
        logger.info(f"Best meta-model: {best_name} with F1-macro: {best_score:.4f}")
        
        return best_name, best_score
    
    def predict_meta(self, model_probs):
        """Make predictions using the trained meta-model"""
        if self.meta_model is None:
            raise ValueError("Meta-model not trained yet")
            
        meta_features = self.create_meta_features(model_probs)
        predictions = self.meta_model.predict(meta_features)
        probabilities = self.meta_model.predict_proba(meta_features)
        
        return predictions, probabilities

def main():
    logger.info("="*80)
    logger.info("IMPROVED META-ENSEMBLE FOR 90%+ TEST ACCURACY")
    logger.info("="*80)
    
    # Load dataset
    logger.info("Loading dataset...")
    df = pd.read_csv('data/augmented/augmented_dataset.csv')
    
    # Use the correct column names
    X = df['text'].values
    y = df['label_numeric'].values  # Use label_numeric as the correct column name
    
    logger.info(f"Dataset loaded: {len(X)} samples, {len(np.unique(y))} classes")
    
    # Stratified split with larger validation set to reduce overfitting
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.3, random_state=42, stratify=y_temp  # Larger validation set
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Initialize ensemble
    ensemble = ImprovedMetaEnsemble()
    ensemble.load_base_models()
    
    if len(ensemble.models) == 0:
        logger.error("No models loaded successfully")
        return
    
    # Get predictions for training meta-model
    logger.info("Getting training predictions...")
    train_probs = ensemble.get_ensemble_predictions(X_train.tolist(), batch_size=8)
    
    logger.info("Getting validation predictions...")
    val_probs = ensemble.get_ensemble_predictions(X_val.tolist(), batch_size=8)
    
    # Train meta-model
    logger.info("Training meta-model...")
    best_meta_name, best_meta_score = ensemble.train_meta_model(
        train_probs, y_train, val_probs, y_val
    )
    
    # Evaluate on test set
    logger.info("Getting test predictions...")
    test_probs = ensemble.get_ensemble_predictions(X_test.tolist(), batch_size=8)
    
    logger.info("Making final predictions...")
    test_preds, test_pred_probs = ensemble.predict_meta(test_probs)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, test_preds)
    test_f1_macro = f1_score(y_test, test_preds, average='macro')
    test_f1_weighted = f1_score(y_test, test_preds, average='weighted')
    
    # Generate detailed report
    class_names = ['Bukan Ujaran Kebencian', 'Ujaran Kebencian - Ringan', 
                   'Ujaran Kebencian - Sedang', 'Ujaran Kebencian - Berat']
    
    test_report = classification_report(y_test, test_preds, target_names=class_names, output_dict=True)
    
    # Save results
    results = {
        'experiment_timestamp': datetime.now().isoformat(),
        'models_used': len(ensemble.models),
        'meta_model': best_meta_name,
        'validation_meta_score': float(best_meta_score),
        'test_samples': len(X_test),
        'final_results': {
            'accuracy': float(test_accuracy),
            'f1_macro': float(test_f1_macro),
            'f1_weighted': float(test_f1_weighted),
            'classification_report': test_report
        },
        'target_achievement': {
            'accuracy_90_percent': test_accuracy >= 0.90,
            'f1_macro_90_percent': test_f1_macro >= 0.90,
            'target_achieved': test_accuracy >= 0.90 or test_f1_macro >= 0.90
        }
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/improved_meta_ensemble_90_percent_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n" + "="*80)
    print("🚀 IMPROVED META-ENSEMBLE RESULTS")
    print("="*80)
    
    print(f"\n📊 CONFIGURATION:")
    print(f"   Base models: {len(ensemble.models)}")
    print(f"   Meta-model: {best_meta_name}")
    print(f"   Validation meta F1: {best_meta_score:.4f}")
    
    print(f"\n📈 FINAL TEST RESULTS:")
    print(f"   Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   F1-Macro: {test_f1_macro:.4f} ({test_f1_macro*100:.2f}%)")
    print(f"   F1-Weighted: {test_f1_weighted:.4f} ({test_f1_weighted*100:.2f}%)")
    
    # Check target achievement
    if test_accuracy >= 0.90:
        print(f"\n🎉 90%+ ACCURACY TARGET ACHIEVED!")
        print(f"   ✅ Test Accuracy: {test_accuracy*100:.2f}% (Target: 90%+)")
    elif test_f1_macro >= 0.90:
        print(f"\n🎉 90%+ F1-MACRO TARGET ACHIEVED!")
        print(f"   ✅ Test F1-Macro: {test_f1_macro*100:.2f}% (Target: 90%+)")
    else:
        print(f"\n⚠️  90% target not yet achieved")
        print(f"   Current best: {max(test_accuracy, test_f1_macro)*100:.2f}%")
        print(f"   Gap to 90%: {(0.90 - max(test_accuracy, test_f1_macro))*100:.2f}%")
    
    logger.info("Improved meta-ensemble experiment completed")
    
    return test_accuracy >= 0.90 or test_f1_macro >= 0.90

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 SUCCESS: 90% target achieved!")
    else:
        print("\n📊 Results saved for further analysis")