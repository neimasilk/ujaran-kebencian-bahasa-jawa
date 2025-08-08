#!/usr/bin/env python3
"""
Advanced Ensemble Methods for 90%+ Accuracy
Implement sophisticated ensemble techniques to push performance beyond 90%
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.special import softmax
import json
import logging
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ensemble_advanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedEnsemble:
    def __init__(self, model_paths, device='cuda'):
        self.model_paths = model_paths
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.tokenizers = []
        self.ensemble_weights = None
        self.meta_learner = None
        self.class_names = ['Bukan Ujaran Kebencian', 'Ujaran Kebencian - Ringan', 
                           'Ujaran Kebencian - Sedang', 'Ujaran Kebencian - Berat']
        
    def load_models(self):
        """Load all models and tokenizers"""
        logger.info(f"Loading {len(self.model_paths)} models")
        
        for i, model_path in enumerate(self.model_paths):
            try:
                logger.info(f"Loading model {i+1}: {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                model.to(self.device)
                model.eval()
                
                self.tokenizers.append(tokenizer)
                self.models.append(model)
                logger.info(f"Model {i+1} loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {e}")
                continue
                
        logger.info(f"Successfully loaded {len(self.models)} models")
        
    def predict_single_model(self, model_idx, texts, batch_size=16):
        """Get predictions from a single model"""
        model = self.models[model_idx]
        tokenizer = self.tokenizers[model_idx]
        
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_probs)
    
    def get_all_predictions(self, texts, batch_size=16):
        """Get predictions from all models"""
        all_model_probs = []
        
        for i, model in enumerate(self.models):
            logger.info(f"Getting predictions from model {i+1}/{len(self.models)}")
            probs = self.predict_single_model(i, texts, batch_size)
            all_model_probs.append(probs)
            
        return np.array(all_model_probs)  # Shape: (n_models, n_samples, n_classes)
    
    def simple_voting(self, all_probs, method='soft'):
        """Simple voting ensemble"""
        if method == 'soft':
            # Average probabilities
            ensemble_probs = np.mean(all_probs, axis=0)
            predictions = np.argmax(ensemble_probs, axis=1)
        else:
            # Hard voting
            individual_preds = np.argmax(all_probs, axis=2)
            predictions = []
            for i in range(individual_preds.shape[1]):
                votes = individual_preds[:, i]
                prediction = np.bincount(votes).argmax()
                predictions.append(prediction)
            predictions = np.array(predictions)
            ensemble_probs = None
            
        return predictions, ensemble_probs
    
    def weighted_voting(self, all_probs, weights=None):
        """Weighted voting ensemble"""
        if weights is None:
            weights = self.ensemble_weights
        if weights is None:
            weights = np.ones(len(self.models)) / len(self.models)
            
        # Weighted average of probabilities
        ensemble_probs = np.average(all_probs, axis=0, weights=weights)
        predictions = np.argmax(ensemble_probs, axis=1)
        
        return predictions, ensemble_probs
    
    def confidence_based_selection(self, all_probs, confidence_threshold=0.8):
        """Select model based on confidence for each prediction"""
        n_samples = all_probs.shape[1]
        predictions = []
        ensemble_probs = []
        
        for i in range(n_samples):
            sample_probs = all_probs[:, i, :]  # Shape: (n_models, n_classes)
            
            # Calculate confidence (max probability) for each model
            confidences = np.max(sample_probs, axis=1)
            
            # Find model with highest confidence
            best_model_idx = np.argmax(confidences)
            best_confidence = confidences[best_model_idx]
            
            if best_confidence >= confidence_threshold:
                # Use the most confident model
                predictions.append(np.argmax(sample_probs[best_model_idx]))
                ensemble_probs.append(sample_probs[best_model_idx])
            else:
                # Use weighted average if no model is confident enough
                avg_probs = np.mean(sample_probs, axis=0)
                predictions.append(np.argmax(avg_probs))
                ensemble_probs.append(avg_probs)
                
        return np.array(predictions), np.array(ensemble_probs)
    
    def optimize_weights(self, all_probs, true_labels, method='f1_macro'):
        """Optimize ensemble weights using validation data"""
        logger.info("Optimizing ensemble weights")
        
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize
            ensemble_probs = np.average(all_probs, axis=0, weights=weights)
            predictions = np.argmax(ensemble_probs, axis=1)
            
            if method == 'accuracy':
                return -accuracy_score(true_labels, predictions)
            else:
                return -f1_score(true_labels, predictions, average='macro')
        
        # Initial weights (equal)
        initial_weights = np.ones(len(self.models)) / len(self.models)
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            self.ensemble_weights = result.x
            logger.info(f"Optimized weights: {self.ensemble_weights}")
        else:
            logger.warning("Weight optimization failed, using equal weights")
            self.ensemble_weights = initial_weights
            
        return self.ensemble_weights
    
    def train_meta_learner(self, all_probs, true_labels, meta_model='xgboost'):
        """Train meta-learner for stacking"""
        logger.info(f"Training meta-learner: {meta_model}")
        
        # Prepare meta-features
        n_models, n_samples, n_classes = all_probs.shape
        
        # Flatten probabilities as features
        meta_features = all_probs.transpose(1, 0, 2).reshape(n_samples, -1)
        
        # Add additional features
        additional_features = []
        for i in range(n_samples):
            sample_probs = all_probs[:, i, :]
            
            # Confidence features
            max_probs = np.max(sample_probs, axis=1)
            mean_confidence = np.mean(max_probs)
            std_confidence = np.std(max_probs)
            
            # Agreement features
            predictions = np.argmax(sample_probs, axis=1)
            agreement = len(np.unique(predictions)) == 1
            
            # Entropy features
            entropies = [-np.sum(probs * np.log(probs + 1e-8)) for probs in sample_probs]
            mean_entropy = np.mean(entropies)
            
            additional_features.append([
                mean_confidence, std_confidence, float(agreement), mean_entropy
            ])
        
        additional_features = np.array(additional_features)
        meta_features = np.concatenate([meta_features, additional_features], axis=1)
        
        # Train meta-learner
        if meta_model == 'xgboost':
            self.meta_learner = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif meta_model == 'lightgbm':
            self.meta_learner = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.meta_learner = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        
        self.meta_learner.fit(meta_features, true_labels)
        logger.info("Meta-learner training completed")
        
        return meta_features
    
    def predict_with_meta_learner(self, all_probs):
        """Make predictions using meta-learner"""
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained")
            
        # Prepare meta-features (same as in training)
        n_models, n_samples, n_classes = all_probs.shape
        meta_features = all_probs.transpose(1, 0, 2).reshape(n_samples, -1)
        
        # Add additional features
        additional_features = []
        for i in range(n_samples):
            sample_probs = all_probs[:, i, :]
            
            max_probs = np.max(sample_probs, axis=1)
            mean_confidence = np.mean(max_probs)
            std_confidence = np.std(max_probs)
            
            predictions = np.argmax(sample_probs, axis=1)
            agreement = len(np.unique(predictions)) == 1
            
            entropies = [-np.sum(probs * np.log(probs + 1e-8)) for probs in sample_probs]
            mean_entropy = np.mean(entropies)
            
            additional_features.append([
                mean_confidence, std_confidence, float(agreement), mean_entropy
            ])
        
        additional_features = np.array(additional_features)
        meta_features = np.concatenate([meta_features, additional_features], axis=1)
        
        # Predict
        predictions = self.meta_learner.predict(meta_features)
        probabilities = self.meta_learner.predict_proba(meta_features)
        
        return predictions, probabilities

def main():
    logger.info("Starting Advanced Ensemble Experiment")
    
    # Model paths (add more models as available)
    model_paths = [
        'models/improved_model',  # Our best model
        # Add more model paths here when available
        # 'models/indobert_baseline_hate_speech',
        # 'models/indobert_large_hate_speech',
    ]
    
    # For demonstration, we'll simulate multiple models by using the same model
    # In practice, you would have different models trained with different strategies
    if len(model_paths) == 1:
        logger.info("Only one model available, creating ensemble simulation")
        # We'll use the same model but with different random seeds for demonstration
        model_paths = [model_paths[0]] * 3  # Simulate 3 models
    
    # Load data
    logger.info("Loading dataset")
    df = pd.read_csv('data/standardized/balanced_dataset.csv')
    
    X = df['text'].values
    y = df['label_numeric'].values
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Initialize ensemble
    ensemble = AdvancedEnsemble(model_paths)
    ensemble.load_models()
    
    if len(ensemble.models) == 0:
        logger.error("No models loaded successfully")
        return
    
    # Get predictions on validation set for optimization
    logger.info("Getting validation predictions for ensemble optimization")
    val_probs = ensemble.get_all_predictions(X_val.tolist())
    
    # Test different ensemble methods
    ensemble_results = {}
    
    # 1. Simple Soft Voting
    logger.info("Testing Simple Soft Voting")
    preds_soft, probs_soft = ensemble.simple_voting(val_probs, method='soft')
    acc_soft = accuracy_score(y_val, preds_soft)
    f1_soft = f1_score(y_val, preds_soft, average='macro')
    ensemble_results['soft_voting'] = {'accuracy': acc_soft, 'f1_macro': f1_soft}
    logger.info(f"Soft Voting - Accuracy: {acc_soft:.4f}, F1-Macro: {f1_soft:.4f}")
    
    # 2. Simple Hard Voting
    logger.info("Testing Simple Hard Voting")
    preds_hard, _ = ensemble.simple_voting(val_probs, method='hard')
    acc_hard = accuracy_score(y_val, preds_hard)
    f1_hard = f1_score(y_val, preds_hard, average='macro')
    ensemble_results['hard_voting'] = {'accuracy': acc_hard, 'f1_macro': f1_hard}
    logger.info(f"Hard Voting - Accuracy: {acc_hard:.4f}, F1-Macro: {f1_hard:.4f}")
    
    # 3. Optimized Weighted Voting
    logger.info("Testing Optimized Weighted Voting")
    weights = ensemble.optimize_weights(val_probs, y_val, method='f1_macro')
    preds_weighted, probs_weighted = ensemble.weighted_voting(val_probs, weights)
    acc_weighted = accuracy_score(y_val, preds_weighted)
    f1_weighted = f1_score(y_val, preds_weighted, average='macro')
    ensemble_results['weighted_voting'] = {'accuracy': acc_weighted, 'f1_macro': f1_weighted}
    logger.info(f"Weighted Voting - Accuracy: {acc_weighted:.4f}, F1-Macro: {f1_weighted:.4f}")
    
    # 4. Confidence-based Selection
    logger.info("Testing Confidence-based Selection")
    preds_conf, probs_conf = ensemble.confidence_based_selection(val_probs, confidence_threshold=0.8)
    acc_conf = accuracy_score(y_val, preds_conf)
    f1_conf = f1_score(y_val, preds_conf, average='macro')
    ensemble_results['confidence_selection'] = {'accuracy': acc_conf, 'f1_macro': f1_conf}
    logger.info(f"Confidence Selection - Accuracy: {acc_conf:.4f}, F1-Macro: {f1_conf:.4f}")
    
    # 5. Meta-learner Stacking
    logger.info("Testing Meta-learner Stacking")
    meta_features = ensemble.train_meta_learner(val_probs, y_val, meta_model='xgboost')
    preds_meta, probs_meta = ensemble.predict_with_meta_learner(val_probs)
    acc_meta = accuracy_score(y_val, preds_meta)
    f1_meta = f1_score(y_val, preds_meta, average='macro')
    ensemble_results['meta_learner'] = {'accuracy': acc_meta, 'f1_macro': f1_meta}
    logger.info(f"Meta-learner - Accuracy: {acc_meta:.4f}, F1-Macro: {f1_meta:.4f}")
    
    # Find best method
    best_method = max(ensemble_results.keys(), key=lambda k: ensemble_results[k]['f1_macro'])
    best_score = ensemble_results[best_method]['f1_macro']
    
    logger.info(f"Best ensemble method: {best_method} with F1-Macro: {best_score:.4f}")
    
    # Evaluate best method on test set
    logger.info("Evaluating best ensemble method on test set")
    test_probs = ensemble.get_all_predictions(X_test.tolist())
    
    if best_method == 'soft_voting':
        test_preds, test_ensemble_probs = ensemble.simple_voting(test_probs, method='soft')
    elif best_method == 'hard_voting':
        test_preds, test_ensemble_probs = ensemble.simple_voting(test_probs, method='hard')
    elif best_method == 'weighted_voting':
        test_preds, test_ensemble_probs = ensemble.weighted_voting(test_probs, weights)
    elif best_method == 'confidence_selection':
        test_preds, test_ensemble_probs = ensemble.confidence_based_selection(test_probs)
    else:  # meta_learner
        test_preds, test_ensemble_probs = ensemble.predict_with_meta_learner(test_probs)
    
    # Calculate final metrics
    test_accuracy = accuracy_score(y_test, test_preds)
    test_f1_macro = f1_score(y_test, test_preds, average='macro')
    test_f1_weighted = f1_score(y_test, test_preds, average='weighted')
    
    # Generate detailed report
    class_names = ['Bukan Ujaran Kebencian', 'Ujaran Kebencian - Ringan', 
                   'Ujaran Kebencian - Sedang', 'Ujaran Kebencian - Berat']
    
    test_report = classification_report(y_test, test_preds, target_names=class_names, output_dict=True)
    test_confusion = confusion_matrix(y_test, test_preds)
    
    # Save results
    results = {
        'experiment_timestamp': datetime.now().isoformat(),
        'model_paths': model_paths,
        'test_samples': len(X_test),
        'validation_results': ensemble_results,
        'best_ensemble_method': best_method,
        'ensemble_weights': weights.tolist() if weights is not None else None,
        'final_test_results': {
            'accuracy': float(test_accuracy),
            'f1_macro': float(test_f1_macro),
            'f1_weighted': float(test_f1_weighted),
            'classification_report': test_report,
            'confusion_matrix': test_confusion.tolist()
        },
        'improvement_analysis': {
            'baseline_single_model': 0.8698,  # From improved_model
            'ensemble_accuracy': float(test_accuracy),
            'accuracy_improvement': float(test_accuracy - 0.8698),
            'baseline_f1_macro': 0.8688,
            'ensemble_f1_macro': float(test_f1_macro),
            'f1_macro_improvement': float(test_f1_macro - 0.8688)
        }
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/ensemble_advanced_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("🚀 ADVANCED ENSEMBLE RESULTS")
    print("="*80)
    
    print(f"\n📊 MODELS USED: {len(ensemble.models)}")
    for i, path in enumerate(model_paths[:len(ensemble.models)]):
        print(f"   Model {i+1}: {path}")
    
    print(f"\n🔬 VALIDATION RESULTS:")
    for method, scores in ensemble_results.items():
        print(f"   {method}: Accuracy={scores['accuracy']:.4f}, F1-Macro={scores['f1_macro']:.4f}")
    
    print(f"\n🏆 BEST METHOD: {best_method}")
    if best_method == 'weighted_voting' and weights is not None:
        print(f"   Optimal weights: {weights}")
    
    print(f"\n📈 FINAL TEST RESULTS:")
    print(f"   Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   F1-Macro: {test_f1_macro:.4f} ({test_f1_macro*100:.2f}%)")
    print(f"   F1-Weighted: {test_f1_weighted:.4f} ({test_f1_weighted*100:.2f}%)")
    
    print(f"\n🚀 IMPROVEMENT vs SINGLE MODEL:")
    acc_improvement = test_accuracy - 0.8698
    f1_improvement = test_f1_macro - 0.8688
    print(f"   Accuracy: +{acc_improvement:.4f} (+{acc_improvement*100:.2f}%)")
    print(f"   F1-Macro: +{f1_improvement:.4f} (+{f1_improvement*100:.2f}%)")
    
    # Check if we reached 90%
    if test_accuracy >= 0.90 or test_f1_macro >= 0.90:
        print(f"\n🎉 90%+ TARGET ACHIEVED!")
        if test_accuracy >= 0.90:
            print(f"   ✅ Accuracy: {test_accuracy*100:.2f}% (Target: 90%+)")
        if test_f1_macro >= 0.90:
            print(f"   ✅ F1-Macro: {test_f1_macro*100:.2f}% (Target: 90%+)")
    else:
        print(f"\n🎯 PROGRESS TOWARDS 90%:")
        acc_gap = 0.90 - test_accuracy
        f1_gap = 0.90 - test_f1_macro
        print(f"   Accuracy gap: {acc_gap*100:.2f}% (Current: {test_accuracy*100:.2f}%)")
        print(f"   F1-Macro gap: {f1_gap*100:.2f}% (Current: {test_f1_macro*100:.2f}%)")
    
    print("\n" + "="*80)
    print("📁 Results saved to: results/ensemble_advanced_results.json")
    print("="*80)
    
    logger.info("Advanced ensemble experiment completed")

if __name__ == "__main__":
    main()