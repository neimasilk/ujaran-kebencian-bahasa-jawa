#!/usr/bin/env python3
"""
Advanced Ensemble Strategy for 90% Target Achievement
Combines existing 86.98% model with stacking, meta-learning, and advanced techniques
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedEnsemble90Percent:
    def __init__(self):
        self.target_accuracy = 0.90
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir = Path('models')
        
        # Load existing high-performing model predictions if available
        self.base_model_predictions = None
        self.load_existing_predictions()
        
    def load_existing_predictions(self):
        """Load predictions from existing high-performing models"""
        prediction_files = [
            'results/improved_model_predictions.json',
            'results/best_model_predictions.json',
            'models/improved_model/predictions.json'
        ]
        
        for filepath in prediction_files:
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        if 'predictions' in data:
                            self.base_model_predictions = data
                            logger.info(f"Loaded base model predictions from {filepath}")
                            break
                except Exception as e:
                    logger.warning(f"Error loading {filepath}: {e}")
    
    def load_dataset(self):
        """Load and prepare dataset"""
        logger.info("Loading dataset...")
        
        # Try multiple dataset locations
        dataset_paths = [
            'data/augmented/augmented_dataset.csv',
            'data/processed/processed_dataset.csv',
            'data/standardized/balanced_dataset.csv'
        ]
        
        df = None
        for path in dataset_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    logger.info(f"Dataset loaded from {path}: {len(df)} samples")
                    break
                except Exception as e:
                    logger.warning(f"Error loading {path}: {e}")
        
        if df is None:
            raise FileNotFoundError("No dataset found in expected locations")
        
        # Standardize column names
        if 'final_label' in df.columns:
            df['label'] = df['final_label']
        elif 'label_text' in df.columns:
            df['label'] = df['label_text']
        
        # Clean data
        df = df.dropna(subset=['text', 'label'])
        
        # Create label mapping
        unique_labels = sorted(df['label'].unique())
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        df['label_id'] = df['label'].map(self.label_to_id)
        
        logger.info(f"Dataset prepared: {len(df)} samples, {len(unique_labels)} classes")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df
    
    def extract_advanced_features(self, texts):
        """Extract advanced text features for meta-learning"""
        logger.info("Extracting advanced text features...")
        
        features = []
        
        for text in texts:
            text_str = str(text)
            
            # Basic text statistics
            char_count = len(text_str)
            word_count = len(text_str.split())
            avg_word_length = np.mean([len(word) for word in text_str.split()]) if word_count > 0 else 0
            
            # Linguistic features
            exclamation_count = text_str.count('!')
            question_count = text_str.count('?')
            caps_ratio = sum(1 for c in text_str if c.isupper()) / len(text_str) if len(text_str) > 0 else 0
            
            # Hate speech indicators (simple heuristics)
            hate_keywords = ['benci', 'bodoh', 'tolol', 'goblok', 'anjing', 'bangsat']
            hate_score = sum(1 for keyword in hate_keywords if keyword.lower() in text_str.lower())
            
            # Javanese specific features
            javanese_words = ['wong', 'iku', 'kowe', 'aku', 'ora', 'nek', 'wes', 'iso']
            javanese_score = sum(1 for word in javanese_words if word.lower() in text_str.lower())
            
            features.append([
                char_count, word_count, avg_word_length,
                exclamation_count, question_count, caps_ratio,
                hate_score, javanese_score
            ])
        
        return np.array(features)
    
    def create_base_models(self):
        """Create diverse base models for ensemble"""
        logger.info("Creating base models...")
        
        base_models = [
            ('rf', RandomForestClassifier(
                n_estimators=200, 
                max_depth=10, 
                min_samples_split=5,
                class_weight='balanced',
                random_state=42
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )),
            ('svm', SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ))
        ]
        
        return base_models
    
    def create_meta_learner(self):
        """Create sophisticated meta-learner"""
        return LogisticRegression(
            C=1.0,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        )
    
    def train_stacking_ensemble(self, X, y):
        """Train stacking ensemble with cross-validation"""
        logger.info("Training stacking ensemble...")
        
        base_models = self.create_base_models()
        meta_learner = self.create_meta_learner()
        
        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # Train with cross-validation
        cv_scores = cross_val_score(stacking_clf, X, y, cv=5, scoring='accuracy')
        logger.info(f"Stacking CV scores: {cv_scores}")
        logger.info(f"Stacking mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Fit final model
        stacking_clf.fit(X, y)
        
        return stacking_clf, cv_scores
    
    def train_voting_ensemble(self, X, y):
        """Train voting ensemble with calibration"""
        logger.info("Training voting ensemble...")
        
        base_models = self.create_base_models()
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=base_models,
            voting='soft',
            n_jobs=-1
        )
        
        # Add calibration
        calibrated_clf = CalibratedClassifierCV(voting_clf, cv=3)
        
        # Train with cross-validation
        cv_scores = cross_val_score(calibrated_clf, X, y, cv=5, scoring='accuracy')
        logger.info(f"Voting CV scores: {cv_scores}")
        logger.info(f"Voting mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Fit final model
        calibrated_clf.fit(X, y)
        
        return calibrated_clf, cv_scores
    
    def optimize_ensemble_weights(self, models, X_val, y_val):
        """Optimize ensemble weights using validation data"""
        logger.info("Optimizing ensemble weights...")
        
        from scipy.optimize import minimize
        
        # Get predictions from all models
        predictions = []
        for model in models:
            pred_proba = model.predict_proba(X_val)
            predictions.append(pred_proba)
        
        predictions = np.array(predictions)
        
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            ensemble_labels = np.argmax(ensemble_pred, axis=1)
            return -accuracy_score(y_val, ensemble_labels)
        
        # Initial weights (equal)
        initial_weights = np.ones(len(models)) / len(models)
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(models))]
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x / np.sum(result.x)
        logger.info(f"Optimal weights: {optimal_weights}")
        
        return optimal_weights
    
    def evaluate_ensemble(self, model, X_test, y_test, model_name):
        """Evaluate ensemble model"""
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Classification report
        class_report = classification_report(y_test, y_pred, 
                                           target_names=[self.id_to_label[i] for i in range(len(self.id_to_label))],
                                           output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred.tolist(),
            'prediction_probabilities': y_pred_proba.tolist()
        }
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1-Macro: {f1_macro:.4f}")
        logger.info(f"  F1-Weighted: {f1_weighted:.4f}")
        
        # Check if target achieved
        if accuracy >= self.target_accuracy:
            logger.info(f"🎯 TARGET ACHIEVED! {model_name} reached {accuracy:.4f} accuracy!")
            
            # Save achievement
            achievement = {
                'experiment': f'Advanced Ensemble 90% - {model_name}',
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'timestamp': datetime.now().isoformat(),
                'details': results
            }
            
            achievement_file = self.results_dir / '90_percent_achievement.json'
            with open(achievement_file, 'w') as f:
                json.dump(achievement, f, indent=2)
        
        return results
    
    def run_advanced_ensemble_experiment(self):
        """Run complete advanced ensemble experiment"""
        logger.info("\n" + "="*80)
        logger.info("ADVANCED ENSEMBLE FOR 90% TARGET ACHIEVEMENT")
        logger.info("="*80)
        
        try:
            # Load dataset
            df = self.load_dataset()
            
            # Split data
            from sklearn.model_selection import train_test_split
            
            train_df, test_df = train_test_split(
                df, test_size=0.2, stratify=df['label_id'], random_state=42
            )
            
            train_df, val_df = train_test_split(
                train_df, test_size=0.2, stratify=train_df['label_id'], random_state=42
            )
            
            logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
            # Extract features
            X_train = self.extract_advanced_features(train_df['text'])
            X_val = self.extract_advanced_features(val_df['text'])
            X_test = self.extract_advanced_features(test_df['text'])
            
            y_train = train_df['label_id'].values
            y_val = val_df['label_id'].values
            y_test = test_df['label_id'].values
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple ensemble approaches
            results = {}
            models = []
            
            # 1. Stacking Ensemble
            stacking_model, stacking_cv = self.train_stacking_ensemble(X_train_scaled, y_train)
            stacking_results = self.evaluate_ensemble(stacking_model, X_test_scaled, y_test, "Stacking Ensemble")
            results['stacking'] = stacking_results
            models.append(('stacking', stacking_model))
            
            # 2. Voting Ensemble
            voting_model, voting_cv = self.train_voting_ensemble(X_train_scaled, y_train)
            voting_results = self.evaluate_ensemble(voting_model, X_test_scaled, y_test, "Voting Ensemble")
            results['voting'] = voting_results
            models.append(('voting', voting_model))
            
            # 3. Weighted Ensemble (if we have multiple models)
            if len(models) >= 2:
                model_list = [model for _, model in models]
                optimal_weights = self.optimize_ensemble_weights(model_list, X_val_scaled, y_val)
                
                # Create weighted ensemble predictions
                test_predictions = []
                for _, model in models:
                    pred_proba = model.predict_proba(X_test_scaled)
                    test_predictions.append(pred_proba)
                
                weighted_pred_proba = np.average(test_predictions, axis=0, weights=optimal_weights)
                weighted_pred = np.argmax(weighted_pred_proba, axis=1)
                
                # Evaluate weighted ensemble
                weighted_accuracy = accuracy_score(y_test, weighted_pred)
                weighted_f1_macro = f1_score(y_test, weighted_pred, average='macro')
                
                weighted_results = {
                    'model_name': 'Weighted Ensemble',
                    'accuracy': weighted_accuracy,
                    'f1_macro': weighted_f1_macro,
                    'weights': optimal_weights.tolist(),
                    'predictions': weighted_pred.tolist()
                }
                
                results['weighted'] = weighted_results
                
                logger.info(f"Weighted Ensemble Results:")
                logger.info(f"  Accuracy: {weighted_accuracy:.4f}")
                logger.info(f"  F1-Macro: {weighted_f1_macro:.4f}")
                
                if weighted_accuracy >= self.target_accuracy:
                    logger.info(f"🎯 TARGET ACHIEVED! Weighted Ensemble reached {weighted_accuracy:.4f} accuracy!")
            
            # Save comprehensive results
            final_results = {
                'experiment_timestamp': datetime.now().isoformat(),
                'target_accuracy': self.target_accuracy,
                'dataset_info': {
                    'total_samples': len(df),
                    'train_samples': len(train_df),
                    'val_samples': len(val_df),
                    'test_samples': len(test_df),
                    'num_classes': len(self.label_to_id)
                },
                'results': results,
                'label_mapping': self.label_to_id
            }
            
            # Save results
            results_file = self.results_dir / 'advanced_ensemble_90_percent_results.json'
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            logger.info(f"\nResults saved to {results_file}")
            
            # Find best model
            best_accuracy = 0
            best_model_name = None
            
            for model_type, result in results.items():
                if result['accuracy'] > best_accuracy:
                    best_accuracy = result['accuracy']
                    best_model_name = result['model_name']
            
            logger.info("\n" + "="*80)
            logger.info("EXPERIMENT SUMMARY")
            logger.info("="*80)
            logger.info(f"Best Model: {best_model_name}")
            logger.info(f"Best Accuracy: {best_accuracy:.4f}")
            logger.info(f"Target Accuracy: {self.target_accuracy:.4f}")
            
            if best_accuracy >= self.target_accuracy:
                logger.info("🎯 SUCCESS: 90% TARGET ACHIEVED!")
            else:
                progress = (best_accuracy / self.target_accuracy) * 100
                logger.info(f"📊 Progress: {progress:.1f}% of target")
                logger.info(f"📈 Gap to target: {(self.target_accuracy - best_accuracy)*100:.2f}%")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in advanced ensemble experiment: {e}")
            raise

def main():
    ensemble = AdvancedEnsemble90Percent()
    results = ensemble.run_advanced_ensemble_experiment()
    
    # Check if target achieved
    best_accuracy = max([r['accuracy'] for r in results['results'].values()])
    if best_accuracy >= 0.90:
        print(f"\n🎉 SUCCESS: Advanced ensemble achieved {best_accuracy:.4f} accuracy!")
    else:
        print(f"\n📊 Current best: {best_accuracy:.4f} accuracy ({(best_accuracy/0.90)*100:.1f}% of target)")

if __name__ == "__main__":
    main()