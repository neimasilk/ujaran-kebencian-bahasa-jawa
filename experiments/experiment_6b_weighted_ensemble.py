"""
EXPERIMENT 6B: Weighted Ensemble dengan XGBoost Meta-Learner
================================================================

Objective: Tingkatkan akurasi dengan advanced ensemble method
- Base models: IndoBERT (Label Smoothing, Combined, CE)
- Meta-learner: XGBoost dengan hyperparameter tuning
- Target: F1-Macro 82%+

Setup:
1. Load existing model predictions
2. Train XGBoost meta-learner dengan cross-validation
3. Evaluate pada test set
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# XGBoost
import xgboost as xgb

# PyTorch for loading existing models
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleConfig:
    """Configuration untuk Experiment 6B"""

    # Models to ensemble (from Experiment 6A)
    # Note: checkpoint-1503 is best (likely Label Smoothing: 81.38%)
    #       checkpoint-2505 is final (Combined: 81.24%)
    models = [
        {
            'name': 'label_smoothing',
            'path': 'models/experiment_6a_focal_loss/checkpoint-1503/',
            'weight': 1.0
        },
        {
            'name': 'combined',
            'path': 'models/experiment_6a_focal_loss/checkpoint-2505/',
            'weight': 0.8
        },
        {
            'name': 'baseline',
            'path': 'models/experiment_5_improved/',
            'weight': 0.6
        }
    ]

    # Data paths
    phase3_path: str = "data/improved/phase3_relabeled.csv"
    phase4_path: str = "data/improved/phase4_generated.csv"

    # Output paths
    output_dir: str = "results/experiment_6b_weighted_ensemble"

    # Training hyperparameters
    seed: int = 42
    max_length: int = 128
    batch_size: int = 32

    # XGBoost parameters
    xgb_params = {
        'max_depth': 5,
        'learning_rate': 0.1,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softprob',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'random_state': 42
    }


class WeightedEnsembleBuilder:
    """Builder untuk weighted ensemble dengan XGBoost"""

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def load_model_and_tokenizer(self, model_path: str, model_name: str):
        """Load model dan tokenizer dari path"""
        logger.info(f"Loading {model_name} from {model_path}")

        if not os.path.exists(model_path):
            logger.warning(f"Model path not found: {model_path}")
            return None, None

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(self.device)
            model.eval()

            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            return None, None

    def prepare_data(self):
        """Load dan prepare dataset"""
        logger.info("Loading dataset...")

        # Load Phase 3 dan 4
        phase3_df = pd.read_csv(self.config.phase3_path)
        phase4_df = pd.read_csv(self.config.phase4_path)

        # Prepare Phase 3
        phase3_train = phase3_df[['text', 'new_label']].copy()
        phase3_train = phase3_train.rename(columns={'new_label': 'label'})

        # Prepare Phase 4
        phase4_train = phase4_df[['text', 'label']].copy()

        # Combine
        combined_df = pd.concat([phase3_train, phase4_train], ignore_index=True)
        combined_df = combined_df.sample(
            frac=1, random_state=self.config.seed
        ).reset_index(drop=True)

        # Split (gunakan split yang sama dengan Experiment 5 dan 6A)
        from sklearn.model_selection import train_test_split

        train_df, temp_df = train_test_split(
            combined_df,
            test_size=0.2,
            random_state=self.config.seed,
            stratify=combined_df['label']
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=self.config.seed,
            stratify=temp_df['label']
        )

        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def get_model_predictions(self, model, tokenizer, df: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities dari model"""
        texts = df['text'].tolist()

        # Tokenize
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )

        # Create dataset
        class PredictionDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings

            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.encodings.items()}

            def __len__(self):
                return len(self.encodings['input_ids'])

        dataset = PredictionDataset(encodings)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size)

        # Get predictions
        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def build_meta_features(self, val_df, test_df):
        """Build meta-features dari semua models"""
        logger.info("Building meta-features...")

        val_meta_features = []
        test_meta_features = []
        n_models_loaded = 0

        for model_config in self.config.models:
            model_path = model_config['path']
            model_name = model_config['name']
            weight = model_config['weight']

            logger.info(f"Processing {model_name}...")

            # Load model
            model, tokenizer = self.load_model_and_tokenizer(model_path, model_name)

            if model is None:
                logger.warning(f"Skipping {model_name} - model not found")
                continue

            n_models_loaded += 1

            # Get predictions
            val_probs = self.get_model_predictions(model, tokenizer, val_df)
            test_probs = self.get_model_predictions(model, tokenizer, test_df)

            # Apply weight
            val_probs_weighted = val_probs * weight
            test_probs_weighted = test_probs * weight

            val_meta_features.append(val_probs_weighted)
            test_meta_features.append(test_probs_weighted)

        # Stack all features
        val_meta = np.hstack(val_meta_features)
        test_meta = np.hstack(test_meta_features)

        logger.info(f"Meta-feature shape: {val_meta.shape}, Models loaded: {n_models_loaded}")

        return val_meta, test_meta, n_models_loaded

    def train_xgb_meta_learner(self, X_meta, y, X_test_meta, y_test):
        """Train XGBoost meta-learner dengan cross-validation"""
        logger.info("Training XGBoost meta-learner...")

        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_meta, label=y)
        dtest = xgb.DMatrix(X_test_meta, label=y_test)

        # Cross-validation untuk tuning
        cv_results = xgb.cv(
            self.config.xgb_params,
            dtrain,
            num_boost_round=200,
            nfold=5,
            stratified=True,
            early_stopping_rounds=20,
            seed=self.config.seed,
            verbose_eval=False
        )

        best_n_rounds = len(cv_results)
        logger.info(f"Best rounds: {best_n_rounds}")

        # Train final model
        model = xgb.XGBClassifier(
            **self.config.xgb_params,
            n_estimators=best_n_rounds
        )

        # Cross-validation score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.seed)
        cv_scores = cross_val_score(model, X_meta, y, cv=cv, scoring='f1_macro')

        logger.info(f"CV F1-Macro: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Train on full data
        model.fit(X_meta, y)

        # Predictions
        preds = model.predict(X_test_meta)

        return model, preds, cv_scores

    def simple_weighted_ensemble(self, val_meta, test_meta, val_labels, test_labels, n_models_loaded):
        """Simple weighted averaging tanpa meta-learner"""
        logger.info("Running simple weighted ensemble...")

        # Average probabilities
        n_classes = 4

        # Each model contributes n_classes features
        # Reshape: (n_samples, n_models * n_classes) -> (n_samples, n_models, n_classes)
        n_samples = val_meta.shape[0]
        val_probs = val_meta.reshape(n_samples, n_models_loaded, n_classes).mean(axis=1)

        n_samples_test = test_meta.shape[0]
        test_probs = test_meta.reshape(n_samples_test, n_models_loaded, n_classes).mean(axis=1)

        # Predictions
        val_preds = val_probs.argmax(axis=1)
        test_preds = test_probs.argmax(axis=1)

        # Metrics
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        test_f1 = f1_score(test_labels, test_preds, average='macro')

        logger.info(f"Simple Weighted - Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}")

        return test_preds, {'f1_macro': test_f1}

    def run_experiment(self):
        """Run complete ensemble experiment"""
        logger.info("=" * 60)
        logger.info("EXPERIMENT 6B: WEIGHTED ENSEMBLE WITH XGBOOST")
        logger.info("=" * 60)

        # Prepare data
        train_df, val_df, test_df = self.prepare_data()

        # Build meta-features
        val_meta, test_meta, n_models_loaded = self.build_meta_features(val_df, test_df)

        val_labels = val_df['label'].values
        test_labels = test_df['label'].values

        results = {}

        # Method 1: Simple Weighted Average
        logger.info("\n" + "=" * 50)
        logger.info("METHOD 1: SIMPLE WEIGHTED AVERAGE")
        logger.info("=" * 50)

        simple_preds, simple_metrics = self.simple_weighted_ensemble(
            val_meta, test_meta, val_labels, test_labels, n_models_loaded
        )

        # Detailed metrics for simple
        simple_f1 = f1_score(test_labels, simple_preds, average='macro')
        simple_acc = accuracy_score(test_labels, simple_preds)
        simple_report = classification_report(
            test_labels, simple_preds,
            target_names=['Neutral', 'Light', 'Moderate', 'Severe'],
            output_dict=True
        )

        results['simple_weighted'] = {
            'f1_macro': simple_f1,
            'accuracy': simple_acc,
            'per_class_f1': {
                'neutral': simple_report['Neutral']['f1-score'],
                'light': simple_report['Light']['f1-score'],
                'moderate': simple_report['Moderate']['f1-score'],
                'severe': simple_report['Severe']['f1-score'],
            }
        }

        # Method 2: XGBoost Meta-Learner
        logger.info("\n" + "=" * 50)
        logger.info("METHOD 2: XGBOOST META-LEARNER")
        logger.info("=" * 50)

        xgb_model, xgb_preds, cv_scores = self.train_xgb_meta_learner(
            val_meta, val_labels, test_meta, test_labels
        )

        # Detailed metrics for XGBoost
        xgb_f1 = f1_score(test_labels, xgb_preds, average='macro')
        xgb_acc = accuracy_score(test_labels, xgb_preds)
        xgb_report = classification_report(
            test_labels, xgb_preds,
            target_names=['Neutral', 'Light', 'Moderate', 'Severe'],
            output_dict=True
        )

        results['xgboost'] = {
            'f1_macro': xgb_f1,
            'accuracy': xgb_acc,
            'cv_f1_mean': float(cv_scores.mean()),
            'cv_f1_std': float(cv_scores.std()),
            'per_class_f1': {
                'neutral': xgb_report['Neutral']['f1-score'],
                'light': xgb_report['Light']['f1-score'],
                'moderate': xgb_report['Moderate']['f1-score'],
                'severe': xgb_report['Severe']['f1-score'],
            }
        }

        # Method 3: XGBoost dengan tuned hyperparameters
        logger.info("\n" + "=" * 50)
        logger.info("METHOD 3: XGBOOST TUNED")
        logger.info("=" * 50)

        # Try different parameters
        tuned_params = self.config.xgb_params.copy()
        tuned_params.update({
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
        })

        tuned_model = xgb.XGBClassifier(**tuned_params, n_estimators=150)

        # Train
        tuned_model.fit(val_meta, val_labels)
        tuned_preds = tuned_model.predict(test_meta)

        # Metrics
        tuned_f1 = f1_score(test_labels, tuned_preds, average='macro')
        tuned_acc = accuracy_score(test_labels, tuned_preds)
        tuned_report = classification_report(
            test_labels, tuned_preds,
            target_names=['Neutral', 'Light', 'Moderate', 'Severe'],
            output_dict=True
        )

        results['xgboost_tuned'] = {
            'f1_macro': tuned_f1,
            'accuracy': tuned_acc,
            'per_class_f1': {
                'neutral': tuned_report['Neutral']['f1-score'],
                'light': tuned_report['Light']['f1-score'],
                'moderate': tuned_report['Moderate']['f1-score'],
                'severe': tuned_report['Severe']['f1-score'],
            }
        }

        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT 6B: SUMMARY")
        print("=" * 60)
        print(f"\nBaseline (Label Smoothing): 81.38%")
        print("\nEnsemble Methods:")
        for method, metrics in results.items():
            print(f"  {method}:")
            print(f"    F1-Macro: {metrics['f1_macro']:.2%}")
            print(f"    Accuracy: {metrics['accuracy']:.2%}")
            print(f"    Per-Class F1:")
            for cls, f1 in metrics['per_class_f1'].items():
                print(f"      {cls}: {f1:.2%}")
            print()

        # Find best method
        best_method = max(results.items(), key=lambda x: x[1]['f1_macro'])
        best_f1 = best_method[1]['f1_macro']
        improvement = best_f1 - 0.8138

        print("=" * 60)
        print(f"BEST METHOD: {best_method[0].upper()}")
        print(f"F1-Macro: {best_f1:.2%}")
        print(f"Improvement: {improvement:+.2%}")
        print("=" * 60)

        # Save results
        output_path = os.path.join(self.config.output_dir, 'results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

        return results


def main():
    """Run Experiment 6B"""
    config = EnsembleConfig()
    builder = WeightedEnsembleBuilder(config)
    results = builder.run_experiment()
    return results


if __name__ == "__main__":
    results = main()
    print("\n[OK] Experiment 6B complete!")
