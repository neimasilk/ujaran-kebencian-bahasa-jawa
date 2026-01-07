"""
EXPERIMENT 6 & 7: Multi-Model Training & Ensemble dengan Dataset Improved 10K+
================================================================================

Experiment 6: Train multiple models dengan dataset improved
- IndoBERT Base (sudah done: 79.19%)
- mBERT
- XLM-RoBERTa

Experiment 7: Ensemble dari semua models
Target: F1-Macro 80-82%
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from transformers.trainer_utils import EvalPrediction
from datasets import Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MultiModelConfig:
    """Configuration untuk Experiment 6 & 7"""

    # Models to train
    models: List[str] = None

    # Data paths
    phase3_path: str = "data/improved/phase3_relabeled.csv"
    phase4_path: str = "data/improved/phase4_generated.csv"

    # Output paths
    output_dir: str = "results/experiment_6_7_multi_model"
    models_dir: str = "models/experiment_6_7_improved"

    # Training hyperparameters
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Random seed
    seed: int = 42

    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    def __post_init__(self):
        if self.models is None:
            self.models = [
                "bert-base-multilingual-cased",  # mBERT
                "xlm-roberta-base",                # XLM-RoBERTa
            ]


class ImprovedDatasetPreparer:
    """Persiapan dataset improved untuk training"""

    def __init__(self, config: MultiModelConfig):
        self.config = config
        self.combined_df = None

    def load_data(self):
        """Load dataset improved dari Phase 3 dan Phase 4"""
        logger.info("Loading improved dataset...")

        # Load Phase 3 (re-labeled)
        phase3_df = pd.read_csv(self.config.phase3_path)
        logger.info(f"Phase 3 loaded: {len(phase3_df)} records")

        # Load Phase 4 (generated)
        phase4_df = pd.read_csv(self.config.phase4_path)
        logger.info(f"Phase 4 loaded: {len(phase4_df)} records")

        # Prepare Phase 3: use 'text' and 'new_label'
        phase3_train = phase3_df[['text', 'new_label']].copy()
        phase3_train = phase3_train.rename(columns={'new_label': 'label'})

        # Prepare Phase 4: use 'text' and 'label'
        phase4_train = phase4_df[['text', 'label']].copy()

        # Combine
        self.combined_df = pd.concat([phase3_train, phase4_train], ignore_index=True)

        # Shuffle
        self.combined_df = self.combined_df.sample(
            frac=1, random_state=self.config.seed
        ).reset_index(drop=True)

        logger.info(f"Combined dataset: {len(self.combined_df)} records")
        logger.info(f"\nLabel distribution:\n{self.combined_df['label'].value_counts()}")

        return self.combined_df

    def split_data(self):
        """Split data menjadi train/val/test"""
        logger.info("Splitting data...")

        # First split: train + temp (val + test)
        train_df, temp_df = train_test_split(
            self.combined_df,
            test_size=(self.config.val_ratio + self.config.test_ratio),
            random_state=self.config.seed,
            stratify=self.combined_df['label']
        )

        # Second split: val and test from temp
        val_ratio_adjusted = self.config.val_ratio / (
            self.config.val_ratio + self.config.test_ratio
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adjusted),
            random_state=self.config.seed,
            stratify=temp_df['label']
        )

        logger.info(f"Train: {len(train_df)} records")
        logger.info(f"Val: {len(val_df)} records")
        logger.info(f"Test: {len(test_df)} records")

        return train_df, val_df, test_df


class MultiModelTrainer:
    """Trainer untuk multiple models"""

    def __init__(self, config: MultiModelConfig):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.trainers = {}
        self.results = {}

        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.models_dir).mkdir(parents=True, exist_ok=True)

    def setup_model(self, model_name: str):
        """Setup model dan tokenizer"""
        logger.info(f"Setting up model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=4
        )

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        self.tokenizers[model_name] = tokenizer
        self.models[model_name] = model

        return model, tokenizer

    def tokenize_data(self, df: pd.DataFrame, model_name: str):
        """Tokenize dataframe"""
        tokenizer = self.tokenizers[model_name]
        texts = df['text'].tolist()
        labels = df['label'].tolist()

        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors=None
        )

        # Add labels
        encodings['labels'] = labels

        return Dataset.from_dict(encodings)

    def train_model(self, model_name: str, train_dataset, val_dataset):
        """Train single model"""
        logger.info(f"Training {model_name}...")

        model = self.models[model_name]

        def compute_metrics(p: EvalPrediction):
            preds = np.argmax(p.predictions, axis=1)
            labels = p.label_ids
            return {
                'accuracy': accuracy_score(labels, preds),
                'f1_macro': f1_score(labels, preds, average='macro'),
                'precision_macro': precision_score(labels, preds, average='macro'),
                'recall_macro': recall_score(labels, preds, average='macro'),
            }

        model_dir = os.path.join(self.config.models_dir, model_name.replace("/", "_"))

        training_args = TrainingArguments(
            output_dir=model_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            seed=self.config.seed,
            logging_dir=os.path.join(model_dir, "logs"),
            logging_steps=50,
            report_to=["none"],
            save_total_limit=2,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizers[model_name],
            data_collator=DataCollatorWithPadding(self.tokenizers[model_name]),
            compute_metrics=compute_metrics,
        )

        train_result = trainer.train()
        self.trainers[model_name] = trainer

        logger.info(f"{model_name} training complete")

        return trainer, train_result

    def evaluate_model(self, model_name: str, test_dataset):
        """Evaluate single model pada test set"""
        logger.info(f"Evaluating {model_name}...")

        trainer = self.trainers[model_name]
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids

        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_micro': f1_score(labels, preds, average='micro'),
            'precision_macro': precision_score(labels, preds, average='macro'),
            'recall_macro': recall_score(labels, preds, average='macro'),
        }

        # Get prediction probabilities for ensemble
        probs = torch.nn.functional.softmax(
            torch.from_numpy(predictions.predictions), dim=-1
        ).numpy()

        logger.info(f"{model_name} - F1-Macro: {metrics['f1_macro']:.4f}")

        return metrics, preds, probs

    def create_ensemble(self, val_probs, val_labels, test_probs):
        """Create ensemble using stacking dengan Logistic Regression"""
        logger.info("Creating ensemble...")

        # Flatten validation probs untuk training meta-learner
        n_models = len(val_probs)
        n_samples = len(val_labels)
        n_classes = 4

        # Reshape: (n_samples, n_models * n_classes)
        val_meta = np.hstack([p.reshape(n_samples, n_classes) for p in val_probs])

        # Train meta-learner
        meta_learner = LogisticRegression(
            max_iter=1000,
            random_state=self.config.seed,
            solver='lbfgs'
        )
        meta_learner.fit(val_meta, val_labels)

        # Make predictions on test set
        n_test = len(test_probs[0])
        test_meta = np.hstack([p.reshape(n_test, n_classes) for p in test_probs])
        ensemble_preds = meta_learner.predict(test_meta)

        logger.info("Ensemble created successfully")

        return meta_learner, ensemble_preds

    def run_all_experiments(self, train_df, val_df, test_df):
        """Run all experiments"""

        # Step 1: Train models
        model_results = {}
        val_probs_list = []
        test_probs_list = []

        for model_name in self.config.models:
            # Setup
            self.setup_model(model_name)

            # Tokenize
            train_dataset = self.tokenize_data(train_df, model_name)
            val_dataset = self.tokenize_data(val_df, model_name)
            test_dataset = self.tokenize_data(test_df, model_name)

            # Train
            trainer, _ = self.train_model(model_name, train_dataset, val_dataset)

            # Evaluate
            metrics, preds, probs = self.evaluate_model(model_name, test_dataset)

            # Get validation probs for ensemble
            val_pred = trainer.predict(val_dataset)
            val_probs = torch.nn.functional.softmax(
                torch.from_numpy(val_pred.predictions), dim=-1
            ).numpy()

            model_results[model_name] = {
                'metrics': metrics,
                'predictions': preds,
                'test_probs': probs,
                'val_probs': val_probs
            }

            val_probs_list.append(val_probs)
            test_probs_list.append(probs)

        # Step 2: Create Ensemble
        val_labels = val_df['label'].values
        meta_learner, ensemble_preds = self.create_ensemble(
            val_probs_list, val_labels, test_probs_list
        )

        # Step 3: Evaluate Ensemble
        test_labels = test_df['label'].values
        ensemble_metrics = {
            'accuracy': accuracy_score(test_labels, ensemble_preds),
            'f1_macro': f1_score(test_labels, ensemble_preds, average='macro'),
            'f1_micro': f1_score(test_labels, ensemble_preds, average='micro'),
            'precision_macro': precision_score(test_labels, ensemble_preds, average='macro'),
            'recall_macro': recall_score(test_labels, ensemble_preds, average='macro'),
        }

        # Per-class metrics
        report = classification_report(
            test_labels, ensemble_preds,
            target_names=['Neutral', 'Light', 'Moderate', 'Severe'],
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(test_labels, ensemble_preds)

        logger.info("\n" + "="*60)
        logger.info("ENSEMBLE RESULTS")
        logger.info("="*60)
        logger.info(f"F1-Macro: {ensemble_metrics['f1_macro']:.4f}")
        logger.info(f"Accuracy: {ensemble_metrics['accuracy']:.4f}")
        logger.info("\nPer-Class F1:")
        for class_name in ['Neutral', 'Light', 'Moderate', 'Severe']:
            logger.info(f"  {class_name}: {report[class_name]['f1-score']:.4f}")

        return {
            'individual_models': model_results,
            'ensemble': {
                'metrics': ensemble_metrics,
                'predictions': ensemble_preds,
                'per_class_f1': {
                    'neutral': report['Neutral']['f1-score'],
                    'light': report['Light']['f1-score'],
                    'moderate': report['Moderate']['f1-score'],
                    'severe': report['Severe']['f1-score'],
                },
                'confusion_matrix': cm.tolist()
            }
        }

    def save_results(self, results: Dict):
        """Save hasil eksperimen"""
        output_path = os.path.join(self.config.output_dir, 'results.json')

        # Convert numpy arrays to lists
        results_serializable = {}
        for key, value in results.items():
            if key == 'individual_models':
                results_serializable[key] = {}
                for model_name, model_result in value.items():
                    results_serializable[key][model_name] = {
                        'metrics': model_result['metrics'],
                    }
            elif key == 'ensemble':
                results_serializable[key] = {
                    'metrics': value['metrics'],
                    'predictions': value['predictions'].tolist() if hasattr(value['predictions'], 'tolist') else value['predictions'],
                    'per_class_f1': value['per_class_f1'],
                    'confusion_matrix': value['confusion_matrix']
                }
            else:
                results_serializable[key] = value

        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

        return results_serializable


def run_experiments(config: MultiModelConfig = None):
    """Run Experiment 6 & 7"""

    if config is None:
        config = MultiModelConfig()

    logger.info("="*60)
    logger.info("EXPERIMENT 6 & 7: MULTI-MODEL & ENSEMBLE")
    logger.info("="*60)

    # Prepare data
    preparer = ImprovedDatasetPreparer(config)
    combined_df = preparer.load_data()
    train_df, val_df, test_df = preparer.split_data()

    # Run experiments
    trainer = MultiModelTrainer(config)
    results = trainer.run_all_experiments(train_df, val_df, test_df)

    # Save results
    trainer.save_results(results)

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT 6 & 7: SUMMARY")
    print("="*60)
    print("\nIndividual Models:")
    for model_name, model_result in results['individual_models'].items():
        print(f"  {model_name}:")
        print(f"    F1-Macro: {model_result['metrics']['f1_macro']:.2%}")
        print(f"    Accuracy: {model_result['metrics']['accuracy']:.2%}")

    print("\nEnsemble:")
    ensemble = results['ensemble']
    print(f"  F1-Macro: {ensemble['metrics']['f1_macro']:.2%} ⭐")
    print(f"  Accuracy: {ensemble['metrics']['accuracy']:.2%}")

    print("\nPer-Class F1 (Ensemble):")
    for class_name, f1 in ensemble['per_class_f1'].items():
        print(f"  {class_name.capitalize()}: {f1:.2%}")

    print("="*60)

    return results, trainer


if __name__ == "__main__":
    results, trainer = run_experiments()
    print("\n✅ Experiments 6 & 7 complete!")
