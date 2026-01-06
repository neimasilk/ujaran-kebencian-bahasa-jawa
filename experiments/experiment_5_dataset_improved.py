"""
EXPERIMENT 5: Training dengan Dataset Improved 10K+
===================================================

Dataset: 10,019 records hasil improvement pipeline (Phase 3 + Phase 4)
- Phase 3: 4,779 records (re-labeled with quality verification)
- Phase 4: 5,240 records (AI-generated with Indonesian context)

Class Balance: 1.38:1 (near-optimal)
Label Confidence: 86.6% average

Hypothesis: Dataset yang lebih representatif akan meningkatkan performa model
Target: F1-Macro 65-70% (improvement dari 62.55%)
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

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
class ExperimentConfig:
    """Configuration untuk Experiment 5"""

    # Model configurations
    model_name: str = "indobenchmark/indobert-base-p1"
    custom_bert_path: Optional[str] = None  # Path to custom BERT if available

    # Data paths
    phase3_path: str = "data/improved/phase3_relabeled.csv"
    phase4_path: str = "data/improved/phase4_generated.csv"

    # Output paths
    output_dir: str = "results/experiment_5_dataset_improved"
    model_output_dir: str = "models/experiment_5_improved"
    log_dir: str = "logs/experiment_5"

    # Training hyperparameters
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1

    # Class weights for imbalanced data
    use_class_weights: bool = True

    # Random seed
    seed: int = 42

    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


class ImprovedDatasetPreparer:
    """Persiapan dataset improved untuk training"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.phase3_df = None
        self.phase4_df = None
        self.combined_df = None

    def load_data(self):
        """Load dataset improved dari Phase 3 dan Phase 4"""
        logger.info("Loading improved dataset...")

        # Load Phase 3 (re-labeled)
        self.phase3_df = pd.read_csv(self.config.phase3_path)
        logger.info(f"Phase 3 loaded: {len(self.phase3_df)} records")

        # Load Phase 4 (generated)
        self.phase4_df = pd.read_csv(self.config.phase4_path)
        logger.info(f"Phase 4 loaded: {len(self.phase4_df)} records")

        # Prepare Phase 3: use 'text' and 'new_label'
        phase3_train = self.phase3_df[['text', 'new_label']].copy()
        phase3_train = phase3_train.rename(columns={'new_label': 'label'})
        phase3_train['source'] = 'phase3_relabeled'

        # Prepare Phase 4: use 'text' and 'label'
        phase4_train = self.phase4_df[['text', 'label']].copy()
        phase4_train['source'] = 'phase4_generated'

        # Combine
        self.combined_df = pd.concat([phase3_train, phase4_train], ignore_index=True)

        # Shuffle
        self.combined_df = self.combined_df.sample(
            frac=1, random_state=self.config.seed
        ).reset_index(drop=True)

        logger.info(f"Combined dataset: {len(self.combined_df)} records")
        logger.info(f"\nLabel distribution:\n{self.combined_df['label'].value_counts()}")

        # Calculate class balance ratio
        label_counts = self.combined_df['label'].value_counts()
        max_count = label_counts.max()
        min_count = label_counts.min()
        balance_ratio = max_count / min_count
        logger.info(f"Class balance ratio: {balance_ratio:.2f}:1")

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


class JavaneseHateSpeechTrainer:
    """Trainer untuk eksperimen dataset improved"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.class_weights = None

        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.model_output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    def setup_model(self):
        """Setup model dan tokenizer"""
        logger.info(f"Setting up model: {self.config.model_name}")

        if self.config.custom_bert_path and os.path.exists(self.config.custom_bert_path):
            logger.info(f"Using custom BERT from: {self.config.custom_bert_path}")
            model_path = self.config.custom_bert_path
        else:
            model_path = self.config.model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=4  # 4 classes: 0=Neutral, 1=Light, 2=Moderate, 3=Severe
        )

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        logger.info(f"Model loaded on: {device}")

    def compute_class_weights(self, labels: List[int]):
        """Hitung class weights untuk imbalanced data"""
        if not self.config.use_class_weights:
            return None

        label_counts = np.bincount(labels)
        total_samples = len(labels)
        num_classes = len(label_counts)

        # Inverse frequency weighting
        class_weights = total_samples / (num_classes * label_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        logger.info(f"Class weights: {class_weights.tolist()}")

        return class_weights

    def tokenize_data(self, df: pd.DataFrame):
        """Tokenize dataframe"""
        texts = df['text'].tolist()
        labels = df['label'].tolist()

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors=None
        )

        # Add labels
        encodings['labels'] = labels

        return Dataset.from_dict(encodings)

    def setup_trainer(self, train_dataset, val_dataset):
        """Setup trainer dengan custom loss function jika class weights digunakan"""

        def compute_metrics(p: EvalPrediction):
            """Compute metrics untuk evaluation"""
            preds = np.argmax(p.predictions, axis=1)
            labels = p.label_ids

            return {
                'accuracy': accuracy_score(labels, preds),
                'f1_macro': f1_score(labels, preds, average='macro'),
                'f1_micro': f1_score(labels, preds, average='micro'),
                'precision_macro': precision_score(labels, preds, average='macro'),
                'recall_macro': recall_score(labels, preds, average='macro'),
            }

        training_args = TrainingArguments(
            output_dir=self.config.model_output_dir,
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
            logging_dir=self.config.log_dir,
            logging_steps=50,
            report_to=["none"],
            save_total_limit=2,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=compute_metrics,
        )

        logger.info("Trainer setup complete")

    def train(self):
        """Train model"""
        logger.info("Starting training...")
        train_result = self.trainer.train()

        logger.info("Training complete")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")

        return train_result

    def evaluate(self, test_dataset):
        """Evaluate model pada test set"""
        logger.info("Evaluating on test set...")

        predictions = self.trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_micro': f1_score(labels, preds, average='micro'),
            'f1_weighted': f1_score(labels, preds, average='weighted'),
            'precision_macro': precision_score(labels, preds, average='macro'),
            'recall_macro': recall_score(labels, preds, average='macro'),
        }

        # Per-class metrics
        report = classification_report(
            labels, preds,
            target_names=['Neutral', 'Light', 'Moderate', 'Severe'],
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(labels, preds)

        logger.info("\n=== Test Results ===")
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")

        logger.info("\n=== Per-Class F1 ===")
        for class_name in ['Neutral', 'Light', 'Moderate', 'Severe']:
            logger.info(f"{class_name}: {report[class_name]['f1-score']:.4f}")

        return metrics, report, cm

    def save_results(self, metrics, report, cm, train_df):
        """Save hasil eksperimen"""
        results = {
            'experiment_name': 'Experiment 5: Dataset Improved 10K+',
            'model': self.config.model_name,
            'dataset_size': len(self.combined_df) if hasattr(self, 'combined_df') else 'N/A',
            'train_size': len(train_df),
            'config': {
                'max_length': self.config.max_length,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'num_epochs': self.config.num_epochs,
                'seed': self.config.seed,
            },
            'test_metrics': metrics,
            'per_class_f1': {
                'neutral': report['Neutral']['f1-score'],
                'light': report['Light']['f1-score'],
                'moderate': report['Moderate']['f1-score'],
                'severe': report['Severe']['f1-score'],
            },
            'confusion_matrix': cm.tolist(),
        }

        # Save results
        output_path = os.path.join(self.config.output_dir, 'results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT 5: DATASET IMPROVED 10K+ - RESULTS")
        print("="*60)
        print(f"Model: {self.config.model_name}")
        print(f"Dataset Size: {results.get('dataset_size', 'N/A')}")
        print(f"\nTest Metrics:")
        print(f"  Accuracy:    {metrics['accuracy']:.2%}")
        print(f"  F1-Macro:    {metrics['f1_macro']:.2%} ⭐")
        print(f"  F1-Weighted: {metrics['f1_weighted']:.2%}")
        print(f"\nPer-Class F1:")
        print(f"  Neutral:  {report['Neutral']['f1-score']:.2%}")
        print(f"  Light:    {report['Light']['f1-score']:.2%}")
        print(f"  Moderate: {report['Moderate']['f1-score']:.2%}")
        print(f"  Severe:   {report['Severe']['f1-score']:.2%}")
        print("="*60)

        return results


def run_experiment(config: ExperimentConfig = None):
    """Run Experiment 5 dengan dataset improved"""

    if config is None:
        config = ExperimentConfig()

    logger.info("="*60)
    logger.info("EXPERIMENT 5: TRAINING DENGAN DATASET IMPROVED 10K+")
    logger.info("="*60)

    # Step 1: Prepare dataset
    preparer = ImprovedDatasetPreparer(config)
    combined_df = preparer.load_data()
    train_df, val_df, test_df = preparer.split_data()

    # Step 2: Setup trainer
    trainer_instance = JavaneseHateSpeechTrainer(config)
    trainer_instance.combined_df = combined_df
    trainer_instance.setup_model()

    # Step 3: Tokenize data
    logger.info("Tokenizing data...")
    train_dataset = trainer_instance.tokenize_data(train_df)
    val_dataset = trainer_instance.tokenize_data(val_df)
    test_dataset = trainer_instance.tokenize_data(test_df)

    # Step 4: Setup trainer
    trainer_instance.setup_trainer(train_dataset, val_dataset)

    # Step 5: Train
    train_result = trainer_instance.train()

    # Step 6: Evaluate
    metrics, report, cm = trainer_instance.evaluate(test_dataset)

    # Step 7: Save results
    results = trainer_instance.save_results(metrics, report, cm, train_df)

    logger.info("Experiment 5 complete!")
    return results, trainer_instance


if __name__ == "__main__":
    # Run experiment
    results, trainer = run_experiment()

    print("\n✅ Experiment 5 selesai!")
    print(f"Hasil lengkap tersimpan di: {trainer.config.output_dir}")
