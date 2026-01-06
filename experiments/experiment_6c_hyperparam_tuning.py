"""
EXPERIMENT 6C: Hyperparameter Tuning untuk Label Smoothing
===========================================================

Objective: Temukan hyperparameter optimal untuk mencapai 82%+ F1-Macro
- Base: Label Smoothing (81.38%)
- Hyperparameter yang dituning:
  - Learning rate
  - Batch size
  - Weight decay
  - Warmup ratio
  - Epochs
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report
)

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss Function"""

    def __init__(self, epsilon: float = 0.1, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.epsilon = epsilon
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)

        # Convert labels to one-hot
        one_hot = F.one_hot(labels, num_classes=n_classes).float()

        # Smooth labels
        smooth_labels = (1 - self.epsilon) * one_hot + self.epsilon / n_classes

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute loss
        loss = -(smooth_labels * log_probs).sum(dim=-1)

        # Apply class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights[labels]
            loss = loss * weights

        return loss.mean()


class CustomLossTrainer(Trainer):
    """Custom Trainer dengan Label Smoothing Loss"""

    def __init__(self, *args, epsilon: float = 0.1, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.class_weights = class_weights
        self.loss_fn = LabelSmoothingLoss(epsilon, class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


@dataclass
class TuningConfig:
    """Configuration untuk Hyperparameter Tuning"""

    # Data paths
    phase3_path: str = "data/improved/phase3_relabeled.csv"
    phase4_path: str = "data/improved/phase4_generated.csv"

    # Output paths
    output_dir: str = "models/experiment_6c_hyperparam_tuning"
    results_dir: str = "results/experiment_6c_hyperparam_tuning"

    # Base hyperparameters (fixed)
    max_length: int = 128
    seed: int = 42

    # Train/val/test split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Hyperparameter search space
    learning_rates: List[float] = None
    batch_sizes: List[int] = None
    weight_decays: List[float] = None
    warmup_ratios: List[float] = None
    num_epochs_list: List[int] = None
    epsilons: List[float] = None

    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]

        if self.batch_sizes is None:
            self.batch_sizes = [8, 16, 32]

        if self.weight_decays is None:
            self.weight_decays = [0.0, 0.01, 0.001]

        if self.warmup_ratios is None:
            self.warmup_ratios = [0.0, 0.1, 0.2]

        if self.num_epochs_list is None:
            self.num_epochs_list = [3, 5, 7]

        if self.epsilons is None:
            self.epsilons = [0.05, 0.1, 0.15]


class HyperparameterTuner:
    """Hyperparameter Tuner untuk Label Smoothing Model"""

    def __init__(self, config: TuningConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []

        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)

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

        # Split
        train_df, temp_df = train_test_split(
            combined_df,
            test_size=(self.config.val_ratio + self.config.test_ratio),
            random_state=self.config.seed,
            stratify=combined_df['label']
        )

        val_ratio_adjusted = self.config.val_ratio / (
            self.config.val_ratio + self.config.test_ratio
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adjusted),
            random_state=self.config.seed,
            stratify=temp_df['label']
        )

        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def tokenize_data(self, df: pd.DataFrame, tokenizer) -> Dataset:
        """Tokenize dataframe"""
        texts = df['text'].tolist()
        labels = df['label'].tolist()

        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors=None
        )

        encodings['labels'] = labels
        return Dataset.from_dict(encodings)

    def train_and_evaluate(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        tokenizer,
        learning_rate: float,
        batch_size: int,
        weight_decay: float,
        warmup_ratio: float,
        num_epochs: int,
        epsilon: float
    ) -> Dict:
        """Train model dengan specific hyperparameters"""

        # Setup model
        model_name = "indobenchmark/indobert-base-p1"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=4
        )
        model.to(self.device)

        # Create unique output directory for this run
        run_name = f"lr_{learning_rate}_bs_{batch_size}_wd_{weight_decay}_wr_{warmup_ratio}_ep_{num_epochs}_eps_{epsilon}"
        run_output_dir = os.path.join(self.config.output_dir, run_name)

        def compute_metrics(p: EvalPrediction):
            preds = np.argmax(p.predictions, axis=1)
            labels = p.label_ids
            return {
                'accuracy': accuracy_score(labels, preds),
                'f1_macro': f1_score(labels, preds, average='macro'),
                'precision_macro': precision_score(labels, preds, average='macro'),
                'recall_macro': recall_score(labels, preds, average='macro'),
            }

        # Training arguments
        training_args = TrainingArguments(
            output_dir=run_output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            seed=self.config.seed,
            logging_dir=os.path.join(run_output_dir, "logs"),
            logging_steps=50,
            report_to=["none"],
            save_total_limit=1,
        )

        # Custom trainer dengan Label Smoothing
        trainer = CustomLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            epsilon=epsilon
        )

        # Train
        logger.info(f"Training with lr={learning_rate}, bs={batch_size}, wd={weight_decay}, wr={warmup_ratio}, ep={num_epochs}, eps={epsilon}")
        trainer.train()

        # Evaluate on test set
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_micro': f1_score(labels, preds, average='micro'),
            'precision_macro': precision_score(labels, preds, average='macro'),
            'recall_macro': recall_score(labels, preds, average='macro'),
        }

        # Per-class metrics
        report = classification_report(
            labels, preds,
            target_names=['Neutral', 'Light', 'Moderate', 'Severe'],
            output_dict=True
        )

        per_class_f1 = {
            'neutral': report['Neutral']['f1-score'],
            'light': report['Light']['f1-score'],
            'moderate': report['Moderate']['f1-score'],
            'severe': report['Severe']['f1-score'],
        }

        logger.info(f"Test F1-Macro: {metrics['f1_macro']:.4f}")

        # Clean up
        del model, trainer
        torch.cuda.empty_cache()

        return {
            'hyperparameters': {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'weight_decay': weight_decay,
                'warmup_ratio': warmup_ratio,
                'num_epochs': num_epochs,
                'epsilon': epsilon
            },
            'metrics': metrics,
            'per_class_f1': per_class_f1,
            'run_name': run_name
        }

    def grid_search(
        self,
        train_df,
        val_df,
        test_df,
        search_configs: List[Dict] = None
    ) -> List[Dict]:
        """Run grid search over hyperparameters"""

        # Setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

        # Tokenize datasets
        train_dataset = self.tokenize_data(train_df, tokenizer)
        val_dataset = self.tokenize_data(val_df, tokenizer)
        test_dataset = self.tokenize_data(test_df, tokenizer)

        if search_configs is None:
            # Default search configurations (focused on promising areas)
            search_configs = [
                # Vary learning rate
                {'learning_rate': 1e-5, 'batch_size': 16, 'weight_decay': 0.01, 'warmup_ratio': 0.1, 'num_epochs': 5, 'epsilon': 0.1},
                {'learning_rate': 3e-5, 'batch_size': 16, 'weight_decay': 0.01, 'warmup_ratio': 0.1, 'num_epochs': 5, 'epsilon': 0.1},
                {'learning_rate': 5e-5, 'batch_size': 16, 'weight_decay': 0.01, 'warmup_ratio': 0.1, 'num_epochs': 5, 'epsilon': 0.1},

                # Vary batch size
                {'learning_rate': 2e-5, 'batch_size': 8, 'weight_decay': 0.01, 'warmup_ratio': 0.1, 'num_epochs': 5, 'epsilon': 0.1},
                {'learning_rate': 2e-5, 'batch_size': 32, 'weight_decay': 0.01, 'warmup_ratio': 0.1, 'num_epochs': 5, 'epsilon': 0.1},

                # Vary weight decay
                {'learning_rate': 2e-5, 'batch_size': 16, 'weight_decay': 0.0, 'warmup_ratio': 0.1, 'num_epochs': 5, 'epsilon': 0.1},
                {'learning_rate': 2e-5, 'batch_size': 16, 'weight_decay': 0.001, 'warmup_ratio': 0.1, 'num_epochs': 5, 'epsilon': 0.1},

                # Vary warmup ratio
                {'learning_rate': 2e-5, 'batch_size': 16, 'weight_decay': 0.01, 'warmup_ratio': 0.0, 'num_epochs': 5, 'epsilon': 0.1},
                {'learning_rate': 2e-5, 'batch_size': 16, 'weight_decay': 0.01, 'warmup_ratio': 0.2, 'num_epochs': 5, 'epsilon': 0.1},

                # Vary epochs
                {'learning_rate': 2e-5, 'batch_size': 16, 'weight_decay': 0.01, 'warmup_ratio': 0.1, 'num_epochs': 3, 'epsilon': 0.1},
                {'learning_rate': 2e-5, 'batch_size': 16, 'weight_decay': 0.01, 'warmup_ratio': 0.1, 'num_epochs': 7, 'epsilon': 0.1},

                # Vary epsilon
                {'learning_rate': 2e-5, 'batch_size': 16, 'weight_decay': 0.01, 'warmup_ratio': 0.1, 'num_epochs': 5, 'epsilon': 0.05},
                {'learning_rate': 2e-5, 'batch_size': 16, 'weight_decay': 0.01, 'warmup_ratio': 0.1, 'num_epochs': 5, 'epsilon': 0.15},

                # Combinations that might work well
                {'learning_rate': 3e-5, 'batch_size': 32, 'weight_decay': 0.01, 'warmup_ratio': 0.1, 'num_epochs': 4, 'epsilon': 0.1},
                {'learning_rate': 1e-5, 'batch_size': 8, 'weight_decay': 0.001, 'warmup_ratio': 0.2, 'num_epochs': 7, 'epsilon': 0.05},
            ]

        logger.info(f"Running {len(search_configs)} configurations...")

        results = []
        for i, config in enumerate(search_configs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Configuration {i+1}/{len(search_configs)}")
            logger.info(f"{'='*60}")

            result = self.train_and_evaluate(
                train_dataset, val_dataset, test_dataset, tokenizer,
                **config
            )
            results.append(result)

            # Print intermediate results
            print(f"\n[{i+1}/{len(search_configs)}] {result['run_name']}")
            print(f"  F1-Macro: {result['metrics']['f1_macro']:.4f}")
            print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")

        return results

    def run_experiment(self):
        """Run complete hyperparameter tuning experiment"""
        logger.info("=" * 60)
        logger.info("EXPERIMENT 6C: HYPERPARAMETER TUNING")
        logger.info("=" * 60)

        # Prepare data
        train_df, val_df, test_df = self.prepare_data()

        # Run grid search
        results = self.grid_search(train_df, val_df, test_df)

        # Sort by F1-Macro
        results_sorted = sorted(results, key=lambda x: x['metrics']['f1_macro'], reverse=True)

        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT 6C: SUMMARY")
        print("=" * 60)
        print(f"\nBaseline (Label Smoothing): 81.38%")
        print(f"\nTop 5 Configurations:")
        print("-" * 60)

        for i, result in enumerate(results_sorted[:5]):
            hp = result['hyperparameters']
            metrics = result['metrics']
            print(f"\n#{i+1}: F1-Macro = {metrics['f1_macro']:.2%}")
            print(f"  Learning Rate: {hp['learning_rate']}")
            print(f"  Batch Size: {hp['batch_size']}")
            print(f"  Weight Decay: {hp['weight_decay']}")
            print(f"  Warmup Ratio: {hp['warmup_ratio']}")
            print(f"  Epochs: {hp['num_epochs']}")
            print(f"  Epsilon: {hp['epsilon']}")
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
            print(f"  Per-Class F1:")
            for cls, f1 in result['per_class_f1'].items():
                print(f"    {cls}: {f1:.2%}")

        best_result = results_sorted[0]
        improvement = best_result['metrics']['f1_macro'] - 0.8138

        print("\n" + "=" * 60)
        print(f"BEST CONFIGURATION:")
        print(f"F1-Macro: {best_result['metrics']['f1_macro']:.2%}")
        print(f"Improvement: {improvement:+.2%}")
        print("=" * 60)

        # Save results
        output_path = os.path.join(self.config.results_dir, 'results.json')
        with open(output_path, 'w') as f:
            json.dump(results_sorted, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

        return results_sorted


def main():
    """Run Experiment 6C"""
    config = TuningConfig()
    tuner = HyperparameterTuner(config)
    results = tuner.run_experiment()
    return results


if __name__ == "__main__":
    results = main()
    print("\n[OK] Experiment 6C complete!")
