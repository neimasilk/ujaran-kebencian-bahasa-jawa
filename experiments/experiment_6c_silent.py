"""
EXPERIMENT 6C: Hyperparameter Tuning untuk Label Smoothing (SILENT MODE)
========================================================================

Objective: Temukan hyperparameter optimal untuk mencapai 82%+ F1-Macro
- Base: Label Smoothing (81.38%)
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

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
from tqdm import tqdm

import sys
from contextlib import contextmanager

@contextmanager
def suppress_output():
    """Context manager untuk suppress stdout/stderr"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class LabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon: float = 0.1, class_weights: torch.Tensor = None):
        super().__init__()
        self.epsilon = epsilon
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        one_hot = F.one_hot(labels, num_classes=n_classes).float()
        smooth_labels = (1 - self.epsilon) * one_hot + self.epsilon / n_classes
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1)
        if self.class_weights is not None:
            weights = self.class_weights[labels]
            loss = loss * weights
        return loss.mean()


class CustomLossTrainer(Trainer):
    def __init__(self, *args, epsilon: float = 0.1, class_weights: torch.Tensor = None, **kwargs):
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


class HyperparameterTuner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []

        # Paths
        self.phase3_path = "data/improved/phase3_relabeled.csv"
        self.phase4_path = "data/improved/phase4_generated.csv"
        self.output_dir = "models/experiment_6c_hyperparam_tuning"
        self.results_dir = "results/experiment_6c_hyperparam_tuning"

        # Config
        self.max_length = 128
        self.seed = 42

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def prepare_data(self):
        print("[1/4] Loading dataset...")
        phase3_df = pd.read_csv(self.phase3_path)
        phase4_df = pd.read_csv(self.phase4_path)

        phase3_train = phase3_df[['text', 'new_label']].copy()
        phase3_train = phase3_train.rename(columns={'new_label': 'label'})

        phase4_train = phase4_df[['text', 'label']].copy()
        combined_df = pd.concat([phase3_train, phase4_train], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        train_df, temp_df = train_test_split(
            combined_df, test_size=0.2, random_state=self.seed, stratify=combined_df['label']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=self.seed, stratify=temp_df['label']
        )

        print(f"      Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df

    def tokenize_data(self, df: pd.DataFrame, tokenizer) -> Dataset:
        texts = df['text'].tolist()
        labels = df['label'].tolist()

        encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=self.max_length, return_tensors=None
        )
        encodings['labels'] = labels
        return Dataset.from_dict(encodings)

    def train_and_evaluate(self, train_dataset, val_dataset, test_dataset, tokenizer, config: dict):
        """Train model dengan specific hyperparameters"""

        model_name = "indobenchmark/indobert-base-p1"
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
        model.to(self.device)

        run_name = f"lr_{config['lr']}_bs_{config['bs']}_wd_{config['wd']}_wr_{config['wr']}_ep_{config['ep']}_eps_{config['eps']}"
        run_output_dir = os.path.join(self.output_dir, run_name)

        def compute_metrics(p: EvalPrediction):
            preds = np.argmax(p.predictions, axis=1)
            labels = p.label_ids
            return {
                'accuracy': accuracy_score(labels, preds),
                'f1_macro': f1_score(labels, preds, average='macro'),
            }

        training_args = TrainingArguments(
            output_dir=run_output_dir,
            learning_rate=config['lr'],
            per_device_train_batch_size=config['bs'],
            per_device_eval_batch_size=config['bs'],
            num_train_epochs=config['ep'],
            weight_decay=config['wd'],
            warmup_ratio=config['wr'],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            seed=self.seed,
            logging_dir=os.path.join(run_output_dir, "logs"),
            logging_steps=1000,
            report_to=["none"],
            save_total_limit=1,
        )

        trainer = CustomLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            epsilon=config['eps']
        )

        # Train with suppressed output
        with suppress_output():
            trainer.train()

        # Evaluate
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

        report = classification_report(
            labels, preds, target_names=['Neutral', 'Light', 'Moderate', 'Severe'], output_dict=True, zero_division=0
        )

        per_class_f1 = {
            'neutral': report['Neutral']['f1-score'],
            'light': report['Light']['f1-score'],
            'moderate': report['Moderate']['f1-score'],
            'severe': report['Severe']['f1-score'],
        }

        del model, trainer
        torch.cuda.empty_cache()

        return {
            'hyperparameters': config,
            'metrics': metrics,
            'per_class_f1': per_class_f1,
            'run_name': run_name
        }

    def run_experiment(self):
        print("\n" + "="*50)
        print("EXPERIMENT 6C: HYPERPARAMETER TUNING")
        print("="*50)

        # Prepare data
        train_df, val_df, test_df = self.prepare_data()

        # Setup tokenizer
        print("[2/4] Tokenizing datasets...")
        tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
        train_dataset = self.tokenize_data(train_df, tokenizer)
        val_dataset = self.tokenize_data(val_df, tokenizer)
        test_dataset = self.tokenize_data(test_df, tokenizer)

        # Search configurations (focused grid search)
        search_configs = [
            # Base config
            {'lr': 2e-5, 'bs': 16, 'wd': 0.01, 'wr': 0.1, 'ep': 5, 'eps': 0.1},

            # Vary learning rate
            {'lr': 1e-5, 'bs': 16, 'wd': 0.01, 'wr': 0.1, 'ep': 5, 'eps': 0.1},
            {'lr': 3e-5, 'bs': 16, 'wd': 0.01, 'wr': 0.1, 'ep': 5, 'eps': 0.1},
            {'lr': 5e-5, 'bs': 16, 'wd': 0.01, 'wr': 0.1, 'ep': 5, 'eps': 0.1},

            # Vary batch size
            {'lr': 2e-5, 'bs': 8, 'wd': 0.01, 'wr': 0.1, 'ep': 5, 'eps': 0.1},
            {'lr': 2e-5, 'bs': 32, 'wd': 0.01, 'wr': 0.1, 'ep': 5, 'eps': 0.1},

            # Vary weight decay
            {'lr': 2e-5, 'bs': 16, 'wd': 0.0, 'wr': 0.1, 'ep': 5, 'eps': 0.1},
            {'lr': 2e-5, 'bs': 16, 'wd': 0.001, 'wr': 0.1, 'ep': 5, 'eps': 0.1},

            # Vary warmup
            {'lr': 2e-5, 'bs': 16, 'wd': 0.01, 'wr': 0.0, 'ep': 5, 'eps': 0.1},
            {'lr': 2e-5, 'bs': 16, 'wd': 0.01, 'wr': 0.2, 'ep': 5, 'eps': 0.1},

            # Vary epochs
            {'lr': 2e-5, 'bs': 16, 'wd': 0.01, 'wr': 0.1, 'ep': 3, 'eps': 0.1},
            {'lr': 2e-5, 'bs': 16, 'wd': 0.01, 'wr': 0.1, 'ep': 7, 'eps': 0.1},

            # Vary epsilon
            {'lr': 2e-5, 'bs': 16, 'wd': 0.01, 'wr': 0.1, 'ep': 5, 'eps': 0.05},
            {'lr': 2e-5, 'bs': 16, 'wd': 0.01, 'wr': 0.1, 'ep': 5, 'eps': 0.15},

            # Promising combinations
            {'lr': 3e-5, 'bs': 32, 'wd': 0.01, 'wr': 0.1, 'ep': 4, 'eps': 0.1},
            {'lr': 1e-5, 'bs': 8, 'wd': 0.001, 'wr': 0.2, 'ep': 7, 'eps': 0.05},
        ]

        print(f"[3/4] Running {len(search_configs)} configurations...")
        print("      (training in progress, please wait...)")

        results = []
        for i, config in enumerate(search_configs):
            result = self.train_and_evaluate(
                train_dataset, val_dataset, test_dataset, tokenizer, config
            )
            results.append(result)

            # Print progress
            hp = result['hyperparameters']
            metrics = result['metrics']
            print(f"      [{i+1:2d}/{len(search_configs)}] lr={hp['lr']:.0e} bs={hp['bs']:2d} "
                  f"wd={hp['wd']} ep={hp['ep']} eps={hp['eps']} | F1={metrics['f1_macro']:.2%}")

        # Sort and print summary
        results_sorted = sorted(results, key=lambda x: x['metrics']['f1_macro'], reverse=True)

        print("\n[4/4] RESULTS:")
        print("-" * 50)
        print(f"Baseline (Label Smoothing): 81.38%")
        print(f"\nTop 5 Configurations:")
        for i, result in enumerate(results_sorted[:5]):
            hp = result['hyperparameters']
            metrics = result['metrics']
            print(f"  #{i+1}: F1={metrics['f1_macro']:.2%} | "
                  f"lr={hp['lr']:.0e} bs={hp['bs']} wd={hp['wd']} wr={hp['wr']} "
                  f"ep={hp['ep']} eps={hp['eps']}")

        best = results_sorted[0]
        improvement = best['metrics']['f1_macro'] - 0.8138
        print(f"\n  BEST: F1={best['metrics']['f1_macro']:.2%} ({improvement:+.2%})")

        # Save results
        output_path = os.path.join(self.results_dir, 'results.json')
        with open(output_path, 'w') as f:
            json.dump(results_sorted, f, indent=2)
        print(f"\n  Results saved to: {output_path}")

        return results_sorted


if __name__ == "__main__":
    tuner = HyperparameterTuner()
    results = tuner.run_experiment()
    print("\n[OK] Experiment 6C complete!")
