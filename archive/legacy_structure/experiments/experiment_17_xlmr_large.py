"""
EXPERIMENT 17: XLM-RoBERTa Large - Proven Multilingual SOTA
===========================================================
XLM-RoBERTa Large is proven effective for Indonesian & Javanese.
Expected improvement: +1-2% over IndoBERT base.
"""
import os
import sys

os.environ['TRANSFORMERS_NO_PROGRESS_BAR'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'

import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass

import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
import torch.nn.functional as F


@dataclass
class Config:
    model_path: str = "FacebookAI/xlm-roberta-large"  # 550M params, SOTA multilingual
    data_path: str = "data/improved/phase3_phase4_combined.csv"
    output_dir: str = "results/experiment_17_xlmr_large"
    model_output_dir: str = "models/experiment_17_xlmr_large"
    max_length: int = 128
    batch_size: int = 8  # Reduced for large model
    learning_rate: float = 1e-5  # Lower LR for large model
    num_epochs: int = 5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.1
    early_stopping_patience: int = 3
    seed: int = 42


class XLMRTrainer(Trainer):
    def __init__(self, *args, label_smoothing=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        log_probs = F.log_softmax(logits, dim=-1)
        num_classes = logits.shape[-1]
        targets_one_hot = F.one_hot(labels, num_classes=num_classes).float()
        smooth_targets = (1 - self.label_smoothing) * targets_one_hot + \
                        self.label_smoothing / num_classes

        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
    }


def prepare_dataset(df, tokenizer, max_length):
    encodings = tokenizer(
        df['text'].tolist(),
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    return Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': df['label'].tolist()
    })


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.model_output_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load data
    df = pd.read_csv(cfg.data_path)
    print(f"[EXP17] Data: {len(df)} samples", flush=True)

    # Split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=cfg.seed, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=cfg.seed, stratify=temp_df['label'])
    print(f"[EXP17] Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}", flush=True)

    # Load XLM-R Large
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path, num_labels=4)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"[EXP17] Model: XLM-RoBERTa Large ({model_params:,} params)", flush=True)

    # Prepare datasets
    train_dataset = prepare_dataset(train_df, tokenizer, cfg.max_length)
    val_dataset = prepare_dataset(val_df, tokenizer, cfg.max_length)
    test_dataset = prepare_dataset(test_df, tokenizer, cfg.max_length)

    # Training args
    training_args = TrainingArguments(
        output_dir=cfg.model_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=f"{cfg.output_dir}/logs",
        logging_steps=100,
        save_total_limit=3,
        seed=cfg.seed,
        fp16=True,
        gradient_checkpointing=True,  # Save memory
        report_to="none",
        disable_tqdm=True,
        log_level="error",
    )

    trainer = XLMRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        label_smoothing=cfg.label_smoothing,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)]
    )

    # Train
    print("[EXP17] Training started...", flush=True)
    train_result = trainer.train()
    print(f"[EXP17] Training completed in {train_result.metrics['train_runtime']:.0f}s", flush=True)

    # Evaluate
    print("[EXP17] Evaluating...", flush=True)
    test_results = trainer.evaluate(test_dataset)

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    f1_macro = test_results['eval_f1_macro']
    accuracy = test_results['eval_accuracy']

    print("\n" + "="*50, flush=True)
    print("EXPERIMENT 17 RESULTS (XLM-RoBERTa Large)", flush=True)
    print("="*50, flush=True)
    print(f"Accuracy:  {accuracy:.4f}", flush=True)
    print(f"F1-Macro:  {f1_macro:.4f}", flush=True)
    print(f"Baseline:  0.8138 (IndoBERT Base)", flush=True)
    print(f"Target:    0.8500 (85%)", flush=True)
    print(f"Delta:     {(f1_macro - 0.8138):+.4f}", flush=True)
    print("="*50, flush=True)

    # Per-class F1
    class_names = ['Neutral', 'Light', 'Moderate', 'Severe']
    print("\nPer-Class F1:", flush=True)
    for i, name in enumerate(class_names):
        f1 = f1_score(labels, preds, labels=[i], average='macro', zero_division=0)
        print(f"  {name}: {f1:.4f}", flush=True)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:", flush=True)
    for i, row in enumerate(cm):
        print(f"  {i}: {row}", flush=True)

    # Save results
    results = {
        'experiment': 'Experiment 17 - XLM-RoBERTa Large',
        'model': 'FacebookAI/xlm-roberta-large',
        'timestamp': datetime.now().isoformat(),
        'samples': len(df),
        'test_metrics': {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
        },
        'per_class_f1': {
            name: float(f1_score(labels, preds, labels=[i], average='macro', zero_division=0))
            for i, name in enumerate(class_names)
        },
        'confusion_matrix': cm.tolist(),
    }

    with open(f'{cfg.output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {cfg.output_dir}/results.json", flush=True)

    return results


if __name__ == "__main__":
    main()
