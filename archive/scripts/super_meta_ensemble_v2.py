#!/usr/bin/env python3
"""
Super Meta-Ensemble Experiment (v2)
Using: Custom Javanese BERT v2 (Trained on Massive Corpus)
Strategy: Multi-Granularity Input (128, 256, 512 tokens) as per Paper
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
BASE_MODEL_PATH = "models/custom_javanese_bert_v2"
OUTPUT_DIR = "results_super_ensemble"

class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

class SuperEnsembleTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Using device: {self.device}")
        print(f"🤖 Base Model: {BASE_MODEL_PATH}")
        
        # Multi-Granularity Configurations (Paper Strategy)
        self.configs = [
            {'name': 'context_128', 'max_length': 128, 'batch_size': 32},
            {'name': 'context_256', 'max_length': 256, 'batch_size': 16},
            {'name': 'context_512', 'max_length': 512, 'batch_size': 8}
        ]
        
        self.results = {
            'experiment_name': 'super_meta_ensemble_v2',
            'timestamp': datetime.now().isoformat(),
            'model_results': {},
            'ensemble_results': {}
        }

    def load_data(self):
        print("📊 Loading balanced dataset...")
        df = pd.read_csv('data/standardized/balanced_dataset.csv')
        df = df.dropna(subset=['text', 'label'])
        
        # Clean text
        df['text'] = df['text'].str.strip().str.replace(r'\s+', ' ', regex=True)
        
        X = df['text'].values
        y = df['label'].values # Numeric 0-3
        
        # Split
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train_variant(self, config, X_train, X_test, y_train, y_test):
        print(f"\n🚀 Training Variant: {config['name']} (Max Len: {config['max_length']})")
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_PATH, num_labels=4
        ).to(self.device)
        
        # Encode
        train_enc = tokenizer(list(X_train), truncation=True, padding=True, max_length=config['max_length'])
        test_enc = tokenizer(list(X_test), truncation=True, padding=True, max_length=config['max_length'])
        
        train_ds = HateSpeechDataset(train_enc, y_train)
        test_ds = HateSpeechDataset(test_enc, y_test)
        
        # Class Weights
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Custom Loss Trainer
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get('logits')
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
                loss = loss_fct(logits.view(-1, 4), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        def compute_metrics(eval_pred):
            pred, labels = eval_pred
            pred = np.argmax(pred, axis=1)
            return {
                'f1_macro': f1_score(labels, pred, average='macro'),
                'accuracy': accuracy_score(labels, pred)
            }

        training_args = TrainingArguments(
            output_dir=f"tmp_super_{config['name']}",
            num_train_epochs=4, # Fine-tuning usually needs fewer epochs
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            learning_rate=2e-5,
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            fp16=True,
            seed=42
        )
        
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        trainer.train()
        
        # Predict
        raw_preds = trainer.predict(test_ds)
        probs = torch.softmax(torch.tensor(raw_preds.predictions), dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        
        score = f1_score(y_test, preds, average='macro')
        print(f"✅ {config['name']} F1-Macro: {score:.4f}")
        
        return probs, score

    def run(self):
        X_train, X_test, y_train, y_test = self.load_data()
        all_probs = []
        
        for config in self.configs:
            probs, score = self.train_variant(config, X_train, X_test, y_train, y_test)
            all_probs.append(probs)
            self.results['model_results'][config['name']] = score
            
        # Meta Ensemble
        print("\n🧠 Training Meta-Learner...")
        meta_X = np.hstack(all_probs)
        
        # Split for Meta
        mx_train, mx_test, my_train, my_test = train_test_split(
            meta_X, y_test, test_size=0.5, random_state=42 # Split test set for meta-training
        )
        
        meta = RandomForestClassifier(n_estimators=200, random_state=42)
        meta.fit(mx_train, my_train)
        
        final_preds = meta.predict(mx_test)
        final_score = f1_score(my_test, final_preds, average='macro')
        acc = accuracy_score(my_test, final_preds)
        
        print("\n" + "="*50)
        print(f"🏆 SUPER ENSEMBLE RESULT: {final_score:.4f} F1-Macro")
        print(f"   Accuracy: {acc:.4f}")
        print("="*50)
        
        # Save
        os.makedirs("results", exist_ok=True)
        with open("results/super_ensemble_results.json", "w") as f:
            self.results['ensemble_results'] = {'f1': final_score, 'acc': acc}
            json.dump(self.results, f, indent=2)

if __name__ == "__main__":
    SuperEnsembleTrainer().run()
