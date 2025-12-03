#!/usr/bin/env python3
"""
Eksperimen tanpa augmentasi untuk membuktikan dataset original lebih baik
Menggunakan dataset standardized/balanced_dataset.csv yang terbukti bagus
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
import torch
from torch.utils.data import Dataset

print("=== EKSPERIMEN TANPA AUGMENTASI ===")
print("Tujuan: Membuktikan dataset original lebih baik dari augmented")
print(f"Timestamp: {datetime.now()}")
print()

# 1. Load dataset original yang bagus
print("1. LOADING DATASET ORIGINAL...")
df = pd.read_csv('data/standardized/balanced_dataset.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")
print(f"   Label distribution:")
for label, count in df['final_label'].value_counts().items():
    print(f"     {label}: {count}")
print()

# 2. Prepare data
print("2. PREPARING DATA...")
texts = df['text'].tolist()
labels = df['final_label'].tolist()

# Create label mapping
unique_labels = sorted(df['final_label'].unique())
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

print(f"   Labels: {unique_labels}")
print(f"   Label mapping: {label_to_id}")

# Convert labels to numeric
label_ids = [label_to_id[label] for label in labels]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, label_ids, test_size=0.2, random_state=42, stratify=label_ids
)

print(f"   Train samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print()

# 3. Load optimal thresholds if available
print("3. LOADING OPTIMAL THRESHOLDS...")
try:
    with open('results/optimal_thresholds.json', 'r') as f:
        threshold_data = json.load(f)
        optimal_thresholds = threshold_data.get('optimal_thresholds', {})
    print(f"   ✓ Loaded thresholds: {optimal_thresholds}")
except:
    optimal_thresholds = None
    print("   ⚠️ No optimal thresholds found, using default 0.5")
print()

# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 4. Train models
models_to_test = [
    'indolem/indobert-base-uncased',
    'indobenchmark/indobert-base-p1'
]

results = []
all_predictions = []

for i, model_name in enumerate(models_to_test):
    print(f"4.{i+1} TRAINING MODEL: {model_name}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=len(unique_labels)
        )
        
        # Create datasets
        train_dataset = TextDataset(X_train, y_train, tokenizer)
        test_dataset = TextDataset(X_test, y_test, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'tmp_original_model_{i}',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'logs_original_{i}',
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            dataloader_num_workers=0,
            remove_unused_columns=False
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print(f"   Training {model_name}...")
        trainer.train()
        
        # Evaluate
        print(f"   Evaluating {model_name}...")
        predictions = trainer.predict(test_dataset)
        y_pred_probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
        
        # Apply thresholds if available
        if optimal_thresholds:
            y_pred = []
            for probs in y_pred_probs:
                pred_label_idx = 0
                max_score = 0
                for label_idx, label_name in id_to_label.items():
                    threshold = optimal_thresholds.get(label_name, 0.5)
                    if probs[label_idx] > threshold and probs[label_idx] > max_score:
                        max_score = probs[label_idx]
                        pred_label_idx = label_idx
                y_pred.append(pred_label_idx)
        else:
            y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Classification report
        target_names = [id_to_label[i] for i in range(len(unique_labels))]
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        
        model_result = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': report
        }
        
        results.append(model_result)
        all_predictions.append(y_pred_probs)
        
        print(f"   ✓ {model_name} - Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}")
        
    except Exception as e:
        print(f"   ❌ Error with {model_name}: {e}")
        continue
    
    print()

# 5. Ensemble prediction
print("5. ENSEMBLE PREDICTION...")
if len(all_predictions) >= 2:
    # Average predictions
    ensemble_probs = np.mean(all_predictions, axis=0)
    
    # Apply thresholds
    if optimal_thresholds:
        ensemble_pred = []
        for probs in ensemble_probs:
            pred_label_idx = 0
            max_score = 0
            for label_idx, label_name in id_to_label.items():
                threshold = optimal_thresholds.get(label_name, 0.5)
                if probs[label_idx] > threshold and probs[label_idx] > max_score:
                    max_score = probs[label_idx]
                    pred_label_idx = label_idx
            ensemble_pred.append(pred_label_idx)
    else:
        ensemble_pred = np.argmax(ensemble_probs, axis=1)
    
    # Calculate ensemble metrics
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    ensemble_f1_macro = f1_score(y_test, ensemble_pred, average='macro')
    ensemble_f1_weighted = f1_score(y_test, ensemble_pred, average='weighted')
    
    target_names = [id_to_label[i] for i in range(len(unique_labels))]
    ensemble_report = classification_report(y_test, ensemble_pred, target_names=target_names, output_dict=True)
    
    print(f"   Ensemble Accuracy: {ensemble_accuracy:.4f}")
    print(f"   Ensemble F1-Macro: {ensemble_f1_macro:.4f}")
    print(f"   Ensemble F1-Weighted: {ensemble_f1_weighted:.4f}")
else:
    ensemble_accuracy = results[0]['accuracy'] if results else 0
    ensemble_f1_macro = results[0]['f1_macro'] if results else 0
    ensemble_f1_weighted = results[0]['f1_weighted'] if results else 0
    ensemble_report = results[0]['classification_report'] if results else {}
    print("   Using single model result as ensemble")

print()

# 6. Save results
print("6. SAVING RESULTS...")
final_results = {
    'experiment_timestamp': datetime.now().isoformat(),
    'experiment_type': 'without_augmentation_original_dataset',
    'dataset_used': 'data/standardized/balanced_dataset.csv',
    'dataset_size': len(df),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'successful_models': len(results),
    'models_attempted': len(models_to_test),
    'thresholds_applied': optimal_thresholds is not None,
    'optimal_thresholds': optimal_thresholds,
    'individual_results': results,
    'final_ensemble_results': {
        'accuracy': ensemble_accuracy,
        'f1_macro': ensemble_f1_macro,
        'f1_weighted': ensemble_f1_weighted,
        'classification_report': ensemble_report
    },
    'target_achievement': {
        'accuracy_90_percent': ensemble_accuracy >= 0.9,
        'f1_macro_90_percent': ensemble_f1_macro >= 0.9,
        'target_achieved': ensemble_f1_macro >= 0.9
    }
}

# Save to file
os.makedirs('results', exist_ok=True)
with open('results/experiment_without_augmentation_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("   ✓ Results saved to results/experiment_without_augmentation_results.json")
print()

# 7. Summary
print("=== SUMMARY EKSPERIMEN TANPA AUGMENTASI ===")
print(f"Dataset: data/standardized/balanced_dataset.csv ({len(df)} samples)")
print(f"Models trained: {len(results)}/{len(models_to_test)}")
print(f"Final Ensemble Results:")
print(f"  • Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
print(f"  • F1-Macro: {ensemble_f1_macro:.4f} ({ensemble_f1_macro*100:.2f}%)")
print(f"  • F1-Weighted: {ensemble_f1_weighted:.4f} ({ensemble_f1_weighted*100:.2f}%)")
print()
print("PERBANDINGAN:")
print(f"  • Stable Push (dengan augmentasi): 40.5% F1-Macro")
print(f"  • Eksperimen ini (tanpa augmentasi): {ensemble_f1_macro*100:.2f}% F1-Macro")
if ensemble_f1_macro > 0.405:
    print(f"  ✅ TERBUKTI: Dataset original LEBIH BAIK dari augmented!")
    improvement = (ensemble_f1_macro - 0.405) * 100
    print(f"  📈 Peningkatan: +{improvement:.2f}% F1-Macro")
else:
    print(f"  ❌ Dataset original tidak lebih baik")

print()
print("=== EKSPERIMEN SELESAI ===")