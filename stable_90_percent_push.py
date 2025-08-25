#!/usr/bin/env python3
"""
Stable Push to 90% F1-Macro Score
Using proven techniques with stable implementation
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
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

def load_best_dataset():
    """Load the best available dataset"""
    print("Loading dataset...")
    
    # Try augmented dataset first
    augmented_path = "data/augmented/augmented_dataset.csv"
    if os.path.exists(augmented_path):
        df = pd.read_csv(augmented_path)
        # Clean data - remove rows with NaN labels
        print(f"Original augmented dataset size: {len(df)}")
        df = df.dropna(subset=['final_label'])
        print(f"After cleaning: {len(df)} samples")
        return df, 'final_label'
    else:
        # Fallback to original balanced dataset
        df = pd.read_csv("data/processed/balanced_evaluation_set.csv")
        print(f"Loaded original dataset: {len(df)} samples")
        return df, 'label'

def apply_optimal_thresholds(predictions, thresholds):
    """Apply optimal thresholds from threshold tuning"""
    if not thresholds:
        return np.argmax(predictions, axis=1)
    
    # Map threshold keys to indices
    threshold_mapping = {
        "Bukan Ujaran Kebencian": 0,
        "Ujaran Kebencian - Ringan": 1,
        "Ujaran Kebencian - Sedang": 2,
        "Ujaran Kebencian - Berat": 3
    }
    
    threshold_values = [0.5, 0.5, 0.5, 0.5]  # Default thresholds
    for label, threshold in thresholds.items():
        if label in threshold_mapping:
            threshold_values[threshold_mapping[label]] = threshold
    
    adjusted_predictions = []
    for pred in predictions:
        # Apply thresholds
        adjusted_pred = np.where(pred > threshold_values, 1, 0)
        # Get the class with highest adjusted score
        if np.sum(adjusted_pred) == 0:
            # If no class meets threshold, use original argmax
            adjusted_predictions.append(np.argmax(pred))
        else:
            # Use the class with highest confidence among those meeting threshold
            class_scores = pred * adjusted_pred
            adjusted_predictions.append(np.argmax(class_scores))
    
    return np.array(adjusted_predictions)

def train_stable_ensemble():
    """Train stable ensemble with proven techniques"""
    print("=== STABLE PUSH TO 90% F1-MACRO ===")
    print(f"Timestamp: {datetime.now()}")
    
    # Load data
    df, label_column = load_best_dataset()
    
    # Prepare data
    texts = df['text'].values
    labels = df[label_column].values
    
    # Create label mapping
    unique_labels = sorted([label for label in df[label_column].unique() if pd.notna(label)])
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    print(f"Labels: {unique_labels}")
    
    # Convert labels to numeric
    numeric_labels = [label_to_id[label] for label in labels]
    
    # Split data with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, numeric_labels, test_size=0.4, random_state=42, stratify=numeric_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Load optimal thresholds
    thresholds = {}
    if os.path.exists("threshold_tuning_results.json"):
        with open("threshold_tuning_results.json", 'r') as f:
            threshold_data = json.load(f)
            thresholds = threshold_data.get("optimal_thresholds", {})
        print(f"Loaded optimal thresholds: {thresholds}")
    
    # Train stable models
    model_configs = [
        {
            "name": "indolem/indobert-base-uncased",
            "epochs": 4,
            "batch_size": 8,
            "learning_rate": 2e-5
        },
        {
            "name": "indobenchmark/indobert-base-p1", 
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 3e-5
        }
    ]
    
    all_predictions = []
    successful_models = 0
    
    for i, config in enumerate(model_configs):
        print(f"\n=== Training Model {i+1}: {config['name']} ===")
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(config['name'])
            model = AutoModelForSequenceClassification.from_pretrained(
                config['name'], 
                num_labels=len(unique_labels),
                ignore_mismatched_sizes=True
            )
            
            # Create datasets
            train_dataset = TextDataset(X_train, y_train, tokenizer)
            val_dataset = TextDataset(X_val, y_val, tokenizer)
            
            # Training arguments - stable configuration
            training_args = TrainingArguments(
                output_dir=f'./tmp_stable_model_{i}',
                num_train_epochs=config['epochs'],
                per_device_train_batch_size=config['batch_size'],
                per_device_eval_batch_size=16,
                warmup_steps=200,
                weight_decay=0.01,
                logging_dir=f'./logs_stable_{i}',
                logging_steps=100,
                eval_strategy="steps",  # Fixed parameter name
                eval_steps=500,
                save_strategy="steps",
                save_steps=1000,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                learning_rate=config['learning_rate'],
                fp16=True,
                dataloader_num_workers=0,
                remove_unused_columns=False,
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )
            
            # Train model
            print(f"Training {config['name']}...")
            trainer.train()
            
            # Get predictions on test set
            test_dataset = TextDataset(X_test, y_test, tokenizer)
            predictions = trainer.predict(test_dataset)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
            
            # Apply optimal thresholds if available
            pred_labels = apply_optimal_thresholds(probs, thresholds)
            
            all_predictions.append(pred_labels)
            successful_models += 1
            
            # Individual model evaluation
            f1_macro = f1_score(y_test, pred_labels, average='macro')
            accuracy = accuracy_score(y_test, pred_labels)
            
            print(f"Model {i+1} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Macro: {f1_macro:.4f}")
            
            # Save model if it's good
            if f1_macro > 0.80:
                model.save_pretrained(f"models/stable_model_{i}")
                tokenizer.save_pretrained(f"models/stable_model_{i}")
                print(f"  Saved model {i+1} (F1: {f1_macro:.4f})")
            
        except Exception as e:
            print(f"Error training {config['name']}: {e}")
            continue
    
    # Ensemble predictions
    print(f"\n=== Creating Ensemble from {successful_models} models ===")
    
    if len(all_predictions) > 1:
        # Stack predictions and use majority voting
        stacked_predictions = np.stack(all_predictions, axis=0)
        ensemble_predictions = []
        
        for i in range(len(y_test)):
            # Get votes from all models
            votes = stacked_predictions[:, i]
            # Use majority vote
            unique, counts = np.unique(votes, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            ensemble_predictions.append(majority_class)
        
        ensemble_predictions = np.array(ensemble_predictions)
    elif len(all_predictions) == 1:
        ensemble_predictions = all_predictions[0]
    else:
        print("No successful models! Using random predictions.")
        ensemble_predictions = np.random.randint(0, len(unique_labels), len(y_test))
    
    # Final evaluation
    final_accuracy = accuracy_score(y_test, ensemble_predictions)
    final_f1_macro = f1_score(y_test, ensemble_predictions, average='macro')
    final_f1_weighted = f1_score(y_test, ensemble_predictions, average='weighted')
    
    print("\n=== FINAL STABLE ENSEMBLE RESULTS ===")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"F1-Macro: {final_f1_macro:.4f}")
    print(f"F1-Weighted: {final_f1_weighted:.4f}")
    
    # Detailed classification report
    target_names = [id_to_label[i] for i in range(len(unique_labels))]
    report = classification_report(y_test, ensemble_predictions, target_names=target_names, output_dict=True)
    
    print("\nDetailed Classification Report:")
    for class_name in target_names:
        metrics = report[class_name]
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-Score: {metrics['f1-score']:.4f}")
    
    # Save results
    results = {
        "experiment_timestamp": datetime.now().isoformat(),
        "experiment_type": "stable_90_percent_push",
        "successful_models": successful_models,
        "models_attempted": len(model_configs),
        "thresholds_applied": bool(thresholds),
        "optimal_thresholds": thresholds,
        "test_samples": len(y_test),
        "final_results": {
            "accuracy": final_accuracy,
            "f1_macro": final_f1_macro,
            "f1_weighted": final_f1_weighted,
            "classification_report": report
        },
        "target_achievement": {
            "accuracy_90_percent": final_accuracy >= 0.90,
            "f1_macro_90_percent": final_f1_macro >= 0.90,
            "target_achieved": final_f1_macro >= 0.90
        }
    }
    
    # Save to results directory
    os.makedirs("results", exist_ok=True)
    results_file = f"results/stable_90_percent_push_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Check if target achieved
    if final_f1_macro >= 0.90:
        print("\n🎉 TARGET ACHIEVED! F1-Macro >= 90% 🎉")
        
        # Save achievement notification
        achievement = {
            "timestamp": datetime.now().isoformat(),
            "achievement": "90_percent_f1_macro_achieved",
            "final_score": final_f1_macro,
            "final_accuracy": final_accuracy,
            "experiment_type": "stable_90_percent_push"
        }
        
        with open("results/90_percent_achievement.json", 'w') as f:
            json.dump(achievement, f, indent=2)
        
        print("Achievement notification saved!")
    else:
        gap = 0.90 - final_f1_macro
        print(f"\nTarget not yet achieved. Gap to 90%: {gap:.4f} ({gap*100:.2f}%)")
        
        # If we're close, suggest next steps
        if final_f1_macro >= 0.85:
            print("\nVery close to target! Consider:")
            print("- Fine-tuning hyperparameters")
            print("- Adding more diverse models to ensemble")
            print("- Advanced threshold optimization")
        elif final_f1_macro >= 0.80:
            print("\nGood progress! Consider:")
            print("- More aggressive data augmentation")
            print("- Cross-validation ensemble")
            print("- Advanced feature engineering")
    
    return final_f1_macro >= 0.90

if __name__ == "__main__":
    success = train_stable_ensemble()
    print(f"\nExperiment completed. Target achieved: {success}")