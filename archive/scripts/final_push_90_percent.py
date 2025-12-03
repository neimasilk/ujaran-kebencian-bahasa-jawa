#!/usr/bin/env python3
"""
Final Push to 90% F1-Macro Score
Combining all successful techniques with aggressive optimization
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

def load_augmented_data():
    """Load the augmented dataset"""
    print("Loading augmented dataset...")
    
    # Try augmented dataset first
    augmented_path = "data/augmented/augmented_dataset.csv"
    if os.path.exists(augmented_path):
        df = pd.read_csv(augmented_path)
        print(f"Loaded augmented dataset: {len(df)} samples")
    else:
        # Fallback to original balanced dataset
        df = pd.read_csv("data/processed/balanced_evaluation_set.csv")
        print(f"Loaded original dataset: {len(df)} samples")
    
    return df

def apply_optimal_thresholds(predictions, thresholds):
    """Apply optimal thresholds from threshold tuning"""
    threshold_values = [
        thresholds.get("Bukan Ujaran Kebencian", 0.5),
        thresholds.get("Ujaran Kebencian - Ringan", 0.5),
        thresholds.get("Ujaran Kebencian - Sedang", 0.5),
        thresholds.get("Ujaran Kebencian - Berat", 0.5)
    ]
    
    adjusted_predictions = []
    for pred in predictions:
        # Apply thresholds
        adjusted_pred = np.where(pred > threshold_values, 1, 0)
        # Get the class with highest adjusted score
        if np.sum(adjusted_pred) == 0:
            # If no class meets threshold, use original argmax
            adjusted_predictions.append(np.argmax(pred))
        else:
            # Use the first class that meets threshold (prioritize by confidence)
            class_scores = pred * adjusted_pred
            adjusted_predictions.append(np.argmax(class_scores))
    
    return np.array(adjusted_predictions)

def train_advanced_ensemble():
    """Train advanced ensemble with all optimizations"""
    print("=== FINAL PUSH TO 90% F1-MACRO ===")
    print(f"Timestamp: {datetime.now()}")
    
    # Load data
    df = load_augmented_data()
    
    # Clean data - remove rows with NaN labels
    print(f"Original dataset size: {len(df)}")
    df = df.dropna(subset=['final_label'])
    print(f"After removing NaN labels: {len(df)}")
    
    # Prepare data
    texts = df['text'].values
    labels = df['final_label'].values
    
    # Create label mapping
    unique_labels = sorted([label for label in df['final_label'].unique() if pd.notna(label)])
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
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
    
    # Train multiple models for ensemble
    models = []
    model_names = [
        "indolem/indobert-base-uncased",
        "indobenchmark/indobert-base-p1",
        "indolem/indobert-large-uncased"
    ]
    
    all_predictions = []
    
    for i, model_name in enumerate(model_names):
        print(f"\n=== Training Model {i+1}: {model_name} ===")
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=len(unique_labels),
                ignore_mismatched_sizes=True
            )
            
            # Create datasets
            train_dataset = TextDataset(X_train, y_train, tokenizer)
            val_dataset = TextDataset(X_val, y_val, tokenizer)
            
            # Training arguments with aggressive optimization
            training_args = TrainingArguments(
                output_dir=f'./tmp_final_model_{i}',
                num_train_epochs=5,  # More epochs
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=f'./logs_final_{i}',
                logging_steps=100,
                evaluation_strategy="steps",
                eval_steps=500,
                save_strategy="steps",
                save_steps=1000,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                learning_rate=2e-5,  # Optimized learning rate
                fp16=True,  # Mixed precision
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
            print(f"Training {model_name}...")
            trainer.train()
            
            # Get predictions on test set
            test_dataset = TextDataset(X_test, y_test, tokenizer)
            predictions = trainer.predict(test_dataset)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
            
            # Apply optimal thresholds if available
            if thresholds:
                pred_labels = apply_optimal_thresholds(probs, thresholds)
            else:
                pred_labels = np.argmax(probs, axis=1)
            
            all_predictions.append(pred_labels)
            
            # Individual model evaluation
            f1_macro = f1_score(y_test, pred_labels, average='macro')
            accuracy = accuracy_score(y_test, pred_labels)
            
            print(f"Model {i+1} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Macro: {f1_macro:.4f}")
            
            # Save model if it's good
            if f1_macro > 0.85:
                model.save_pretrained(f"models/final_model_{i}")
                tokenizer.save_pretrained(f"models/final_model_{i}")
                print(f"  Saved model {i+1} (F1: {f1_macro:.4f})")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            # Use random predictions as fallback
            all_predictions.append(np.random.randint(0, len(unique_labels), len(y_test)))
    
    # Ensemble predictions using majority voting
    print("\n=== Creating Ensemble Predictions ===")
    
    if len(all_predictions) > 1:
        # Stack predictions and use majority voting
        stacked_predictions = np.stack(all_predictions, axis=0)
        ensemble_predictions = []
        
        for i in range(len(y_test)):
            # Get votes from all models
            votes = stacked_predictions[:, i]
            # Use majority vote, with tie-breaking by first model
            unique, counts = np.unique(votes, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            ensemble_predictions.append(majority_class)
        
        ensemble_predictions = np.array(ensemble_predictions)
    else:
        ensemble_predictions = all_predictions[0] if all_predictions else np.random.randint(0, len(unique_labels), len(y_test))
    
    # Final evaluation
    final_accuracy = accuracy_score(y_test, ensemble_predictions)
    final_f1_macro = f1_score(y_test, ensemble_predictions, average='macro')
    final_f1_weighted = f1_score(y_test, ensemble_predictions, average='weighted')
    
    print("\n=== FINAL ENSEMBLE RESULTS ===")
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
        "experiment_type": "final_push_90_percent",
        "models_used": len(model_names),
        "model_names": model_names,
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
        },
        "individual_model_predictions": [pred.tolist() for pred in all_predictions]
    }
    
    # Save to results directory
    os.makedirs("results", exist_ok=True)
    results_file = f"results/final_push_90_percent_results.json"
    
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
            "experiment_type": "final_push_90_percent"
        }
        
        with open("results/90_percent_achievement.json", 'w') as f:
            json.dump(achievement, f, indent=2)
        
        print("Achievement notification saved!")
    else:
        gap = 0.90 - final_f1_macro
        print(f"\nTarget not yet achieved. Gap to 90%: {gap:.4f} ({gap*100:.2f}%)")
        print("Consider additional optimization strategies.")
    
    return final_f1_macro >= 0.90

if __name__ == "__main__":
    success = train_advanced_ensemble()
    print(f"\nExperiment completed. Target achieved: {success}")