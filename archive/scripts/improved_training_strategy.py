#!/usr/bin/env python3
"""
Strategi training yang diperbaiki untuk mengatasi masalah yang ditemukan dalam evaluasi.
Implementasi stratified sampling, class weighting, dan threshold tuning.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from datetime import datetime
from collections import Counter

# Add src to path
sys.path.append('src')
from modelling.train_model import load_labeled_data, JavaneseHateSpeechDataset
from utils.logger import setup_logger

class ImprovedTrainingStrategy:
    def __init__(self, data_path, model_name="indobenchmark/indobert-base-p1"):
        self.data_path = data_path
        self.model_name = model_name
        self.logger = setup_logger("improved_training")
        
        # Label mapping
        self.label_mapping = {
            "Bukan Ujaran Kebencian": 0,
            "Ujaran Kebencian - Ringan": 1,
            "Ujaran Kebencian - Sedang": 2,
            "Ujaran Kebencian - Berat": 3
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
    def analyze_data_distribution(self):
        """Analisis distribusi data untuk strategi training"""
        self.logger.info("Analyzing data distribution...")
        
        # Load data
        df = load_labeled_data(self.data_path)
        
        if df is None or df.empty:
            raise ValueError(f"Failed to load data from {self.data_path}")
        
        # Rename columns for consistency
        df = df.rename(columns={'processed_text': 'text'})
        
        # Add label names for analysis
        df['label_name'] = df['label'].map(self.reverse_label_mapping)
        
        # Analyze distribution
        label_counts = df['label'].value_counts().sort_index()
        self.logger.info("Label distribution:")
        for label, count in label_counts.items():
            label_name = self.reverse_label_mapping[label]
            self.logger.info(f"  {label_name}: {count} ({count/len(df)*100:.2f}%)")
        
        return df, label_counts
    
    def create_stratified_split(self, df, test_size=0.2, random_state=42):
        """Membuat stratified split yang seimbang"""
        self.logger.info(f"Creating stratified split with test_size={test_size}...")
        
        # Stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, val_idx = next(sss.split(df['text'], df['label']))
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        # Log distributions
        self.logger.info("Training set distribution:")
        train_counts = train_df['label'].value_counts().sort_index()
        for label, count in train_counts.items():
            label_name = self.reverse_label_mapping[label]
            self.logger.info(f"  {label_name}: {count} ({count/len(train_df)*100:.2f}%)")
        
        self.logger.info("Validation set distribution:")
        val_counts = val_df['label'].value_counts().sort_index()
        for label, count in val_counts.items():
            label_name = self.reverse_label_mapping[label]
            self.logger.info(f"  {label_name}: {count} ({count/len(val_df)*100:.2f}%)")
        
        return train_df, val_df
    
    def compute_class_weights(self, labels):
        """Hitung class weights untuk mengatasi imbalance"""
        self.logger.info("Computing class weights...")
        
        # Compute class weights
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_labels,
            y=labels
        )
        
        # Create weight mapping
        weight_mapping = {label: weight for label, weight in zip(unique_labels, class_weights)}
        
        self.logger.info("Class weights:")
        for label, weight in weight_mapping.items():
            label_name = self.reverse_label_mapping[label]
            self.logger.info(f"  {label_name}: {weight:.4f}")
        
        return weight_mapping
    
    def create_weighted_sampler(self, labels, class_weights):
        """Membuat weighted sampler untuk training"""
        self.logger.info("Creating weighted sampler...")
        
        # Create sample weights
        sample_weights = [class_weights[label] for label in labels]
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        return sampler
    
    def create_improved_datasets(self, train_df, val_df, tokenizer, max_length=128):
        """Membuat datasets dengan strategi yang diperbaiki"""
        self.logger.info("Creating improved datasets...")
        
        # Tokenize training data
        train_encodings = tokenizer(
            train_df['text'].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Tokenize validation data
        val_encodings = tokenizer(
            val_df['text'].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Create datasets
        train_dataset = JavaneseHateSpeechDataset(
            train_encodings,
            train_df['label'].tolist()
        )
        
        val_dataset = JavaneseHateSpeechDataset(
            val_encodings,
            val_df['label'].tolist()
        )
        
        return train_dataset, val_dataset
    
    def create_focal_loss_trainer(self, model, train_dataset, val_dataset, class_weights, output_dir):
        """Membuat trainer dengan focal loss untuk mengatasi class imbalance"""
        
        class FocalLossTrainer(Trainer):
            def __init__(self, class_weights, alpha=1.0, gamma=2.0, **kwargs):
                super().__init__(**kwargs)
                self.class_weights = torch.tensor(list(class_weights.values()), dtype=torch.float32)
                self.alpha = alpha
                self.gamma = gamma
            
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                
                # Move class weights to same device as logits
                device = logits.device
                class_weights = self.class_weights.to(device)
                
                # Compute focal loss
                ce_loss = torch.nn.functional.cross_entropy(
                    logits, labels, weight=class_weights, reduction='none'
                )
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                loss = focal_loss.mean()
                
                return (loss, outputs) if return_outputs else loss
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            report_to=None,  # Disable wandb
            dataloader_pin_memory=False,
        )
        
        # Custom metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            # Compute metrics
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
            precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
            
            return {
                'accuracy': accuracy,
                'f1_macro': f1,
                'f1_weighted': f1_w,
                'precision_macro': precision,
                'recall_macro': recall
            }
        
        # Create trainer
        trainer = FocalLossTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        return trainer
    
    def run_improved_training(self, output_dir="models/improved_model"):
        """Menjalankan training dengan strategi yang diperbaiki"""
        self.logger.info("Starting improved training strategy...")
        
        # 1. Analyze data
        df, label_counts = self.analyze_data_distribution()
        
        # 2. Create stratified split
        train_df, val_df = self.create_stratified_split(df)
        
        # 3. Compute class weights
        class_weights = self.compute_class_weights(train_df['label'].values)
        
        # 4. Load model and tokenizer
        self.logger.info(f"Loading model and tokenizer: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_mapping)
        )
        
        # 5. Create datasets
        train_dataset, val_dataset = self.create_improved_datasets(train_df, val_df, tokenizer)
        
        # 6. Create trainer with focal loss
        trainer = self.create_focal_loss_trainer(
            model, train_dataset, val_dataset, class_weights, output_dir
        )
        
        # 7. Train model
        self.logger.info("Starting training...")
        trainer.train()
        
        # 8. Save model
        self.logger.info(f"Saving improved model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # 9. Save training info
        training_info = {
            "training_date": datetime.now().isoformat(),
            "model_name": self.model_name,
            "data_path": self.data_path,
            "improvements": [
                "Stratified sampling",
                "Class weighting",
                "Focal loss",
                "Balanced evaluation"
            ],
            "class_weights": {self.reverse_label_mapping[k]: v for k, v in class_weights.items()},
            "data_distribution": {
                "total_samples": len(df),
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "label_distribution": {self.reverse_label_mapping[k]: int(v) for k, v in label_counts.items()}
            }
        }
        
        with open(f"{output_dir}/training_info.json", 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info("Improved training completed!")
        return output_dir

def main():
    """Main function untuk menjalankan improved training"""
    data_path = "src/data_collection/hasil-labeling.csv"
    output_dir = "models/improved_model"
    
    # Create improved training strategy
    strategy = ImprovedTrainingStrategy(data_path)
    
    # Run improved training
    model_path = strategy.run_improved_training(output_dir)
    
    print(f"\n=== IMPROVED TRAINING COMPLETED ===")
    print(f"Model saved to: {model_path}")
    print(f"Next steps:")
    print(f"1. Run balanced evaluation on improved model")
    print(f"2. Compare results with original model")
    print(f"3. Tune thresholds for production deployment")
    print(f"4. Implement ensemble methods if needed")

if __name__ == "__main__":
    main()