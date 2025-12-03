#!/usr/bin/env python3
"""
Threshold tuning untuk mengoptimalkan performa model hate speech detection.
Script ini akan mencari threshold optimal untuk setiap kelas.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Add src to path
sys.path.append('src')
from modelling.evaluate_model import prepare_evaluation_data

class ThresholdTuner:
    def __init__(self, model_path, balanced_data_path):
        self.model_path = model_path
        self.balanced_data_path = balanced_data_path
        
        # Label mapping
        self.label_mapping = {
            "Bukan Ujaran Kebencian": 0,
            "Ujaran Kebencian - Ringan": 1,
            "Ujaran Kebencian - Sedang": 2,
            "Ujaran Kebencian - Berat": 3
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
    def load_model_and_data(self):
        """Load model, tokenizer, dan data"""
        print("Loading model and tokenizer...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
        
        # Load balanced evaluation data
        print("Loading balanced evaluation data...")
        df = pd.read_csv(self.balanced_data_path)
        
        # Check if we have the right columns
        if 'label' in df.columns:
            # Map labels to numeric
            df['label_numeric'] = df['label'].map(self.label_mapping)
            self.eval_texts = df['text'].tolist()
            self.eval_labels = df['label_numeric'].tolist()
        elif 'final_label' in df.columns:
            # Use final_label column and map to numeric
            df['label_numeric'] = df['final_label'].map(self.label_mapping)
            self.eval_texts = df['text'].tolist()
            self.eval_labels = df['label_numeric'].tolist()
        else:
            raise ValueError(f"Expected 'label' or 'final_label' column in {self.balanced_data_path}")
        
        print(f"Loaded {len(self.eval_texts)} evaluation samples")
        
    def get_model_predictions(self, batch_size=16):
        """Dapatkan prediksi probabilitas dari model"""
        print("Getting model predictions...")
        
        # Prepare evaluation dataset
        eval_dataset = prepare_evaluation_data(self.eval_texts, self.tokenizer, self.eval_labels)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device if using GPU
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                
                # Get model outputs
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Convert to probabilities
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)
    
    def find_optimal_thresholds(self, probabilities, true_labels):
        """Cari threshold optimal untuk setiap kelas"""
        print("Finding optimal thresholds...")
        
        optimal_thresholds = {}
        threshold_results = {}
        
        for class_idx in range(len(self.label_mapping)):
            class_name = self.reverse_label_mapping[class_idx]
            print(f"\nOptimizing threshold for class: {class_name}")
            
            # Binary classification for this class
            binary_labels = (true_labels == class_idx).astype(int)
            class_probabilities = probabilities[:, class_idx]
            
            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(binary_labels, class_probabilities)
            
            # Calculate F1 scores for each threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # Find optimal threshold (max F1)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            optimal_f1 = f1_scores[optimal_idx]
            optimal_precision = precision[optimal_idx]
            optimal_recall = recall[optimal_idx]
            
            optimal_thresholds[class_idx] = optimal_threshold
            threshold_results[class_name] = {
                'threshold': float(optimal_threshold),
                'f1_score': float(optimal_f1),
                'precision': float(optimal_precision),
                'recall': float(optimal_recall)
            }
            
            print(f"  Optimal threshold: {optimal_threshold:.4f}")
            print(f"  F1 score: {optimal_f1:.4f}")
            print(f"  Precision: {optimal_precision:.4f}")
            print(f"  Recall: {optimal_recall:.4f}")
        
        return optimal_thresholds, threshold_results
    
    def apply_thresholds(self, probabilities, thresholds):
        """Terapkan threshold yang dioptimalkan untuk prediksi"""
        predictions = []
        
        for prob_vector in probabilities:
            # Default prediction (highest probability)
            default_pred = np.argmax(prob_vector)
            
            # Check if any class meets its threshold
            threshold_predictions = []
            for class_idx, threshold in thresholds.items():
                if prob_vector[class_idx] >= threshold:
                    threshold_predictions.append((class_idx, prob_vector[class_idx]))
            
            if threshold_predictions:
                # Choose class with highest probability among those meeting threshold
                best_class = max(threshold_predictions, key=lambda x: x[1])[0]
                predictions.append(best_class)
            else:
                # Fallback to default prediction
                predictions.append(default_pred)
        
        return np.array(predictions)
    
    def evaluate_with_thresholds(self, probabilities, true_labels, thresholds):
        """Evaluasi performa dengan threshold yang dioptimalkan"""
        print("\nEvaluating with optimized thresholds...")
        
        # Apply thresholds
        threshold_predictions = self.apply_thresholds(probabilities, thresholds)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(true_labels, threshold_predictions)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, threshold_predictions, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels, threshold_predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            true_labels, threshold_predictions, average=None
        )
        
        # Classification report
        class_names = [self.reverse_label_mapping[i] for i in range(len(self.label_mapping))]
        report = classification_report(
            true_labels, threshold_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, threshold_predictions)
        
        results = {
            'accuracy': accuracy,
            'macro_avg': {
                'precision': precision_macro,
                'recall': recall_macro,
                'f1_score': f1_macro
            },
            'weighted_avg': {
                'precision': precision_weighted,
                'recall': recall_weighted,
                'f1_score': f1_weighted
            },
            'per_class_metrics': {},
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        for i, class_name in enumerate(class_names):
            results['per_class_metrics'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(support[i])
            }
        
        return results, threshold_predictions
    
    def create_threshold_comparison(self, probabilities, true_labels, default_predictions, threshold_predictions):
        """Buat perbandingan antara prediksi default dan dengan threshold"""
        print("Creating threshold comparison...")
        
        from sklearn.metrics import accuracy_score, f1_score
        
        # Default metrics
        default_accuracy = accuracy_score(true_labels, default_predictions)
        default_f1_macro = f1_score(true_labels, default_predictions, average='macro')
        default_f1_weighted = f1_score(true_labels, default_predictions, average='weighted')
        
        # Threshold metrics
        threshold_accuracy = accuracy_score(true_labels, threshold_predictions)
        threshold_f1_macro = f1_score(true_labels, threshold_predictions, average='macro')
        threshold_f1_weighted = f1_score(true_labels, threshold_predictions, average='weighted')
        
        comparison = {
            'default_model': {
                'accuracy': default_accuracy,
                'f1_macro': default_f1_macro,
                'f1_weighted': default_f1_weighted
            },
            'threshold_tuned': {
                'accuracy': threshold_accuracy,
                'f1_macro': threshold_f1_macro,
                'f1_weighted': threshold_f1_weighted
            },
            'improvements': {
                'accuracy_improvement': threshold_accuracy - default_accuracy,
                'f1_macro_improvement': threshold_f1_macro - default_f1_macro,
                'f1_weighted_improvement': threshold_f1_weighted - default_f1_weighted
            }
        }
        
        return comparison
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def save_results(self, thresholds, threshold_results, evaluation_results, comparison, output_path):
        """Simpan hasil threshold tuning"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'data_path': self.balanced_data_path,
            'optimal_thresholds': {self.reverse_label_mapping[k]: v for k, v in thresholds.items()},
            'threshold_optimization_results': threshold_results,
            'evaluation_with_thresholds': evaluation_results,
            'performance_comparison': comparison
        }
        
        # Convert numpy types to Python native types
        results = self.convert_numpy_types(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("Results saved successfully!")
    
    def run_threshold_tuning(self, output_path="threshold_tuning_results.json"):
        """Jalankan proses threshold tuning lengkap"""
        print("=== THRESHOLD TUNING FOR HATE SPEECH DETECTION ===")
        
        # 1. Load model and data
        self.load_model_and_data()
        
        # 2. Get model predictions
        default_predictions, probabilities, true_labels = self.get_model_predictions()
        
        # 3. Find optimal thresholds
        optimal_thresholds, threshold_results = self.find_optimal_thresholds(probabilities, true_labels)
        
        # 4. Evaluate with thresholds
        evaluation_results, threshold_predictions = self.evaluate_with_thresholds(
            probabilities, true_labels, optimal_thresholds
        )
        
        # 5. Create comparison
        comparison = self.create_threshold_comparison(
            probabilities, true_labels, default_predictions, threshold_predictions
        )
        
        # 6. Save results
        self.save_results(
            optimal_thresholds, threshold_results, evaluation_results, comparison, output_path
        )
        
        # 7. Print summary
        print("\n=== THRESHOLD TUNING SUMMARY ===")
        print(f"Default model accuracy: {comparison['default_model']['accuracy']:.4f}")
        print(f"Threshold-tuned accuracy: {comparison['threshold_tuned']['accuracy']:.4f}")
        print(f"Accuracy improvement: {comparison['improvements']['accuracy_improvement']:.4f}")
        print(f"\nDefault F1 (macro): {comparison['default_model']['f1_macro']:.4f}")
        print(f"Threshold-tuned F1 (macro): {comparison['threshold_tuned']['f1_macro']:.4f}")
        print(f"F1 macro improvement: {comparison['improvements']['f1_macro_improvement']:.4f}")
        
        return optimal_thresholds, evaluation_results

def main():
    """Main function untuk threshold tuning"""
    model_path = "models/trained_model"  # Path ke model yang sudah ditraining
    balanced_data_path = "data/processed/balanced_evaluation_set.csv"
    output_path = "threshold_tuning_results.json"
    
    # Create threshold tuner
    tuner = ThresholdTuner(model_path, balanced_data_path)
    
    # Run threshold tuning
    optimal_thresholds, results = tuner.run_threshold_tuning(output_path)
    
    print(f"\n=== THRESHOLD TUNING COMPLETED ===")
    print(f"Results saved to: {output_path}")
    print(f"\nOptimal thresholds:")
    for class_idx, threshold in optimal_thresholds.items():
        class_name = tuner.reverse_label_mapping[class_idx]
        print(f"  {class_name}: {threshold:.4f}")

if __name__ == "__main__":
    main()