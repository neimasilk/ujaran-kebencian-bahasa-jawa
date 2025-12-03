#!/usr/bin/env python3
"""
Evaluasi model menggunakan dataset yang seimbang dan representatif
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import json
from datetime import datetime

# Add src to path
sys.path.append('src')
from modelling.evaluate_model import prepare_evaluation_data
from utils.logger import setup_logger

def load_balanced_evaluation_data(file_path):
    """Load dataset evaluasi yang seimbang"""
    print(f"Loading balanced evaluation dataset dari: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Total samples: {len(df)}")
    
    # Mapping label ke numeric (sama seperti training)
    label_mapping = {
        "Bukan Ujaran Kebencian": 0,
        "Ujaran Kebencian - Ringan": 1,
        "Ujaran Kebencian - Sedang": 2,
        "Ujaran Kebencian - Berat": 3
    }
    
    # Convert labels
    df['label_numeric'] = df['final_label'].map(label_mapping)
    
    # Remove any rows with unmapped labels
    df = df.dropna(subset=['label_numeric'])
    df['label_numeric'] = df['label_numeric'].astype(int)
    
    print("Distribusi label:")
    label_dist = df['final_label'].value_counts()
    for label, count in label_dist.items():
        print(f"  {label}: {count} ({count/len(df)*100:.2f}%)")
    
    return df['text'].tolist(), df['label_numeric'].tolist(), df['final_label'].tolist()

def evaluate_model_balanced(model_path, eval_data_path, output_file):
    """Evaluasi model dengan dataset seimbang"""
    print(f"=== EVALUASI MODEL DENGAN DATASET SEIMBANG ===")
    print(f"Model: {model_path}")
    print(f"Data: {eval_data_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load model dan tokenizer
        print("Loading model dan tokenizer...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        # Load evaluation data
        eval_texts, eval_labels, eval_label_names = load_balanced_evaluation_data(eval_data_path)
        
        # Prepare evaluation dataset
        print("Preparing evaluation dataset...")
        eval_dataset = prepare_evaluation_data(eval_texts, tokenizer, eval_labels)
        
        # Run predictions
        print("Running predictions...")
        predictions = []
        raw_outputs = []
        
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(eval_dataset), batch_size):
                # Get batch items
                batch_items = [eval_dataset[j] for j in range(i, min(i + batch_size, len(eval_dataset)))]
                
                # Prepare batch tensors
                input_ids = torch.stack([item['input_ids'] for item in batch_items]).to(device)
                attention_mask = torch.stack([item['attention_mask'] for item in batch_items]).to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get predictions
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                batch_raw_outputs = logits.cpu().numpy()
                
                predictions.extend(batch_predictions)
                raw_outputs.extend(batch_raw_outputs)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"Processed {i + len(batch_items)}/{len(eval_dataset)} samples")
        
        # Compute metrics
        print("Computing metrics...")
        
        accuracy = accuracy_score(eval_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(eval_labels, predictions, average=None)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(eval_labels, predictions, average='macro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(eval_labels, predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(eval_labels, predictions)
        
        # Classification report
        label_names = ["Bukan Ujaran Kebencian", "Ujaran Kebencian - Ringan", "Ujaran Kebencian - Sedang", "Ujaran Kebencian - Berat"]
        class_report = classification_report(eval_labels, predictions, target_names=label_names, output_dict=True)
        
        # Print results
        print("\n=== HASIL EVALUASI SEIMBANG ===")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"F1-Score Macro: {f1_macro:.4f} ({f1_macro*100:.2f}%)")
        print(f"F1-Score Weighted: {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
        print(f"Precision Weighted: {precision_weighted:.4f} ({precision_weighted*100:.2f}%)")
        print(f"Recall Weighted: {recall_weighted:.4f} ({recall_weighted*100:.2f}%)")
        
        print("\n=== DISTRIBUSI PREDIKSI ===")
        pred_counts = np.bincount(predictions, minlength=4)
        for i, count in enumerate(pred_counts):
            print(f"{label_names[i]}: {count}/{len(predictions)} ({count/len(predictions)*100:.1f}%)")
        
        print("\n=== METRIK PER KELAS ===")
        for i, label in enumerate(label_names):
            if i < len(precision):
                print(f"{label}:")
                print(f"  Precision: {precision[i]:.4f} ({precision[i]*100:.2f}%)")
                print(f"  Recall: {recall[i]:.4f} ({recall[i]*100:.2f}%)")
                print(f"  F1-Score: {f1[i]:.4f} ({f1[i]*100:.2f}%)")
                print(f"  Support: {support[i]}")
        
        print("\n=== CONFUSION MATRIX ===")
        print("Actual \\ Predicted:", end="")
        for label in label_names:
            print(f"\t{label[:10]}", end="")
        print()
        
        for i, label in enumerate(label_names):
            print(f"{label[:15]}", end="")
            for j in range(len(label_names)):
                if i < len(cm) and j < len(cm[i]):
                    print(f"\t{cm[i][j]}", end="")
                else:
                    print(f"\t0", end="")
            print()
        
        # Save results
        results = {
            "evaluation_date": datetime.now().isoformat(),
            "model_path": model_path,
            "eval_data_path": eval_data_path,
            "dataset_info": {
                "total_samples": len(eval_labels),
                "balanced": True,
                "samples_per_class": 200
            },
            "metrics": {
                "accuracy": float(accuracy),
                "f1_macro": float(f1_macro),
                "f1_weighted": float(f1_weighted),
                "precision_macro": float(precision_macro),
                "precision_weighted": float(precision_weighted),
                "recall_macro": float(recall_macro),
                "recall_weighted": float(recall_weighted)
            },
            "per_class_metrics": {
                label_names[i]: {
                    "precision": float(precision[i]) if i < len(precision) else 0.0,
                    "recall": float(recall[i]) if i < len(recall) else 0.0,
                    "f1_score": float(f1[i]) if i < len(f1) else 0.0,
                    "support": int(support[i]) if i < len(support) else 0
                } for i in range(len(label_names))
            },
            "confusion_matrix": cm.tolist(),
            "prediction_distribution": {
                label_names[i]: int(pred_counts[i]) if i < len(pred_counts) else 0
                for i in range(len(label_names))
            },
            "classification_report": class_report
        }
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nHasil evaluasi disimpan ke: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Paths
    model_path = "models/trained_model"
    eval_data_path = "data/processed/balanced_evaluation_set.csv"
    output_file = "models/trained_model/balanced_evaluation_results.json"
    
    # Run evaluation
    results = evaluate_model_balanced(model_path, eval_data_path, output_file)
    
    if results:
        print("\n=== EVALUASI SELESAI ===")
        print("Hasil evaluasi menunjukkan performa model pada dataset yang seimbang.")
        print("Bandingkan dengan evaluasi sebelumnya untuk melihat perbedaan yang signifikan.")
    else:
        print("\n=== EVALUASI GAGAL ===")
        print("Terjadi error selama evaluasi. Periksa log di atas untuk detail.")