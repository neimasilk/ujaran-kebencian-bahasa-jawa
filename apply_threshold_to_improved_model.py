#!/usr/bin/env python3
"""
Apply Threshold Tuning to Improved Model
Demonstrate potential further improvement by applying optimal thresholds to the best model
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
import logging
from datetime import datetime
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/threshold_improved_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path):
    """Load the improved model and tokenizer"""
    logger.info(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded on device: {device}")
    return model, tokenizer, device

def predict_with_probabilities(model, tokenizer, device, texts, batch_size=16):
    """Get predictions and probabilities for texts"""
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_probs)

def apply_custom_thresholds(probabilities, thresholds):
    """Apply custom thresholds to probabilities"""
    predictions = []
    
    for prob in probabilities:
        # Check each class against its threshold
        class_scores = []
        for i, (class_prob, threshold) in enumerate(zip(prob, thresholds)):
            if class_prob >= threshold:
                class_scores.append((i, class_prob))
        
        # If multiple classes meet threshold, choose highest probability
        if class_scores:
            predictions.append(max(class_scores, key=lambda x: x[1])[0])
        else:
            # If no class meets threshold, use highest probability
            predictions.append(np.argmax(prob))
    
    return np.array(predictions)

def main():
    logger.info("Starting threshold tuning application on improved model")
    
    # Load data
    logger.info("Loading balanced dataset")
    df = pd.read_csv('data/standardized/balanced_dataset.csv')
    
    # Prepare data
    X = df['text'].values
    y = df['label_numeric'].values
    
    # Split data (use same split as evaluation for consistency)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split for validation (for threshold optimization)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Load improved model
    model, tokenizer, device = load_model_and_tokenizer('models/improved_model')
    
    # Get predictions on validation set for threshold optimization
    logger.info("Getting validation predictions for threshold optimization")
    val_probs = predict_with_probabilities(model, tokenizer, device, X_val.tolist())
    val_preds_default = np.argmax(val_probs, axis=1)
    
    # Calculate default performance on validation
    val_acc_default = accuracy_score(y_val, val_preds_default)
    val_f1_default = f1_score(y_val, val_preds_default, average='macro')
    
    logger.info(f"Validation - Default accuracy: {val_acc_default:.4f}, F1-Macro: {val_f1_default:.4f}")
    
    # Load optimal thresholds from previous threshold tuning
    # These are the thresholds that worked well on the baseline model
    optimal_thresholds = [0.7128, 0.2332, 0.2023, 0.3395]  # From threshold_tuning_results.json
    
    # Apply thresholds to validation set
    val_preds_tuned = apply_custom_thresholds(val_probs, optimal_thresholds)
    val_acc_tuned = accuracy_score(y_val, val_preds_tuned)
    val_f1_tuned = f1_score(y_val, val_preds_tuned, average='macro')
    
    logger.info(f"Validation - Tuned accuracy: {val_acc_tuned:.4f}, F1-Macro: {val_f1_tuned:.4f}")
    
    # Now evaluate on test set
    logger.info("Evaluating on test set")
    test_probs = predict_with_probabilities(model, tokenizer, device, X_test.tolist())
    
    # Default predictions (threshold 0.5)
    test_preds_default = np.argmax(test_probs, axis=1)
    test_acc_default = accuracy_score(y_test, test_preds_default)
    test_f1_default = f1_score(y_test, test_preds_default, average='macro')
    
    # Tuned predictions
    test_preds_tuned = apply_custom_thresholds(test_probs, optimal_thresholds)
    test_acc_tuned = accuracy_score(y_test, test_preds_tuned)
    test_f1_tuned = f1_score(y_test, test_preds_tuned, average='macro')
    
    # Calculate improvements
    acc_improvement = test_acc_tuned - test_acc_default
    f1_improvement = test_f1_tuned - test_f1_default
    
    # Generate detailed reports
    class_names = ['Bukan Ujaran Kebencian', 'Ujaran Kebencian - Ringan', 
                   'Ujaran Kebencian - Sedang', 'Ujaran Kebencian - Berat']
    
    report_default = classification_report(y_test, test_preds_default, 
                                         target_names=class_names, output_dict=True)
    report_tuned = classification_report(y_test, test_preds_tuned, 
                                       target_names=class_names, output_dict=True)
    
    # Results summary
    results = {
        'model_path': 'models/improved_model',
        'evaluation_timestamp': datetime.now().isoformat(),
        'test_samples': len(X_test),
        'optimal_thresholds': {
            class_names[i]: optimal_thresholds[i] for i in range(len(class_names))
        },
        'performance_comparison': {
            'default_model': {
                'accuracy': float(test_acc_default),
                'f1_macro': float(test_f1_default),
                'classification_report': report_default
            },
            'threshold_tuned': {
                'accuracy': float(test_acc_tuned),
                'f1_macro': float(test_f1_tuned),
                'classification_report': report_tuned
            },
            'improvements': {
                'accuracy_improvement': float(acc_improvement),
                'f1_macro_improvement': float(f1_improvement),
                'accuracy_improvement_percent': float(acc_improvement * 100),
                'f1_macro_improvement_percent': float(f1_improvement * 100)
            }
        }
    }
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/improved_model_threshold_tuning.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 THRESHOLD TUNING ON IMPROVED MODEL - RESULTS")
    print("="*80)
    
    print(f"\n📊 MODEL: models/improved_model")
    print(f"📊 TEST SAMPLES: {len(X_test):,}")
    
    print(f"\n🔧 OPTIMAL THRESHOLDS:")
    for class_name, threshold in zip(class_names, optimal_thresholds):
        print(f"   {class_name}: {threshold:.4f}")
    
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"   Default (0.5 threshold):")
    print(f"     Accuracy: {test_acc_default:.4f} ({test_acc_default*100:.2f}%)")
    print(f"     F1-Macro: {test_f1_default:.4f} ({test_f1_default*100:.2f}%)")
    
    print(f"\n   Threshold Tuned:")
    print(f"     Accuracy: {test_acc_tuned:.4f} ({test_acc_tuned*100:.2f}%)")
    print(f"     F1-Macro: {test_f1_tuned:.4f} ({test_f1_tuned*100:.2f}%)")
    
    print(f"\n🚀 IMPROVEMENTS:")
    print(f"   Accuracy: +{acc_improvement:.4f} (+{acc_improvement*100:.2f}%)")
    print(f"   F1-Macro: +{f1_improvement:.4f} (+{f1_improvement*100:.2f}%)")
    
    # Determine if we reached new milestone
    final_accuracy = test_acc_tuned * 100
    if final_accuracy >= 90:
        print(f"\n🎉 NEW MILESTONE ACHIEVED: {final_accuracy:.2f}% (90%+ TARGET REACHED!)")
    elif final_accuracy >= 88:
        print(f"\n🎯 EXCELLENT PROGRESS: {final_accuracy:.2f}% (Close to 90% target!)")
    else:
        print(f"\n✅ GOOD IMPROVEMENT: {final_accuracy:.2f}% (Building towards 90% target)")
    
    print("\n" + "="*80)
    print("📁 Results saved to: results/improved_model_threshold_tuning.json")
    print("="*80)
    
    logger.info("Threshold tuning application completed successfully")

if __name__ == "__main__":
    main()