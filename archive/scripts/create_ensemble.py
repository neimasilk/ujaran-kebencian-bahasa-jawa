import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import argparse
from scipy.optimize import minimize

class EnsemblePredictor:
    def __init__(self, model_paths, weights=None):
        self.model_paths = model_paths
        self.models = []
        self.tokenizers = []
        self.weights = weights if weights is not None else [1.0/len(model_paths)] * len(model_paths)
        
        # Load all models
        for path in model_paths:
            print(f"Loading model from {path}...")
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            
            self.tokenizers.append(tokenizer)
            self.models.append(model)
    
    def predict_batch(self, texts, batch_size=16):
        """Predict using ensemble of models"""
        all_predictions = []
        
        # Get predictions from each model
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            model_predictions = []
            
            for j in range(0, len(texts), batch_size):
                batch_texts = texts[j:j+batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors='pt'
                )
                
                # Move to device
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Predict
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    model_predictions.extend(probabilities.cpu().numpy())
            
            all_predictions.append(np.array(model_predictions))
        
        # Weighted ensemble
        ensemble_probs = np.zeros_like(all_predictions[0])
        for i, (preds, weight) in enumerate(zip(all_predictions, self.weights)):
            ensemble_probs += weight * preds
        
        # Get final predictions
        final_predictions = np.argmax(ensemble_probs, axis=1)
        
        return final_predictions, ensemble_probs

def load_and_preprocess_data(file_path, confidence_threshold=0.7):
    """Load and preprocess the dataset"""
    df = pd.read_csv(file_path)
    print(f"Original dataset size: {len(df)}")
    
    # Filter by confidence if available
    if 'confidence_score' in df.columns:
        df = df[df['confidence_score'] >= confidence_threshold]
        print(f"After confidence filtering (>={confidence_threshold}): {len(df)}")
    
    # Remove rows with missing text or labels
    df = df.dropna(subset=['text', 'final_label'])
    
    # Remove empty text
    df = df[df['text'].str.strip() != '']
    
    # Map labels to integers
    label_mapping = {
        'Bukan Ujaran Kebencian': 0,
        'Ujaran Kebencian - Ringan': 1,
        'Ujaran Kebencian - Sedang': 1,
        'Ujaran Kebencian - Berat': 1
    }
    
    df['label'] = df['final_label'].map(label_mapping)
    df = df.dropna(subset=['label'])  # Remove unmapped labels
    df['label'] = df['label'].astype(int)
    
    print(f"Final dataset size: {len(df)}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df

def optimize_ensemble_weights(model_paths, val_texts, val_labels):
    """Optimize ensemble weights using validation data"""
    print("Optimizing ensemble weights...")
    
    # Get predictions from each model
    all_model_probs = []
    
    for path in model_paths:
        print(f"Getting predictions from {path}...")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        model_probs = []
        batch_size = 16
        
        for i in range(0, len(val_texts), batch_size):
            batch_texts = val_texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                model_probs.extend(probabilities.cpu().numpy())
        
        all_model_probs.append(np.array(model_probs))
    
    # Optimize weights
    def objective(weights):
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Calculate ensemble predictions
        ensemble_probs = np.zeros_like(all_model_probs[0])
        for i, (probs, weight) in enumerate(zip(all_model_probs, weights)):
            ensemble_probs += weight * probs
        
        predictions = np.argmax(ensemble_probs, axis=1)
        f1 = f1_score(val_labels, predictions, average='macro')
        
        return -f1  # Minimize negative F1
    
    # Initial weights (equal)
    initial_weights = np.ones(len(model_paths)) / len(model_paths)
    
    # Constraints: weights sum to 1 and are non-negative
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(len(model_paths))]
    
    # Optimize
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x / np.sum(result.x)  # Normalize
    
    print(f"Optimal weights: {optimal_weights}")
    print(f"Best F1-macro: {-result.fun:.4f}")
    
    return optimal_weights

def evaluate_ensemble(ensemble, texts, labels):
    """Evaluate ensemble performance"""
    predictions, probabilities = ensemble.predict_batch(texts)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    results = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'per_class_metrics': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        },
        'confusion_matrix': cm.tolist(),
        'ensemble_weights': [float(w) for w in ensemble.weights]
    }
    
    return results

def main():
    # Load ensemble results to get best models
    with open('models/ensemble/ensemble_results.json', 'r') as f:
        ensemble_results = json.load(f)
    
    # Sort by F1 score and select top 5 models
    ensemble_results.sort(key=lambda x: x['f1_macro'], reverse=True)
    top_models = ensemble_results[:5]
    
    print("Selected models for ensemble:")
    for i, model in enumerate(top_models):
        print(f"{i+1}. {model['model_id']} ({model['model_name']}) - F1: {model['f1_macro']:.4f}")
    
    model_paths = [model['output_dir'] for model in top_models]
    
    # Load data
    print("\nLoading data...")
    df = load_and_preprocess_data('src/data_collection/hasil-labeling.csv')
    
    # Split data (same split as training)
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    val_texts = val_df['text'].tolist()
    val_labels = val_df['label'].tolist()
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Optimize ensemble weights
    optimal_weights = optimize_ensemble_weights(model_paths, val_texts, val_labels)
    
    # Create ensemble with optimized weights
    print("\nCreating optimized ensemble...")
    ensemble = EnsemblePredictor(model_paths, optimal_weights)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = evaluate_ensemble(ensemble, val_texts, val_labels)
    
    print("=== VALIDATION RESULTS ===")
    print(f"Accuracy: {val_results['accuracy']:.4f}")
    print(f"F1-Macro: {val_results['f1_macro']:.4f}")
    print(f"F1-Weighted: {val_results['f1_weighted']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_ensemble(ensemble, test_texts, test_labels)
    
    print("\n=== TEST RESULTS ===")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"F1-Macro: {test_results['f1_macro']:.4f}")
    print(f"F1-Weighted: {test_results['f1_weighted']:.4f}")
    
    print("\n=== PER-CLASS METRICS (TEST) ===")
    class_names = ['Non-Hate', 'Hate']
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  Precision: {test_results['per_class_metrics']['precision'][i]:.4f}")
        print(f"  Recall: {test_results['per_class_metrics']['recall'][i]:.4f}")
        print(f"  F1: {test_results['per_class_metrics']['f1'][i]:.4f}")
        print(f"  Support: {test_results['per_class_metrics']['support'][i]}")
    
    print("\n=== CONFUSION MATRIX (TEST) ===")
    print("Predicted ->")
    print("Actual |  Non-Hate  Hate")
    print(f"Non-Hate | {test_results['confusion_matrix'][0][0]:8d} {test_results['confusion_matrix'][0][1]:5d}")
    print(f"Hate     | {test_results['confusion_matrix'][1][0]:8d} {test_results['confusion_matrix'][1][1]:5d}")
    
    print("\n=== ENSEMBLE WEIGHTS ===")
    for i, (model, weight) in enumerate(zip(top_models, optimal_weights)):
        print(f"{model['model_id']}: {weight:.4f}")
    
    # Save results
    final_results = {
        'validation_results': val_results,
        'test_results': test_results,
        'selected_models': top_models,
        'optimal_weights': optimal_weights.tolist()
    }
    
    os.makedirs('models/final_ensemble', exist_ok=True)
    with open('models/final_ensemble/ensemble_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: models/final_ensemble/ensemble_results.json")
    
    # Save ensemble configuration
    ensemble_config = {
        'model_paths': model_paths,
        'weights': optimal_weights.tolist(),
        'performance': {
            'test_f1_macro': test_results['f1_macro'],
            'test_accuracy': test_results['accuracy']
        }
    }
    
    with open('models/final_ensemble/ensemble_config.json', 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    print(f"Ensemble configuration saved to: models/final_ensemble/ensemble_config.json")

if __name__ == "__main__":
    main()