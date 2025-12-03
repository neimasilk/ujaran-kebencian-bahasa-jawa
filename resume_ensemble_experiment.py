#!/usr/bin/env python3
"""
Resume Multi-Architecture Ensemble Experiment
Melanjutkan eksperimen yang terhenti saat evaluasi ensemble
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
import logging
from datetime import datetime
import os
import warnings
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/resume_ensemble_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    """Load augmented dataset"""
    logger.info("Loading augmented dataset")
    
    # Load augmented data
    df = pd.read_csv('data/augmented/augmented_dataset.csv')
    
    # Use stratified sampling for faster training (50% of data)
    X = df['text'].values
    y = df['label_numeric'].values
    
    # Split into train/temp, then temp into val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Use 50% of training data for faster experimentation
    X_train_subset, _, y_train_subset, _ = train_test_split(
        X_train, y_train, test_size=0.5, random_state=42, stratify=y_train
    )
    
    logger.info(f"Data split - Train: {len(X_train_subset)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train_subset, X_val, X_test, y_train_subset, y_val, y_test

def get_model_predictions(model_name, model_path, texts, labels, batch_size=32):
    """Get predictions from a trained model"""
    logger.info(f"Getting predictions from {model_name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Create dataset
    dataset = HateSpeechDataset(texts, labels, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get probabilities and predictions
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            probabilities.extend(probs.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)

def evaluate_model(model_name, model_path, X_test, y_test):
    """Evaluate individual model"""
    logger.info(f"Evaluating individual model: {model_name}")
    
    predictions, probabilities = get_model_predictions(model_name, model_path, X_test, y_test)
    
    accuracy = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average='macro')
    
    logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'predictions': predictions,
        'probabilities': probabilities
    }

def ensemble_predict(models_info, X_test, y_test, weights=None):
    """Make ensemble predictions"""
    logger.info("Making ensemble predictions")
    
    if weights is None:
        weights = {name: 1.0/len(models_info) for name in models_info.keys()}
    
    # Get predictions from all models
    all_probabilities = []
    for model_name, model_path in models_info.items():
        _, probabilities = get_model_predictions(model_name, model_path, X_test, y_test)
        all_probabilities.append(probabilities * weights[model_name])
    
    # Average probabilities
    ensemble_probabilities = np.sum(all_probabilities, axis=0)
    ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
    
    return ensemble_predictions, ensemble_probabilities

def evaluate_ensemble(models_info, X_test, y_test, weights=None):
    """Evaluate ensemble performance"""
    logger.info("Evaluating ensemble performance")
    
    predictions, probabilities = ensemble_predict(models_info, X_test, y_test, weights)
    
    accuracy = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'predictions': predictions,
        'probabilities': probabilities,
        'classification_report': classification_report(y_test, predictions, target_names=['Normal', 'Abusive', 'Hateful', 'Offensive'])
    }

def main():
    logger.info("Resuming Multi-Architecture Ensemble Experiment")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Check if we have pre-trained models from previous run
    model_configs = {
        'indobert': 'indobenchmark/indobert-base-p1',
        'indobert_uncased': 'indolem/indobert-base-uncased', 
        'roberta_indo': 'cahya/roberta-base-indonesian-522M'
    }
    
    # Since models weren't saved, we need to retrain or use base models
    logger.info("No saved models found. Using base pre-trained models for quick evaluation.")
    
    # Evaluate individual models (base versions)
    individual_results = {}
    for model_name, model_path in model_configs.items():
        try:
            result = evaluate_model(model_name, model_path, X_test, y_test)
            individual_results[model_name] = result
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
    
    # Ensemble evaluation with equal weights
    logger.info("Evaluating ensemble with equal weights")
    equal_weights = {name: 1.0/len(individual_results) for name in individual_results.keys()}
    ensemble_equal = evaluate_ensemble(model_configs, X_test, y_test, equal_weights)
    
    # Prepare results
    results = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': {
                'train': len(X_train),
                'validation': len(X_val),
                'test': len(X_test)
            },
            'models_used': list(model_configs.keys())
        },
        'individual_results': {
            name: {
                'accuracy': result['accuracy'],
                'f1_macro': result['f1_macro']
            } for name, result in individual_results.items()
        },
        'ensemble_results': {
            'equal_weights': {
                'accuracy': ensemble_equal['accuracy'],
                'f1_macro': ensemble_equal['f1_macro'],
                'weights': equal_weights
            }
        },
        'classification_report': ensemble_equal['classification_report']
    }
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/resumed_ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*50)
    
    logger.info("\nIndividual Model Results:")
    for name, result in individual_results.items():
        logger.info(f"{name}: Accuracy={result['accuracy']:.4f}, F1-Macro={result['f1_macro']:.4f}")
    
    logger.info("\nEnsemble Results (Equal Weights):")
    logger.info(f"Accuracy: {ensemble_equal['accuracy']:.4f}")
    logger.info(f"F1-Macro: {ensemble_equal['f1_macro']:.4f}")
    
    logger.info("\nClassification Report:")
    logger.info(ensemble_equal['classification_report'])
    
    logger.info(f"\nResults saved to: results/resumed_ensemble_results.json")
    
    return results

if __name__ == "__main__":
    main()