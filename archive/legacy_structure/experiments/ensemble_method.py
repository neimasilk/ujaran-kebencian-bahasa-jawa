#!/usr/bin/env python3
"""
Ensemble Method for Javanese Hate Speech Detection
Combines predictions from multiple models for improved performance

Author: AI Assistant
Date: 2025-07-24
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ensemble_method.log'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Ensure immediate output
import sys
sys.stdout.flush()
sys.stderr.flush()

class EnsembleConfig:
    """Configuration for ensemble method"""
    
    # Model paths (these should point to your trained models)
    MODEL_PATHS = [
        "models/indobert_baseline_hate_speech",  # IndoBERT Base
        "models/indobert_large_hate_speech",     # IndoBERT Large
        "models/mbert_javanese_hate_speech",     # mBERT
    ]
    
    # Ensemble weights (can be adjusted based on individual model performance)
    MODEL_WEIGHTS = [0.4, 0.3, 0.3]  # Should sum to 1.0
    
    # Data path
    DATA_PATH = "data/standardized/balanced_dataset.csv"
    
    # Label mapping
    LABEL_MAPPING = {
        0: "Bukan Ujaran Kebencian",
        1: "Ujaran Kebencian - Ringan",
        2: "Ujaran Kebencian - Sedang",
        3: "Ujaran Kebencian - Berat"
    }

class HateSpeechDataset(Dataset):
    """Dataset for hate speech classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
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

def load_and_preprocess_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the dataset"""
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Use standardized dataset columns
    if 'label_numeric' in df.columns:
        df = df[['text', 'label_numeric']].copy()
        df = df.rename(columns={'label_numeric': 'label'})
    elif 'final_label' in df.columns:
        label_mapping = {
            'Bukan Ujaran Kebencian': 0,
            'Ujaran Kebencian - Ringan': 1,
            'Ujaran Kebencian - Sedang': 2,
            'Ujaran Kebencian - Berat': 3
        }
        df['label_id'] = df['final_label'].map(label_mapping)
        df = df[['text', 'label_id']].copy()
        df = df.rename(columns={'label_id': 'label'})
    else:
        raise ValueError("Dataset must contain either 'label_numeric' or 'final_label' column")
    
    # Clean data
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)
    
    return df['text'].values, df['label'].values

def create_stratified_split(texts: np.ndarray, labels: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """Create stratified train-test split"""
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def load_model(model_path: str):
    """Load a trained model and its tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        return None, None

def get_model_predictions(model, tokenizer, texts: List[str], device: str = 'cpu') -> np.ndarray:
    """Get predictions from a single model"""
    model.to(device)
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions.append(probabilities.cpu().numpy()[0])
    
    return np.array(predictions)

def ensemble_predict(models_data: List[Tuple], texts: List[str], weights: List[float], device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """Make ensemble predictions by combining multiple models"""
    all_predictions = []
    
    # Get predictions from each model
    for model, tokenizer in models_data:
        if model is not None and tokenizer is not None:
            predictions = get_model_predictions(model, tokenizer, texts, device)
            all_predictions.append(predictions)
        else:
            # If model failed to load, add zero predictions
            logger.warning("Skipping failed model in ensemble")
            all_predictions.append(np.zeros((len(texts), 4)))
    
    # Combine predictions using weighted average
    weighted_predictions = np.zeros_like(all_predictions[0])
    total_weight = 0.0
    
    for i, (predictions, weight) in enumerate(zip(all_predictions, weights)):
        weighted_predictions += predictions * weight
        total_weight += weight
    
    # Normalize by total weight
    if total_weight > 0:
        weighted_predictions /= total_weight
    
    # Get final predictions (argmax)
    final_predictions = np.argmax(weighted_predictions, axis=1)
    
    return final_predictions, weighted_predictions

def compute_metrics(true_labels: np.ndarray, predictions: np.ndarray) -> Dict:
    """Compute evaluation metrics"""
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    accuracy = accuracy_score(true_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }

def main():
    """Main ensemble method function"""
    logger.info("=" * 60)
    logger.info("ENSEMBLE METHOD FOR JAVANESE HATE SPEECH DETECTION")
    logger.info("=" * 60)
    
    config = EnsembleConfig()
    
    try:
        # Load and preprocess data
        texts, labels = load_and_preprocess_data(config.DATA_PATH)
        X_train, X_test, y_train, y_test = create_stratified_split(texts, labels)
        
        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Load all models
        models_data = []
        for model_path in config.MODEL_PATHS:
            logger.info(f"Loading model from {model_path}")
            model, tokenizer = load_model(model_path)
            models_data.append((model, tokenizer))
        
        # Validate weights
        if len(config.MODEL_WEIGHTS) != len(models_data):
            logger.warning("Model weights don't match number of models. Using equal weights.")
            config.MODEL_WEIGHTS = [1.0 / len(models_data)] * len(models_data)
        
        # Make ensemble predictions on test set
        logger.info("Making ensemble predictions on test set")
        final_predictions, _ = ensemble_predict(models_data, X_test.tolist(), config.MODEL_WEIGHTS, device)
        
        # Compute metrics
        metrics = compute_metrics(y_test, final_predictions)
        
        # Print results
        logger.info("=" * 60)
        logger.info("ENSEMBLE MODEL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1-Score Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"Precision Macro: {metrics['precision_macro']:.4f}")
        logger.info(f"Recall Macro: {metrics['recall_macro']:.4f}")
        
        # Compare with individual models (if available)
        logger.info("=" * 60)
        logger.info("INDIVIDUAL MODEL COMPARISON")
        logger.info("=" * 60)
        
        for i, (model, tokenizer) in enumerate(models_data):
            if model is not None and tokenizer is not None:
                # Get predictions from individual model
                model_predictions, _ = ensemble_predict(
                    [(model, tokenizer)], X_test.tolist(), [1.0], device
                )
                
                # Compute metrics
                model_metrics = compute_metrics(y_test, model_predictions)
                
                logger.info(f"Model {i+1} ({config.MODEL_PATHS[i]}):")
                logger.info(f"  Accuracy: {model_metrics['accuracy']:.4f}")
                logger.info(f"  F1-Score Macro: {model_metrics['f1_macro']:.4f}")
                
                # Compare with ensemble
                improvement = metrics['f1_macro'] - model_metrics['f1_macro']
                logger.info(f"  Improvement with ensemble: {improvement:+.4f}")
                logger.info("-" * 40)
        
        # Save results
        results = {
            'ensemble_metrics': metrics,
            'model_paths': config.MODEL_PATHS,
            'model_weights': config.MODEL_WEIGHTS,
            'test_set_size': len(X_test)
        }
        
        results_file = Path('ensemble_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        logger.info("=" * 60)
        logger.info("ENSEMBLE METHOD COMPLETED")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Ensemble method failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()