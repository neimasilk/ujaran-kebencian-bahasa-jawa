#!/usr/bin/env python3
"""
Script evaluasi cepat untuk model yang sudah dilatih
"""

import sys
import os
sys.path.append('src')

from modelling.train_model import load_labeled_data
from modelling.evaluate_model import load_model_and_tokenizer, prepare_evaluation_data, predict, compute_metrics
from utils.logger import setup_logger
import pandas as pd
import json

def main():
    # Setup logging
    logger = setup_logger("quick_evaluation", level="INFO")
    
    # Paths
    model_path = "models/trained_model"
    data_path = "src/data_collection/hasil-labeling.csv"
    output_path = "models/trained_model/evaluation_results.json"
    
    logger.info("=== QUICK MODEL EVALUATION ===")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Data path: {data_path}")
    
    # Load model
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_path)
    if not model or not tokenizer:
        logger.error("Failed to load model")
        return
    
    # Load and prepare data (same as training)
    logger.info("Loading labeled data...")
    df = load_labeled_data(data_path)
    if df is None or len(df) == 0:
        logger.error("Failed to load data")
        return
    
    logger.info(f"Loaded {len(df)} samples")
    
    # Take a subset for quick evaluation (first 1000 samples)
    eval_df = df.head(1000).copy()
    texts = eval_df['processed_text'].tolist()
    labels = eval_df['label'].tolist()
    
    logger.info(f"Evaluating on {len(texts)} samples...")
    
    # Prepare evaluation dataset
    eval_dataset = prepare_evaluation_data(texts, tokenizer, labels=labels)
    if not eval_dataset:
        logger.error("Failed to prepare evaluation data")
        return
    
    # Run predictions
    logger.info("Running predictions...")
    predictions, raw_outputs = predict(model, eval_dataset)
    if predictions is None:
        logger.error("Failed to get predictions")
        return
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(labels, predictions)
    
    # Display results
    logger.info("\n=== EVALUATION RESULTS ===")
    if metrics:
        for key, value in metrics.items():
            if key != 'classification_report':
                if isinstance(value, (int, float)):
                    logger.info(f"{key}: {value:.4f}")
                else:
                    logger.info(f"{key}: {value}")
        
        if 'classification_report' in metrics:
            logger.info("\n=== CLASSIFICATION REPORT ===")
            logger.info(f"\n{metrics['classification_report']}")
    
    # Save results
    results = {
        "model_path": model_path,
        "data_path": data_path,
        "num_samples": len(texts),
        "metrics": metrics,
        "predictions": predictions.tolist()[:10]  # Save first 10 predictions as sample
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to: {output_path}")
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()