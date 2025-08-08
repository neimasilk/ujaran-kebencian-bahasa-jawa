#!/usr/bin/env python3
"""
Test script untuk mengidentifikasi masalah model loading
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test loading model yang digunakan dalam ensemble"""
    
    # Model configs yang sama dengan ensemble
    model_configs = {
        'indobert_uncased': {
            'name': 'indobenchmark/indobert-base-p1',
            'description': 'IndoBERT Base Uncased'
        },
        'roberta_indo': {
            'name': 'cahya/roberta-base-indonesian-522M',
            'description': 'RoBERTa Indonesian'
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Test loading each model
    for model_key, config in model_configs.items():
        try:
            logger.info(f"Testing {model_key}: {config['name']}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config['name'])
            logger.info(f"✓ Tokenizer loaded for {model_key}")
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                config['name'],
                num_labels=4,
                problem_type="single_label_classification"
            )
            model.to(device)
            logger.info(f"✓ Model loaded for {model_key}")
            
            # Test simple prediction
            test_text = "Ini adalah teks uji coba"
            inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                logger.info(f"✓ Prediction test passed for {model_key}")
                logger.info(f"  Prediction shape: {predictions.shape}")
            
        except Exception as e:
            logger.error(f"✗ Failed to load {model_key}: {str(e)}")
            logger.error(f"  Error type: {type(e).__name__}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")

def test_dataset_loading():
    """Test loading dataset"""
    try:
        logger.info("Testing dataset loading...")
        df = pd.read_csv('data/augmented/augmented_dataset.csv')
        logger.info(f"✓ Dataset loaded: {df.shape}")
        logger.info(f"  Columns: {df.columns.tolist()}")
        logger.info(f"  Label distribution: {df.final_label.value_counts().to_dict()}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load dataset: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("=== Model Loading Test ===")
    
    # Test dataset first
    if test_dataset_loading():
        logger.info("\n=== Testing Model Loading ===")
        test_model_loading()
    else:
        logger.error("Dataset loading failed, skipping model tests")
    
    logger.info("\n=== Test Complete ===")