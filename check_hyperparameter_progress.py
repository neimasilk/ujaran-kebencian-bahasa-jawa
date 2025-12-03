#!/usr/bin/env python3
"""
Check hyperparameter tuning progress and checkpoint status

Author: AI Assistant
Date: 2025-01-24
"""

import json
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_progress():
    """Check hyperparameter tuning progress"""
    
    # Paths
    checkpoint_file = Path("experiments/results/hyperparameter_tuning/checkpoint.json")
    results_file = Path("experiments/results/hyperparameter_tuning/intermediate_results.json")
    final_results_file = Path("experiments/results/hyperparameter_tuning/final_results.json")
    best_config_file = Path("experiments/results/hyperparameter_tuning/best_configuration.json")
    
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING PROGRESS CHECK")
    logger.info("=" * 60)
    
    # Check if final results exist (completed)
    if final_results_file.exists() and best_config_file.exists():
        logger.info("✓ HYPERPARAMETER TUNING COMPLETED!")
        
        with open(best_config_file, 'r') as f:
            best_config = json.load(f)
        
        with open(final_results_file, 'r') as f:
            final_results = json.load(f)
        
        # Find best result
        best_result = max(final_results, key=lambda x: x['f1_macro'])
        
        logger.info(f"Best F1-Macro Score: {best_result['f1_macro']:.4f}")
        logger.info(f"Best Accuracy: {best_result['accuracy']:.4f}")
        logger.info(f"Best Configuration:")
        logger.info(f"  - Learning Rate: {best_config['learning_rate']}")
        logger.info(f"  - Batch Size: {best_config['batch_size']}")
        logger.info(f"  - Epochs: {best_config['num_epochs']}")
        logger.info(f"  - Warmup Ratio: {best_config['warmup_ratio']}")
        logger.info(f"Total experiments completed: {len(final_results)}")
        
        return
    
    # Check checkpoint status
    if checkpoint_file.exists():
        logger.info("📋 CHECKPOINT FOUND - Process can be resumed")
        
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        completed = checkpoint['completed_experiments']
        total = 72  # 4 * 3 * 2 * 3
        progress = (completed / total) * 100
        
        logger.info(f"Progress: {completed}/{total} experiments ({progress:.1f}%)")
        logger.info(f"Checkpoint timestamp: {checkpoint['timestamp']}")
        
        if checkpoint['best_f1'] > 0:
            logger.info(f"Current best F1-Macro: {checkpoint['best_f1']:.4f}")
            if checkpoint['best_config']:
                logger.info(f"Current best config: LR={checkpoint['best_config']['learning_rate']}, "
                           f"BS={checkpoint['best_config']['batch_size']}, "
                           f"EP={checkpoint['best_config']['num_epochs']}, "
                           f"WR={checkpoint['best_config']['warmup_ratio']}")
        
        logger.info("\nTo resume: python experiments/hyperparameter_tuning.py")
        
    elif results_file.exists():
        logger.info("📊 INTERMEDIATE RESULTS FOUND")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        completed = len(results)
        total = 72
        progress = (completed / total) * 100
        
        logger.info(f"Progress: {completed}/{total} experiments ({progress:.1f}%)")
        
        if results:
            best_result = max(results, key=lambda x: x['f1_macro'])
            logger.info(f"Best F1-Macro so far: {best_result['f1_macro']:.4f}")
            logger.info(f"Best config so far: LR={best_result['learning_rate']}, "
                       f"BS={best_result['batch_size']}, "
                       f"EP={best_result['num_epochs']}, "
                       f"WR={best_result['warmup_ratio']}")
    
    else:
        logger.info("❌ NO PROGRESS FILES FOUND")
        logger.info("Hyperparameter tuning has not been started or no progress saved yet.")
        logger.info("\nTo start: python experiments/hyperparameter_tuning.py")
    
    logger.info("=" * 60)

def show_experiment_details():
    """Show detailed experiment configuration"""
    logger.info("\nEXPERIMENT CONFIGURATION:")
    logger.info("- Learning Rates: [1e-5, 2e-5, 3e-5, 5e-5]")
    logger.info("- Batch Sizes: [8, 16, 32]")
    logger.info("- Epochs: [3, 5]")
    logger.info("- Warmup Ratios: [0.05, 0.1, 0.15]")
    logger.info("- Total combinations: 4 × 3 × 2 × 3 = 72 experiments")
    logger.info("- Model: indobenchmark/indobert-base-p1")
    logger.info("- Dataset: data/standardized/balanced_dataset.csv")

if __name__ == "__main__":
    check_progress()
    show_experiment_details()