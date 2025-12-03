#!/usr/bin/env python3
"""
Script untuk melatih model hate speech detection.
Script ini akan memuat data yang sudah dilabeli dan melatih model IndoBERT
untuk klasifikasi ujaran kebencian dalam bahasa Jawa.

Usage:
    python train_model.py --data_path ./hasil-labeling.csv --output_dir ./models/hate_speech_model
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from modelling.train_model import load_labeled_data, prepare_datasets, train_model
from utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(
        description="Train hate speech detection model for Javanese text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_model.py --data_path ./hasil-labeling.csv --output_dir ./models/hate_speech_model
  
  # Training with custom parameters
  python train_model.py --data_path ./hasil-labeling.csv --output_dir ./models/hate_speech_model --epochs 3 --batch_size 16 --learning_rate 2e-5
"""
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="./hasil-labeling.csv",
        help="Path to the labeled data CSV file (default: ./hasil-labeling.csv)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/hate_speech_model",
        help="Directory to save the trained model (default: ./models/hate_speech_model)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size (default: 8)"
    )
    
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Evaluation batch size (default: 8)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for validation (default: 0.2)"
    )
    
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps (default: 10)"
    )
    
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep (default: 2)"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing model directory"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("training", level=args.log_level)
    
    # Validate inputs
    if not os.path.exists(args.data_path):
        logger.error(f"Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    # Check if output directory exists
    if os.path.exists(args.output_dir) and not args.force:
        logger.error(f"Output directory already exists: {args.output_dir}")
        logger.error("Use --force to overwrite or choose a different output directory")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting model training...")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Training parameters:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Eval batch size: {args.eval_batch_size}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - Weight decay: {args.weight_decay}")
    logger.info(f"  - Test size: {args.test_size}")
    
    try:
        # Load labeled data
        logger.info("Loading labeled data...")
        df = load_labeled_data(args.data_path)
        
        if df is None:
            logger.error("Failed to load labeled data")
            sys.exit(1)
            
        logger.info(f"Loaded {len(df)} samples")
        
        # Prepare datasets
        logger.info("Preparing datasets...")
        train_dataset, val_dataset = prepare_datasets(df, test_size=args.test_size)
        
        if train_dataset is None or val_dataset is None:
            logger.error("Failed to prepare datasets")
            sys.exit(1)
            
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Train model
        logger.info("Starting model training...")
        model = train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            save_total_limit=args.save_total_limit
        )
        
        if model is None:
            logger.error("Training failed")
            sys.exit(1)
            
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {args.output_dir}")
        
        # Save training configuration
        config_path = os.path.join(args.output_dir, "training_config.txt")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("Training Configuration\n")
            f.write("===================\n\n")
            f.write(f"Data path: {args.data_path}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Eval batch size: {args.eval_batch_size}\n")
            f.write(f"Learning rate: {args.learning_rate}\n")
            f.write(f"Weight decay: {args.weight_decay}\n")
            f.write(f"Test size: {args.test_size}\n")
            f.write(f"Training samples: {len(train_dataset)}\n")
            f.write(f"Validation samples: {len(val_dataset)}\n")
            
        logger.info(f"Training configuration saved to: {config_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()