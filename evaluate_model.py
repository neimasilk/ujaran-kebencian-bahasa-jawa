#!/usr/bin/env python3
"""
Script untuk mengevaluasi model hate speech detection yang sudah dilatih.
Script ini akan memuat model yang sudah dilatih dan mengevaluasi performanya
pada data test atau data baru.

Usage:
    python evaluate_model.py --model_path ./models/hate_speech_model --data_path ./data/test_data.csv
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from modelling.evaluate_model import evaluate_model
from utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained hate speech detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with test data (with labels)
  python evaluate_model.py --model_path ./models/hate_speech_model --data_path ./data/test_data.csv --output_dir ./evaluation_results
  
  # Evaluate with new data (without labels)
  python evaluate_model.py --model_path ./models/hate_speech_model --data_path ./data/new_data.csv --output_dir ./evaluation_results --no_labels
"""
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the evaluation data CSV file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results (default: ./evaluation_results)"
    )
    
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Set this flag if the data doesn't contain labels (prediction only)"
    )
    
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text column in the CSV file (default: text)"
    )
    
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Name of the label column in the CSV file (default: label)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("evaluation", level=args.log_level)
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        sys.exit(1)
        
    if not os.path.exists(args.data_path):
        logger.error(f"Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting model evaluation...")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Has labels: {not args.no_labels}")
    
    try:
        # Run evaluation
        results = evaluate_model(
            model_path=args.model_path,
            eval_data_path_or_texts=args.data_path,
            text_column=args.text_column,
            label_column=args.label_column if not args.no_labels else None,
            output_file=os.path.join(args.output_dir, "evaluation_results.json")
        )
        
        if results is None:
            logger.error("Evaluation failed")
            sys.exit(1)
            
        logger.info("Evaluation completed successfully!")
        
        if not args.no_labels:
            logger.info("\n=== EVALUATION RESULTS ===")
            logger.info(f"Accuracy: {results.get('accuracy', 'N/A'):.4f}")
            logger.info(f"F1 Score (Macro): {results.get('f1_macro', 'N/A'):.4f}")
            logger.info(f"Precision (Macro): {results.get('precision_macro', 'N/A'):.4f}")
            logger.info(f"Recall (Macro): {results.get('recall_macro', 'N/A'):.4f}")
            
            # Show class-wise metrics if available
            if 'classification_report' in results:
                logger.info("\n=== CLASSIFICATION REPORT ===")
                logger.info(f"\n{results['classification_report']}")
        else:
            logger.info("Predictions saved to output directory")
            
        logger.info(f"\nDetailed results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()