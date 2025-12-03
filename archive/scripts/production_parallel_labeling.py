#!/usr/bin/env python3
"""
Production Parallel Labeling Script
Script produksi untuk parallel labeling tanpa demo testing.

Usage:
    python production_parallel_labeling.py --input data.csv --output results.csv --real
    python production_parallel_labeling.py --input data.csv --output results.csv --mock  # for testing
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collection.parallel_deepseek_pipeline import ParallelDeepSeekLabelingPipeline
from utils.logger import setup_logger

# Setup logging
logger = setup_logger("production_parallel", level="INFO")

def main():
    """Main production function - langsung labeling tanpa demo."""
    parser = argparse.ArgumentParser(description="Production Parallel Labeling")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    parser.add_argument("--workers", type=int, default=3, help="Number of workers (default: 3)")
    parser.add_argument("--mock", action="store_true", help="Use mock mode for testing")
    parser.add_argument("--real", action="store_true", help="Use real API")
    parser.add_argument("--force", action="store_true", help="Override existing locks")
    
    args = parser.parse_args()
    
    # Determine mode
    if args.real:
        mock_mode = False
        logger.info("🔴 PRODUCTION MODE: Using real DeepSeek API")
        logger.info("💰 API costs will be incurred - make sure you have sufficient credits")
    else:
        mock_mode = True
        logger.info("🟢 MOCK MODE: Using mock API for testing")
    
    logger.info(f"📂 Input: {args.input}")
    logger.info(f"📁 Output: {args.output}")
    logger.info(f"👥 Workers: {args.workers}")
    
    if args.force:
        logger.info("⚠️  FORCE MODE: Override existing locks")
    
    try:
        # Initialize pipeline
        pipeline = ParallelDeepSeekLabelingPipeline(
            mock_mode=mock_mode, 
            max_workers=args.workers
        )
        
        logger.info("🚀 Starting parallel labeling...")
        logger.info("💡 Press Ctrl+C to STOP and SAVE progress")
        start_time = time.time()
        
        # Run pipeline langsung tanpa demo
        report = pipeline.run_pipeline(args.input, args.output)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Labeling completed in {total_time:.2f}s")
        
        # Display simple report
        logger.info("\n📊 RESULTS:")
        for section, content in report.items():
            logger.info(f"  {section}: {content}")
        
        # Verify output
        if Path(args.output).exists():
            result_df = pd.read_csv(args.output)
            logger.info(f"\n✅ Output saved: {len(result_df)} samples in {args.output}")
            
            # Show sample results
            logger.info("\n📋 Sample Results:")
            for i, row in result_df.head(3).iterrows():
                logger.info(f"  Text: {row['text'][:50]}...")
                logger.info(f"  Label: {row['final_label']} (confidence: {row.get('confidence', 'N/A')})")
                logger.info("")
        else:
            logger.error(f"❌ Output file not created: {args.output}")
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Process STOPPED by user (Ctrl+C)")
        logger.info("💾 Progress SAVED - run the same command to resume")
        sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Labeling failed: {e}")
        raise

if __name__ == "__main__":
    main()