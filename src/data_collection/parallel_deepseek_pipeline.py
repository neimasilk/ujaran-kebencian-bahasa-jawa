#!/usr/bin/env python3
"""
Parallel DeepSeek Labeling Pipeline untuk labeling data secara paralel.

Pipeline ini mengimplementasikan versi paralel dari deepseek_labeling_pipeline.py dengan:
- Concurrent processing menggunakan asyncio dan ThreadPoolExecutor
- Progress tracking real-time
- Better error handling dan recovery
- Optimized rate limiting untuk multiple workers
- Comprehensive performance metrics

Usage:
    python parallel_deepseek_pipeline.py --input raw-dataset.csv --output labeled-dataset.csv --workers 5
    python parallel_deepseek_pipeline.py --mock --workers 10  # Testing dengan mock client
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# Add src to path untuk import
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Settings
from utils.deepseek_labeling import DeepSeekLabelingStrategy
from utils.deepseek_client_parallel import create_parallel_deepseek_client, LabelingResult
from utils.logger import setup_logger

# Setup logging
logger = setup_logger("parallel_deepseek_pipeline")

class ProgressTracker:
    """Class untuk tracking progress dengan tqdm."""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.completed = 0
        self.errors = 0
        self.start_time = time.time()
        self.pbar = tqdm(total=total, desc=desc, unit="texts")
    
    def update(self, completed: int, total: int, result: LabelingResult):
        """Update progress bar."""
        if result.error:
            self.errors += 1
        
        # Update progress bar
        increment = completed - self.completed
        if increment > 0:
            self.pbar.update(increment)
            self.completed = completed
            
            # Update description dengan stats
            elapsed = time.time() - self.start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            
            self.pbar.set_postfix({
                'rate': f'{rate:.1f}/s',
                'errors': self.errors,
                'eta': f'{eta:.0f}s'
            })
    
    def close(self):
        """Close progress bar."""
        self.pbar.close()

class ParallelDeepSeekLabelingPipeline:
    """Pipeline paralel untuk labeling dengan DeepSeek API."""
    
    def __init__(self, mock_mode: bool = False, settings: Settings = None, max_workers: int = 5):
        """Initialize parallel pipeline.
        
        Args:
            mock_mode: Jika True, menggunakan mock client untuk testing
            settings: Instance Settings
            max_workers: Maximum number of concurrent workers
        """
        self.settings = settings or Settings()
        self.mock_mode = mock_mode
        self.max_workers = max_workers
        self.strategy = DeepSeekLabelingStrategy()
        self.client = create_parallel_deepseek_client(
            mock=mock_mode, 
            settings=self.settings,
            max_workers=max_workers
        )
        
        # Mapping kategori
        self.label_mapping = {
            0: "Bukan Ujaran Kebencian",
            1: "Ujaran Kebencian - Ringan",
            2: "Ujaran Kebencian - Sedang", 
            3: "Ujaran Kebencian - Berat"
        }
        
        logger.info(f"Initialized parallel pipeline with {max_workers} workers (mock={mock_mode})")
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load dataset dari file CSV.
        
        Args:
            file_path: Path ke file dataset
            
        Returns:
            DataFrame dengan kolom 'text' dan 'label'
        """
        try:
            # Coba load dengan header
            df = pd.read_csv(file_path)
            
            # Jika tidak ada header yang sesuai, assume format: text,label
            if df.columns.tolist() != ['text', 'label']:
                df = pd.read_csv(file_path, names=['text', 'label'])
            
            logger.info(f"Loaded dataset: {len(df)} samples")
            logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset from {file_path}: {e}")
            raise
    
    async def process_negative_data_parallel(self, negative_df: pd.DataFrame) -> pd.DataFrame:
        """Process data negative menggunakan parallel DeepSeek API.
        
        Args:
            negative_df: DataFrame dengan data berlabel 'negative'
            
        Returns:
            DataFrame dengan hasil labeling detail
        """
        if negative_df.empty:
            logger.info("No negative data to process")
            return pd.DataFrame()
        
        logger.info(f"Processing {len(negative_df)} negative samples with {self.max_workers} parallel workers")
        
        # Prepare texts untuk parallel processing
        texts = negative_df['text'].tolist()
        
        # Setup progress tracking
        progress_tracker = ProgressTracker(len(texts), "Labeling texts")
        
        try:
            # Process all texts in parallel
            start_time = time.time()
            results = await self.client.label_batch_parallel(
                texts, 
                progress_callback=progress_tracker.update
            )
            processing_time = time.time() - start_time
            
            progress_tracker.close()
            
            # Convert results ke DataFrame format
            result_data = []
            for idx, (original_idx, result) in enumerate(zip(negative_df.index, results)):
                result_data.append({
                    'original_index': original_idx,
                    'text': result.text,
                    'label': 'negative',  # Original label
                    'final_label': self.label_mapping[result.label_id],
                    'confidence_score': result.confidence,
                    'response_time': result.response_time,
                    'labeling_method': 'deepseek_api_parallel',
                    'error': result.error
                })
            
            # Convert ke DataFrame
            result_df = pd.DataFrame(result_data)
            
            # Log statistics
            if not result_df.empty:
                label_dist = result_df['final_label'].value_counts()
                error_count = result_df['error'].notna().sum()
                avg_confidence = result_df['confidence_score'].mean()
                avg_response_time = result_df['response_time'].mean()
                total_time = result_df['response_time'].sum()
                
                logger.info(f"Parallel labeling completed in {processing_time:.2f}s:")
                logger.info(f"  Total API time: {total_time:.2f}s (speedup: {total_time/processing_time:.1f}x)")
                logger.info(f"  Average confidence: {avg_confidence:.3f}")
                logger.info(f"  Average response time: {avg_response_time:.3f}s")
                logger.info(f"  Errors: {error_count}/{len(result_df)} ({error_count/len(result_df)*100:.1f}%)")
                
                for label, count in label_dist.items():
                    logger.info(f"  {label}: {count} samples")
            
            return result_df
            
        except Exception as e:
            progress_tracker.close()
            logger.error(f"Error in parallel processing: {e}")
            raise
    
    def process_negative_data_parallel_sync(self, negative_df: pd.DataFrame) -> pd.DataFrame:
        """Synchronous wrapper untuk parallel processing.
        
        Args:
            negative_df: DataFrame dengan data berlabel 'negative'
            
        Returns:
            DataFrame dengan hasil labeling detail
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.process_negative_data_parallel(negative_df))
        finally:
            loop.close()
    
    def prepare_positive_data(self, positive_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data positive dengan label final.
        
        Args:
            positive_df: DataFrame dengan data berlabel 'positive'
            
        Returns:
            DataFrame dengan label final
        """
        if positive_df.empty:
            return pd.DataFrame()
        
        result_df = positive_df.copy()
        result_df['final_label'] = self.label_mapping[0]  # "Bukan Ujaran Kebencian"
        result_df['confidence_score'] = 1.0
        result_df['response_time'] = 0.0
        result_df['labeling_method'] = 'rule_based_positive'
        result_df['error'] = None
        
        logger.info(f"Prepared {len(result_df)} positive samples as 'Bukan Ujaran Kebencian'")
        return result_df
    
    def run_pipeline(self, input_file: str, output_file: str) -> Dict:
        """Run complete parallel labeling pipeline.
        
        Args:
            input_file: Path ke input dataset
            output_file: Path untuk output hasil labeling
            
        Returns:
            Dictionary berisi laporan hasil
        """
        start_time = time.time()
        
        try:
            # Load dataset
            logger.info(f"Loading dataset from {input_file}")
            df = self.load_dataset(input_file)
            
            # Split positive/negative
            positive_df = df[df['label'] == 'positive'].copy()
            negative_df = df[df['label'] == 'negative'].copy()
            
            logger.info(f"Split data: {len(positive_df)} positive, {len(negative_df)} negative")
            
            # Process positive data (rule-based)
            positive_results = self.prepare_positive_data(positive_df)
            
            # Process negative data (parallel API)
            negative_results = self.process_negative_data_parallel_sync(negative_df)
            
            # Combine results
            if not positive_results.empty and not negative_results.empty:
                final_df = pd.concat([positive_results, negative_results], ignore_index=True)
            elif not positive_results.empty:
                final_df = positive_results
            elif not negative_results.empty:
                final_df = negative_results
            else:
                final_df = pd.DataFrame()
            
            # Save results
            if not final_df.empty:
                final_df.to_csv(output_file, index=False)
                logger.info(f"Results saved to {output_file}")
            
            # Generate report
            processing_time = time.time() - start_time
            report = self.generate_comprehensive_report(final_df, processing_time)
            
            return report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def generate_comprehensive_report(self, final_df: pd.DataFrame, 
                                    processing_time: float) -> Dict:
        """Generate laporan komprehensif hasil labeling.
        
        Args:
            final_df: DataFrame hasil labeling final
            processing_time: Waktu total processing dalam detik
            
        Returns:
            Dictionary berisi laporan lengkap
        """
        if final_df.empty:
            return {}
        
        # Basic statistics
        total_samples = len(final_df)
        label_distribution = final_df['final_label'].value_counts().to_dict()
        method_distribution = final_df['labeling_method'].value_counts().to_dict()
        
        # Confidence statistics
        avg_confidence = final_df['confidence_score'].mean()
        min_confidence = final_df['confidence_score'].min()
        max_confidence = final_df['confidence_score'].max()
        low_confidence_count = len(final_df[final_df['confidence_score'] < 0.6])
        
        # Performance statistics
        api_samples = len(final_df[final_df['labeling_method'].str.contains('deepseek_api')])
        rule_samples = len(final_df[final_df['labeling_method'] == 'rule_based_positive'])
        
        if api_samples > 0:
            api_df = final_df[final_df['labeling_method'].str.contains('deepseek_api')]
            avg_response_time = api_df['response_time'].mean()
            total_api_time = api_df['response_time'].sum()
            error_count = api_df['error'].notna().sum()
        else:
            avg_response_time = 0
            total_api_time = 0
            error_count = 0
        
        # Calculate theoretical speedup
        if api_samples > 0 and self.max_workers > 1:
            sequential_time = total_api_time
            parallel_efficiency = processing_time / (sequential_time / self.max_workers) if sequential_time > 0 else 1
            theoretical_speedup = sequential_time / processing_time if processing_time > 0 else 1
        else:
            parallel_efficiency = 1
            theoretical_speedup = 1
        
        report = {
            "summary": {
                "total_samples": total_samples,
                "processing_time_seconds": round(processing_time, 2),
                "samples_per_second": round(total_samples / processing_time, 2) if processing_time > 0 else 0,
                "max_workers": self.max_workers,
                "mock_mode": self.mock_mode
            },
            "label_distribution": label_distribution,
            "method_distribution": method_distribution,
            "confidence_stats": {
                "average": round(avg_confidence, 3),
                "minimum": round(min_confidence, 3),
                "maximum": round(max_confidence, 3),
                "low_confidence_count": low_confidence_count,
                "low_confidence_percentage": round(low_confidence_count / total_samples * 100, 2)
            },
            "performance_stats": {
                "api_samples": api_samples,
                "rule_based_samples": rule_samples,
                "average_api_response_time": round(avg_response_time, 3),
                "total_api_time": round(total_api_time, 2),
                "error_count": error_count,
                "error_rate": round(error_count / api_samples * 100, 2) if api_samples > 0 else 0
            },
            "parallel_stats": {
                "theoretical_speedup": round(theoretical_speedup, 2),
                "parallel_efficiency": round(parallel_efficiency, 2),
                "time_saved_seconds": round(max(0, total_api_time - processing_time), 2)
            }
        }
        
        # Log report
        logger.info("=== PARALLEL LABELING REPORT ===")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Processing time: {processing_time:.2f}s")
        logger.info(f"Throughput: {total_samples/processing_time:.2f} samples/sec")
        logger.info(f"Theoretical speedup: {theoretical_speedup:.2f}x")
        logger.info(f"Time saved: {max(0, total_api_time - processing_time):.2f}s")
        
        return report

def main():
    """Main function untuk command line interface."""
    parser = argparse.ArgumentParser(description="Parallel DeepSeek Labeling Pipeline")
    parser.add_argument("--input", "-i", default="src/data_collection/raw-dataset.csv",
                       help="Input dataset file path")
    parser.add_argument("--output", "-o", default="hasil-labeling-parallel.csv",
                       help="Output labeled dataset file path")
    parser.add_argument("--workers", "-w", type=int, default=5,
                       help="Number of parallel workers (default: 5)")
    parser.add_argument("--mock", action="store_true",
                       help="Use mock client for testing")
    parser.add_argument("--sample", type=int, default=None,
                       help="Process only first N samples (for testing)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ParallelDeepSeekLabelingPipeline(
        mock_mode=args.mock,
        max_workers=args.workers
    )
    
    # Load and optionally sample data
    if args.sample:
        logger.info(f"Loading and sampling {args.sample} records for testing")
        df = pipeline.load_dataset(args.input)
        df = df.head(args.sample)
        
        # Save sampled data to temp file
        temp_input = f"temp_sample_{args.sample}.csv"
        df.to_csv(temp_input, index=False)
        input_file = temp_input
    else:
        input_file = args.input
    
    try:
        # Run pipeline
        report = pipeline.run_pipeline(input_file, args.output)
        
        # Print summary
        print("\n=== PIPELINE COMPLETED ===")
        print(f"Processed: {report['summary']['total_samples']} samples")
        print(f"Time: {report['summary']['processing_time_seconds']}s")
        print(f"Throughput: {report['summary']['samples_per_second']} samples/sec")
        print(f"Workers: {report['summary']['max_workers']}")
        print(f"Speedup: {report['parallel_stats']['theoretical_speedup']}x")
        print(f"Output: {args.output}")
        
    finally:
        # Cleanup temp file if created
        if args.sample and Path(temp_input).exists():
            Path(temp_input).unlink()

if __name__ == "__main__":
    main()