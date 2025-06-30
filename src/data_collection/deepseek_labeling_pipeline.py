#!/usr/bin/env python3
"""Pipeline utama untuk labeling data menggunakan strategi cost-efficient dengan DeepSeek API.

Script ini mengimplementasikan strategi pembagian positif/negatif untuk menghemat biaya:
- Data 'positive' → langsung dilabeli sebagai "Bukan Ujaran Kebencian"
- Data 'negative' → dilabeli detail menggunakan DeepSeek API

Usage:
    python deepseek_labeling_pipeline.py --input raw-dataset.csv --output labeled-dataset.csv
    python deepseek_labeling_pipeline.py --mock  # Untuk testing tanpa API
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add src to path untuk import
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Settings
from utils.deepseek_labeling import DeepSeekLabelingStrategy
from utils.deepseek_client import create_deepseek_client, LabelingResult
from utils.logger import setup_logger

# Setup logging
logger = setup_logger("deepseek_pipeline")

class DeepSeekLabelingPipeline:
    """Pipeline utama untuk labeling dengan DeepSeek API."""
    
    def __init__(self, mock_mode: bool = False, settings: Settings = None):
        """Inisialisasi pipeline.
        
        Args:
            mock_mode: Jika True, menggunakan mock client untuk testing
            settings: Instance Settings
        """
        self.settings = settings or Settings()
        self.mock_mode = mock_mode
        self.strategy = DeepSeekLabelingStrategy()
        self.client = create_deepseek_client(mock=mock_mode, settings=self.settings)
        
        # Mapping kategori
        self.label_mapping = {
            0: "Bukan Ujaran Kebencian",
            1: "Ujaran Kebencian - Ringan",
            2: "Ujaran Kebencian - Sedang", 
            3: "Ujaran Kebencian - Berat"
        }
    
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
    
    def process_negative_data(self, negative_df: pd.DataFrame) -> pd.DataFrame:
        """Proses data negative menggunakan DeepSeek API.
        
        Args:
            negative_df: DataFrame dengan data berlabel 'negative'
            
        Returns:
            DataFrame dengan hasil labeling detail
        """
        if negative_df.empty:
            logger.info("No negative data to process")
            return pd.DataFrame()
        
        logger.info(f"Processing {len(negative_df)} negative samples with DeepSeek API")
        
        # Prepare batches
        batch_size = self.settings.deepseek_batch_size
        results = []
        
        for i in range(0, len(negative_df), batch_size):
            batch_df = negative_df.iloc[i:i+batch_size]
            batch_texts = batch_df['text'].tolist()
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(negative_df)-1)//batch_size + 1}")
            
            # Label batch dengan DeepSeek
            batch_results = self.client.label_batch(batch_texts)
            
            # Convert results ke DataFrame format
            for idx, (original_idx, result) in enumerate(zip(batch_df.index, batch_results)):
                results.append({
                    'original_index': original_idx,
                    'text': result.text,
                    'label': 'negative',  # Original label
                    'final_label': self.label_mapping[result.label_id],
                    'confidence_score': result.confidence,
                    'response_time': result.response_time,
                    'labeling_method': 'deepseek_api',
                    'error': result.error
                })
            
            # Print progress
            if batch_results:
                avg_confidence = sum(r.confidence for r in batch_results) / len(batch_results)
                logger.info(f"Batch completed. Average confidence: {avg_confidence:.3f}")
        
        # Convert ke DataFrame
        result_df = pd.DataFrame(results)
        
        # Log statistics
        if not result_df.empty:
            label_dist = result_df['final_label'].value_counts()
            logger.info(f"Negative data labeling completed:")
            for label, count in label_dist.items():
                logger.info(f"  {label}: {count} samples")
            
            # Log errors
            error_count = result_df['error'].notna().sum()
            if error_count > 0:
                logger.warning(f"Encountered {error_count} errors during labeling")
        
        return result_df
    
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
        api_samples = len(final_df[final_df['labeling_method'] == 'deepseek_api'])
        rule_samples = len(final_df[final_df['labeling_method'] == 'rule_based_positive'])
        
        if api_samples > 0:
            avg_response_time = final_df[final_df['labeling_method'] == 'deepseek_api']['response_time'].mean()
            total_api_time = final_df[final_df['labeling_method'] == 'deepseek_api']['response_time'].sum()
        else:
            avg_response_time = 0
            total_api_time = 0
        
        # Cost analysis
        cost_saving_percentage = (rule_samples / total_samples * 100) if total_samples > 0 else 0
        
        # Error analysis
        error_count = final_df['error'].notna().sum()
        error_rate = (error_count / total_samples * 100) if total_samples > 0 else 0
        
        report = {
            'summary': {
                'total_samples': total_samples,
                'processing_time_seconds': round(processing_time, 2),
                'samples_per_second': round(total_samples / processing_time, 2) if processing_time > 0 else 0
            },
            'label_distribution': label_distribution,
            'labeling_methods': {
                'rule_based_positive': rule_samples,
                'deepseek_api': api_samples,
                'method_distribution_percentage': {
                    method: round(count / total_samples * 100, 2) 
                    for method, count in method_distribution.items()
                }
            },
            'confidence_analysis': {
                'average_confidence': round(avg_confidence, 3),
                'min_confidence': round(min_confidence, 3),
                'max_confidence': round(max_confidence, 3),
                'low_confidence_samples': low_confidence_count,
                'low_confidence_percentage': round(low_confidence_count / total_samples * 100, 2)
            },
            'performance_metrics': {
                'average_api_response_time': round(avg_response_time, 3),
                'total_api_time': round(total_api_time, 2),
                'api_efficiency': round(api_samples / total_api_time, 2) if total_api_time > 0 else 0
            },
            'cost_analysis': {
                'samples_processed_by_rule': rule_samples,
                'samples_processed_by_api': api_samples,
                'estimated_cost_saving_percentage': round(cost_saving_percentage, 2),
                'api_usage_ratio': round(api_samples / total_samples, 3) if total_samples > 0 else 0
            },
            'error_analysis': {
                'total_errors': error_count,
                'error_rate_percentage': round(error_rate, 2),
                'success_rate_percentage': round(100 - error_rate, 2)
            }
        }
        
        return report
    
    def run_pipeline(self, input_file: str, output_file: str) -> Dict:
        """Menjalankan pipeline labeling lengkap.
        
        Args:
            input_file: Path ke file input dataset
            output_file: Path ke file output hasil labeling
            
        Returns:
            Dictionary berisi laporan hasil
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting DeepSeek labeling pipeline")
        logger.info(f"Input: {input_file}")
        logger.info(f"Output: {output_file}")
        logger.info(f"Mock mode: {self.mock_mode}")
        
        try:
            # 1. Load dataset
            df = self.load_dataset(input_file)
            
            # 2. Split data berdasarkan label awal
            positive_data, negative_data = self.strategy.filter_data_by_initial_label(
                df, text_column='text', label_column='label'
            )
            
            # 3. Process positive data (rule-based)
            positive_result = self.prepare_positive_data(positive_data)
            
            # 4. Process negative data (DeepSeek API)
            negative_result = self.process_negative_data(negative_data)
            
            # 5. Combine results
            if not positive_result.empty and not negative_result.empty:
                final_df = pd.concat([positive_result, negative_result], ignore_index=True)
            elif not positive_result.empty:
                final_df = positive_result
            elif not negative_result.empty:
                final_df = negative_result
            else:
                final_df = pd.DataFrame()
            
            # 6. Save results
            if not final_df.empty:
                # Ensure output directory exists
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save dengan kolom yang diperlukan
                output_columns = ['text', 'label', 'final_label', 'confidence_score', 
                                'labeling_method', 'response_time']
                final_df[output_columns].to_csv(output_file, index=False)
                logger.info(f"Results saved to: {output_file}")
            
            # 7. Generate report
            processing_time = time.time() - start_time
            report = self.generate_comprehensive_report(final_df, processing_time)
            
            logger.info("Pipeline completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def print_report(report: Dict) -> None:
    """Print laporan hasil labeling dalam format yang mudah dibaca.
    
    Args:
        report: Dictionary berisi laporan hasil
    """
    print("\n" + "="*60)
    print("           LAPORAN HASIL LABELING DEEPSEEK")
    print("="*60)
    
    # Summary
    summary = report.get('summary', {})
    print(f"\n📊 RINGKASAN:")
    print(f"   Total sampel: {summary.get('total_samples', 0):,}")
    print(f"   Waktu proses: {summary.get('processing_time_seconds', 0):.2f} detik")
    print(f"   Kecepatan: {summary.get('samples_per_second', 0):.2f} sampel/detik")
    
    # Label distribution
    label_dist = report.get('label_distribution', {})
    print(f"\n🏷️  DISTRIBUSI LABEL:")
    for label, count in label_dist.items():
        percentage = (count / summary.get('total_samples', 1)) * 100
        print(f"   {label}: {count:,} ({percentage:.1f}%)")
    
    # Cost analysis
    cost_analysis = report.get('cost_analysis', {})
    print(f"\n💰 ANALISIS BIAYA:")
    print(f"   Sampel rule-based: {cost_analysis.get('samples_processed_by_rule', 0):,}")
    print(f"   Sampel API: {cost_analysis.get('samples_processed_by_api', 0):,}")
    print(f"   Penghematan biaya: {cost_analysis.get('estimated_cost_saving_percentage', 0):.1f}%")
    
    # Performance
    performance = report.get('performance_metrics', {})
    print(f"\n⚡ PERFORMA API:")
    print(f"   Rata-rata response time: {performance.get('average_api_response_time', 0):.3f} detik")
    print(f"   Total waktu API: {performance.get('total_api_time', 0):.2f} detik")
    
    # Confidence
    confidence = report.get('confidence_analysis', {})
    print(f"\n🎯 ANALISIS CONFIDENCE:")
    print(f"   Rata-rata confidence: {confidence.get('average_confidence', 0):.3f}")
    print(f"   Sampel confidence rendah (<0.6): {confidence.get('low_confidence_samples', 0):,} ({confidence.get('low_confidence_percentage', 0):.1f}%)")
    
    # Errors
    errors = report.get('error_analysis', {})
    print(f"\n❌ ANALISIS ERROR:")
    print(f"   Total error: {errors.get('total_errors', 0):,}")
    print(f"   Success rate: {errors.get('success_rate_percentage', 0):.1f}%")
    
    print("\n" + "="*60)


def main():
    """Main function untuk menjalankan pipeline."""
    parser = argparse.ArgumentParser(
        description="DeepSeek Labeling Pipeline untuk Ujaran Kebencian Bahasa Jawa"
    )
    parser.add_argument(
        "--input", 
        default="raw-dataset.csv",
        help="Path ke file input dataset (default: raw-dataset.csv)"
    )
    parser.add_argument(
        "--output", 
        default="labeled-dataset.csv",
        help="Path ke file output hasil labeling (default: labeled-dataset.csv)"
    )
    parser.add_argument(
        "--mock", 
        action="store_true",
        help="Gunakan mock client untuk testing (tidak memanggil API sebenarnya)"
    )
    parser.add_argument(
        "--sample", 
        type=int,
        help="Proses hanya N sampel pertama (untuk testing)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    settings = Settings()
    pipeline = DeepSeekLabelingPipeline(mock_mode=args.mock, settings=settings)
    
    # Jika sample mode, buat subset dataset
    if args.sample:
        logger.info(f"Sample mode: processing only first {args.sample} samples")
        df = pipeline.load_dataset(args.input)
        sample_df = df.head(args.sample)
        sample_input = f"sample_{args.sample}_{args.input}"
        sample_df.to_csv(sample_input, index=False)
        args.input = sample_input
    
    try:
        # Run pipeline
        report = pipeline.run_pipeline(args.input, args.output)
        
        # Print report
        print_report(report)
        
        # Save report
        report_file = args.output.replace('.csv', '_report.json')
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to: {report_file}")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()