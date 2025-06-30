"""Pipeline labeling dengan mekanisme checkpoint dan resume untuk persistence.

Script ini mengimplementasikan sistem checkpoint yang memungkinkan:
- Resume otomatis jika proses terputus (lampu mati, pause, dll)
- Menyimpan progress secara berkala
- Skip data yang sudah dilabeli sebelumnya
- Recovery dari checkpoint terakhir

Usage:
    python persistent_labeling_pipeline.py --input raw-dataset.csv --output labeled-dataset.csv
    python persistent_labeling_pipeline.py --resume  # Melanjutkan dari checkpoint terakhir
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add src to path untuk import
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Settings
from utils.deepseek_labeling import DeepSeekLabelingStrategy
from utils.deepseek_client import create_deepseek_client, LabelingResult
from utils.logger import setup_logger
from utils.cost_optimizer import CostOptimizer

# Setup logging
logger = setup_logger("persistent_pipeline")

class CheckpointManager:
    """Manager untuk checkpoint dan resume functionality."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory untuk menyimpan checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, 
                       checkpoint_id: str,
                       processed_indices: List[int],
                       results: List[Dict],
                       metadata: Dict) -> str:
        """Save checkpoint ke file.
        
        Args:
            checkpoint_id: Unique ID untuk checkpoint
            processed_indices: List index yang sudah diproses
            results: List hasil labeling
            metadata: Metadata tambahan (input_file, output_file, dll)
            
        Returns:
            Path ke checkpoint file
        """
        checkpoint_data = {
            'checkpoint_id': checkpoint_id,
            'timestamp': time.time(),
            'processed_indices': processed_indices,
            'results': results,
            'metadata': metadata
        }
        
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Checkpoint saved: {checkpoint_file}")
        return str(checkpoint_file)
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict]:
        """Load checkpoint dari file.
        
        Args:
            checkpoint_id: ID checkpoint yang akan di-load
            
        Returns:
            Checkpoint data atau None jika tidak ditemukan
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        if not checkpoint_file.exists():
            return None
            
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            logger.info(f"Checkpoint loaded: {checkpoint_file}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_file}: {e}")
            return None
    
    def list_checkpoints(self) -> List[str]:
        """List semua checkpoint yang tersedia.
        
        Returns:
            List checkpoint IDs
        """
        checkpoint_files = list(self.checkpoint_dir.glob("*.json"))
        return [f.stem for f in checkpoint_files]
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get checkpoint terbaru berdasarkan timestamp.
        
        Returns:
            Checkpoint ID terbaru atau None
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
            
        latest_checkpoint = None
        latest_timestamp = 0
        
        for checkpoint_id in checkpoints:
            checkpoint_data = self.load_checkpoint(checkpoint_id)
            if checkpoint_data and checkpoint_data.get('timestamp', 0) > latest_timestamp:
                latest_timestamp = checkpoint_data['timestamp']
                latest_checkpoint = checkpoint_id
                
        return latest_checkpoint


class PersistentLabelingPipeline:
    """Pipeline labeling dengan checkpoint dan resume capability."""
    
    def __init__(self, 
                 mock_mode: bool = False, 
                 settings: Settings = None,
                 checkpoint_interval: int = 10,
                 cost_strategy: str = "warn_expensive"):
        """Initialize pipeline.
        
        Args:
            mock_mode: Jika True, menggunakan mock client
            settings: Instance Settings
            checkpoint_interval: Interval untuk save checkpoint (per batch)
            cost_strategy: Strategi optimasi biaya ('discount_only', 'always', 'warn_expensive')
        """
        self.settings = settings or Settings()
        self.mock_mode = mock_mode
        self.strategy = DeepSeekLabelingStrategy()
        self.client = create_deepseek_client(mock=mock_mode, settings=self.settings)
        self.checkpoint_manager = CheckpointManager()
        self.checkpoint_interval = checkpoint_interval
        self.cost_strategy = cost_strategy
        
        # Setup cost optimizer
        self.cost_optimizer = CostOptimizer()
        
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
    
    def process_with_checkpoints(self, 
                                df: pd.DataFrame,
                                checkpoint_id: str,
                                output_file: str,
                                resume_data: Optional[Dict] = None) -> Dict:
        """Process dataset dengan checkpoint mechanism.
        
        Args:
            df: DataFrame input
            checkpoint_id: ID untuk checkpoint
            output_file: Path output file
            resume_data: Data untuk resume (jika ada)
            
        Returns:
            Dictionary hasil processing
        """
        start_time = time.time()
        
        # Initialize atau resume
        if resume_data:
            processed_indices = set(resume_data['processed_indices'])
            results = resume_data['results']
            logger.info(f"Resuming from checkpoint: {len(processed_indices)} samples already processed")
        else:
            processed_indices = set()
            results = []
            logger.info("Starting fresh processing")
        
        # Split data berdasarkan label
        positive_data, negative_data = self.strategy.filter_data_by_initial_label(
            df, text_column='text', label_column='label'
        )
        
        # Process positive data (rule-based) - cepat, tidak perlu checkpoint
        positive_results = self._process_positive_data_fast(positive_data, processed_indices)
        results.extend(positive_results)
        
        # Process negative data dengan checkpoint
        negative_results = self._process_negative_data_with_checkpoints(
            negative_data, processed_indices, checkpoint_id, output_file
        )
        results.extend(negative_results)
        
        # Final save
        self._save_results(results, output_file)
        
        # Generate report
        processing_time = time.time() - start_time
        final_df = pd.DataFrame(results)
        report = self._generate_report(final_df, processing_time)
        
        return report
    
    def _process_positive_data_fast(self, 
                                   positive_df: pd.DataFrame, 
                                   processed_indices: set) -> List[Dict]:
        """Process positive data dengan rule-based (cepat).
        
        Args:
            positive_df: DataFrame positive data
            processed_indices: Set index yang sudah diproses
            
        Returns:
            List hasil processing
        """
        results = []
        
        for idx, row in positive_df.iterrows():
            if idx in processed_indices:
                continue
                
            results.append({
                'original_index': idx,
                'text': row['text'],
                'label': row['label'],
                'final_label': self.label_mapping[0],
                'confidence_score': 1.0,
                'response_time': 0.0,
                'labeling_method': 'rule_based_positive',
                'error': None
            })
            processed_indices.add(idx)
        
        if results:
            logger.info(f"Processed {len(results)} positive samples (rule-based)")
        
        return results
    
    def _process_negative_data_with_checkpoints(self, 
                                              negative_df: pd.DataFrame,
                                              processed_indices: set,
                                              checkpoint_id: str,
                                              output_file: str) -> List[Dict]:
        """Process negative data dengan checkpoint mechanism dan monitoring biaya.
        
        Args:
            negative_df: DataFrame negative data
            processed_indices: Set index yang sudah diproses
            checkpoint_id: ID checkpoint
            output_file: Path output file
            
        Returns:
            List hasil processing
        """
        results = []
        batch_size = self.settings.deepseek_batch_size
        batch_count = 0
        
        # Filter data yang belum diproses
        remaining_df = negative_df[~negative_df.index.isin(processed_indices)]
        
        if remaining_df.empty:
            logger.info("All negative data already processed")
            return results
        
        logger.info(f"Processing {len(remaining_df)} remaining negative samples")
        
        # Cek strategi biaya sebelum memulai
        should_process, reason = self.cost_optimizer.should_process_now(self.cost_strategy)
        logger.info(f"💰 {reason}")
        
        if not should_process and self.cost_strategy == "discount_only":
            logger.info("⏳ Menunggu periode diskon...")
            if not self.mock_mode:
                self.cost_optimizer.wait_for_discount_period()
        
        try:
            for i in range(0, len(remaining_df), batch_size):
                batch_df = remaining_df.iloc[i:i+batch_size]
                batch_texts = batch_df['text'].tolist()
                
                batch_num = i//batch_size + 1
                total_batches = (len(remaining_df)-1)//batch_size + 1
                logger.info(f"Processing batch {batch_num}/{total_batches}")
                
                # Label batch dengan DeepSeek
                batch_results = self.client.label_batch(batch_texts)
                
                # Convert results
                batch_processed = []
                for idx, (original_idx, result) in enumerate(zip(batch_df.index, batch_results)):
                    result_dict = {
                        'original_index': original_idx,
                        'text': result.text,
                        'label': 'negative',
                        'final_label': self.label_mapping[result.label_id],
                        'confidence_score': result.confidence,
                        'response_time': result.response_time,
                        'labeling_method': 'deepseek_api',
                        'error': result.error
                    }
                    batch_processed.append(result_dict)
                    processed_indices.add(original_idx)
                
                results.extend(batch_processed)
                batch_count += 1
                
                # Save checkpoint setiap interval
                if batch_count % self.checkpoint_interval == 0:
                    all_results = self._load_existing_results(output_file) + results
                    self.checkpoint_manager.save_checkpoint(
                        checkpoint_id=checkpoint_id,
                        processed_indices=list(processed_indices),
                        results=all_results,
                        metadata={
                            'output_file': output_file,
                            'batch_count': batch_count,
                            'total_processed': len(processed_indices)
                        }
                    )
                    
                    # Save intermediate results
                    self._save_results(all_results, output_file)
                    logger.info(f"Checkpoint saved after batch {batch_num}")
                
                # Progress info
                if batch_results:
                    avg_confidence = sum(r.confidence for r in batch_results) / len(batch_results)
                    logger.info(f"Batch {batch_num} completed. Avg confidence: {avg_confidence:.3f}")
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user. Saving checkpoint...")
            all_results = self._load_existing_results(output_file) + results
            self.checkpoint_manager.save_checkpoint(
                checkpoint_id=checkpoint_id,
                processed_indices=list(processed_indices),
                results=all_results,
                metadata={
                    'output_file': output_file,
                    'interrupted': True,
                    'total_processed': len(processed_indices)
                }
            )
            self._save_results(all_results, output_file)
            raise
        
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            # Save emergency checkpoint
            all_results = self._load_existing_results(output_file) + results
            emergency_checkpoint = f"{checkpoint_id}_emergency_{int(time.time())}"
            self.checkpoint_manager.save_checkpoint(
                checkpoint_id=emergency_checkpoint,
                processed_indices=list(processed_indices),
                results=all_results,
                metadata={
                    'output_file': output_file,
                    'error': str(e),
                    'total_processed': len(processed_indices)
                }
            )
            raise
        
        return results
    
    def _load_existing_results(self, output_file: str) -> List[Dict]:
        """Load existing results dari output file jika ada.
        
        Args:
            output_file: Path ke output file
            
        Returns:
            List existing results
        """
        if not Path(output_file).exists():
            return []
        
        try:
            df = pd.read_csv(output_file)
            return df.to_dict('records')
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
            return []
    
    def _save_results(self, results: List[Dict], output_file: str) -> None:
        """Save results ke file.
        
        Args:
            results: List hasil labeling
            output_file: Path output file
        """
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save dengan kolom yang diperlukan
        output_columns = ['text', 'label', 'final_label', 'confidence_score', 
                         'labeling_method', 'response_time']
        
        # Filter kolom yang ada
        available_columns = [col for col in output_columns if col in df.columns]
        df[available_columns].to_csv(output_file, index=False)
        
        logger.info(f"Results saved to: {output_file} ({len(results)} samples)")
    
    def _generate_report(self, final_df: pd.DataFrame, processing_time: float) -> Dict:
        """Generate laporan hasil.
        
        Args:
            final_df: DataFrame hasil final
            processing_time: Waktu processing
            
        Returns:
            Dictionary laporan
        """
        if final_df.empty:
            return {}
        
        total_samples = len(final_df)
        label_distribution = final_df['final_label'].value_counts().to_dict()
        method_distribution = final_df['labeling_method'].value_counts().to_dict()
        
        avg_confidence = final_df['confidence_score'].mean()
        error_count = final_df['error'].notna().sum() if 'error' in final_df.columns else 0
        
        return {
            'summary': {
                'total_samples': total_samples,
                'processing_time_seconds': round(processing_time, 2),
                'samples_per_second': round(total_samples / processing_time, 2) if processing_time > 0 else 0
            },
            'label_distribution': label_distribution,
            'labeling_methods': method_distribution,
            'confidence_analysis': {
                'average_confidence': round(avg_confidence, 3)
            },
            'error_analysis': {
                'total_errors': error_count,
                'success_rate_percentage': round((total_samples - error_count) / total_samples * 100, 2)
            }
        }
    
    def run_pipeline(self, 
                    input_file: str, 
                    output_file: str, 
                    resume: bool = False) -> Dict:
        """Run pipeline dengan checkpoint support.
        
        Args:
            input_file: Path input file
            output_file: Path output file
            resume: Jika True, coba resume dari checkpoint
            
        Returns:
            Dictionary laporan hasil
        """
        # Generate checkpoint ID berdasarkan input/output files
        checkpoint_id = f"labeling_{Path(input_file).stem}_{Path(output_file).stem}"
        
        logger.info(f"Starting persistent labeling pipeline")
        logger.info(f"Input: {input_file}")
        logger.info(f"Output: {output_file}")
        logger.info(f"Checkpoint ID: {checkpoint_id}")
        logger.info(f"Resume mode: {resume}")
        
        # Load dataset
        df = self.load_dataset(input_file)
        
        # Check for resume
        resume_data = None
        if resume:
            resume_data = self.checkpoint_manager.load_checkpoint(checkpoint_id)
            if resume_data:
                logger.info(f"Found checkpoint with {len(resume_data['processed_indices'])} processed samples")
            else:
                logger.info("No checkpoint found, starting fresh")
        
        # Process dengan checkpoints
        report = self.process_with_checkpoints(df, checkpoint_id, output_file, resume_data)
        
        logger.info("Pipeline completed successfully")
        return report


def print_report(report: Dict) -> None:
    """Print laporan hasil.
    
    Args:
        report: Dictionary laporan
    """
    print("\n" + "="*60)
    print("        LAPORAN HASIL LABELING (PERSISTENT)")
    print("="*60)
    
    summary = report.get('summary', {})
    print(f"\n📊 RINGKASAN:")
    print(f"   Total sampel: {summary.get('total_samples', 0):,}")
    print(f"   Waktu proses: {summary.get('processing_time_seconds', 0):.2f} detik")
    print(f"   Kecepatan: {summary.get('samples_per_second', 0):.2f} sampel/detik")
    
    label_dist = report.get('label_distribution', {})
    print(f"\n🏷️  DISTRIBUSI LABEL:")
    for label, count in label_dist.items():
        percentage = (count / summary.get('total_samples', 1)) * 100
        print(f"   {label}: {count:,} ({percentage:.1f}%)")
    
    methods = report.get('labeling_methods', {})
    print(f"\n⚙️  METODE LABELING:")
    for method, count in methods.items():
        print(f"   {method}: {count:,}")
    
    confidence = report.get('confidence_analysis', {})
    print(f"\n🎯 CONFIDENCE:")
    print(f"   Rata-rata: {confidence.get('average_confidence', 0):.3f}")
    
    errors = report.get('error_analysis', {})
    print(f"\n✅ SUCCESS RATE:")
    print(f"   {errors.get('success_rate_percentage', 0):.1f}%")
    
    print("\n" + "="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Persistent DeepSeek Labeling Pipeline dengan Checkpoint Support"
    )
    parser.add_argument(
        "--input", 
        default="src/data_collection/raw-dataset.csv",
        help="Path ke file input dataset"
    )
    parser.add_argument(
        "--output", 
        default="labeled-dataset-persistent.csv",
        help="Path ke file output hasil labeling"
    )
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume dari checkpoint terakhir"
    )
    parser.add_argument(
        "--mock", 
        action="store_true",
        help="Gunakan mock client untuk testing"
    )
    parser.add_argument(
        "--checkpoint-interval", 
        type=int,
        default=10,
        help="Interval save checkpoint (per batch, default: 10)"
    )
    parser.add_argument(
        "--list-checkpoints", 
        action="store_true",
        help="List semua checkpoint yang tersedia"
    )
    
    args = parser.parse_args()
    
    # List checkpoints jika diminta
    if args.list_checkpoints:
        checkpoint_manager = CheckpointManager()
        checkpoints = checkpoint_manager.list_checkpoints()
        if checkpoints:
            print("\nCheckpoint yang tersedia:")
            for checkpoint_id in checkpoints:
                checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_id)
                if checkpoint_data:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', 
                                           time.localtime(checkpoint_data['timestamp']))
                    processed = len(checkpoint_data.get('processed_indices', []))
                    print(f"  {checkpoint_id}: {processed} samples processed ({timestamp})")
        else:
            print("\nTidak ada checkpoint yang tersedia.")
        return
    
    # Initialize pipeline
    settings = Settings()
    pipeline = PersistentLabelingPipeline(
        mock_mode=args.mock, 
        settings=settings,
        checkpoint_interval=args.checkpoint_interval
    )
    
    try:
        # Run pipeline
        report = pipeline.run_pipeline(args.input, args.output, args.resume)
        
        # Print report
        print_report(report)
        
        # Save report
        report_file = args.output.replace('.csv', '_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to: {report_file}")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user. Progress saved in checkpoint.")
        print("\n⚠️  Pipeline dihentikan. Progress sudah disimpan di checkpoint.")
        print("   Gunakan --resume untuk melanjutkan proses.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline gagal: {e}")
        print("   Cek checkpoint untuk recovery.")
        sys.exit(1)


if __name__ == "__main__":
    main()