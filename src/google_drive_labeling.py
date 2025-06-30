#!/usr/bin/env python3
"""
Google Drive Labeling Pipeline
Labeling dengan DeepSeek API + Google Drive persistence

Features:
- Cancel/Resume anytime dengan checkpoint
- Optimized untuk jam promo DeepSeek
- Hasil otomatis tersimpan ke Google Drive
- Multi-device sync

Author: AI Assistant
Date: 2025-07-01
"""

import os
import sys
import time
import signal
import argparse
import logging
import json
from datetime import datetime, time
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config.settings import Settings
from utils.cloud_checkpoint_manager import CloudCheckpointManager
from data_collection.persistent_labeling_pipeline import PersistentLabelingPipeline
from utils.deepseek_client import create_deepseek_client
from utils.logger import setup_logger

class GoogleDriveLabelingPipeline:
    """
    Pipeline untuk labeling dengan Google Drive integration
    """
    
    def __init__(self, 
                 dataset_path: str,
                 output_name: str = "google-drive-labeling",
                 batch_size: int = 10,
                 checkpoint_interval: int = 50,
                 cloud_sync_interval: int = 100):
        """
        Initialize Google Drive Labeling Pipeline
        
        Args:
            dataset_path: Path ke dataset CSV
            output_name: Nama untuk output files
            batch_size: Ukuran batch untuk processing
            checkpoint_interval: Interval untuk save checkpoint lokal
            cloud_sync_interval: Interval untuk sync ke Google Drive
        """
        self.dataset_path = dataset_path
        self.output_name = output_name
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.cloud_sync_interval = cloud_sync_interval
        
        # Generate checkpoint ID yang konsisten
        self.checkpoint_id = f"labeling_{Path(dataset_path).stem}_{output_name}"
        
        # Setup settings
        self.settings = Settings()
        
        # Setup logger
        self.logger = setup_logger('google_drive_labeling')
        
        # Initialize components
        self.cloud_manager = CloudCheckpointManager(
            project_folder='ujaran-kebencian-labeling',
            local_cache_dir='src/checkpoints'
        )
        
        self.deepseek_client = None
        self.labeling_pipeline = None
        self.interrupted = False
        
        # Setup signal handlers untuk graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals untuk graceful shutdown"""
        self.logger.info(f"\n🛑 Received signal {signum} (Ctrl+C). Initiating graceful shutdown...")
        self.logger.info("💾 Saving checkpoint and syncing to Google Drive...")
        self.interrupted = True
        
        # Force immediate sync to cloud
        try:
            self.sync_to_cloud(force=True)
            self.logger.info("✅ Emergency sync to Google Drive completed")
        except Exception as e:
            self.logger.error(f"❌ Emergency sync failed: {str(e)}")
        
        # Exit gracefully
        self.logger.info("🔄 Process stopped. Run 'python labeling.py' again to resume.")
        import sys
        sys.exit(0)
    
    def setup(self) -> bool:
        """
        Setup semua komponen yang diperlukan
        
        Returns:
            bool: True jika setup berhasil
        """
        try:
            # 1. Setup Google Drive authentication
            self.logger.info("🔐 Setting up Google Drive authentication...")
            if not self.cloud_manager.authenticate():
                if self.cloud_manager._offline_mode:
                    self.logger.warning("⚠️ Running in offline mode. Results will be saved locally only.")
                else:
                    self.logger.error("❌ Google Drive authentication failed")
                    return False
            else:
                self.logger.info("✅ Google Drive authentication successful")
                
                # 1.5. Verify and recover Google Drive folder structure
                self.logger.info("🔍 Verifying Google Drive folder structure...")
                if not self.cloud_manager.verify_and_recover_folders():
                    self.logger.warning("⚠️ Could not verify folder structure, continuing anyway...")
            
            # 2. Setup DeepSeek client
            self.logger.info("🤖 Setting up DeepSeek API client...")
            self.deepseek_client = create_deepseek_client()
            if not self.deepseek_client:
                self.logger.error("❌ DeepSeek client setup failed")
                return False
            self.logger.info("✅ DeepSeek API client ready")
            
            # 3. Setup labeling pipeline
            self.logger.info("⚙️ Setting up labeling pipeline...")
            self.labeling_pipeline = PersistentLabelingPipeline(
                mock_mode=False,
                settings=self.settings,
                checkpoint_interval=self.checkpoint_interval,
                cost_strategy="warn_expensive"
            )
            self.logger.info("✅ Labeling pipeline ready")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Setup failed: {str(e)}")
            return False
    
    def check_promo_hours(self) -> bool:
        """
        Check apakah sedang jam promo DeepSeek
        
        Returns:
            bool: True jika sedang jam promo
        """
        current_hour = datetime.now().hour
        
        # Jam promo DeepSeek (contoh: 00:00-06:00 dan 18:00-23:59)
        promo_hours = list(range(0, 6)) + list(range(18, 24))
        
        is_promo = current_hour in promo_hours
        
        if is_promo:
            self.logger.info(f"🎉 Promo time detected! Current hour: {current_hour}")
        else:
            self.logger.info(f"⏰ Not promo time. Current hour: {current_hour}. Promo hours: 00-06, 18-23")
        
        return is_promo
    
    def wait_for_promo(self):
        """
        Wait sampai jam promo dimulai
        """
        while not self.check_promo_hours() and not self.interrupted:
            current_hour = datetime.now().hour
            
            # Calculate next promo hour
            if current_hour < 18:
                next_promo = 18
                wait_hours = next_promo - current_hour
            else:
                next_promo = 24  # Midnight (00:00)
                wait_hours = next_promo - current_hour
            
            self.logger.info(f"⏳ Waiting for promo hours. Next promo in {wait_hours} hours (at {next_promo:02d}:00)")
            
            # Wait 30 minutes and check again
            for _ in range(30):
                if self.interrupted:
                    return
                time.sleep(60)  # Sleep 1 minute
    
    def sync_to_cloud(self, force: bool = False):
        """
        Sync checkpoint dan results ke Google Drive
        
        Args:
            force: Force sync meskipun belum waktunya
        """
        if self.cloud_manager._offline_mode:
            self.logger.warning("⚠️ Offline mode - skipping cloud sync")
            return
        
        if not self.cloud_manager._authenticated:
            self.logger.warning("⚠️ Not authenticated - skipping cloud sync")
            return
        
        try:
            sync_count = 0
            
            # Sync checkpoint
            checkpoint_file = f"checkpoints/labeling_{self.output_name}.json"
            if os.path.exists(checkpoint_file):
                self.logger.info("☁️ Syncing checkpoint to Google Drive...")
                checkpoint_name = f"checkpoint_{self.output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                # Read checkpoint data
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                # Upload using save_checkpoint method
                success = self.cloud_manager.save_checkpoint(checkpoint_data, f"labeling_{self.output_name}")
                if success:
                    sync_count += 1
                    self.logger.info(f"✅ Checkpoint synced: {checkpoint_name}")
                else:
                    self.logger.error(f"❌ Failed to sync checkpoint: {checkpoint_name}")
            else:
                self.logger.info(f"📄 No checkpoint file found: {checkpoint_file}")
            
            # Sync results
            results_file = f"{self.output_name}.csv"
            if os.path.exists(results_file):
                self.logger.info("☁️ Syncing results to Google Drive...")
                results_name = f"results_{self.output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                # Upload using upload_dataset method
                success = self.cloud_manager.upload_dataset(results_file, results_name)
                if success:
                    sync_count += 1
                    self.logger.info(f"✅ Results synced: {results_name}")
                else:
                    self.logger.error(f"❌ Failed to sync results: {results_name}")
            else:
                # Try to create CSV from checkpoint if it doesn't exist
                checkpoint_file = f"checkpoints/labeling_{self.output_name}.json"
                if os.path.exists(checkpoint_file):
                    self.logger.info(f"📄 Creating CSV from checkpoint: {checkpoint_file}")
                    try:
                        import pandas as pd
                        
                        # Load checkpoint data
                        with open(checkpoint_file, 'r', encoding='utf-8') as f:
                            checkpoint_data = json.load(f)
                        
                        # Extract results
                        results = checkpoint_data.get('results', [])
                        if results:
                            # Create DataFrame
                            df = pd.DataFrame(results)
                            
                            # Define output columns
                            output_columns = ['text', 'label', 'final_label', 'confidence_score', 
                                            'labeling_method', 'response_time']
                            
                            # Filter available columns
                            available_columns = [col for col in output_columns if col in df.columns]
                            
                            # Save CSV
                            df[available_columns].to_csv(results_file, index=False)
                            self.logger.info(f"✅ CSV created from checkpoint: {results_file} ({len(results)} samples)")
                            
                            # Now sync the created CSV
                            self.logger.info("☁️ Syncing created results to Google Drive...")
                            results_name = f"results_{self.output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            
                            success = self.cloud_manager.upload_dataset(results_file, results_name)
                            if success:
                                sync_count += 1
                                self.logger.info(f"✅ Results synced: {results_name}")
                            else:
                                self.logger.error(f"❌ Failed to sync results: {results_name}")
                        else:
                            self.logger.info("📄 No results found in checkpoint")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to create CSV from checkpoint: {e}")
                else:
                    self.logger.info(f"📄 No results file found: {results_file}")
            
            if sync_count > 0:
                self.logger.info(f"✅ Cloud sync completed - {sync_count} files synced")
            else:
                self.logger.info("ℹ️ No files to sync")
            
        except Exception as e:
            self.logger.error(f"❌ Cloud sync failed: {str(e)}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
    
    def run_labeling(self, wait_for_promo: bool = True, resume: bool = True):
        """
        Run labeling process dengan Google Drive integration
        
        Args:
            wait_for_promo: Wait untuk jam promo sebelum mulai
            resume: Resume dari checkpoint jika ada
        """
        try:
            # 1. Wait for promo hours jika diminta
            if wait_for_promo:
                if not self.check_promo_hours():
                    self.logger.info("⏰ Waiting for promo hours to start labeling...")
                    self.wait_for_promo()
                    
                    if self.interrupted:
                        self.logger.info("🛑 Process interrupted while waiting for promo")
                        return
            
            # 2. Load dataset
            self.logger.info(f"📂 Loading dataset: {self.dataset_path}")
            df = self.labeling_pipeline.load_dataset(self.dataset_path)
            
            # 3. Use consistent checkpoint ID dan output file
            checkpoint_id = self.checkpoint_id
            output_file = f"{self.output_name}.csv"
            
            # 4. Check for resume data
            resume_data = None
            if resume and not self.cloud_manager._offline_mode:
                self.logger.info("📥 Checking for cloud checkpoint...")
                try:
                    latest_checkpoint = self.cloud_manager.get_latest_checkpoint()
                    if latest_checkpoint:
                        # Validate checkpoint integrity
                        if self.cloud_manager.validate_checkpoint(latest_checkpoint):
                            self.logger.info("✅ Valid cloud checkpoint found")
                            # Display clear resume information
                            self.cloud_manager.display_resume_info(latest_checkpoint)
                            resume_data = latest_checkpoint
                        else:
                            self.logger.warning("⚠️ Cloud checkpoint validation failed, starting fresh")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not download cloud checkpoint: {str(e)}")
                    
            # Check local checkpoint if cloud failed
            if resume_data is None and resume:
                self.logger.info("📥 Checking for local checkpoint...")
                try:
                    local_checkpoint = self.labeling_pipeline.load_checkpoint(checkpoint_id)
                    if local_checkpoint:
                        if self.cloud_manager.validate_checkpoint(local_checkpoint):
                            self.logger.info("✅ Valid local checkpoint found")
                            self.cloud_manager.display_resume_info(local_checkpoint)
                            resume_data = local_checkpoint
                        else:
                            self.logger.warning("⚠️ Local checkpoint validation failed, starting fresh")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load local checkpoint: {str(e)}")
            
            # 5. Start labeling process
            self.logger.info("🚀 Starting labeling process...")
            
            # Setup periodic sync during processing
            import threading
            import time
            
            def periodic_sync():
                """Sync to cloud every 5 minutes during processing"""
                while not self.interrupted:
                    time.sleep(300)  # 5 minutes
                    if not self.interrupted:
                        self.sync_to_cloud(force=False)
            
            # Start periodic sync thread
            sync_thread = threading.Thread(target=periodic_sync, daemon=True)
            sync_thread.start()
            
            # Process dengan checkpoints
            try:
                report = self.labeling_pipeline.process_with_checkpoints(
                    df, checkpoint_id, output_file, resume_data
                )
                
                self.logger.info("🏁 Labeling completed successfully!")
                
                # Final sync to cloud
                self.sync_to_cloud(force=True)
                
                return report
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Process interrupted during labeling")
                # Emergency sync
                self.sync_to_cloud(force=True)
                raise
            
        except Exception as e:
            self.logger.error(f"❌ Labeling process failed: {str(e)}")
            # Emergency sync
            self.sync_to_cloud(force=True)
            raise
    
    def status(self):
        """
        Show current status dan progress
        """
        print("\n" + "="*60)
        print("📊 GOOGLE DRIVE LABELING STATUS")
        print("="*60)
        
        # Check promo status
        is_promo = self.check_promo_hours()
        promo_status = "🎉 PROMO TIME" if is_promo else "⏰ Regular Time"
        print(f"Time Status: {promo_status}")
        
        # Check Google Drive status
        if self.cloud_manager._offline_mode:
            print("Cloud Status: ⚠️ Offline Mode")
        else:
            print("Cloud Status: ☁️ Connected")
        
        # Check local files
        checkpoint_file = f"checkpoints/{self.checkpoint_id}.json"
        results_file = f"{self.output_name}.csv"
        
        if os.path.exists(checkpoint_file):
            checkpoint_size = os.path.getsize(checkpoint_file)
            print(f"Local Checkpoint: ✅ {checkpoint_size} bytes")
        else:
            print("Local Checkpoint: ❌ Not found")
        
        if os.path.exists(results_file):
            results_size = os.path.getsize(results_file)
            print(f"Local Results: ✅ {results_size} bytes")
        else:
            print("Local Results: ❌ Not found")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Google Drive Labeling Pipeline')
    parser.add_argument('--dataset', default='src/data_collection/raw-dataset.csv',
                       help='Path to dataset CSV file')
    parser.add_argument('--output', default='google-drive-labeling',
                       help='Output name for results')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for processing')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                       help='Interval for local checkpoints')
    parser.add_argument('--cloud-sync-interval', type=int, default=100,
                       help='Interval for cloud sync')
    parser.add_argument('--no-promo-wait', action='store_true',
                       help='Start immediately without waiting for promo hours')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh without resuming from checkpoint')
    parser.add_argument('--status', action='store_true',
                       help='Show current status')
    parser.add_argument('--setup', action='store_true',
                       help='Show setup instructions')
    
    args = parser.parse_args()
    
    if args.setup:
        print("\n" + "="*60)
        print("🔧 GOOGLE DRIVE LABELING SETUP")
        print("="*60)
        print("\n1. Install dependencies:")
        print("   pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        print("\n2. Setup Google Drive API:")
        print("   - Go to https://console.cloud.google.com/")
        print("   - Create new project")
        print("   - Enable Google Drive API")
        print("   - Create OAuth 2.0 Credentials (Desktop Application)")
        print("   - Download as 'credentials.json'")
        print("   - Place in project root")
        print("\n3. Run labeling:")
        print("   python src/google_drive_labeling.py")
        print("\n4. Features:")
        print("   ✅ Cancel/Resume anytime (Ctrl+C)")
        print("   ✅ Automatic promo hour detection")
        print("   ✅ Google Drive auto-sync")
        print("   ✅ Multi-device support")
        print("\n" + "="*60)
        return
    
    # Initialize pipeline
    pipeline = GoogleDriveLabelingPipeline(
        dataset_path=args.dataset,
        output_name=args.output,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        cloud_sync_interval=args.cloud_sync_interval
    )
    
    if args.status:
        pipeline.status()
        return
    
    # Setup pipeline
    if not pipeline.setup():
        print("❌ Setup failed. Use --setup for instructions.")
        return
    
    # Run labeling
    try:
        pipeline.run_labeling(
            wait_for_promo=not args.no_promo_wait,
            resume=not args.no_resume
        )
        print("\n🎉 Labeling completed successfully!")
        print("📁 Results saved to Google Drive and locally")
        
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        print("💾 Progress has been saved and synced to Google Drive")
        print("🔄 Use the same command to resume from where you left off")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("💾 Emergency save completed")
        print("🔄 Check logs and try resuming")

if __name__ == '__main__':
    main()