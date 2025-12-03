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
from datetime import datetime
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
                 checkpoint_interval: int = 5,
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
        print(f"DEBUG: Settings type after instantiation: {type(self.settings)}")
        print(f"DEBUG: Settings has deepseek_base_url: {hasattr(self.settings, 'deepseek_base_url')}")
        if hasattr(self.settings, 'deepseek_base_url'):
            print(f"DEBUG: deepseek_base_url value: {self.settings.deepseek_base_url}")
        
        # Setup logger
        self.logger = setup_logger('google_drive_labeling')
        
        # Initialize components
        print(f"DEBUG: Settings type before CloudCheckpointManager: {type(self.settings)}")
        self.cloud_manager = CloudCheckpointManager(
            project_folder='ujaran-kebencian-labeling',
            local_cache_dir='src/checkpoints'
        )
        print(f"DEBUG: Settings type after CloudCheckpointManager: {type(self.settings)}")
        
        self.deepseek_client = None
        self.labeling_pipeline = None
        self.interrupted = False
        self.current_checkpoint_id = None
        self.machine_id = self._generate_machine_id()
        self.lock_acquired = False
        
        # Pipeline settings
        self.pipeline_settings = {
            'wait_for_promo': True,
            'auto_sync': True,
            'max_retries': 3,
            'lock_timeout_minutes': 120  # 2 hours default timeout
        }
        
        # Setup signal handlers untuk graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Register cleanup function untuk unexpected exits
        import atexit
        atexit.register(self._cleanup_on_exit)
    
    def _generate_machine_id(self) -> str:
        """Generate unique machine ID untuk distributed locking"""
        import socket
        import hashlib
        
        hostname = socket.gethostname()
        timestamp = str(int(time.time()))
        machine_info = f"{hostname}_{timestamp}"
        
        return hashlib.md5(machine_info.encode()).hexdigest()[:8]
    
    def _cleanup_on_exit(self):
        """Cleanup function yang dipanggil saat exit"""
        if self.lock_acquired:
            try:
                self.cloud_manager.release_labeling_lock(self.machine_id)
                self.logger.info("🔓 Lock released on exit")
            except Exception as e:
                self.logger.error(f"❌ Failed to release lock on exit: {str(e)}")
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals untuk graceful shutdown"""
        self.logger.info(f"\n🛑 Received signal {signum} (Ctrl+C). Initiating graceful shutdown...")
        self.logger.info("💾 Saving checkpoint and syncing to Google Drive...")
        self.interrupted = True
        
        # Release lock if acquired
        if self.lock_acquired:
            try:
                self.cloud_manager.release_labeling_lock(self.machine_id)
                self.logger.info("🔓 Lock released")
            except Exception as e:
                self.logger.error(f"❌ Failed to release lock: {str(e)}")
        
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
            print(f"DEBUG: Settings type before cloud auth: {type(self.settings)}")
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
            print(f"DEBUG: Settings type after Google Drive auth: {type(self.settings)}")
            
            # 2. Setup DeepSeek client
            self.logger.info("🤖 Setting up DeepSeek API client...")
            print(f"DEBUG: Settings type before DeepSeek client: {type(self.settings)}")
            self.deepseek_client = create_deepseek_client()
            print(f"DEBUG: Settings type after DeepSeek client: {type(self.settings)}")
            if not self.deepseek_client:
                self.logger.error("❌ DeepSeek client setup failed")
                return False
            self.logger.info("✅ DeepSeek API client ready")
            
            # 3. Setup labeling pipeline
            self.logger.info("⚙️ Setting up labeling pipeline...")
            print(f"DEBUG before PersistentLabelingPipeline: settings type: {type(self.settings)}")
            print(f"DEBUG before PersistentLabelingPipeline: settings is dict: {isinstance(self.settings, dict)}")
            self.labeling_pipeline = PersistentLabelingPipeline(
                mock_mode=False,
                settings=self.settings,
                checkpoint_interval=self.checkpoint_interval,
                cost_strategy="warn_expensive",
                cloud_manager=self.cloud_manager
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
            
            # Sync checkpoint - gunakan checkpoint_id yang konsisten
            checkpoint_file = f"src/checkpoints/{self.checkpoint_id}.json"
            if os.path.exists(checkpoint_file):
                self.logger.info("☁️ Syncing checkpoint to Google Drive...")
                checkpoint_name = f"checkpoint_{self.output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                # Read checkpoint data
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                # Upload using save_checkpoint method dengan checkpoint_id yang konsisten
                success = self.cloud_manager.save_checkpoint(checkpoint_data, self.checkpoint_id)
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
                checkpoint_file = f"src/checkpoints/{self.checkpoint_id}.json"
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
    
    def run_labeling(self, wait_for_promo: bool = True, resume: bool = True, force: bool = False):
        """
        Run labeling process dengan Google Drive integration
        
        Args:
            wait_for_promo: Wait untuk jam promo sebelum mulai
            resume: Resume dari checkpoint jika ada
            force: Force start even without cloud checkpoint
        """
        try:
            # Log force mode if enabled
            if force:
                self.logger.warning("⚠️ FORCE MODE ENABLED: Will bypass STRICT CLOUD-FIRST POLICY if needed")
                self.logger.warning("🚀 This allows starting fresh labeling without cloud checkpoint")
            # 1. Acquire distributed lock
            self.logger.info(f"🔒 Acquiring labeling lock for machine: {self.machine_id}")
            if not self.cloud_manager.acquire_labeling_lock(self.machine_id, self.pipeline_settings['lock_timeout_minutes']):
                self.logger.error("❌ Could not acquire lock. Another process might be running.")
                self.logger.info("💡 Use --force to override lock (if you're sure no other process is running)")
                sys.exit(1)
            
            self.lock_acquired = True
            self.logger.info(f"✅ Lock acquired by machine: {self.machine_id}")
            
            # 2. Wait for promo hours jika diminta
            if wait_for_promo:
                if not self.check_promo_hours():
                    self.logger.info("⏰ Waiting for promo hours to start labeling...")
                    self.wait_for_promo()
                    
                    if self.interrupted:
                        self.logger.info("🛑 Process interrupted while waiting for promo")
                        return
            
            # 3. Load dataset
            self.logger.info(f"📂 Loading dataset: {self.dataset_path}")
            df = self.labeling_pipeline.load_dataset(self.dataset_path)
            
            # 4. Use consistent checkpoint ID dan output file
            checkpoint_id = self.checkpoint_id
            output_file = f"{self.output_name}.csv"
            
            # 5. Check for resume data - STRICT CLOUD-FIRST POLICY
            resume_data = None
            if resume:
                if self.cloud_manager._offline_mode:
                    if force:
                        self.logger.warning("⚠️ FORCE MODE: Bypassing STRICT CLOUD-FIRST POLICY (offline mode)")
                        self.logger.warning("🚀 Starting fresh labeling process without cloud checkpoint")
                        resume_data = None
                    else:
                        self.logger.error("🚫 STRICT CLOUD-FIRST POLICY: Cannot resume in offline mode")
                        self.logger.error("🌐 Please ensure internet connection and Google Drive authentication")
                        self.logger.error("💡 Use --force flag to start fresh labeling process")
                        return
                
                self.logger.info("📥 Checking for cloud checkpoint (STRICT CLOUD-FIRST)...")
                try:
                    latest_checkpoint = self.cloud_manager.get_latest_checkpoint()
                    if latest_checkpoint:
                        # Validate checkpoint integrity
                        if self.cloud_manager.validate_checkpoint(latest_checkpoint):
                            self.logger.info("✅ Valid cloud checkpoint found")
                            
                            # CONFLICT DETECTION: Check for local checkpoint conflicts
                            try:
                                local_checkpoint = self.labeling_pipeline.load_checkpoint(checkpoint_id)
                                if local_checkpoint and local_checkpoint.get('timestamp') != latest_checkpoint.get('timestamp'):
                                    self.logger.warning("⚠️ CONFLICT DETECTED: Local checkpoint differs from cloud")
                                    self.logger.warning("🌐 Cloud checkpoint will be used as single source of truth")
                                    self.logger.warning("🗑️ Local checkpoint will be ignored to prevent conflicts")
                            except Exception:
                                pass  # No local checkpoint or error reading it
                            
                            # Display clear resume information
                            self.cloud_manager.display_resume_info(latest_checkpoint)
                            resume_data = latest_checkpoint
                        else:
                            if force:
                                self.logger.warning("⚠️ FORCE MODE: Bypassing STRICT CLOUD-FIRST POLICY (invalid checkpoint)")
                                self.logger.warning("🚀 Starting fresh labeling process without cloud checkpoint")
                                resume_data = None
                            else:
                                self.logger.error("❌ Cloud checkpoint validation failed")
                                self.logger.error("🚫 STRICT CLOUD-FIRST POLICY: Cannot proceed with invalid checkpoint")
                                self.logger.error("💡 Use --force flag to start fresh labeling process")
                                return
                    else:
                        if force:
                            self.logger.warning("⚠️ FORCE MODE: Bypassing STRICT CLOUD-FIRST POLICY (no checkpoint)")
                            self.logger.warning("🚀 Starting fresh labeling process without cloud checkpoint")
                            resume_data = None
                        else:
                            self.logger.error("❌ No cloud checkpoint found")
                            self.logger.error("🚫 STRICT CLOUD-FIRST POLICY: Cannot resume without cloud checkpoint")
                            self.logger.error("💡 Use --force flag to start fresh labeling process")
                            return
                except Exception as e:
                    if force:
                        self.logger.warning(f"⚠️ FORCE MODE: Could not access cloud checkpoint: {str(e)}")
                        self.logger.warning("⚠️ FORCE MODE: Bypassing STRICT CLOUD-FIRST POLICY (cloud access error)")
                        self.logger.warning("🚀 Starting fresh labeling process without cloud checkpoint")
                        resume_data = None
                    else:
                        self.logger.error(f"❌ Could not access cloud checkpoint: {str(e)}")
                        self.logger.error("🚫 STRICT CLOUD-FIRST POLICY: Cannot proceed without cloud access")
                        self.logger.error("💡 Check internet connection and Google Drive authentication")
                        self.logger.error("💡 Use --force flag to start fresh labeling process")
                        return
            
            # 6. Start labeling process
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
                
                # Release lock
                if self.lock_acquired:
                    self.cloud_manager.release_labeling_lock(self.machine_id)
                    self.lock_acquired = False
                    self.logger.info("🔓 Lock released")
                
                return report
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Process interrupted during labeling")
                # Emergency sync
                self.sync_to_cloud(force=True)
                # Release lock
                if self.lock_acquired:
                    self.cloud_manager.release_labeling_lock(self.machine_id)
                    self.lock_acquired = False
                    self.logger.info("🔓 Lock released")
                raise
            
        except Exception as e:
            self.logger.error(f"❌ Labeling failed: {str(e)}")
            # Emergency sync
            self.sync_to_cloud(force=True)
            # Release lock
            if self.lock_acquired:
                self.cloud_manager.release_labeling_lock(self.machine_id)
                self.lock_acquired = False
                self.logger.info("🔓 Lock released")
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
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                       help='Number of batches between checkpoints (default: 5)')
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
    parser.add_argument('--force', action='store_true',
                       help='Force start even if lock exists (use with caution)')
    
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
        print("   ✅ Distributed locking (prevents conflicts)")
        print("\n5. Multi-device usage:")
        print("   - Only one device can run labeling at a time")
        print("   - Use --force to override stuck locks")
        print("   - Automatic lock release on completion/interruption")
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
    
    # Handle force flag
    if args.force:
        pipeline.logger.warning("⚠️ Force mode enabled - overriding any existing locks")
        try:
            pipeline.cloud_manager.force_release_lock()
        except Exception as e:
            pipeline.logger.warning(f"⚠️ Could not release existing lock: {str(e)}")
    
    # Run labeling
    try:
        pipeline.run_labeling(
            wait_for_promo=not args.no_promo_wait,
            resume=not args.no_resume,
            force=args.force
        )
        print("\n🎉 Labeling completed successfully!")
        print("📁 Results saved to Google Drive and locally")
        
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        print("💾 Progress has been saved and synced to Google Drive")
        print("🔄 Use the same command to resume from where you left off")
        print("💡 If stuck with lock error, use --force flag")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("💾 Emergency save completed")
        print("🔄 Check logs and try resuming")
        print("💡 If stuck with lock error, use --force flag")

if __name__ == '__main__':
    main()