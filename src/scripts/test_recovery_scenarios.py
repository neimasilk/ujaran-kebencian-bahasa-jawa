#!/usr/bin/env python3
"""
Test Recovery Scenarios for Google Drive Labeling System

This script tests various interruption and recovery scenarios:
1. Graceful interruption (Ctrl+C) after several batches
2. Resume with clear information about last processed batch
3. Hard interruption simulation (power loss/terminal kill)
4. Multi-device recovery using cloud checkpoints

Usage:
    python test_recovery_scenarios.py --scenario [graceful|hard|multi_device]
"""

import argparse
import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from google_drive_labeling import GoogleDriveLabelingPipeline
from utils.cloud_checkpoint_manager import CloudCheckpointManager

class RecoveryTester:
    def __init__(self):
        self.test_dataset = "data/sample_test_data.csv"
        self.output_name = "test_recovery_output"
        self.checkpoint_id = "test_recovery_checkpoint"
        
    def setup_test_environment(self):
        """Setup test environment dengan sample data"""
        print("🔧 Setting up test environment...")
        
        # Create sample test data if not exists
        if not os.path.exists(self.test_dataset):
            self.create_sample_data()
            
        # Clean previous test results
        self.cleanup_previous_tests()
        
    def create_sample_data(self):
        """Create sample CSV data for testing"""
        import pandas as pd
        
        os.makedirs("src/data", exist_ok=True)
        
        # Create sample data with 50 rows for testing
        sample_data = {
            'text': [f"Sample text for testing recovery scenario {i}" for i in range(50)],
            'id': [f"test_id_{i}" for i in range(50)]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(self.test_dataset, index=False)
        print(f"✅ Created sample test data: {self.test_dataset}")
        
    def cleanup_previous_tests(self):
        """Clean up previous test results"""
        # Remove local checkpoint files
        checkpoint_files = [
            f"checkpoints/{self.checkpoint_id}.json",
            f"results/{self.output_name}.csv"
        ]
        
        for file_path in checkpoint_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Removed previous test file: {file_path}")
                
    def test_graceful_interruption(self):
        """Test graceful interruption scenario (Ctrl+C)"""
        print("\n🧪 Testing Graceful Interruption Scenario")
        print("=" * 50)
        
        # Start labeling process
        pipeline = GoogleDriveLabelingPipeline(
            dataset_path=self.test_dataset,
            output_name=self.output_name
        )
        
        if not pipeline.setup():
            print("❌ Pipeline setup failed")
            return False
            
        # Start labeling in a separate thread
        labeling_thread = threading.Thread(
            target=pipeline.run_labeling,
            kwargs={'wait_for_promo': False, 'resume': False}
        )
        labeling_thread.daemon = True
        labeling_thread.start()
        
        # Wait for a few batches to be processed
        print("⏳ Waiting for 10 seconds to process some batches...")
        time.sleep(10)
        
        # Send interrupt signal
        print("\n🛑 Sending Ctrl+C signal...")
        os.kill(os.getpid(), signal.SIGINT)
        
        # Wait for graceful shutdown
        labeling_thread.join(timeout=30)
        
        print("✅ Graceful interruption test completed")
        return True
        
    def test_resume_functionality(self):
        """Test resume functionality after interruption"""
        print("\n🧪 Testing Resume Functionality")
        print("=" * 50)
        
        # Start new pipeline instance to test resume
        pipeline = GoogleDriveLabelingPipeline(
            dataset_path=self.test_dataset,
            output_name=self.output_name
        )
        
        if not pipeline.setup():
            print("❌ Pipeline setup failed")
            return False
            
        print("🔄 Starting resume process...")
        pipeline.run_labeling(wait_for_promo=False, resume=True)
        
        print("✅ Resume functionality test completed")
        return True
        
    def test_hard_interruption(self):
        """Test hard interruption scenario (simulated power loss)"""
        print("\n🧪 Testing Hard Interruption Scenario")
        print("=" * 50)
        
        # Start labeling process as subprocess
        script_content = f"""
import sys
sys.path.insert(0, 'src')
from google_drive_labeling import GoogleDriveLabelingPipeline

pipeline = GoogleDriveLabelingPipeline(
    dataset_path="{self.test_dataset}",
    output_name="{self.output_name}",
    checkpoint_id="{self.checkpoint_id}"
)

if pipeline.setup():
    pipeline.run_labeling(wait_for_promo=False, resume=False)
"""
        
        # Write temporary script
        temp_script = "temp_labeling_script.py"
        with open(temp_script, 'w') as f:
            f.write(script_content)
            
        try:
            # Start subprocess
            print("🚀 Starting labeling subprocess...")
            process = subprocess.Popen([sys.executable, temp_script])
            
            # Wait for some processing
            print("⏳ Waiting for 15 seconds to process some batches...")
            time.sleep(15)
            
            # Kill process abruptly (simulating power loss)
            print("\n💀 Killing process abruptly (simulating power loss)...")
            process.kill()
            process.wait()
            
            print("✅ Hard interruption simulation completed")
            
        finally:
            # Clean up temp script
            if os.path.exists(temp_script):
                os.remove(temp_script)
                
        return True
        
    def test_multi_device_recovery(self):
        """Test multi-device recovery using cloud checkpoints"""
        print("\n🧪 Testing Multi-Device Recovery Scenario")
        print("=" * 50)
        
        # Simulate device 1: Create and upload checkpoint
        print("📱 Simulating Device 1: Creating checkpoint...")
        cloud_manager = CloudCheckpointManager()
        
        if not cloud_manager.authenticate():
            print("❌ Cloud authentication failed")
            return False
            
        # Create a test checkpoint
        test_checkpoint = {
            'checkpoint_id': self.checkpoint_id,
            'processed_count': 25,
            'total_count': 50,
            'last_processed_index': 24,
            'batch_size': 5,
            'timestamp': time.time(),
            'output_file': f"{self.output_name}.csv",
            'dataset_path': self.test_dataset
        }
        
        # Save to cloud
        if cloud_manager.save_checkpoint(test_checkpoint, self.checkpoint_id):
            print("✅ Device 1: Checkpoint saved to cloud")
        else:
            print("❌ Device 1: Failed to save checkpoint")
            return False
            
        # Simulate device 2: Load and resume from cloud
        print("\n💻 Simulating Device 2: Loading from cloud...")
        
        # Create new pipeline instance (simulating different device)
        pipeline = GoogleDriveLabelingPipeline(
            dataset_path=self.test_dataset,
            output_name=self.output_name
        )
        
        if not pipeline.setup():
            print("❌ Device 2: Pipeline setup failed")
            return False
            
        # Test resume from cloud
        print("🔄 Device 2: Attempting to resume from cloud checkpoint...")
        pipeline.run_labeling(wait_for_promo=False, resume=True)
        
        print("✅ Multi-device recovery test completed")
        return True
        
    def test_folder_recovery(self):
        """Test Google Drive folder recovery after deletion"""
        print("\n🧪 Testing Folder Recovery Scenario")
        print("=" * 50)
        
        cloud_manager = CloudCheckpointManager()
        
        if not cloud_manager.authenticate():
            print("❌ Cloud authentication failed")
            return False
            
        print("🔍 Testing folder verification and recovery...")
        
        # Test folder verification
        if cloud_manager.verify_and_recover_folders():
            print("✅ Folder structure verified/recovered successfully")
        else:
            print("❌ Folder recovery failed")
            return False
            
        return True
        
    def run_all_tests(self):
        """Run all recovery scenarios"""
        print("🧪 Running All Recovery Scenarios")
        print("=" * 60)
        
        self.setup_test_environment()
        
        tests = [
            ("Folder Recovery", self.test_folder_recovery),
            ("Graceful Interruption", self.test_graceful_interruption),
            ("Resume Functionality", self.test_resume_functionality),
            ("Hard Interruption", self.test_hard_interruption),
            ("Multi-Device Recovery", self.test_multi_device_recovery)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                print(f"\n{'='*20} {test_name} {'='*20}")
                results[test_name] = test_func()
                time.sleep(2)  # Brief pause between tests
            except KeyboardInterrupt:
                print(f"\n⚠️ {test_name} interrupted by user")
                results[test_name] = False
                break
            except Exception as e:
                print(f"\n❌ {test_name} failed with error: {str(e)}")
                results[test_name] = False
                
        # Print summary
        print("\n" + "="*60)
        print("📊 TEST RESULTS SUMMARY")
        print("="*60)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:<25} {status}")
            
        passed = sum(results.values())
        total = len(results)
        print(f"\nOverall: {passed}/{total} tests passed")
        
        return passed == total

def main():
    parser = argparse.ArgumentParser(description='Test recovery scenarios for Google Drive labeling system')
    parser.add_argument('--scenario', choices=['graceful', 'hard', 'multi_device', 'folder', 'all'], 
                       default='all', help='Recovery scenario to test')
    
    args = parser.parse_args()
    
    tester = RecoveryTester()
    
    if args.scenario == 'graceful':
        tester.setup_test_environment()
        success = tester.test_graceful_interruption()
        if success:
            tester.test_resume_functionality()
    elif args.scenario == 'hard':
        tester.setup_test_environment()
        success = tester.test_hard_interruption()
        if success:
            tester.test_resume_functionality()
    elif args.scenario == 'multi_device':
        tester.setup_test_environment()
        tester.test_multi_device_recovery()
    elif args.scenario == 'folder':
        tester.test_folder_recovery()
    else:
        tester.run_all_tests()

if __name__ == "__main__":
    main()