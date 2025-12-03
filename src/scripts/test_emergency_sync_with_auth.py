#!/usr/bin/env python3
"""
Test Emergency Sync dengan Authentication untuk Google Drive Labeling
Script ini menguji apakah emergency sync bisa menemukan dan upload file dengan nama yang benar
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from google_drive_labeling import GoogleDriveLabelingPipeline

def test_emergency_sync_with_auth():
    """Test emergency sync functionality dengan authentication"""
    print("🧪 Testing Emergency Sync with Authentication")
    print("="*50)
    
    # Initialize pipeline dengan parameter yang sama seperti labeling.py
    pipeline = GoogleDriveLabelingPipeline(
        dataset_path='src/data_collection/raw-dataset.csv',
        output_name='hasil-labeling',
        batch_size=10
    )
    
    print(f"📋 Pipeline initialized:")
    print(f"   Dataset: {pipeline.dataset_path}")
    print(f"   Output: {pipeline.output_name}")
    print(f"   Checkpoint ID: {pipeline.checkpoint_id}")
    print()
    
    # Setup authentication and components
    print("🔐 Setting up authentication and components...")
    try:
        success = pipeline.setup()
        if success:
            print("✅ Setup successful")
        else:
            print("❌ Setup failed")
            return
    except Exception as e:
        print(f"❌ Setup failed: {str(e)}")
        return
    
    # Check expected checkpoint file path
    expected_checkpoint = f"checkpoints/labeling_{pipeline.output_name}.json"
    print(f"🔍 Expected checkpoint file: {expected_checkpoint}")
    
    if os.path.exists(expected_checkpoint):
        size = os.path.getsize(expected_checkpoint)
        print(f"✅ Checkpoint file exists: {size} bytes")
    else:
        print("❌ Checkpoint file not found")
        return
    
    # Check results file
    results_file = f"{pipeline.output_name}.csv"
    print(f"🔍 Expected results file: {results_file}")
    
    if os.path.exists(results_file):
        size = os.path.getsize(results_file)
        print(f"✅ Results file exists: {size} bytes")
    else:
        print("❌ Results file not found")
        return
    
    print()
    
    # Test sync_to_cloud method
    print("🔄 Testing sync_to_cloud method with authentication...")
    try:
        pipeline.sync_to_cloud(force=True)
        print("✅ sync_to_cloud completed successfully")
        print("📤 Files should now be uploaded to Google Drive")
    except Exception as e:
        print(f"❌ sync_to_cloud failed: {str(e)}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
    
    print()
    print("🏁 Test completed")
    print("📁 Check your Google Drive folder 'ujaran-kebencian-labeling' for uploaded files")

if __name__ == "__main__":
    test_emergency_sync_with_auth()