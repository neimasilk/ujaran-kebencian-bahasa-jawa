#!/usr/bin/env python3
"""
Test script untuk memverifikasi perbaikan Google Drive upload

Script ini akan:
1. Test authentication ke Google Drive
2. Test setup folder structure
3. Test upload file ke Google Drive
4. Test save checkpoint ke Google Drive

Author: AI Assistant
Date: 2025-01-27
"""

import os
import sys
import json
import tempfile
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.cloud_checkpoint_manager import CloudCheckpointManager

def test_authentication():
    """Test Google Drive authentication"""
    print("\n" + "="*60)
    print("🔐 TESTING GOOGLE DRIVE AUTHENTICATION")
    print("="*60)
    
    manager = CloudCheckpointManager()
    
    # Test authentication
    auth_success = manager.authenticate()
    
    if auth_success:
        print("✅ Authentication successful")
        print(f"📁 Project folder: {manager.project_folder}")
        print(f"💾 Checkpoint folder ID: {manager.checkpoint_folder_id}")
        print(f"📊 Datasets folder ID: {manager.datasets_folder_id}")
        print(f"📈 Results folder ID: {manager.results_folder_id}")
        return manager
    else:
        print("❌ Authentication failed")
        return None

def test_checkpoint_save(manager):
    """Test checkpoint save to Google Drive"""
    print("\n" + "="*60)
    print("💾 TESTING CHECKPOINT SAVE")
    print("="*60)
    
    # Create test checkpoint data
    test_checkpoint = {
        'checkpoint_id': f'test_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'processed_indices': [0, 1, 2, 3, 4],
        'timestamp': datetime.now().isoformat(),
        'metadata': {
            'total_samples': 100,
            'last_batch': 1,
            'test_data': True
        }
    }
    
    # Test save checkpoint
    checkpoint_id = test_checkpoint['checkpoint_id']
    success = manager.save_checkpoint(test_checkpoint, checkpoint_id)
    
    if success:
        print(f"✅ Checkpoint saved successfully: {checkpoint_id}")
        
        # Test load checkpoint
        loaded_checkpoint = manager.load_checkpoint(checkpoint_id)
        if loaded_checkpoint:
            print(f"✅ Checkpoint loaded successfully: {checkpoint_id}")
            print(f"📊 Processed samples: {len(loaded_checkpoint['processed_indices'])}")
            return True
        else:
            print(f"❌ Failed to load checkpoint: {checkpoint_id}")
            return False
    else:
        print(f"❌ Failed to save checkpoint: {checkpoint_id}")
        return False

def test_file_upload(manager):
    """Test file upload to Google Drive"""
    print("\n" + "="*60)
    print("📤 TESTING FILE UPLOAD")
    print("="*60)
    
    # Create test CSV file
    test_data = [
        ['text', 'label'],
        ['Test text 1', 'not_hate'],
        ['Test text 2', 'hate'],
        ['Test text 3', 'not_hate']
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
        import csv
        writer = csv.writer(temp_file)
        writer.writerows(test_data)
        temp_file_path = temp_file.name
    
    try:
        # Test upload to results folder
        success = manager.upload_file(temp_file_path, 'results')
        
        if success:
            print(f"✅ File uploaded successfully to results folder")
            return True
        else:
            print(f"❌ Failed to upload file to results folder")
            return False
            
    finally:
        # Cleanup temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def test_folder_verification(manager):
    """Test folder structure verification"""
    print("\n" + "="*60)
    print("📁 TESTING FOLDER VERIFICATION")
    print("="*60)
    
    # Test folder verification
    folders_ready = manager.verify_and_recover_folders()
    
    if folders_ready:
        print("✅ Folder structure verified successfully")
        return True
    else:
        print("❌ Folder structure verification failed")
        return False

def main():
    """Main test function"""
    print("🧪 GOOGLE DRIVE INTEGRATION TEST")
    print("This script will test the Google Drive upload fixes")
    print("Make sure you have credentials.json in the project root")
    
    # Test authentication
    manager = test_authentication()
    if not manager:
        print("\n❌ Cannot proceed without authentication")
        return False
    
    # Test folder verification
    if not test_folder_verification(manager):
        print("\n❌ Folder verification failed")
        return False
    
    # Test checkpoint save
    if not test_checkpoint_save(manager):
        print("\n❌ Checkpoint save test failed")
        return False
    
    # Test file upload
    if not test_file_upload(manager):
        print("\n❌ File upload test failed")
        return False
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Google Drive integration is working correctly")
    print("✅ Files should now be uploaded to Google Drive properly")
    print("="*60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)