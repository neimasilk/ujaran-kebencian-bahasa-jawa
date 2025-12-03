#!/usr/bin/env python3
"""
Test script untuk memverifikasi Google Drive synchronization

Usage:
    python test_gdrive_sync.py
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from utils.cloud_checkpoint_manager import CloudCheckpointManager

def test_authentication():
    """Test Google Drive authentication"""
    print("🔐 Testing Google Drive authentication...")
    
    manager = CloudCheckpointManager(
        credentials_file='credentials.json',
        token_file='token.json',
        project_folder='ujaran-kebencian-test',
        local_cache_dir='src/test_checkpoints'
    )
    
    success = manager.authenticate()
    if success:
        print("✅ Authentication successful")
        return manager
    else:
        print("❌ Authentication failed")
        return None

def test_checkpoint_sync(manager):
    """Test checkpoint synchronization"""
    print("\n💾 Testing checkpoint sync...")
    
    # Create test checkpoint
    test_data = {
        'checkpoint_id': f'test_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'processed_indices': [0, 1, 2, 3, 4],
        'results': [
            {'text': 'test 1', 'label': 'neutral'},
            {'text': 'test 2', 'label': 'hate'},
        ],
        'timestamp': datetime.now().isoformat(),
        'metadata': {
            'total_samples': 5,
            'processed_count': 2
        }
    }
    
    checkpoint_id = test_data['checkpoint_id']
    
    # Save checkpoint
    success = manager.save_checkpoint(test_data, checkpoint_id)
    if success:
        print(f"✅ Test checkpoint saved: {checkpoint_id}")
    else:
        print(f"❌ Failed to save test checkpoint: {checkpoint_id}")
        return False
    
    # Load checkpoint back
    loaded_data = manager.load_checkpoint(checkpoint_id)
    if loaded_data:
        print(f"✅ Test checkpoint loaded: {checkpoint_id}")
        return True
    else:
        print(f"❌ Failed to load test checkpoint: {checkpoint_id}")
        return False

def test_dataset_upload(manager):
    """Test dataset file upload"""
    print("\n📤 Testing dataset upload...")
    
    # Create test CSV file
    test_csv_path = 'test_dataset.csv'
    test_csv_content = '''text,label,confidence
"Ini adalah teks netral",neutral,0.95
"Teks yang mengandung kebencian",hate,0.87
"Teks positif dan baik",positive,0.92
'''
    
    with open(test_csv_path, 'w', encoding='utf-8') as f:
        f.write(test_csv_content)
    
    # Upload to Google Drive
    cloud_filename = f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    success = manager.upload_dataset(test_csv_path, cloud_filename)
    
    # Cleanup local test file
    os.unlink(test_csv_path)
    
    if success:
        print(f"✅ Test dataset uploaded: {cloud_filename}")
        return True
    else:
        print(f"❌ Failed to upload test dataset: {cloud_filename}")
        return False

def test_status_check(manager):
    """Test status information"""
    print("\n📊 Checking status...")
    
    status = manager.get_status()
    print(f"Authentication: {status['authenticated']}")
    print(f"Offline mode: {status['offline_mode']}")
    print(f"Total checkpoints: {status['total_checkpoints']}")
    print(f"Local checkpoints: {status['local_checkpoints']}")
    print(f"Cloud checkpoints: {status['cloud_checkpoints']}")
    print(f"Latest checkpoint: {status['latest_checkpoint']}")
    
    return True

def main():
    """Main test function"""
    print("🧪 Google Drive Sync Test")
    print("=" * 50)
    
    # Test authentication
    manager = test_authentication()
    if not manager:
        print("\n❌ Cannot proceed without authentication")
        return False
    
    # Run tests
    tests = [
        ("Checkpoint Sync", lambda: test_checkpoint_sync(manager)),
        ("Dataset Upload", lambda: test_dataset_upload(manager)),
        ("Status Check", lambda: test_status_check(manager))
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Google Drive sync is working correctly.")
        return True
    else:
        print("\n⚠️ Some tests failed. Please check the configuration.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)