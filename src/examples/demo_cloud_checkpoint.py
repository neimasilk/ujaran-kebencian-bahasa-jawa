#!/usr/bin/env python3
"""
Demo untuk Cloud Checkpoint Manager
Testing Google Drive integration untuk checkpoint persistence

Author: AI Assistant
Date: 2025-01-27
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils.cloud_checkpoint_manager import CloudCheckpointManager

def demo_basic_operations():
    """
    Demo basic operations dari CloudCheckpointManager
    """
    print("🚀 Demo: Basic Cloud Checkpoint Operations\n")
    
    # Initialize manager
    manager = CloudCheckpointManager(
        project_folder='ujaran-kebencian-demo',
        local_cache_dir='src/demo-checkpoints'
    )
    
    # Test authentication
    print("1. Testing Authentication...")
    auth_success = manager.authenticate()
    
    if not auth_success:
        print("⚠️ Running in offline mode - will test local operations only")
    
    # Get initial status
    print("\n2. Initial Status:")
    status = manager.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Create test checkpoints
    print("\n3. Creating Test Checkpoints...")
    
    test_checkpoints = []
    for i in range(1, 4):
        checkpoint_data = {
            "checkpoint_id": f"demo_checkpoint_{i:03d}",
            "timestamp": datetime.now().isoformat(),
            "processed_indices": list(range(1, i * 10 + 1)),
            "total_samples": 100,
            "current_batch": i,
            "progress_percentage": (i * 10),
            "metadata": {
                "model_version": "v1.0",
                "dataset_version": "demo",
                "processing_mode": "test"
            }
        }
        
        checkpoint_id = f"demo_checkpoint_{i:03d}"
        success = manager.save_checkpoint(checkpoint_data, checkpoint_id)
        
        if success:
            test_checkpoints.append(checkpoint_id)
            print(f"   ✅ Created checkpoint: {checkpoint_id}")
        else:
            print(f"   ❌ Failed to create checkpoint: {checkpoint_id}")
        
        # Small delay untuk different timestamps
        time.sleep(0.1)
    
    # List all checkpoints
    print("\n4. Listing All Checkpoints:")
    checkpoints = manager.list_checkpoints()
    
    if checkpoints:
        for checkpoint in checkpoints:
            print(f"   📄 {checkpoint['id']} ({checkpoint['source']}) - {checkpoint['timestamp']} - {checkpoint['size']} bytes")
    else:
        print("   No checkpoints found")
    
    # Test loading checkpoints
    print("\n5. Testing Checkpoint Loading:")
    
    for checkpoint_id in test_checkpoints:
        loaded_data = manager.load_checkpoint(checkpoint_id)
        if loaded_data:
            progress = loaded_data.get('progress_percentage', 0)
            processed = len(loaded_data.get('processed_indices', []))
            print(f"   ✅ Loaded {checkpoint_id}: {progress}% complete, {processed} samples processed")
        else:
            print(f"   ❌ Failed to load {checkpoint_id}")
    
    # Test getting latest checkpoint
    print("\n6. Getting Latest Checkpoint:")
    latest = manager.get_latest_checkpoint()
    if latest:
        print(f"   📄 Latest: {latest['checkpoint_id']} - {latest['progress_percentage']}% complete")
    else:
        print("   No checkpoints available")
    
    # Test sync (jika cloud available)
    if auth_success:
        print("\n7. Testing Checkpoint Sync:")
        sync_success = manager.sync_checkpoints()
        if sync_success:
            print("   ✅ Sync completed successfully")
        else:
            print("   ❌ Sync failed")
    
    # Final status
    print("\n8. Final Status:")
    final_status = manager.get_status()
    for key, value in final_status.items():
        print(f"   {key}: {value}")
    
    return manager, test_checkpoints

def demo_persistence_scenario():
    """
    Demo skenario persistence: simulate interruption dan resume
    """
    print("\n🔄 Demo: Persistence Scenario (Interruption & Resume)\n")
    
    # Simulate initial processing
    print("1. Simulating Initial Processing Session...")
    
    manager1 = CloudCheckpointManager(
        project_folder='ujaran-kebencian-demo',
        local_cache_dir='src/demo-checkpoints'
    )
    manager1.authenticate()
    
    # Simulate processing dengan periodic checkpoints
    total_samples = 50
    batch_size = 10
    
    for batch in range(1, 4):  # Process 3 batches, then "interrupt"
        start_idx = (batch - 1) * batch_size + 1
        end_idx = min(batch * batch_size, total_samples)
        
        processed_indices = list(range(1, end_idx + 1))
        
        checkpoint_data = {
            "checkpoint_id": f"processing_session_001_batch_{batch:03d}",
            "timestamp": datetime.now().isoformat(),
            "processed_indices": processed_indices,
            "total_samples": total_samples,
            "current_batch": batch,
            "progress_percentage": round((len(processed_indices) / total_samples) * 100, 2),
            "session_id": "session_001",
            "last_processed_index": end_idx
        }
        
        checkpoint_id = f"processing_session_001_batch_{batch:03d}"
        manager1.save_checkpoint(checkpoint_data, checkpoint_id)
        
        print(f"   📄 Saved checkpoint: batch {batch}, processed {len(processed_indices)}/{total_samples} samples ({checkpoint_data['progress_percentage']}%)")
        
        time.sleep(0.1)
    
    print("   ⚠️ Simulating interruption after batch 3...")
    
    # Simulate resume pada "different machine"
    print("\n2. Simulating Resume on Different Machine...")
    
    manager2 = CloudCheckpointManager(
        project_folder='ujaran-kebencian-demo',
        local_cache_dir='src/demo-checkpoints-machine2'  # Different local cache
    )
    manager2.authenticate()
    
    # Find latest checkpoint
    latest_checkpoint = manager2.get_latest_checkpoint()
    
    if latest_checkpoint:
        last_processed = latest_checkpoint['last_processed_index']
        processed_indices = latest_checkpoint['processed_indices']
        session_id = latest_checkpoint.get('session_id', 'unknown')
        
        print(f"   ✅ Resumed from checkpoint: {latest_checkpoint['checkpoint_id']}")
        print(f"   📊 Progress: {len(processed_indices)}/{total_samples} samples ({latest_checkpoint['progress_percentage']}%)")
        print(f"   🔄 Continuing from index {last_processed + 1}...")
        
        # Continue processing
        for batch in range(4, 6):  # Complete remaining batches
            start_idx = (batch - 1) * batch_size + 1
            end_idx = min(batch * batch_size, total_samples)
            
            # Add new processed indices
            new_indices = list(range(last_processed + 1, end_idx + 1))
            processed_indices.extend(new_indices)
            
            checkpoint_data = {
                "checkpoint_id": f"processing_session_001_batch_{batch:03d}",
                "timestamp": datetime.now().isoformat(),
                "processed_indices": processed_indices,
                "total_samples": total_samples,
                "current_batch": batch,
                "progress_percentage": round((len(processed_indices) / total_samples) * 100, 2),
                "session_id": session_id,
                "last_processed_index": end_idx
            }
            
            checkpoint_id = f"processing_session_001_batch_{batch:03d}"
            manager2.save_checkpoint(checkpoint_data, checkpoint_id)
            
            print(f"   📄 Saved checkpoint: batch {batch}, processed {len(processed_indices)}/{total_samples} samples ({checkpoint_data['progress_percentage']}%)")
            
            last_processed = end_idx
            time.sleep(0.1)
        
        print(f"   🎉 Processing completed! Total: {len(processed_indices)}/{total_samples} samples")
        
    else:
        print("   ❌ No checkpoint found for resume")
    
    return manager2

def demo_cleanup():
    """
    Demo cleanup operations
    """
    print("\n🗑️ Demo: Cleanup Operations\n")
    
    manager = CloudCheckpointManager(
        project_folder='ujaran-kebencian-demo',
        local_cache_dir='src/demo-checkpoints'
    )
    manager.authenticate()
    
    # List current checkpoints
    print("1. Current Checkpoints:")
    checkpoints = manager.list_checkpoints()
    print(f"   Total: {len(checkpoints)} checkpoints")
    
    # Cleanup old checkpoints (keep only 3 latest)
    print("\n2. Cleaning up old checkpoints (keeping 3 latest)...")
    deleted_count = manager.cleanup_old_checkpoints(keep_count=3)
    print(f"   🗑️ Deleted {deleted_count} old checkpoints")
    
    # List remaining checkpoints
    print("\n3. Remaining Checkpoints:")
    remaining_checkpoints = manager.list_checkpoints()
    for checkpoint in remaining_checkpoints:
        print(f"   📄 {checkpoint['id']} ({checkpoint['source']}) - {checkpoint['timestamp']}")
    
    return manager

def setup_instructions():
    """
    Print setup instructions
    """
    print("""
📋 Setup Instructions untuk Cloud Checkpoint Demo:

1. Install Dependencies:
   pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

2. Setup Google Drive API:
   - Go to https://console.cloud.google.com/
   - Create new project atau pilih existing project
   - Enable Google Drive API
   - Create OAuth 2.0 Credentials (Desktop Application)
   - Download credentials sebagai 'credentials.json'
   - Place file di root directory project ini

3. Run Demo:
   python demo_cloud_checkpoint.py

4. Demo Options:
   python demo_cloud_checkpoint.py --demo basic      # Basic operations
   python demo_cloud_checkpoint.py --demo persistence # Persistence scenario
   python demo_cloud_checkpoint.py --demo cleanup    # Cleanup operations
   python demo_cloud_checkpoint.py --demo all        # All demos

Note: Pada first run, browser akan terbuka untuk OAuth consent.
Grant permissions untuk Drive access.
    """)

def main():
    parser = argparse.ArgumentParser(description='Demo Cloud Checkpoint Manager')
    parser.add_argument('--demo', choices=['basic', 'persistence', 'cleanup', 'all'], 
                       default='all', help='Demo type to run')
    parser.add_argument('--setup', action='store_true', help='Show setup instructions')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_instructions()
        return
    
    # Check credentials
    if not os.path.exists('credentials.json'):
        print("❌ credentials.json not found!")
        print("\nRun: python demo_cloud_checkpoint.py --setup")
        print("for setup instructions.\n")
        return
    
    print("🌟 Cloud Checkpoint Manager Demo")
    print("=" * 50)
    
    try:
        if args.demo in ['basic', 'all']:
            manager, test_checkpoints = demo_basic_operations()
        
        if args.demo in ['persistence', 'all']:
            manager = demo_persistence_scenario()
        
        if args.demo in ['cleanup', 'all']:
            manager = demo_cleanup()
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Tips:")
        print("   - Checkpoints are saved both locally and in Google Drive")
        print("   - Local cache provides offline access")
        print("   - Cloud sync enables cross-device persistence")
        print("   - Automatic fallback to offline mode if cloud unavailable")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()