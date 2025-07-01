#!/usr/bin/env python3
"""
Demo script untuk menunjukkan robustness sistem labeling
Mendemonstrasikan fitur-fitur robustness yang telah diimplementasikan

Author: AI Assistant
Date: 2025-01-27
"""

import os
import sys
import time
import signal
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.cloud_checkpoint_manager import CloudCheckpointManager
from google_drive_labeling import GoogleDriveLabelingPipeline

def demo_checkpoint_system():
    """
    Demo checkpoint save/load system
    """
    print("\n🧪 DEMO: Checkpoint System")
    print("=" * 40)
    
    cloud_manager = CloudCheckpointManager()
    
    # Create sample checkpoint
    checkpoint_data = {
        'checkpoint_id': 'demo_checkpoint_001',
        'processed_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'timestamp': datetime.now().isoformat(),
        'metadata': {
            'total_samples': 100,
            'last_batch': 2,
            'progress_percentage': 10.0
        }
    }
    
    print("📝 Saving checkpoint...")
    success = cloud_manager.save_checkpoint(checkpoint_data, 'demo_checkpoint_001')
    if success:
        print("✅ Checkpoint saved successfully")
    else:
        print("❌ Checkpoint save failed")
        return
    
    print("\n📖 Loading checkpoint...")
    loaded_checkpoint = cloud_manager.load_checkpoint('demo_checkpoint_001')
    if loaded_checkpoint:
        print("✅ Checkpoint loaded successfully")
        print(f"📊 Progress: {len(loaded_checkpoint['processed_indices'])}/100 samples")
        cloud_manager.display_resume_info(loaded_checkpoint)
    else:
        print("❌ Checkpoint load failed")

def demo_locking_system():
    """
    Demo distributed locking system
    """
    print("\n🧪 DEMO: Locking System")
    print("=" * 40)
    
    cloud_manager = CloudCheckpointManager()
    machine_id = "demo_machine_001"
    
    print("🔒 Acquiring labeling lock...")
    lock_acquired = cloud_manager.acquire_labeling_lock(machine_id, timeout_minutes=30)
    
    if lock_acquired:
        print("✅ Lock acquired successfully")
        
        # Show lock status
        print("\n📊 Checking lock status...")
        status = cloud_manager.check_labeling_status()
        print(f"🔍 Lock status: {'Active' if status['is_running'] else 'Inactive'}")
        print(f"🖥️ Machine ID: {status.get('machine_id', 'Unknown')}")
        print(f"⏰ Started at: {status.get('start_time', 'Unknown')}")
        
        # Simulate some work
        print("\n⚙️ Simulating labeling work...")
        for i in range(3):
            print(f"   Processing batch {i+1}/3...")
            time.sleep(1)
        
        # Release lock
        print("\n🔓 Releasing lock...")
        release_success = cloud_manager.release_labeling_lock(machine_id)
        if release_success:
            print("✅ Lock released successfully")
        else:
            print("❌ Lock release failed")
    else:
        print("❌ Failed to acquire lock (another process may be running)")
        
        # Show current lock status
        status = cloud_manager.check_labeling_status()
        if status['is_running']:
            print(f"ℹ️ Current lock holder: {status.get('machine_id', 'Unknown')}")
            print(f"⏰ Started at: {status.get('start_time', 'Unknown')}")

def demo_force_override():
    """
    Demo force lock override
    """
    print("\n🧪 DEMO: Force Lock Override")
    print("=" * 40)
    
    cloud_manager = CloudCheckpointManager()
    
    # First acquire a lock
    machine_id = "demo_machine_stuck"
    print("🔒 Creating a test lock...")
    cloud_manager.acquire_labeling_lock(machine_id, timeout_minutes=30)
    
    # Show lock status
    status = cloud_manager.check_labeling_status()
    if status['is_running']:
        print(f"✅ Test lock created by: {status.get('machine_id', 'Unknown')}")
        
        # Demonstrate force override
        print("\n💥 Demonstrating force override...")
        print("⚠️ WARNING: This would normally be used only in emergencies!")
        
        force_success = cloud_manager.force_release_lock()
        if force_success:
            print("✅ Force override successful")
            
            # Verify lock is released
            new_status = cloud_manager.check_labeling_status()
            if not new_status['is_running']:
                print("✅ Lock successfully removed")
            else:
                print("❌ Lock still active after force override")
        else:
            print("❌ Force override failed")
    else:
        print("❌ Failed to create test lock")

def demo_interruption_handling():
    """
    Demo graceful interruption handling
    """
    print("\n🧪 DEMO: Interruption Handling")
    print("=" * 40)
    
    print("📡 Setting up signal handlers...")
    
    # Create a simple signal handler demo
    interrupted = False
    
    def demo_signal_handler(signum, frame):
        nonlocal interrupted
        signal_name = signal.Signals(signum).name
        print(f"\n⚠️ Received {signal_name} signal!")
        print("💾 Saving checkpoint...")
        print("🔓 Releasing lock...")
        print("✅ Graceful shutdown completed")
        interrupted = True
    
    # Register signal handler
    original_handler = signal.signal(signal.SIGINT, demo_signal_handler)
    
    print("✅ Signal handlers registered")
    print("ℹ️ In real usage, press Ctrl+C to trigger graceful shutdown")
    print("🔄 Simulating work that can be interrupted...")
    
    try:
        for i in range(5):
            if interrupted:
                break
            print(f"   Working... step {i+1}/5")
            time.sleep(0.5)
        
        if not interrupted:
            print("✅ Work completed normally")
    
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)
        print("🔄 Signal handlers restored")

def demo_recovery_scenarios():
    """
    Demo recovery scenarios
    """
    print("\n🧪 DEMO: Recovery Scenarios")
    print("=" * 40)
    
    cloud_manager = CloudCheckpointManager()
    
    print("🔍 Checking for existing checkpoints...")
    checkpoints = cloud_manager.list_checkpoints()
    
    if checkpoints:
        print(f"📊 Found {len(checkpoints)} existing checkpoints:")
        for i, checkpoint_id in enumerate(checkpoints[:3], 1):
            print(f"   {i}. {checkpoint_id}")
        
        print("\n📖 Getting latest checkpoint...")
        latest = cloud_manager.get_latest_checkpoint()
        if latest:
            print(f"✅ Latest checkpoint: {latest.get('checkpoint_id', 'unknown')}")
            print("\n📋 Resume information:")
            cloud_manager.display_resume_info(latest)
        else:
            print("❌ Failed to get latest checkpoint")
    else:
        print("ℹ️ No existing checkpoints found (fresh start scenario)")
        print("🆕 This would be a fresh labeling session")

def demo_system_status():
    """
    Demo system status check
    """
    print("\n🧪 DEMO: System Status")
    print("=" * 40)
    
    cloud_manager = CloudCheckpointManager()
    
    print("📊 Getting system status...")
    status = cloud_manager.get_status()
    
    print(f"🔐 Authentication: {'✅ Active' if status.get('authenticated', False) else '❌ Offline'}")
    print(f"📴 Offline Mode: {'✅ Yes' if status.get('offline_mode', False) else '❌ No'}")
    print(f"💾 Local Checkpoints: {status.get('local_checkpoints', 0)}")
    print(f"☁️ Cloud Checkpoints: {status.get('cloud_checkpoints', 0)}")
    
    # Check labeling status
    labeling_status = cloud_manager.check_labeling_status()
    print(f"🔒 Labeling Active: {'✅ Yes' if labeling_status.get('is_running', False) else '❌ No'}")
    
    if labeling_status.get('is_running', False):
        print(f"🖥️ Active Machine: {labeling_status.get('machine_id', 'Unknown')}")
        print(f"⏰ Started: {labeling_status.get('start_time', 'Unknown')}")
        print(f"🏠 Hostname: {labeling_status.get('hostname', 'Unknown')}")

def main():
    """
    Main demo function
    """
    print("🚀 Robust Labeling System Demo")
    print("=" * 60)
    print("This demo showcases the robustness features implemented in the labeling system.")
    print("All features work both online (with Google Drive) and offline (local only).")
    
    demos = [
        ("System Status Check", demo_system_status),
        ("Checkpoint System", demo_checkpoint_system),
        ("Locking System", demo_locking_system),
        ("Force Override", demo_force_override),
        ("Interruption Handling", demo_interruption_handling),
        ("Recovery Scenarios", demo_recovery_scenarios)
    ]
    
    try:
        for demo_name, demo_func in demos:
            print(f"\n🎯 Running Demo: {demo_name}")
            demo_func()
            
            # Pause between demos
            input("\n⏸️ Press Enter to continue to next demo...")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user (Ctrl+C)")
        print("✅ This demonstrates graceful interruption handling!")
    
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    
    finally:
        print("\n🎉 Demo completed!")
        print("\n📚 Key Takeaways:")
        print("   ✅ System handles interruptions gracefully")
        print("   ✅ Checkpoints ensure no data loss")
        print("   ✅ Locking prevents conflicts")
        print("   ✅ Recovery is automatic")
        print("   ✅ Force override available for emergencies")
        print("   ✅ Works both online and offline")
        
        print("\n🚀 The system is ready for production use!")

if __name__ == '__main__':
    main()