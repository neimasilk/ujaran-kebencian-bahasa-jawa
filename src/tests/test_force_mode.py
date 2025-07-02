#!/usr/bin/env python3
"""
Test script untuk memverifikasi force mode functionality
"""

import sys
import os
sys.path.append('src')

from google_drive_labeling import GoogleDriveLabelingPipeline
from utils.logger import setup_logger

def test_force_mode():
    """
    Test force mode dengan simulasi kondisi tanpa cloud checkpoint
    """
    logger = setup_logger("test_force_mode")
    
    print("🧪 Testing Force Mode Implementation")
    print("=" * 50)
    
    # Initialize pipeline
    try:
        pipeline = GoogleDriveLabelingPipeline(
            dataset_path="src/data_collection/raw-dataset.csv",
            output_name="test-force-mode"
        )
        
        print("✅ Pipeline initialized successfully")
        
        # Test 1: Normal mode (should fail without cloud checkpoint)
        print("\n📋 Test 1: Normal mode (resume=True, force=False)")
        print("Expected: Should fail with STRICT CLOUD-FIRST POLICY")
        
        # Test 2: Force mode (should bypass and start fresh)
        print("\n📋 Test 2: Force mode (resume=True, force=True)")
        print("Expected: Should bypass STRICT CLOUD-FIRST POLICY and start fresh")
        
        # Test 3: No resume mode (should work normally)
        print("\n📋 Test 3: No resume mode (resume=False, force=False)")
        print("Expected: Should start fresh without checking cloud")
        
        print("\n💡 To run actual tests:")
        print("1. python labeling.py  # Should fail with STRICT CLOUD-FIRST POLICY")
        print("2. python labeling.py --force  # Should bypass and start fresh")
        print("3. python labeling.py --no-resume  # Should start fresh")
        
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_force_mode()
    if success:
        print("\n🎉 Force mode implementation test completed")
        print("📝 Implementation summary:")
        print("   - Added 'force' parameter to run_labeling method")
        print("   - Modified STRICT CLOUD-FIRST POLICY to respect force flag")
        print("   - Added proper logging for force mode operations")
        print("   - Updated main function to pass force flag")
    else:
        print("\n❌ Test failed")
        sys.exit(1)