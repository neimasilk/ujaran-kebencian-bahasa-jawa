#!/usr/bin/env python3
"""
Script untuk test training dengan GPU
Script ini akan menjalankan training singkat untuk memastikan GPU berfungsi dengan baik.

Usage:
    python test_gpu_training.py
"""

import sys
import os
from pathlib import Path
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from utils.logger import setup_logger

def test_gpu_setup():
    """Test GPU setup dan availability"""
    print("=== GPU Setup Test ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test GPU computation
        device = torch.device('cuda')
        print(f"Using device: {device}")
        
        # Test tensor operations on GPU
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        
        print(f"GPU computation test: {z.device} - SUCCESS")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        print("GPU cache cleared")
        
        return True
    else:
        print("❌ GPU tidak tersedia")
        print("Install PyTorch dengan CUDA support:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False

def test_model_loading():
    """Test loading model ke GPU"""
    print("\n=== Model Loading Test ===")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
        print("✅ Tokenizer loaded successfully")
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            "indobenchmark/indobert-base-p1",
            num_labels=4
        )
        print("✅ Model loaded successfully")
        
        # Move model to GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model.to(device)
            print(f"✅ Model moved to {device}")
            
            # Test model inference on GPU
            test_text = "Iki conto teks Jawa kanggo testing."
            inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
            
            # Move inputs to GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            print(f"✅ Model inference on GPU successful")
            print(f"Prediction shape: {predictions.shape}")
            print(f"GPU Memory used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            return True
        else:
            print("⚠️ GPU tidak tersedia, menggunakan CPU")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_training_components():
    """Test komponen training"""
    print("\n=== Training Components Test ===")
    
    try:
        from modelling.train_model import load_labeled_data, prepare_datasets
        
        # Create dummy data for testing
        dummy_data = pd.DataFrame({
            'processed_text': [
                'Iki teks Jawa sing apik',
                'Teks liyane kanggo testing',
                'Conto teks katelu',
                'Teks kaping papat',
                'Teks kaping lima'
            ],
            'label': [0, 1, 2, 3, 0],  # Changed from 'final_label' to 'label'
            'confidence_score': [0.9, 0.8, 0.85, 0.95, 0.88],
            'error': [None, None, None, None, None]
        })
        
        print("✅ Dummy data created")
        
        # Test prepare_datasets
        train_ds, val_ds = prepare_datasets(dummy_data, test_size=0.2)
        
        if train_ds and val_ds:
            print(f"✅ Datasets prepared - Train: {len(train_ds)}, Val: {len(val_ds)}")
            
            # Test data loading
            sample = train_ds[0]
            print(f"✅ Sample data shape: {sample['input_ids'].shape}")
            
            return True
        else:
            print("❌ Failed to prepare datasets")
            return False
            
    except Exception as e:
        print(f"❌ Error testing training components: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 GPU Training Test Suite")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logger("gpu_test", level="INFO")
    
    # Test results
    results = []
    
    # Test 1: GPU Setup
    results.append(test_gpu_setup())
    
    # Test 2: Model Loading
    results.append(test_model_loading())
    
    # Test 3: Training Components
    results.append(test_training_components())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    test_names = ["GPU Setup", "Model Loading", "Training Components"]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    all_passed = all(results)
    
    if all_passed:
        print("\n🎉 Semua test berhasil! GPU siap untuk training.")
        print("\n🚀 Untuk memulai training:")
        print("python train_model.py --data_path ./hasil-labeling.csv --output_dir ./models/hate_speech_model")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n💡 Optimasi untuk {gpu_name} ({gpu_memory:.0f}GB):")
            print("python train_model.py --batch_size 32 --eval_batch_size 64 --epochs 3")
    else:
        print("\n❌ Beberapa test gagal. Periksa konfigurasi GPU dan dependencies.")
        print("\n🔧 Troubleshooting:")
        print("1. Install PyTorch dengan CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("2. Pastikan driver NVIDIA terbaru terinstal")
        print("3. Restart terminal setelah instalasi")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)