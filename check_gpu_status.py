#!/usr/bin/env python3
"""
Script untuk memeriksa status GPU dan device yang digunakan
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_status():
    """Periksa status GPU dan CUDA"""
    logger.info("=" * 50)
    logger.info("GPU STATUS CHECK")
    logger.info("=" * 50)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # GPU information
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        
        logger.info(f"GPU Count: {gpu_count}")
        logger.info(f"Current Device: {current_device}")
        logger.info(f"GPU Name: {gpu_name}")
        
        # Memory information
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        logger.info(f"Memory Allocated: {memory_allocated:.2f} GB")
        logger.info(f"Memory Reserved: {memory_reserved:.2f} GB")
        logger.info(f"Total Memory: {memory_total:.2f} GB")
    else:
        logger.info("GPU not available - using CPU")
    
    # Check default device
    device = torch.device('cuda' if cuda_available else 'cpu')
    logger.info(f"Default Device: {device}")
    
    return cuda_available, device

def test_model_device():
    """Test apakah model menggunakan GPU"""
    logger.info("\n" + "=" * 50)
    logger.info("MODEL DEVICE TEST")
    logger.info("=" * 50)
    
    try:
        # Load model
        model_name = "indobenchmark/indobert-base-p1"
        logger.info(f"Loading model: {model_name}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=4
        )
        
        # Check model device
        model_device = next(model.parameters()).device
        logger.info(f"Model Device (before): {model_device}")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            model_device_after = next(model.parameters()).device
            logger.info(f"Model Device (after .cuda()): {model_device_after}")
        else:
            logger.info("GPU not available - model remains on CPU")
        
        # Test tensor operations
        test_tensor = torch.randn(1, 128)
        if torch.cuda.is_available():
            test_tensor = test_tensor.cuda()
            logger.info(f"Test Tensor Device: {test_tensor.device}")
        
        logger.info("Model device test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model device test: {str(e)}")

def main():
    """Main function"""
    cuda_available, device = check_gpu_status()
    test_model_device()
    
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    
    if cuda_available:
        logger.info("✅ GPU tersedia dan dapat digunakan untuk training")
        logger.info("✅ Model akan menggunakan GPU secara otomatis")
        logger.info("✅ fp16 training akan diaktifkan")
    else:
        logger.info("❌ GPU tidak tersedia - training menggunakan CPU")
        logger.info("❌ fp16 training tidak akan diaktifkan")
    
    logger.info(f"Device yang akan digunakan: {device}")

if __name__ == "__main__":
    main()