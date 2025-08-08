#!/usr/bin/env python3
"""
Safely stop hyperparameter tuning and save checkpoint

Author: AI Assistant
Date: 2025-01-24
"""

import os
import signal
import psutil
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_hyperparameter_process():
    """Find running hyperparameter tuning process"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'hyperparameter_tuning.py' in cmdline:
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def stop_hyperparameter_tuning():
    """Safely stop hyperparameter tuning process"""
    logger.info("Looking for hyperparameter tuning process...")
    
    proc = find_hyperparameter_process()
    if not proc:
        logger.info("No hyperparameter tuning process found.")
        return
    
    logger.info(f"Found hyperparameter tuning process (PID: {proc.pid})")
    logger.info("Sending interrupt signal to save checkpoint...")
    
    try:
        # Send SIGINT (Ctrl+C) to allow graceful shutdown
        if os.name == 'nt':  # Windows
            proc.send_signal(signal.CTRL_C_EVENT)
        else:  # Unix/Linux
            proc.send_signal(signal.SIGINT)
        
        logger.info("Interrupt signal sent. Waiting for process to save checkpoint...")
        
        # Wait for process to finish gracefully (max 30 seconds)
        try:
            proc.wait(timeout=30)
            logger.info("Process stopped gracefully. Checkpoint should be saved.")
        except psutil.TimeoutExpired:
            logger.warning("Process didn't stop within 30 seconds. Forcing termination...")
            proc.terminate()
            proc.wait(timeout=10)
            logger.info("Process terminated.")
            
    except Exception as e:
        logger.error(f"Error stopping process: {str(e)}")
        return
    
    # Check if checkpoint file exists
    checkpoint_file = Path("experiments/results/hyperparameter_tuning/checkpoint.json")
    if checkpoint_file.exists():
        logger.info(f"✓ Checkpoint file found: {checkpoint_file}")
        logger.info("You can resume the hyperparameter tuning by running:")
        logger.info("python experiments/hyperparameter_tuning.py")
    else:
        logger.warning("⚠ Checkpoint file not found. Process may not have saved properly.")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("SAFELY STOPPING HYPERPARAMETER TUNING")
    logger.info("=" * 60)
    
    stop_hyperparameter_tuning()
    
    logger.info("=" * 60)
    logger.info("STOP OPERATION COMPLETED")
    logger.info("=" * 60)