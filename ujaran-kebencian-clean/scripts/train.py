#!/usr/bin/env python3
"""Training script for Javanese hate speech detection model."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.models.train_model import main as train_main
from app.utils.logger import logger

if __name__ == "__main__":
    logger.info("Starting model training...")
    try:
        train_main()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)