#!/usr/bin/env python3
"""Evaluation script for Javanese hate speech detection model."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.models.evaluate_model import main as evaluate_main
from app.utils.logger import logger

if __name__ == "__main__":
    logger.info("Starting model evaluation...")
    try:
        evaluate_main()
        logger.info("Evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)