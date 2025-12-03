"""Logging utilities for the hate speech detection system."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def setup_logger(
    name: str, 
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, level.upper()))
    
    # Default format
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(file_handler)
    
    return logger

def get_model_logger() -> logging.Logger:
    """Get logger for model training and evaluation."""
    return setup_logger(
        "model", 
        log_file="logs/model.log",
        level="DEBUG"
    )

def get_api_logger() -> logging.Logger:
    """Get logger for API operations."""
    return setup_logger(
        "api", 
        log_file="logs/api.log",
        level="INFO"
    )

def get_data_logger() -> logging.Logger:
    """Get logger for data processing operations."""
    return setup_logger(
        "data", 
        log_file="logs/data.log",
        level="INFO"
    )

def log_model_metrics(logger: logging.Logger, metrics: dict, epoch: Optional[int] = None):
    """Log model training metrics in a structured format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics (accuracy, loss, etc.)
        epoch: Optional epoch number
    """
    epoch_str = f"Epoch {epoch}: " if epoch is not None else ""
    
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"{epoch_str}{metrics_str}")

def log_prediction(logger: logging.Logger, text: str, prediction: str, confidence: float):
    """Log prediction results.
    
    Args:
        logger: Logger instance
        text: Input text (truncated for privacy)
        prediction: Model prediction
        confidence: Prediction confidence score
    """
    # Truncate text for privacy and log size
    text_preview = text[:50] + "..." if len(text) > 50 else text
    logger.info(
        f"Prediction - Text: '{text_preview}', "
        f"Result: {prediction}, Confidence: {confidence:.4f}"
    )