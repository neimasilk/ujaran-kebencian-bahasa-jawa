#!/usr/bin/env python3
"""
Script untuk menjalankan API server Javanese Hate Speech Detection

Usage:
    python run_server.py [--host HOST] [--port PORT] [--reload]

Example:
    python run_server.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import uvicorn
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Javanese Hate Speech Detection API Server"
    )
    parser.add_argument(
        "--host", 
        default="127.0.0.1", 
        help="Host to bind the server (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind the server (default: 8000)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1, 
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--log-level", 
        default="info", 
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level (default: info)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting Javanese Hate Speech Detection API...")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Reload: {args.reload}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Log Level: {args.log_level}")
    
    # Check if model exists
    model_path = "models/bert_jawa_hate_speech"
    if not os.path.exists(model_path):
        logger.warning(f"Model path tidak ditemukan: {model_path}")
        logger.warning("API akan berjalan tapi prediksi tidak akan tersedia.")
        logger.warning("Pastikan model sudah dilatih dan disimpan di path yang benar.")
    else:
        logger.info(f"Model ditemukan di: {model_path}")
    
    try:
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,  # reload mode only works with 1 worker
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Server dihentikan oleh user")
    except Exception as e:
        logger.error(f"Error menjalankan server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()