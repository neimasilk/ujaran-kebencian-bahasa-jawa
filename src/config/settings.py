"""Configuration settings for the hate speech detection system."""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Konfigurasi aplikasi menggunakan pydantic BaseSettings.
    
    Semua setting dapat di-override melalui environment variables.
    Contoh: MODEL_NAME=custom-bert python app.py
    """
    
    model_config = {"extra": "ignore", "env_file": ".env", "env_file_encoding": "utf-8"}
    
    # Model Configuration
    model_name: str = "indolem/indobert-base-uncased"
    model_max_length: int = 512
    num_labels: int = 4
    model_cache_dir: str = "./src/models/cache"
    
    # Data Configuration  
    data_dir: str = "./src/data"
    raw_dataset_path: str = "./src/data_collection/raw-dataset.csv"
    processed_dataset_path: str = "./src/data/processed/dataset.csv"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_timeout: int = 30
    
    # DeepSeek API Configuration
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"  # Points to DeepSeek-V3
    deepseek_max_tokens: int = 100
    deepseek_temperature: float = 0.1
    deepseek_batch_size: int = 10  # Smaller batch for testing
    deepseek_rate_limit: int = 100  # requests per minute
    
    # Logging Configuration
    log_level: str = "INFO"
    log_dir: str = "./logs"
    log_file: str = "hate_speech_detection.log"
    log_max_size: str = "10MB"
    log_backup_count: int = 5
    
    # Security Configuration
    api_key_header: str = "X-API-Key"
    allowed_origins: List[str] = ["*"]
    rate_limit_per_minute: int = 60
    
    # Model Training
    model_save_path: str = "src/models/"
    checkpoint_dir: str = "src/checkpoints/"
    
    # Labels Configuration
    label_mapping: dict = {
        "bukan_ujaran_kebencian": 0,
        "ujaran_kebencian_ringan": 1,
        "ujaran_kebencian_sedang": 2,
        "ujaran_kebencian_berat": 3
    }
    


# Global settings instance
settings = Settings()

# Create necessary directories with absolute paths
from pathlib import Path
base_dir = Path(__file__).parent.parent  # Points to src directory

# Create directories relative to src
log_dir = base_dir / "logs" if not settings.log_file else (base_dir / settings.log_file).parent
os.makedirs(log_dir, exist_ok=True)

# Ensure model_save_path and checkpoint_dir are absolute
model_path = base_dir / settings.model_save_path.replace("src/", "").replace("./src/", "")
checkpoint_path = base_dir / settings.checkpoint_dir.replace("src/", "").replace("./src/", "")

os.makedirs(model_path, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)