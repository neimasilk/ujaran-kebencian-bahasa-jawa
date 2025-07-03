#!/usr/bin/env python3
"""
Production-ready deployment script untuk Javanese hate speech detection.
Mengintegrasikan semua perbaikan: improved model, threshold tuning, dan monitoring.
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from dataclasses import dataclass

# Add src to path
sys.path.append('src')
from utils.logger import setup_logger

@dataclass
class PredictionResult:
    """Struktur hasil prediksi"""
    text: str
    predicted_label: str
    confidence: float
    all_probabilities: Dict[str, float]
    threshold_applied: bool
    processing_time: float
    timestamp: str

class ProductionHateSpeechDetector:
    """Production-ready hate speech detector dengan semua optimasi"""
    
    def __init__(self, 
                 model_path: str,
                 threshold_config_path: Optional[str] = None,
                 device: str = "auto"):
        """
        Initialize production detector
        
        Args:
            model_path: Path ke trained model
            threshold_config_path: Path ke threshold configuration
            device: Device untuk inference (auto, cpu, cuda)
        """
        self.model_path = model_path
        self.threshold_config_path = threshold_config_path
        self.logger = setup_logger("production_detector")
        
        # Label mapping
        self.label_mapping = {
            0: "Bukan Ujaran Kebencian",
            1: "Ujaran Kebencian - Ringan", 
            2: "Ujaran Kebencian - Sedang",
            3: "Ujaran Kebencian - Berat"
        }
        
        # Setup device
        self.device = self._setup_device(device)
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Load threshold configuration
        self._load_threshold_config()
        
        # Initialize monitoring
        self.prediction_count = 0
        self.total_processing_time = 0.0
        
        self.logger.info("Production hate speech detector initialized successfully")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                self.logger.info("Using CPU")
        
        return torch.device(device)
    
    def _load_model_and_tokenizer(self):
        """Load model dan tokenizer"""
        self.logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_threshold_config(self):
        """Load threshold configuration jika tersedia"""
        self.thresholds = None
        
        if self.threshold_config_path and os.path.exists(self.threshold_config_path):
            try:
                with open(self.threshold_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Extract thresholds
                if 'optimal_thresholds' in config:
                    self.thresholds = {}
                    for label_name, threshold in config['optimal_thresholds'].items():
                        # Find label index
                        for idx, name in self.label_mapping.items():
                            if name == label_name:
                                self.thresholds[idx] = threshold
                                break
                    
                    self.logger.info(f"Loaded optimized thresholds: {self.thresholds}")
                else:
                    self.logger.warning("No optimal_thresholds found in config")
            
            except Exception as e:
                self.logger.warning(f"Failed to load threshold config: {e}")
        else:
            self.logger.info("No threshold configuration provided, using default thresholds")
    
    def _preprocess_text(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Preprocess text untuk model"""
        # Basic text cleaning
        text = text.strip()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def _apply_thresholds(self, probabilities: np.ndarray) -> Tuple[int, bool]:
        """Apply optimized thresholds jika tersedia"""
        if self.thresholds is None:
            # Default: argmax
            return int(np.argmax(probabilities)), False
        
        # Check threshold untuk setiap kelas
        threshold_candidates = []
        for class_idx, threshold in self.thresholds.items():
            if probabilities[class_idx] >= threshold:
                threshold_candidates.append((class_idx, probabilities[class_idx]))
        
        if threshold_candidates:
            # Pilih kelas dengan probabilitas tertinggi yang memenuhi threshold
            best_class = max(threshold_candidates, key=lambda x: x[1])[0]
            return best_class, True
        else:
            # Fallback ke argmax
            return int(np.argmax(probabilities)), False
    
    def predict(self, text: str) -> PredictionResult:
        """Predict hate speech untuk single text"""
        start_time = datetime.now()
        
        try:
            # Preprocess
            inputs = self._preprocess_text(text)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # Apply thresholds
            predicted_class, threshold_applied = self._apply_thresholds(probabilities)
            predicted_label = self.label_mapping[predicted_class]
            confidence = float(probabilities[predicted_class])
            
            # Create probability dictionary
            all_probabilities = {
                label: float(prob) for label, prob in zip(self.label_mapping.values(), probabilities)
            }
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update monitoring
            self.prediction_count += 1
            self.total_processing_time += processing_time
            
            # Create result
            result = PredictionResult(
                text=text,
                predicted_label=predicted_label,
                confidence=confidence,
                all_probabilities=all_probabilities,
                threshold_applied=threshold_applied,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed for text: {text[:50]}... Error: {e}")
            raise
    
    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[PredictionResult]:
        """Predict hate speech untuk batch texts"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                result = self.predict(text)
                results.append(result)
        
        return results
    
    def get_monitoring_stats(self) -> Dict:
        """Get monitoring statistics"""
        avg_processing_time = (
            self.total_processing_time / self.prediction_count 
            if self.prediction_count > 0 else 0
        )
        
        return {
            'total_predictions': self.prediction_count,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_processing_time,
            'predictions_per_second': 1 / avg_processing_time if avg_processing_time > 0 else 0,
            'device': str(self.device),
            'threshold_enabled': self.thresholds is not None
        }
    
    def export_prediction_config(self, output_path: str):
        """Export konfigurasi untuk deployment"""
        config = {
            'model_path': self.model_path,
            'threshold_config_path': self.threshold_config_path,
            'device': str(self.device),
            'label_mapping': self.label_mapping,
            'thresholds': self.thresholds,
            'export_date': datetime.now().isoformat(),
            'monitoring_stats': self.get_monitoring_stats()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Configuration exported to {output_path}")

class ProductionAPI:
    """Simple API wrapper untuk production deployment"""
    
    def __init__(self, detector: ProductionHateSpeechDetector):
        self.detector = detector
        self.logger = setup_logger("production_api")
    
    def health_check(self) -> Dict:
        """Health check endpoint"""
        try:
            # Test prediction
            test_result = self.detector.predict("Test text")
            
            return {
                'status': 'healthy',
                'model_loaded': True,
                'device': str(self.detector.device),
                'threshold_enabled': self.detector.thresholds is not None,
                'test_prediction_time': test_result.processing_time,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_text(self, text: str) -> Dict:
        """API endpoint untuk single prediction"""
        try:
            result = self.detector.predict(text)
            
            return {
                'success': True,
                'prediction': {
                    'label': result.predicted_label,
                    'confidence': result.confidence,
                    'all_probabilities': result.all_probabilities,
                    'threshold_applied': result.threshold_applied
                },
                'metadata': {
                    'processing_time': result.processing_time,
                    'timestamp': result.timestamp
                }
            }
        except Exception as e:
            self.logger.error(f"API prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_stats(self) -> Dict:
        """API endpoint untuk monitoring stats"""
        return {
            'success': True,
            'stats': self.detector.get_monitoring_stats(),
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Demo production deployment"""
    print("=== PRODUCTION HATE SPEECH DETECTION DEMO ===")
    
    # Configuration
    model_path = "models/trained_model"  # Ganti dengan improved model jika sudah ditraining
    threshold_config_path = "threshold_tuning_results.json"  # Jika sudah ada
    
    try:
        # Initialize detector
        print("Initializing production detector...")
        detector = ProductionHateSpeechDetector(
            model_path=model_path,
            threshold_config_path=threshold_config_path if os.path.exists(threshold_config_path) else None
        )
        
        # Initialize API
        api = ProductionAPI(detector)
        
        # Health check
        print("\nPerforming health check...")
        health = api.health_check()
        print(f"Health status: {health['status']}")
        
        # Demo predictions
        test_texts = [
            "Selamat pagi, semoga hari ini menyenangkan",
            "Kamu bodoh sekali, tidak berguna",
            "Aku benci banget sama orang kayak kamu",
            "Mari kita bekerja sama untuk kemajuan bersama"
        ]
        
        print("\nDemo predictions:")
        for i, text in enumerate(test_texts, 1):
            print(f"\n{i}. Text: {text}")
            result = api.predict_text(text)
            
            if result['success']:
                pred = result['prediction']
                print(f"   Prediction: {pred['label']}")
                print(f"   Confidence: {pred['confidence']:.4f}")
                print(f"   Threshold applied: {pred['threshold_applied']}")
                print(f"   Processing time: {result['metadata']['processing_time']:.4f}s")
            else:
                print(f"   Error: {result['error']}")
        
        # Show stats
        print("\nMonitoring stats:")
        stats = api.get_stats()
        if stats['success']:
            for key, value in stats['stats'].items():
                print(f"  {key}: {value}")
        
        # Export configuration
        config_path = "production_config.json"
        detector.export_prediction_config(config_path)
        print(f"\nConfiguration exported to: {config_path}")
        
        print("\n=== PRODUCTION DEPLOYMENT READY ===")
        print("Next steps:")
        print("1. Deploy detector dalam production environment")
        print("2. Setup monitoring dan logging")
        print("3. Implement rate limiting dan caching")
        print("4. Setup automated model updates")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()