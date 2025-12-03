#!/usr/bin/env python3
"""
Script untuk testing API Javanese Hate Speech Detection

Usage:
    python test_api.py [--base-url BASE_URL]

Example:
    python test_api.py --base-url http://localhost:8000
"""

import requests
import json
import argparse
import time
import logging
from typing import List, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def test_health(self) -> bool:
        """Test health endpoint"""
        try:
            logger.info("Testing health endpoint...")
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Health check: {data}")
                return data.get('model_loaded', False)
            else:
                logger.error(f"Health check failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing health: {e}")
            return False
    
    def test_root(self) -> bool:
        """Test root endpoint"""
        try:
            logger.info("Testing root endpoint...")
            response = self.session.get(f"{self.base_url}/")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Root response: {data}")
                return True
            else:
                logger.error(f"Root test failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing root: {e}")
            return False
    
    def test_model_info(self) -> bool:
        """Test model info endpoint"""
        try:
            logger.info("Testing model info endpoint...")
            response = self.session.get(f"{self.base_url}/model-info")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Model info: {json.dumps(data, indent=2)}")
                return True
            elif response.status_code == 503:
                logger.warning("Model belum dimuat")
                return False
            else:
                logger.error(f"Model info test failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing model info: {e}")
            return False
    
    def test_single_prediction(self, text: str) -> Dict:
        """Test single prediction"""
        try:
            logger.info(f"Testing single prediction for: '{text[:50]}...'")
            
            payload = {"text": text}
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Prediction result: {json.dumps(data, indent=2)}")
                return data
            else:
                logger.error(f"Single prediction failed: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error testing single prediction: {e}")
            return {}
    
    def test_batch_prediction(self, texts: List[str]) -> Dict:
        """Test batch prediction"""
        try:
            logger.info(f"Testing batch prediction for {len(texts)} texts")
            
            payload = {"texts": texts}
            response = self.session.post(
                f"{self.base_url}/batch-predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Batch prediction completed in {data['total_processing_time']:.3f}s")
                for i, pred in enumerate(data['predictions']):
                    logger.info(f"  Text {i+1}: {pred['predicted_label']} (confidence: {pred['confidence']:.3f})")
                return data
            else:
                logger.error(f"Batch prediction failed: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error testing batch prediction: {e}")
            return {}
    
    def test_error_handling(self) -> bool:
        """Test error handling"""
        logger.info("Testing error handling...")
        
        # Test empty text
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json={"text": ""},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 400:
                logger.info("✓ Empty text error handling works")
            else:
                logger.warning(f"Empty text should return 400, got {response.status_code}")
        except Exception as e:
            logger.error(f"Error testing empty text: {e}")
        
        # Test invalid JSON
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                data="invalid json",
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 422:
                logger.info("✓ Invalid JSON error handling works")
            else:
                logger.warning(f"Invalid JSON should return 422, got {response.status_code}")
        except Exception as e:
            logger.error(f"Error testing invalid JSON: {e}")
        
        # Test large batch
        try:
            large_batch = ["test text"] * 101  # Over the limit
            response = self.session.post(
                f"{self.base_url}/batch-predict",
                json={"texts": large_batch},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 400:
                logger.info("✓ Large batch error handling works")
            else:
                logger.warning(f"Large batch should return 400, got {response.status_code}")
        except Exception as e:
            logger.error(f"Error testing large batch: {e}")
        
        return True
    
    def run_comprehensive_test(self):
        """Run comprehensive API test"""
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE API TEST")
        logger.info(f"Base URL: {self.base_url}")
        logger.info("=" * 60)
        
        # Test basic endpoints
        logger.info("\n1. Testing basic endpoints...")
        root_ok = self.test_root()
        health_ok = self.test_health()
        
        if not health_ok:
            logger.warning("Model tidak dimuat. Beberapa test akan dilewati.")
            return
        
        model_info_ok = self.test_model_info()
        
        # Test predictions
        logger.info("\n2. Testing predictions...")
        
        # Sample Javanese texts for testing
        test_texts = [
            "Sugeng enjing, piye kabare?",  # Good morning, how are you?
            "Kowe ki bodho tenan!",  # You are really stupid!
            "Aku seneng banget karo kowe",  # I really like you
            "Matamu ki ireng kaya setan",  # Your eyes are black like a demon
            "Ayo padha sinau bareng-bareng"  # Let's study together
        ]
        
        # Single predictions
        for text in test_texts:
            result = self.test_single_prediction(text)
            time.sleep(0.5)  # Small delay between requests
        
        # Batch prediction
        logger.info("\n3. Testing batch prediction...")
        batch_result = self.test_batch_prediction(test_texts)
        
        # Error handling
        logger.info("\n4. Testing error handling...")
        self.test_error_handling()
        
        logger.info("\n" + "=" * 60)
        logger.info("COMPREHENSIVE TEST COMPLETED")
        logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Test Javanese Hate Speech Detection API"
    )
    parser.add_argument(
        "--base-url", 
        default="http://localhost:8000", 
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--text", 
        help="Single text to test (if not provided, runs comprehensive test)"
    )
    parser.add_argument(
        "--batch", 
        nargs="+", 
        help="Multiple texts for batch testing"
    )
    
    args = parser.parse_args()
    
    tester = APITester(args.base_url)
    
    if args.text:
        # Test single text
        tester.test_single_prediction(args.text)
    elif args.batch:
        # Test batch
        tester.test_batch_prediction(args.batch)
    else:
        # Run comprehensive test
        tester.run_comprehensive_test()

if __name__ == "__main__":
    main()