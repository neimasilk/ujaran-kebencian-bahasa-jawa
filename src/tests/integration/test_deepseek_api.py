"""Integration tests for DeepSeek API.

Tests the actual DeepSeek API integration with real API calls.
This test requires valid DEEPSEEK_API_KEY in environment.

Author: AI Assistant
Date: 2025-01-01
"""

import unittest
import sys
import pandas as pd
from pathlib import Path
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import Settings
from utils.deepseek_client import create_deepseek_client
from utils.logger import setup_logger

class TestDeepSeekAPIIntegration(unittest.TestCase):
    """Integration tests for DeepSeek API."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.logger = setup_logger("test_deepseek_api")
        cls.settings = Settings()
        
        # Skip tests if no API key
        if not cls.settings.deepseek_api_key:
            raise unittest.SkipTest("DeepSeek API key not found. Set DEEPSEEK_API_KEY in .env file.")
        
        # Load test dataset
        dataset_path = Path(__file__).parent.parent.parent / "data_collection" / "raw-dataset.csv"
        if not dataset_path.exists():
            raise unittest.SkipTest(f"Test dataset not found: {dataset_path}")
        
        # Load and prepare test data
        df = pd.read_csv(dataset_path, header=None, names=['text', 'label'])
        cls.negative_samples = df[df['label'] == 'negative'].head(5).to_dict('records')
        cls.positive_samples = df[df['label'] == 'positive'].head(3).to_dict('records')
        
        if not cls.negative_samples:
            raise unittest.SkipTest("No negative samples found in dataset")
    
    def setUp(self):
        """Set up each test."""
        self.client = create_deepseek_client(mock=False, settings=self.settings)
    
    def test_api_connection(self):
        """Test basic API connection."""
        self.assertIsNotNone(self.client)
        self.logger.info("✅ DeepSeek client created successfully")
    
    def test_single_negative_sample(self):
        """Test labeling a single negative sample."""
        if not self.negative_samples:
            self.skipTest("No negative samples available")
        
        sample = self.negative_samples[0]
        text = sample['text']
        
        self.logger.info(f"Testing negative sample: {text[:50]}...")
        
        result = self.client.label_single_text(text)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIsNone(result.error, f"API error: {result.error}")
        self.assertIsNotNone(result.label_id)
        self.assertIn(result.label_id, [0, 1, 2, 3], "Label ID should be 0-3")
        self.assertIsNotNone(result.confidence)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertIsNotNone(result.response_time)
        self.assertGreater(result.response_time, 0)
        
        self.logger.info(f"✅ Result: Label {result.label_id}, Confidence {result.confidence:.3f}")
    
    def test_single_positive_sample(self):
        """Test labeling a single positive sample."""
        if not self.positive_samples:
            self.skipTest("No positive samples available")
        
        sample = self.positive_samples[0]
        text = sample['text']
        
        self.logger.info(f"Testing positive sample: {text[:50]}...")
        
        result = self.client.label_single_text(text)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIsNone(result.error, f"API error: {result.error}")
        self.assertIsNotNone(result.label_id)
        self.assertIn(result.label_id, [0, 1, 2, 3], "Label ID should be 0-3")
        
        self.logger.info(f"✅ Result: Label {result.label_id}, Confidence {result.confidence:.3f}")
    
    def test_batch_negative_samples(self):
        """Test labeling multiple negative samples."""
        if len(self.negative_samples) < 3:
            self.skipTest("Not enough negative samples for batch test")
        
        results = []
        
        for i, sample in enumerate(self.negative_samples[:3]):
            text = sample['text']
            self.logger.info(f"Processing negative sample {i+1}/3: {text[:30]}...")
            
            result = self.client.label_single_text(text)
            
            # Basic assertions for each result
            self.assertIsNotNone(result)
            if result.error:
                self.logger.warning(f"API error for sample {i+1}: {result.error}")
            else:
                self.assertIn(result.label_id, [0, 1, 2, 3])
                results.append(result)
        
        # At least some results should be successful
        self.assertGreater(len(results), 0, "At least one sample should be processed successfully")
        
        # Calculate averages
        if results:
            avg_confidence = sum(r.confidence for r in results) / len(results)
            avg_response_time = sum(r.response_time for r in results) / len(results)
            
            self.logger.info(f"✅ Batch test completed: {len(results)} successful")
            self.logger.info(f"   Average confidence: {avg_confidence:.3f}")
            self.logger.info(f"   Average response time: {avg_response_time:.2f}s")
    
    def test_api_usage_stats(self):
        """Test API usage statistics."""
        # Make at least one API call
        if self.negative_samples:
            sample = self.negative_samples[0]
            self.client.label_single_text(sample['text'])
        
        stats = self.client.get_usage_stats()
        
        # Check that stats is a dictionary
        self.assertIsInstance(stats, dict)
        
        # Log stats for debugging
        self.logger.info("API Usage Stats:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
    
    def test_error_handling(self):
        """Test error handling with invalid input."""
        # Test empty string
        result = self.client.label_single_text("")
        # Should either handle gracefully or return error
        self.assertIsNotNone(result)
        
        # Test very long string (if applicable)
        long_text = "a" * 10000
        result = self.client.label_single_text(long_text)
        self.assertIsNotNone(result)
        
        self.logger.info("✅ Error handling test completed")

def run_integration_test():
    """Run integration tests with detailed output."""
    # Setup logging
    logger = setup_logger("test_deepseek_integration")
    
    logger.info("🚀 Starting DeepSeek API Integration Tests")
    logger.info("="*60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDeepSeekAPIIntegration)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        logger.error("\nFailures:")
        for test, traceback in result.failures:
            logger.error(f"  {test}: {traceback}")
    
    if result.errors:
        logger.error("\nErrors:")
        for test, traceback in result.errors:
            logger.error(f"  {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)