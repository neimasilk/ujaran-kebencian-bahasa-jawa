#!/usr/bin/env python3
"""
Script untuk menguji DeepSeek API dengan 10 kasus negatif (hate speech) saja.
Menggunakan DeepSeek V3 model untuk labeling detail.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config.settings import Settings
from utils.deepseek_client import create_deepseek_client
from utils.logger import setup_logger

# Setup logging
logger = setup_logger("test_deepseek_negative")

def main():
    """Test DeepSeek API dengan 10 kasus negatif."""
    
    # Load settings
    settings = Settings()
    
    # Validate API key
    if not settings.deepseek_api_key:
        logger.error("DeepSeek API key tidak ditemukan. Set DEEPSEEK_API_KEY di .env file.")
        return
    
    logger.info(f"Using DeepSeek model: {settings.deepseek_model}")
    logger.info(f"API endpoint: {settings.deepseek_base_url}")
    
    # Load dataset
    dataset_path = Path("src/data_collection/raw-dataset.csv")
    if not dataset_path.exists():
        logger.error(f"Dataset tidak ditemukan: {dataset_path}")
        return
    
    # Load CSV without header, assign column names
    df = pd.read_csv(dataset_path, header=None, names=['text', 'label'])
    logger.info(f"Dataset loaded: {len(df)} total records")
    
    # Filter hanya data negatif (hate speech)
    negative_df = df[df['label'] == 'negative'].head(10)
    logger.info(f"Selected {len(negative_df)} negative samples for testing")
    
    if negative_df.empty:
        logger.error("Tidak ada data negatif ditemukan dalam dataset")
        return
    
    # Create DeepSeek client (real API, not mock)
    client = create_deepseek_client(mock=False, settings=settings)
    
    # Test labeling
    logger.info("Starting DeepSeek API testing...")
    results = []
    
    for idx, row in negative_df.iterrows():
        text = row['text']
        logger.info(f"Processing sample {len(results)+1}/10: {text[:50]}...")
        
        try:
            # Label dengan DeepSeek API
            result = client.label_single_text(text)
            
            # Mapping label ke nama kategori
            label_names = {
                0: "Bukan Ujaran Kebencian",
                1: "Ujaran Kebencian Ringan", 
                2: "Ujaran Kebencian Sedang",
                3: "Ujaran Kebencian Berat"
            }
            
            result_data = {
                'original_text': text,
                'original_label': row['label'],
                'deepseek_label_id': result.label_id,
                'deepseek_label_name': label_names.get(result.label_id, 'Unknown'),
                'confidence': result.confidence,
                'response_time': result.response_time,
                'error': result.error
            }
            
            results.append(result_data)
            
            # Print hasil
            print(f"\n--- Sample {len(results)} ---")
            print(f"Text: {text}")
            print(f"Original: {row['label']}")
            print(f"DeepSeek: {result_data['deepseek_label_name']} (ID: {result.label_id})")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Response time: {result.response_time:.2f}s")
            if result.error:
                print(f"Error: {result.error}")
            
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            result_data = {
                'original_text': text,
                'original_label': row['label'],
                'deepseek_label_id': None,
                'deepseek_label_name': 'Error',
                'confidence': None,
                'response_time': None,
                'error': str(e)
            }
            results.append(result_data)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = "test_deepseek_negative_10_results.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY HASIL TESTING DEEPSEEK V3")
    print("="*60)
    
    successful = len([r for r in results if r['error'] is None])
    failed = len(results) - successful
    
    print(f"Total samples: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        avg_confidence = sum([r['confidence'] for r in results if r['confidence'] is not None]) / successful
        avg_response_time = sum([r['response_time'] for r in results if r['response_time'] is not None]) / successful
        
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Average response time: {avg_response_time:.2f}s")
        
        # Label distribution
        label_dist = {}
        for r in results:
            if r['deepseek_label_name'] and r['deepseek_label_name'] != 'Error':
                label_dist[r['deepseek_label_name']] = label_dist.get(r['deepseek_label_name'], 0) + 1
        
        print("\nLabel distribution:")
        for label, count in label_dist.items():
            print(f"  {label}: {count}")
    
    # API usage stats
    stats = client.get_usage_stats()
    print("\nAPI Usage Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()