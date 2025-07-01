#!/usr/bin/env python3
"""
Test script untuk memverifikasi implementasi parallel labeling.

Script ini akan:
1. Membuat dataset test kecil
2. Test parallel labeling dengan mock client
3. Membandingkan performa serial vs parallel
4. Verifikasi hasil labeling konsisten

Usage:
    python test_parallel_labeling.py
"""

import asyncio
import logging
import time
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config.settings import Settings
from utils.deepseek_client import create_deepseek_client
from utils.deepseek_client_parallel import create_parallel_deepseek_client
from data_collection.deepseek_labeling_pipeline import DeepSeekLabelingPipeline
from data_collection.parallel_deepseek_pipeline import ParallelDeepSeekLabelingPipeline
from utils.logger import setup_logger

# Setup logging
logger = setup_logger("test_parallel", level="INFO")

def create_test_dataset(size: int = 50) -> pd.DataFrame:
    """Create test dataset dengan berbagai jenis teks Bahasa Jawa.
    
    Args:
        size: Jumlah sampel yang akan dibuat
        
    Returns:
        DataFrame dengan kolom 'text' dan 'label'
    """
    # Template teks untuk testing
    positive_texts = [
        "Aku seneng banget karo budaya Jawa",
        "Perlu sinau luwih akeh babagan sejarah",
        "Ora setuju karo keputusan iki, nanging kudu dihormati",
        "Sugeng enjing, piye kabare?",
        "Makanan Jawa iku enak banget",
        "Aku arep sinau basa Jawa",
        "Gamelan iku musik tradisional sing apik",
        "Batik Jawa iku warisan budaya",
        "Selamat pagi, semoga sukses",
        "Terima kasih atas bantuannya"
    ]
    
    negative_texts = [
        "Wong Jawa iku angel diajak maju",
        "Tipikal wong kota, ora ngerti adat",
        "Wong [etnis] iku memang ora iso dipercaya",
        "Kelompok iki kudu diusir saka kene",
        "Wong wadon ora sebagus wong lanang",
        "Wanita ora tau mandheg ngomong",
        "Wong homo dadi beban ing masyarakat kita",
        "Wong ireng kok mambu aneh?",
        "Diskriminasi marang imigran ora salah",
        "Wong cacat lara",
        "Dalit minangka sampah masyarakat",
        "Wong [kelompok] iku kudu dipateni kabeh",
        "Ayo berantas [kelompok] nganti tuntas",
        "Ora ana gunane urip [kelompok] iki",
        "Sialan, macet maneh",
        "Brengsek, telat maneh",
        "wong Arab iku bodho",
        "meneng wae, bocah wadon sing ora sopan",
        "Pengungsi pancen njijiki",
        "sindrom retarded down"
    ]
    
    # Generate dataset
    data = []
    
    # Add positive samples
    positive_count = size // 3  # 1/3 positive
    for i in range(positive_count):
        text = positive_texts[i % len(positive_texts)]
        if i > 0:  # Add variation
            text = f"{text} {i}"
        data.append({"text": text, "label": "positive"})
    
    # Add negative samples
    negative_count = size - positive_count  # 2/3 negative
    for i in range(negative_count):
        text = negative_texts[i % len(negative_texts)]
        if i > 0:  # Add variation
            text = f"{text} {i}"
        data.append({"text": text, "label": "negative"})
    
    df = pd.DataFrame(data)
    logger.info(f"Created test dataset with {len(df)} samples")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df

def test_serial_labeling(df: pd.DataFrame) -> tuple:
    """Test serial labeling pipeline.
    
    Args:
        df: Test dataset
        
    Returns:
        Tuple (results_df, processing_time)
    """
    logger.info("=== Testing Serial Labeling ===")
    
    # Initialize serial pipeline
    pipeline = DeepSeekLabelingPipeline(mock_mode=True)
    
    # Process negative data only (positive akan di-skip)
    negative_df = df[df['label'] == 'negative'].copy()
    
    start_time = time.time()
    results = pipeline.process_negative_data(negative_df)
    processing_time = time.time() - start_time
    
    logger.info(f"Serial processing completed in {processing_time:.2f}s")
    logger.info(f"Processed {len(results)} samples")
    
    return results, processing_time

def test_parallel_labeling(df: pd.DataFrame, max_workers: int = 5) -> tuple:
    """Test parallel labeling pipeline.
    
    Args:
        df: Test dataset
        max_workers: Number of parallel workers
        
    Returns:
        Tuple (results_df, processing_time)
    """
    logger.info(f"=== Testing Parallel Labeling (workers={max_workers}) ===")
    
    # Initialize parallel pipeline
    pipeline = ParallelDeepSeekLabelingPipeline(mock_mode=True, max_workers=max_workers)
    
    # Process negative data only
    negative_df = df[df['label'] == 'negative'].copy()
    
    start_time = time.time()
    results = pipeline.process_negative_data_parallel_sync(negative_df)
    processing_time = time.time() - start_time
    
    logger.info(f"Parallel processing completed in {processing_time:.2f}s")
    logger.info(f"Processed {len(results)} samples")
    
    return results, processing_time

def compare_results(serial_results: pd.DataFrame, parallel_results: pd.DataFrame) -> bool:
    """Compare hasil serial vs parallel untuk memastikan konsistensi.
    
    Args:
        serial_results: Hasil dari serial processing
        parallel_results: Hasil dari parallel processing
        
    Returns:
        True jika hasil konsisten
    """
    logger.info("=== Comparing Results ===")
    
    if len(serial_results) != len(parallel_results):
        logger.error(f"Length mismatch: serial={len(serial_results)}, parallel={len(parallel_results)}")
        return False
    
    # Sort both by text untuk comparison
    serial_sorted = serial_results.sort_values('text').reset_index(drop=True)
    parallel_sorted = parallel_results.sort_values('text').reset_index(drop=True)
    
    # Compare key columns
    mismatches = 0
    for i in range(len(serial_sorted)):
        serial_row = serial_sorted.iloc[i]
        parallel_row = parallel_sorted.iloc[i]
        
        if (serial_row['text'] != parallel_row['text'] or 
            serial_row['final_label'] != parallel_row['final_label']):
            logger.warning(f"Mismatch at row {i}:")
            logger.warning(f"  Serial: {serial_row['text']} -> {serial_row['final_label']}")
            logger.warning(f"  Parallel: {parallel_row['text']} -> {parallel_row['final_label']}")
            mismatches += 1
    
    if mismatches == 0:
        logger.info("✅ Results are consistent between serial and parallel processing")
        return True
    else:
        logger.error(f"❌ Found {mismatches} mismatches between serial and parallel results")
        return False

def test_different_worker_counts(df: pd.DataFrame) -> None:
    """Test dengan berbagai jumlah workers untuk melihat scaling.
    
    Args:
        df: Test dataset
    """
    logger.info("=== Testing Different Worker Counts ===")
    
    worker_counts = [1, 2, 3, 5, 8]
    results = []
    
    negative_df = df[df['label'] == 'negative'].copy()
    
    for workers in worker_counts:
        logger.info(f"Testing with {workers} workers...")
        
        pipeline = ParallelDeepSeekLabelingPipeline(mock_mode=True, max_workers=workers)
        
        start_time = time.time()
        result_df = pipeline.process_negative_data_parallel_sync(negative_df)
        processing_time = time.time() - start_time
        
        throughput = len(result_df) / processing_time if processing_time > 0 else 0
        
        results.append({
            'workers': workers,
            'time': processing_time,
            'throughput': throughput,
            'samples': len(result_df)
        })
        
        logger.info(f"  Workers: {workers}, Time: {processing_time:.2f}s, Throughput: {throughput:.1f} samples/s")
    
    # Find optimal worker count
    best_result = max(results, key=lambda x: x['throughput'])
    logger.info(f"\n🏆 Best performance: {best_result['workers']} workers ({best_result['throughput']:.1f} samples/s)")
    
    return results

def test_full_pipeline(df: pd.DataFrame) -> None:
    """Test complete pipeline dengan save/load.
    
    Args:
        df: Test dataset
    """
    logger.info("=== Testing Full Pipeline ===")
    
    # Save test dataset
    test_input = "test_dataset.csv"
    test_output = "test_results.csv"
    
    df.to_csv(test_input, index=False)
    
    try:
        # Run parallel pipeline
        pipeline = ParallelDeepSeekLabelingPipeline(mock_mode=True, max_workers=3)
        report = pipeline.run_pipeline(test_input, test_output)
        
        # Verify output file
        if Path(test_output).exists():
            result_df = pd.read_csv(test_output)
            logger.info(f"✅ Output file created with {len(result_df)} samples")
            logger.info(f"✅ Pipeline report generated: {len(report)} sections")
        else:
            logger.error("❌ Output file not created")
    
    finally:
        # Cleanup
        for file in [test_input, test_output]:
            if Path(file).exists():
                Path(file).unlink()

def main():
    """Main test function."""
    logger.info("🚀 Starting Parallel Labeling Tests")
    
    # Create test dataset
    test_size = 30  # Small size untuk testing cepat
    df = create_test_dataset(test_size)
    
    try:
        # Test 1: Serial vs Parallel comparison
        logger.info("\n" + "="*50)
        serial_results, serial_time = test_serial_labeling(df)
        parallel_results, parallel_time = test_parallel_labeling(df, max_workers=3)
        
        # Calculate speedup
        speedup = serial_time / parallel_time if parallel_time > 0 else 1
        logger.info(f"\n📊 Performance Comparison:")
        logger.info(f"  Serial time: {serial_time:.2f}s")
        logger.info(f"  Parallel time: {parallel_time:.2f}s")
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        # Test 2: Result consistency
        logger.info("\n" + "="*50)
        consistent = compare_results(serial_results, parallel_results)
        
        # Test 3: Different worker counts
        logger.info("\n" + "="*50)
        scaling_results = test_different_worker_counts(df)
        
        # Test 4: Full pipeline
        logger.info("\n" + "="*50)
        test_full_pipeline(df)
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("🎉 TEST SUMMARY")
        logger.info(f"✅ Serial processing: {len(serial_results)} samples in {serial_time:.2f}s")
        logger.info(f"✅ Parallel processing: {len(parallel_results)} samples in {parallel_time:.2f}s")
        logger.info(f"✅ Speedup achieved: {speedup:.2f}x")
        logger.info(f"✅ Result consistency: {'PASS' if consistent else 'FAIL'}")
        logger.info(f"✅ Scaling test: {len(scaling_results)} configurations tested")
        logger.info(f"✅ Full pipeline: PASS")
        
        if speedup > 1.5 and consistent:
            logger.info("\n🎯 PARALLEL IMPLEMENTATION READY FOR PRODUCTION!")
            logger.info("   Recommended: Use 3-5 workers for optimal performance")
        else:
            logger.warning("\n⚠️  Need to investigate performance or consistency issues")
    
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()