#!/usr/bin/env python3
"""
Data Preparation for IndoBERT Large Experiment
Menggunakan hasil-labeling.csv sebagai dataset utama
Mengacak urutan data dan mempersiapkan format yang sesuai untuk training

Author: AI Research Assistant
Date: 3 Juli 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from collections import Counter
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_hasil_labeling(file_path: str) -> pd.DataFrame:
    """Load hasil-labeling.csv dataset"""
    logger.info(f"Loading hasil-labeling dataset from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df

def analyze_dataset_distribution(df: pd.DataFrame) -> dict:
    """Analyze dataset distribution and characteristics"""
    logger.info("Analyzing dataset distribution...")
    
    analysis = {
        'total_samples': len(df),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'label_distribution': {},
        'labeling_method_distribution': {},
        'confidence_score_stats': {}
    }
    
    # Label distribution
    if 'final_label' in df.columns:
        label_counts = df['final_label'].value_counts()
        analysis['label_distribution'] = label_counts.to_dict()
        
        logger.info("Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {label}: {count} ({percentage:.2f}%)")
    
    # Labeling method distribution
    if 'labeling_method' in df.columns:
        method_counts = df['labeling_method'].value_counts()
        analysis['labeling_method_distribution'] = method_counts.to_dict()
        
        logger.info("Labeling method distribution:")
        for method, count in method_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {method}: {count} ({percentage:.2f}%)")
    
    # Confidence score statistics
    if 'confidence_score' in df.columns:
        conf_stats = df['confidence_score'].describe()
        analysis['confidence_score_stats'] = conf_stats.to_dict()
        
        logger.info("Confidence score statistics:")
        logger.info(f"  Mean: {conf_stats['mean']:.3f}")
        logger.info(f"  Std: {conf_stats['std']:.3f}")
        logger.info(f"  Min: {conf_stats['min']:.3f}")
        logger.info(f"  Max: {conf_stats['max']:.3f}")
    
    return analysis

def shuffle_dataset(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Shuffle dataset to randomize order"""
    logger.info(f"Shuffling dataset with random_state={random_state}")
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Shuffle the dataframe
    shuffled_df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    logger.info(f"Dataset shuffled. Original order changed.")
    logger.info(f"First 5 original indices after shuffle: {shuffled_df.get('original_index', range(5))[:5].tolist() if 'original_index' in shuffled_df.columns else 'N/A'}")
    
    return shuffled_df

def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for training"""
    logger.info("Cleaning and preparing data...")
    
    # Create a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # Remove rows with missing text or final_label
    initial_count = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(subset=['text', 'final_label'])
    final_count = len(cleaned_df)
    
    if initial_count != final_count:
        logger.info(f"Removed {initial_count - final_count} rows with missing text or final_label")
    
    # Clean text column
    cleaned_df['text'] = cleaned_df['text'].astype(str)
    cleaned_df['text'] = cleaned_df['text'].str.strip()
    
    # Remove empty texts
    empty_text_count = len(cleaned_df[cleaned_df['text'].str.len() == 0])
    if empty_text_count > 0:
        cleaned_df = cleaned_df[cleaned_df['text'].str.len() > 0]
        logger.info(f"Removed {empty_text_count} rows with empty text")
    
    # Remove duplicates based on text
    duplicate_count = cleaned_df.duplicated(subset=['text']).sum()
    if duplicate_count > 0:
        cleaned_df = cleaned_df.drop_duplicates(subset=['text'], keep='first')
        logger.info(f"Removed {duplicate_count} duplicate texts")
    
    logger.info(f"Final cleaned dataset shape: {cleaned_df.shape}")
    
    return cleaned_df

def create_label_mapping():
    """Create consistent label mapping"""
    return {
        'Bukan Ujaran Kebencian': 0,
        'Ujaran Kebencian - Ringan': 1,
        'Ujaran Kebencian - Sedang': 2,
        'Ujaran Kebencian - Berat': 3
    }

def prepare_final_dataset_from_hasil_labeling():
    """Prepare final dataset from hasil-labeling.csv"""
    logger.info("=" * 60)
    logger.info("PREPARING DATASET FROM HASIL-LABELING.CSV")
    logger.info("=" * 60)
    
    # Paths
    hasil_labeling_path = "src/data_collection/hasil-labeling.csv"
    output_path = "data/processed/final_dataset.csv"
    analysis_output_path = "data/processed/dataset_analysis.json"
    
    # Load dataset
    df = load_hasil_labeling(hasil_labeling_path)
    
    # Analyze original dataset
    original_analysis = analyze_dataset_distribution(df)
    
    # Clean and prepare data
    cleaned_df = clean_and_prepare_data(df)
    
    # Shuffle dataset to randomize order
    shuffled_df = shuffle_dataset(cleaned_df, random_state=42)
    
    # Create label mapping
    label_mapping = create_label_mapping()
    
    # Map labels to numeric values
    shuffled_df['label_numeric'] = shuffled_df['final_label'].map(label_mapping)
    
    # Handle any unmapped labels
    unmapped = shuffled_df[shuffled_df['label_numeric'].isna()]
    if len(unmapped) > 0:
        logger.warning(f"Found {len(unmapped)} unmapped labels:")
        unique_unmapped = unmapped['final_label'].unique()
        logger.warning(f"Unmapped labels: {unique_unmapped}")
        
        # Remove unmapped labels
        shuffled_df = shuffled_df.dropna(subset=['label_numeric'])
        logger.info(f"Removed {len(unmapped)} rows with unmapped labels")
    
    # Prepare final dataset
    final_df = shuffled_df[[
        'text', 'final_label', 'label_numeric', 'confidence_score', 
        'labeling_method', 'response_time'
    ]].copy()
    
    # Rename columns for consistency
    final_df = final_df.rename(columns={
        'final_label': 'label',
        'label_numeric': 'label_id'
    })
    
    # Analyze final dataset
    final_analysis = analyze_dataset_distribution(final_df.rename(columns={'label': 'final_label'}))
    
    # Log final class distribution
    logger.info("\nFinal class distribution after shuffling:")
    class_counts = final_df['label'].value_counts()
    total_samples = len(final_df)
    
    for label, count in class_counts.items():
        percentage = (count / total_samples) * 100
        logger.info(f"  {label}: {count} ({percentage:.2f}%)")
    
    # Calculate class imbalance ratio
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    logger.info(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Save final dataset
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_df.to_csv(output_path, index=False)
    logger.info(f"\nFinal dataset saved to {output_path}")
    logger.info(f"Final dataset shape: {final_df.shape}")
    
    # Create stratified train-test split
    logger.info("\nCreating stratified train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        final_df['text'], final_df['label_id'],
        test_size=0.2, stratify=final_df['label_id'], random_state=42
    )
    
    logger.info(f"Train set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Log train set distribution
    train_label_counts = pd.Series(y_train).value_counts().sort_index()
    test_label_counts = pd.Series(y_test).value_counts().sort_index()
    
    logger.info("\nTrain set distribution:")
    for label_id, count in train_label_counts.items():
        label_name = [k for k, v in label_mapping.items() if v == label_id][0]
        percentage = (count / len(y_train)) * 100
        logger.info(f"  {label_name}: {count} ({percentage:.2f}%)")
    
    logger.info("\nTest set distribution:")
    for label_id, count in test_label_counts.items():
        label_name = [k for k, v in label_mapping.items() if v == label_id][0]
        percentage = (count / len(y_test)) * 100
        logger.info(f"  {label_name}: {count} ({percentage:.2f}%)")
    
    # Save train-test split
    train_df = pd.DataFrame({
        'text': X_train,
        'label': y_train
    })
    test_df = pd.DataFrame({
        'text': X_test,
        'label': y_test
    })
    
    train_df.to_csv('data/processed/train_set.csv', index=False)
    test_df.to_csv('data/processed/test_set.csv', index=False)
    
    logger.info("Train and test sets saved")
    
    # Save comprehensive analysis
    comprehensive_analysis = {
        'preparation_metadata': {
            'source_file': hasil_labeling_path,
            'preparation_date': datetime.now().isoformat(),
            'random_seed': 42,
            'shuffled': True,
            'test_size': 0.2
        },
        'original_dataset_analysis': original_analysis,
        'final_dataset_analysis': final_analysis,
        'label_mapping': label_mapping,
        'class_imbalance_ratio': float(imbalance_ratio),
        'train_test_split': {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_distribution': train_label_counts.to_dict(),
            'test_distribution': test_label_counts.to_dict()
        }
    }
    
    with open(analysis_output_path, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_analysis, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Comprehensive analysis saved to {analysis_output_path}")
    
    # Generate summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("DATASET PREPARATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Source: {hasil_labeling_path}")
    logger.info(f"Original samples: {original_analysis['total_samples']}")
    logger.info(f"Final samples: {len(final_df)}")
    logger.info(f"Data reduction: {original_analysis['total_samples'] - len(final_df)} samples")
    logger.info(f"Shuffled: Yes (random_state=42)")
    logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")
    logger.info(f"Train/Test split: {len(X_train)}/{len(X_test)} (80%/20%)")
    logger.info("Dataset ready for IndoBERT Large experiment!")
    
    return final_df, comprehensive_analysis

if __name__ == "__main__":
    prepare_final_dataset_from_hasil_labeling()