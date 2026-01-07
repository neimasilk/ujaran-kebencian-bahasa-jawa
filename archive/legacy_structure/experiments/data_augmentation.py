#!/usr/bin/env python3
"""
Data Augmentation for Javanese Hate Speech Detection Dataset
Increases dataset diversity to improve model performance

Author: AI Assistant
Date: 2025-07-24
"""

import os
import re
import json
import random
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_augmentation.log'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

class DataAugmentationConfig:
    """Configuration for data augmentation"""
    
    # Input data path
    INPUT_DATA_PATH = "data/standardized/balanced_dataset.csv"
    
    # Output data path
    OUTPUT_DATA_PATH = "data/standardized/augmented_dataset.csv"
    
    # Augmentation techniques to apply
    AUGMENTATION_TECHNIQUES = [
        'synonym_replacement',
        'random_insertion',
        'random_swap',
        'random_deletion'
    ]
    
    # Augmentation rates (how much of the data to augment)
    AUGMENTATION_RATE = 0.5  # 50% of data will be augmented
    
    # Number of augmented samples per original sample
    AUGMENTATION_COUNT = 2  # Create 2 augmented versions per original sample
    
    # Label mapping
    LABEL_MAPPING = {
        0: "Bukan Ujaran Kebencian",
        1: "Ujaran Kebencian - Ringan",
        2: "Ujaran Kebencian - Sedang",
        3: "Ujaran Kebencian - Berat"
    }

class JavaneseTextAugmenter:
    """Text augmenter specifically for Javanese language"""
    
    def __init__(self):
        # Initialize stemmer for Indonesian (close to Javanese)
        try:
            self.stemmer = StemmerFactory().create_stemmer()
        except:
            self.stemmer = None
            logger.warning("Sastrawi stemmer not available. Proceeding without stemming.")
        
        # Stopwords in Indonesian (similar to Javanese)
        try:
            self.stopwords = set(stopwords.words('indonesian'))
        except:
            self.stopwords = set()
            logger.warning("NLTK stopwords not available. Proceeding without stopwords.")
        
        # Simple synonym dictionary for common Javanese/Indonesian words
        self.synonym_dict = {
            # General terms
            'tidak': ['ora', 'mboten', 'oraono'],
            'benar': ['lurus', 'betul', 'bener'],
            'salah': ['pantat', 'salip', 'mbotenlurus'],
            'baik': ['apik', 'bener', 'pinilih'],
            'buruk': ['awon', 'alit', 'gawe'],
            'besar': ['gede', 'ageng', 'pinter'],
            'kecil': ['cilik', 'alit', 'sithik'],
            'cepat': ['gancang', 'tancap', 'langkung'],
            'lambat': ['ala', 'lamban', 'waton'],
            
            # Hate speech related terms (be careful with these)
            'benci': ['kesel', 'muak', 'ora seneng'],
            'bodoh': ['goblok', 'tolol', 'ora ndhonor'],
            'gila': ['goblog', 'ndhisiki', 'ora waras'],
            
            # Positive terms
            'suka': ['seneng', 'rejeki', 'sreg'],
            'senang': ['bahagia', 'gembira', 'sreg'],
            'indah': ['apik', 'cakep', 'rapi'],
        }
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word"""
        return self.synonym_dict.get(word, [])
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace n words in the text with their synonyms"""
        words = text.split()
        new_words = words.copy()
        
        # Find words that can be replaced
        replaceable_indices = []
        for i, word in enumerate(words):
            if word in self.synonym_dict:
                replaceable_indices.append(i)
        
        # Randomly select n words to replace
        n = min(n, len(replaceable_indices))
        random_indices = random.sample(replaceable_indices, n) if replaceable_indices else []
        
        # Replace selected words
        for i in random_indices:
            synonyms = self._get_synonyms(words[i])
            if synonyms:
                new_words[i] = random.choice(synonyms)
        
        return ' '.join(new_words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Insert n random synonyms into the text"""
        words = text.split()
        new_words = words.copy()
        
        # Find words that have synonyms
        synonym_words = [word for word in words if word in self.synonym_dict]
        
        if not synonym_words:
            return text
        
        for _ in range(n):
            # Select a random word that has synonyms
            random_word = random.choice(synonym_words)
            synonyms = self._get_synonyms(random_word)
            
            if synonyms:
                # Select a random synonym
                random_synonym = random.choice(synonyms)
                
                # Insert the synonym at a random position
                random_idx = random.randint(0, len(new_words) - 1)
                new_words.insert(random_idx, random_synonym)
        
        return ' '.join(new_words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """Swap n pairs of words in the text"""
        words = text.split()
        new_words = words.copy()
        
        if len(words) < 2:
            return text
        
        for _ in range(n):
            # Select two random indices
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            
            # Swap words
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p"""
        words = text.split()
        
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        # If all words were deleted, return a random word
        if not new_words:
            return random.choice(words)
        
        return ' '.join(new_words)

def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from CSV file"""
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Use standardized dataset columns
    if 'label_numeric' in df.columns:
        df = df[['text', 'label_numeric']].copy()
        df = df.rename(columns={'label_numeric': 'label'})
    elif 'final_label' in df.columns:
        label_mapping = {
            'Bukan Ujaran Kebencian': 0,
            'Ujaran Kebencian - Ringan': 1,
            'Ujaran Kebencian - Sedang': 2,
            'Ujaran Kebencian - Berat': 3
        }
        df['label_id'] = df['final_label'].map(label_mapping)
        df = df[['text', 'label_id']].copy()
        df = df.rename(columns={'label_id': 'label'})
    else:
        raise ValueError("Dataset must contain either 'label_numeric' or 'final_label' column")
    
    # Clean data
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)
    
    logger.info(f"Loaded {len(df)} samples")
    return df

def augment_dataset(df: pd.DataFrame, config: DataAugmentationConfig) -> pd.DataFrame:
    """Augment the dataset with various techniques"""
    logger.info("Starting data augmentation")
    
    augmenter = JavaneseTextAugmenter()
    
    # Calculate how many samples to augment
    num_to_augment = int(len(df) * config.AUGMENTATION_RATE)
    logger.info(f"Augmenting {num_to_augment} samples")
    
    # Select samples to augment
    df_to_augment = df.sample(n=num_to_augment, random_state=42)
    df_not_augmented = df.drop(df_to_augment.index)
    
    # List to store augmented samples
    augmented_samples = []
    
    # Apply augmentation techniques
    for idx, row in df_to_augment.iterrows():
        original_text = row['text']
        label = row['label']
        
        # Create multiple augmented versions
        for _ in range(config.AUGMENTATION_COUNT):
            # Randomly select an augmentation technique
            technique = random.choice(config.AUGMENTATION_TECHNIQUES)
            
            # Apply the technique
            if technique == 'synonym_replacement':
                augmented_text = augmenter.synonym_replacement(original_text, n=random.randint(1, 3))
            elif technique == 'random_insertion':
                augmented_text = augmenter.random_insertion(original_text, n=random.randint(1, 2))
            elif technique == 'random_swap':
                augmented_text = augmenter.random_swap(original_text, n=random.randint(1, 2))
            elif technique == 'random_deletion':
                augmented_text = augmenter.random_deletion(original_text, p=random.uniform(0.05, 0.2))
            else:
                augmented_text = original_text  # Fallback
            
            # Add to augmented samples
            augmented_samples.append({
                'text': augmented_text,
                'label': label
            })
    
    # Create DataFrame from augmented samples
    df_augmented = pd.DataFrame(augmented_samples)
    
    # Combine original non-augmented samples with augmented ones
    df_final = pd.concat([df_not_augmented, df_augmented], ignore_index=True)
    
    logger.info(f"Original samples: {len(df)}")
    logger.info(f"Augmented samples: {len(df_augmented)}")
    logger.info(f"Final dataset size: {len(df_final)}")
    
    return df_final

def save_dataset(df: pd.DataFrame, output_path: str):
    """Save the augmented dataset to CSV"""
    logger.info(f"Saving augmented dataset to {output_path}")
    
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info("Dataset saved successfully")

def analyze_dataset(df_original: pd.DataFrame, df_augmented: pd.DataFrame, config: DataAugmentationConfig):
    """Analyze the original and augmented datasets"""
    logger.info("Analyzing datasets")
    
    # Class distribution
    original_dist = df_original['label'].value_counts().sort_index()
    augmented_dist = df_augmented['label'].value_counts().sort_index()
    
    logger.info("Class distribution comparison:")
    logger.info("Label\tOriginal\tAugmented")
    for label in sorted(original_dist.index):
        orig_count = original_dist.get(label, 0)
        aug_count = augmented_dist.get(label, 0)
        logger.info(f"{label}\t{orig_count}\t\t{aug_count}")
    
    # Text length statistics
    original_lengths = df_original['text'].str.len()
    augmented_lengths = df_augmented['text'].str.len()
    
    logger.info("\nText length statistics:")
    logger.info(f"Original - Mean: {original_lengths.mean():.2f}, Std: {original_lengths.std():.2f}")
    logger.info(f"Augmented - Mean: {augmented_lengths.mean():.2f}, Std: {augmented_lengths.std():.2f}")

def main():
    """Main data augmentation function"""
    logger.info("=" * 60)
    logger.info("DATA AUGMENTATION FOR JAVANESE HATE SPEECH DETECTION")
    logger.info("=" * 60)
    
    config = DataAugmentationConfig()
    
    try:
        # Load original dataset
        df_original = load_data(config.INPUT_DATA_PATH)
        
        # Augment dataset
        df_augmented = augment_dataset(df_original, config)
        
        # Analyze datasets
        analyze_dataset(df_original, df_augmented, config)
        
        # Save augmented dataset
        save_dataset(df_augmented, config.OUTPUT_DATA_PATH)
        
        logger.info("=" * 60)
        logger.info("DATA AUGMENTATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Augmented dataset saved to: {config.OUTPUT_DATA_PATH}")
        logger.info("You can now use this augmented dataset for training your models")
        
    except Exception as e:
        logger.error(f"Data augmentation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()