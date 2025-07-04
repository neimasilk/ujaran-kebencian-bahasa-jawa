import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os

def shuffle_dataset(input_file, output_file, random_state=42):
    """
    Shuffle the dataset to randomize the order of samples
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        random_state (int): Random seed for reproducibility
    """
    print(f"Loading dataset from: {input_file}")
    
    # Load the dataset
    df = pd.read_csv(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Label distribution before shuffling:")
    print(df['label'].value_counts())
    
    # Shuffle the dataset
    print(f"\nShuffling dataset with random_state={random_state}...")
    df_shuffled = shuffle(df, random_state=random_state)
    
    # Reset index
    df_shuffled = df_shuffled.reset_index(drop=True)
    
    print(f"\nShuffled dataset shape: {df_shuffled.shape}")
    print(f"Label distribution after shuffling:")
    print(df_shuffled['label'].value_counts())
    
    # Verify shuffling by checking first 10 labels
    print(f"\nFirst 10 labels after shuffling: {df_shuffled['label'].head(10).tolist()}")
    print(f"Last 10 labels after shuffling: {df_shuffled['label'].tail(10).tolist()}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save shuffled dataset
    print(f"\nSaving shuffled dataset to: {output_file}")
    df_shuffled.to_csv(output_file, index=False)
    
    print("Dataset shuffling completed successfully!")
    
    return df_shuffled

if __name__ == "__main__":
    # Define file paths
    input_file = "d:/documents/ujaran-kebencian-bahasa-jawa/src/data_collection/hasil-labeling.csv"
    output_file = "d:/documents/ujaran-kebencian-bahasa-jawa/data/processed/final_dataset_shuffled.csv"
    
    # Shuffle the dataset
    shuffled_df = shuffle_dataset(input_file, output_file, random_state=42)
    
    print(f"\nShuffled dataset saved successfully!")
    print(f"You can now use the shuffled dataset: {output_file}")