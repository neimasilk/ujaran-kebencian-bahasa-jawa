import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def create_synthetic_dataset():
    print("Creating synthetic dataset...")
    os.makedirs('data/standardized', exist_ok=True)
    os.makedirs('data/augmented', exist_ok=True)
    
    # Create random Javanese-like text
    vocab = ["aku", "kowe", "dheweke", "mangan", "turu", "lunga", "pasar", "sego", "goreng", "enak", 
             "asu", "babi", "jiancok", "goblok", "apik", "elek", "saiki", "wingi", "sesuk"]
    
    data = []
    labels = {
        0: "Bukan Ujaran Kebencian",
        1: "Ujaran Kebencian - Ringan",
        2: "Ujaran Kebencian - Sedang",
        3: "Ujaran Kebencian - Berat"
    }
    
    # Generate 100 samples
    for i in range(100):
        text_len = np.random.randint(5, 20)
        text = " ".join(np.random.choice(vocab, text_len))
        label_numeric = np.random.randint(0, 4)
        
        data.append({
            'text': text,
            'label_numeric': label_numeric,
            'final_label': labels[label_numeric]
        })
        
    df = pd.DataFrame(data)
    
    # Save to both locations referenced in scripts
    df.to_csv('data/standardized/balanced_dataset.csv', index=False)
    df.to_csv('data/augmented/augmented_dataset.csv', index=False)
    print(f"Created dataset with {len(df)} samples.")

def setup_model():
    print("Setting up base model...")
    output_dir = 'models/improved_model'
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a tiny model for demonstration purposes to avoid memory issues
    model_name = "prajjwal1/bert-tiny" 
    
    print(f"Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    
    print(f"Saving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model setup complete.")

if __name__ == "__main__":
    create_synthetic_dataset()
    setup_model()
