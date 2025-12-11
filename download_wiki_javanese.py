#!/usr/bin/env python3
"""
Script to download Javanese Wikipedia for Domain-Adaptive Pre-Training.
"""

from datasets import load_dataset
import os
import re

def clean_wiki_text(text):
    # Remove titles (lines starting with =)
    text = re.sub(r'=+ .+ =+', '', text)
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def download_and_save_wiki():
    print("🌍 Downloading Javanese Wikipedia...")
    try:
        # Load Javanese Wikipedia (jv)
        # Using newer 'wikimedia/wikipedia' dataset
        dataset = load_dataset("wikimedia/wikipedia", "20231101.jv", split="train")
        
        print(f"📚 Downloaded {len(dataset)} articles.")
        
        output_file = 'data/corpus/wiki_javanese.txt'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"💾 Saving to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in dataset:
                text = clean_wiki_text(article['text'])
                if len(text) > 50:  # Minimum length
                    f.write(text + '\n')
                    
        print("✅ Wikipedia data saved!")
        return output_file
        
    except Exception as e:
        print(f"❌ Error downloading Wikipedia: {e}")
        return None

if __name__ == "__main__":
    download_and_save_wiki()
