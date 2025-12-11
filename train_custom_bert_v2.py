#!/usr/bin/env python3
"""
Custom Javanese BERT v2: Domain-Adaptive Pre-Training (DAPT)
Combines:
1. Standardized Hate Speech Dataset (Context)
2. Javanese Wikipedia (Grammar/Vocabulary)
3. Synthetic AI Data (Slang/Code-Switching) - Optional
"""

import os
import glob
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments,
    LineByLineTextDataset
)
import torch
import warnings

warnings.filterwarnings('ignore')

def combine_corpora():
    print("📚 Combining text sources...")
    
    sources = [
        'data/corpus/javanese_text.txt',      # From original dataset
        'data/corpus/wiki_javanese.txt',      # From Wikipedia
        'data/corpus/synthetic_javanese.txt'  # From AI (DeepSeek)
    ]
    
    combined_path = 'data/corpus/combined_corpus.txt'
    total_lines = 0
    
    with open(combined_path, 'w', encoding='utf-8') as outfile:
        for source in sources:
            if os.path.exists(source):
                print(f"   ➕ Adding {source}...")
                with open(source, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        if len(line.strip()) > 10:
                            outfile.write(line)
                            total_lines += 1
            else:
                print(f"   ⚠️ Source not found (skipping): {source}")
                
    print(f"✅ Combined corpus saved to {combined_path} ({total_lines} lines)")
    return combined_path

def train_dapt_model(corpus_path):
    # Start from our previous custom model if it exists, otherwise base
    if os.path.exists("models/custom_javanese_roberta"):
        model_name = "models/custom_javanese_roberta"
        print("🚀 Continuing training from 'models/custom_javanese_roberta'")
    else:
        model_name = "flax-community/indonesian-roberta-base"
        print("🚀 Starting fresh from 'flax-community/indonesian-roberta-base'")
        
    output_dir = "models/custom_javanese_bert_v2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    print("📦 Loading dataset (this might take a moment)...")
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=corpus_path,
        block_size=128
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
        output_dir="tmp_dapt_training",
        overwrite_output_dir=True,
        num_train_epochs=3, # Fewer epochs needed because corpus is huge now
        per_device_train_batch_size=32,
        save_steps=1000,
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
        logging_steps=500,
        seed=42
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    print("🔥 Starting Domain-Adaptive Pre-Training...")
    trainer.train()
    
    print(f"💾 Saving v2 model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Custom Javanese BERT v2 Complete!")

if __name__ == "__main__":
    combined_file = combine_corpora()
    train_dapt_model(combined_file)
