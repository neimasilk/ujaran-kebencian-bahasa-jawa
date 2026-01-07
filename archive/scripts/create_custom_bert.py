#!/usr/bin/env python3
"""
Script to create a "Custom Javanese BERT/RoBERTa" via Task-Adaptive Pre-Training (TAPT).
We take a base Indonesian model and further pre-train it on our Javanese dataset
using Masked Language Modeling (MLM).
"""

import os
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from transformers import LineByLineTextDataset
import torch
import warnings

warnings.filterwarnings('ignore')

def create_corpus():
    print("📝 Creating Javanese text corpus from dataset...")
    # Load the balanced dataset
    df = pd.read_csv('data/standardized/balanced_dataset.csv')

    # Extract only the text
    texts = df['text'].astype(str).tolist()

    # Save to a text file for the model to read
    os.makedirs('data/corpus', exist_ok=True)
    corpus_path = 'data/corpus/javanese_text.txt'

    with open(corpus_path, 'w', encoding='utf-8') as f:
        for text in texts:
            # Clean newlines to ensure one sequence per line
            clean_text = text.replace('\n', ' ').strip()
            if len(clean_text) > 10:  # Only meaningful text
                f.write(clean_text + '\n')

    print(f"✅ Corpus saved to {corpus_path} ({len(texts)} lines)")
    return corpus_path

def train_custom_model(corpus_path):
    model_name = "flax-community/indonesian-roberta-base"
    output_dir = "models/custom_javanese_roberta"

    print(f"🤖 Initializing base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    print("📚 Preparing dataset for Masked Language Modeling (MLM)...")
    # Using LineByLineTextDataset is simple for this scale
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=corpus_path,
        block_size=128  # Context window for pre-training
    )

    # MLM Data Collator: Randomly masks 15% of tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="tmp_custom_bert_training",
        overwrite_output_dir=True,
        num_train_epochs=10,  # More epochs for pre-training usually needed
        per_device_train_batch_size=32,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,  # Use GPU acceleration
        logging_steps=100,
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("🚀 Starting TAPT (Task-Adaptive Pre-Training)...")
    print("   This teaches the model Javanese vocabulary and structure.")
    trainer.train()

    print(f"💾 Saving custom model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Custom Javanese Model Created!")

if __name__ == "__main__":
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("⚠️ WARNING: CUDA not detected. Training will be slow!")
    else:
        print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")

    corpus_file = create_corpus()
    train_custom_model(corpus_file)
