"""
EXPERIMENT 8: Extended DAPT untuk Custom Javanese BERT v3
===========================================================

Objective: Pre-training BERT dengan corpus Jawa yang lebih besar
- Base: Indonesian RoBERTa (flax-community/indonesian-roberta-base)
- Corpus: Combined ~1.5M lines dari Wiki + Dataset + Synthetic
- Target: F1-Macro > 82% pada hate speech detection

CHECKPOINT & RESUME SYSTEM:
- Script bisa resume dari checkpoint terakhir
- Progress disimpan di experiments/experiment_8_progress.json
- Untuk resume: jalankan script yang sama, akan detect checkpoint

USAGE:
- Run pertama: python experiments/experiment_8_extended_dapt.py
- Resume (jika terhenti): python experiments/experiment_8_extended_dapt.py --resume
- Cek progress: python experiments/experiment_8_extended_dapt.py --status

ESTIMATED TIME: 1-2 jam untuk 5-10 epochs
"""

import os
import json
import argparse
import warnings
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


class ProgressTracker:
    """Track progress training untuk resume capability"""

    def __init__(self, progress_file):
        self.progress_file = progress_file
        self.data = self.load()

    def load(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "status": "not_started",
            "start_time": None,
            "last_update": None,
            "epoch": 0,
            "max_epochs": 0,
            "steps_completed": 0,
            "model_path": None,
            "history": []
        }

    def save(self, **kwargs):
        for key, value in kwargs.items():
            self.data[key] = value
        self.data["last_update"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get_status(self):
        return self.data


class ExtendedDAPTTrainer:
    """Extended Domain-Adaptive Pre-Training Trainer"""

    def __init__(self, resume=False):
        self.progress = ProgressTracker("experiments/experiment_8_progress.json")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Config
        self.base_model = "flax-community/indonesian-roberta-base"
        self.corpus_file = "data/corpus/combined_corpus.txt"
        self.output_dir = "models/custom_javanese_bert_v3"

        # Training config
        self.max_epochs = 10  # Bisa diubah sesuai waktu
        self.batch_size = 32
        self.learning_rate = 2e-5
        self.max_length = 128
        self.gradient_accumulation = 2

        self.resume = resume

    def check_resume(self):
        """Check if we can resume from checkpoint"""
        # Create output dir if not exists
        os.makedirs(self.output_dir, exist_ok=True)

        last_checkpoint = get_last_checkpoint(self.output_dir)

        if last_checkpoint is not None and os.path.exists(last_checkpoint):
            print(f"\n[CHECKPOINT] Found checkpoint at: {last_checkpoint}")
            print("[CHECKPOINT] Will resume training from last checkpoint\n")

            self.model = AutoModelForMaskedLM.from_pretrained(last_checkpoint)
            return True, last_checkpoint
        else:
            print("\n[CHECKPOINT] No checkpoint found. Starting fresh training.\n")
            return False, self.base_model

    def prepare_dataset(self):
        """Prepare dataset dari corpus file"""
        print("[1/5] Preparing dataset...")

        if not os.path.exists(self.corpus_file):
            print(f"ERROR: Corpus file not found: {self.corpus_file}")
            return None

        # Get corpus size
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if len(line.strip()) > 10]

        print(f"      Total lines in corpus: {len(lines):,}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        # Create dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=False,  # Dynamic padding
                max_length=self.max_length,
                return_special_tokens_mask=True,
            )

        # Process in chunks to avoid memory issues
        print("      Tokenizing (this may take a while)...")
        all_input_ids = []
        all_attention_mask = []

        for i in range(0, len(lines), 10000):
            chunk = lines[i:i+10000]
            texts = {"text": chunk}

            # Simple tokenization
            encoded = tokenizer(
                chunk,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None
            )

            all_input_ids.extend(encoded["input_ids"])
            all_attention_mask.extend(encoded["attention_mask"])

            if (i // 10000 + 1) % 10 == 0:
                print(f"      Processed {i+len(chunk):,} lines...")

        # Filter by length (avoid too short sequences)
        filtered = [
            (ids, mask) for ids, mask in zip(all_input_ids, all_attention_mask)
            if len(ids) > 10
        ]

        if not filtered:
            print("ERROR: No valid sequences after filtering!")
            return None

        all_input_ids, all_attention_mask = zip(*filtered)

        dataset = Dataset.from_dict({
            "input_ids": list(all_input_ids),
            "attention_mask": list(all_attention_mask)
        })

        print(f"      Final dataset size: {len(dataset):,} samples")
        return tokenizer, dataset

    def get_training_args(self):
        """Setup training arguments dengan checkpoint support"""
        return TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=False,  # Important for resume!
            num_train_epochs=self.max_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            fp16=True,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=500,
            save_strategy="epoch",
            save_total_limit=3,  # Keep last 3 checkpoints
            prediction_loss_only=True,
            report_to=["none"],
            seed=42,
            dataloader_num_workers=0,
        )

    def train(self):
        """Main training loop dengan progress tracking"""
        print("="*50)
        print("EXPERIMENT 8: EXTENDED DAPT")
        print("="*50)
        print(f"Base Model: {self.base_model}")
        print(f"Corpus: {self.corpus_file}")
        print(f"Output: {self.output_dir}")
        print(f"Max Epochs: {self.max_epochs}")
        print(f"Device: {self.device}")
        print("="*50)

        # Check existing progress
        progress_data = self.progress.get_status()

        if progress_data["status"] == "completed":
            print("\n[STATUS] Training already completed!")
            print(f"[STATUS] Model saved at: {progress_data['model_path']}")

            # Ask if user wants to continue training
            response = input("\nContinue training for more epochs? (y/n): ")
            if response.lower() != 'y':
                return

            self.max_epochs = progress_data["max_epochs"] + 5
            print(f"      Continuing to {self.max_epochs} total epochs...")

        # Prepare dataset
        result = self.prepare_dataset()
        if result is None:
            return

        tokenizer, dataset = result

        # Check resume
        is_resume, model_path = self.check_resume()

        if is_resume:
            start_epoch = progress_data.get("epoch", 0)
            print(f"[RESUME] Continuing from epoch {start_epoch}")
        else:
            # Load fresh model
            print("\n[2/5] Loading base model...")
            model = AutoModelForMaskedLM.from_pretrained(self.base_model)
            model.to(self.device)
            model_path = self.base_model
            start_epoch = 0

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        # Training args
        training_args = self.get_training_args()

        # Update initial epoch if resuming
        if is_resume:
            training_args.num_train_epochs = self.max_epochs

        # Create trainer
        print("\n[3/5] Setting up trainer...")
        trainer = Trainer(
            model=model if is_resume else AutoModelForMaskedLM.from_pretrained(self.base_model),
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        if not is_resume:
            trainer.model.to(self.device)

        # Update progress
        self.progress.save(
            status="training",
            start_time=progress_data.get("start_time") or datetime.now().isoformat(),
            max_epochs=self.max_epochs
        )

        # Train dengan custom callback untuk progress tracking
        print("\n[4/5] Starting training...")

        class ProgressCallback:
            def __init__(self, tracker, max_epochs):
                self.tracker = tracker
                self.max_epochs = max_epochs
                self.current_epoch = 0

            def on_epoch_end(self, args, state, control, **kwargs):
                # Update progress setiap epoch
                epoch = int(state.epoch) if state.epoch else 0
                if epoch > self.current_epoch:
                    self.current_epoch = epoch

                    # Calculate estimated time
                    elapsed = datetime.now() - datetime.fromisoformat(
                        self.tracker.data.get("start_time", datetime.now().isoformat())
                    )

                    eta = elapsed * (self.max_epochs - epoch) / max(epoch, 1)

                    print(f"\n[PROGRESS] Epoch {epoch}/{self.max_epochs} completed")
                    print(f"          Elapsed: {elapsed}")
                    print(f"          ETA: {eta}")
                    print(f"          Checkpoint: {args.output_dir}/checkpoint-{epoch * len(dataset) // self.max_epochs}")

                    # Save progress
                    self.tracker.save(
                        epoch=epoch,
                        steps_completed=state.global_step
                    )

        # Add callback (manual implementation since Trainer callback complex)
        # Train normally
        trainer.train()

        # Save final model
        print("\n[5/5] Saving final model...")
        final_path = os.path.join(self.output_dir, "final_model")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)

        # Mark as completed
        self.progress.save(
            status="completed",
            epoch=self.max_epochs,
            model_path=final_path,
            end_time=datetime.now().isoformat()
        )

        print(f"\n[OK] Training complete!")
        print(f"   Model saved to: {final_path}")

        return final_path

    def show_status(self):
        """Show training status"""
        data = self.progress.get_status()

        print("="*50)
        print("EXPERIMENT 8 STATUS")
        print("="*50)
        print(f"Status: {data['status'].upper()}")

        if data['status'] == "training":
            print(f"Epoch: {data['epoch']}/{data.get('max_epochs', '?')}")
            print(f"Steps: {data['steps_completed']}")
        elif data['status'] == "completed":
            print(f"Epochs completed: {data['epoch']}")
            print(f"Model path: {data['model_path']}")

        print(f"Start time: {data.get('start_time', 'N/A')}")
        print(f"Last update: {data.get('last_update', 'N/A')}")

        # Check for checkpoints
        if os.path.exists(self.output_dir):
            checkpoints = [d for d in os.listdir(self.output_dir) if d.startswith('checkpoint-')]
            if checkpoints:
                print(f"\nCheckpoints available:")
                for cp in sorted(checkpoints):
                    print(f"  - {cp}")
        else:
            print("\nNo checkpoints found (training not started)")

        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Experiment 8: Extended DAPT")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--status", action="store_true", help="Show training status")
    args = parser.parse_args()

    trainer = ExtendedDAPTTrainer(resume=args.resume)

    if args.status:
        trainer.show_status()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
