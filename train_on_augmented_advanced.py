import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
import os
from datetime import datetime
import logging
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

class JavaneseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def train_model_on_augmented_data(model_name, augmented_data_path, test_size=0.2, val_size=0.1):
    """Train model on augmented dataset with advanced techniques"""
    
    logger.info(f"Training {model_name} on augmented dataset")
    
    # Load augmented dataset
    logger.info(f"Loading augmented dataset from {augmented_data_path}")
    df = pd.read_csv(augmented_data_path)
    
    logger.info(f"Dataset loaded: {len(df)} samples")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Check for required columns
    if 'text' not in df.columns or 'label_numeric' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label_numeric' columns")
    
    # Prepare data
    texts = df['text'].tolist()
    labels = df['label_numeric'].tolist()
    
    # Split data: train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,  # 4 classes for hate speech classification
        problem_type="single_label_classification"
    )
    
    # Create datasets
    train_dataset = JavaneseDataset(X_train, y_train, tokenizer)
    val_dataset = JavaneseDataset(X_val, y_val, tokenizer)
    test_dataset = JavaneseDataset(X_test, y_test, tokenizer)
    
    # Training arguments with advanced settings
    model_safe_name = model_name.replace('/', '_')
    output_dir = f"models/{model_safe_name}_augmented_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,
        seed=42,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_smoothing_factor=0.1,  # Label smoothing for better generalization
        learning_rate=2e-5,
        lr_scheduler_type="cosine",  # Cosine learning rate schedule
        gradient_accumulation_steps=2,  # Gradient accumulation
    )
    
    # Initialize trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    # Get predictions for detailed analysis
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # Calculate detailed metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1_macro = f1_score(y_test, y_pred, average='macro')
    test_f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Classification report
    class_names = ['Bukan Ujaran Kebencian', 'Ujaran Kebencian - Ringan', 
                   'Ujaran Kebencian - Sedang', 'Ujaran Kebencian - Berat']
    
    classification_rep = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Prepare results
    results = {
        'model_name': model_name,
        'model_safe_name': model_safe_name,
        'output_dir': output_dir,
        'dataset_info': {
            'total_samples': len(df),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'augmented_samples': len(df[df.get('augmented', False) == True]) if 'augmented' in df.columns else 'unknown'
        },
        'test_metrics': {
            'accuracy': float(test_accuracy),
            'f1_macro': float(test_f1_macro),
            'f1_weighted': float(test_f1_weighted)
        },
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix.tolist(),
        'training_args': {
            'num_train_epochs': training_args.num_train_epochs,
            'learning_rate': training_args.learning_rate,
            'batch_size': training_args.per_device_train_batch_size,
            'weight_decay': training_args.weight_decay,
            'label_smoothing_factor': training_args.label_smoothing_factor,
            'lr_scheduler_type': training_args.lr_scheduler_type
        },
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Test Results - Accuracy: {test_accuracy:.4f}, F1-Macro: {test_f1_macro:.4f}")
    
    return results, trainer

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check for augmented dataset
    augmented_data_path = 'data/augmented/augmented_dataset.csv'
    
    if not os.path.exists(augmented_data_path):
        logger.error(f"Augmented dataset not found at {augmented_data_path}")
        logger.info("Please run advanced_data_augmentation.py first")
        return
    
    # Models to train
    models_to_train = [
        'indolem/indobert-base-uncased',
        'indobenchmark/indobert-base-p1',
        'flax-community/indonesian-roberta-base'
    ]
    
    all_results = []
    
    for model_name in models_to_train:
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"Training {model_name} on augmented dataset")
            logger.info(f"{'='*80}")
            
            results, trainer = train_model_on_augmented_data(model_name, augmented_data_path)
            all_results.append(results)
            
            # Save individual results
            model_safe_name = model_name.replace('/', '_')
            result_file = f"results/{model_safe_name}_augmented_results.json"
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {result_file}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    # Save combined results
    combined_results = {
        'experiment_name': 'Training on Augmented Dataset',
        'timestamp': datetime.now().isoformat(),
        'models_trained': len(all_results),
        'results': all_results,
        'best_model': None
    }
    
    # Find best model
    if all_results:
        best_model = max(all_results, key=lambda x: x['test_metrics']['f1_macro'])
        combined_results['best_model'] = {
            'model_name': best_model['model_name'],
            'accuracy': best_model['test_metrics']['accuracy'],
            'f1_macro': best_model['test_metrics']['f1_macro'],
            'f1_weighted': best_model['test_metrics']['f1_weighted']
        }
    
    # Save combined results
    combined_file = f"results/augmented_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING ON AUGMENTED DATASET - SUMMARY")
    logger.info(f"{'='*80}")
    
    if all_results:
        logger.info(f"Models trained: {len(all_results)}")
        logger.info(f"Best model: {combined_results['best_model']['model_name']}")
        logger.info(f"Best accuracy: {combined_results['best_model']['accuracy']:.4f}")
        logger.info(f"Best F1-Macro: {combined_results['best_model']['f1_macro']:.4f}")
        
        logger.info("\nAll Results:")
        for result in all_results:
            logger.info(f"  {result['model_name']}: Acc={result['test_metrics']['accuracy']:.4f}, F1={result['test_metrics']['f1_macro']:.4f}")
    
    logger.info(f"\nResults saved to: {combined_file}")
    logger.info(f"{'='*80}")

if __name__ == "__main__":
    main()