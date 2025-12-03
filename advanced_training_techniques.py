import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from transformers.trainer_utils import EvalPrediction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MixupDataset(Dataset):
    """Dataset with Mixup augmentation"""
    def __init__(self, encodings, labels, alpha=0.2):
        self.encodings = encodings
        self.labels = labels
        self.alpha = alpha
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Apply mixup with probability 0.5
        if np.random.random() > 0.5 and len(self.labels) > 1:
            # Select random sample for mixing
            mix_idx = np.random.randint(0, len(self.labels))
            if mix_idx != idx:
                lam = np.random.beta(self.alpha, self.alpha)
                
                # Mix input embeddings (approximation)
                for key in ['input_ids', 'attention_mask']:
                    if key in item:
                        mix_item = torch.tensor(self.encodings[key][mix_idx])
                        # For discrete tokens, use lambda as probability
                        if np.random.random() < lam:
                            item[key] = item[key]
                        else:
                            item[key] = mix_item
                
                # Mix labels
                item['mixup_lambda'] = lam
                item['mixup_target'] = torch.tensor(self.labels[mix_idx], dtype=torch.long)
        
        return item
    
    def __len__(self):
        return len(self.labels)

class JavaneseDataset(Dataset):
    """Standard Javanese dataset"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.labels)

class CustomTrainer(Trainer):
    """Custom trainer with Focal Loss and Mixup support"""
    def __init__(self, *args, use_focal_loss=False, focal_alpha=1, focal_gamma=2, 
                 label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_focal_loss = use_focal_loss
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma) if use_focal_loss else None
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get('logits')
        
        if self.use_focal_loss:
            loss = self.focal_loss(logits, labels)
        else:
            # Use label smoothing
            if self.label_smoothing > 0:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        # Handle mixup if present
        if 'mixup_lambda' in inputs and 'mixup_target' in inputs:
            lam = inputs['mixup_lambda']
            target_a = labels
            target_b = inputs['mixup_target']
            
            if self.use_focal_loss:
                loss_a = self.focal_loss(logits, target_a)
                loss_b = self.focal_loss(logits, target_b)
            else:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                loss_a = loss_fct(logits, target_a)
                loss_b = loss_fct(logits, target_b)
            
            loss = lam * loss_a + (1 - lam) * loss_b
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred: EvalPrediction):
    """Compute accuracy and F1 metrics"""
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

def train_with_advanced_techniques(model_name, train_texts, train_labels, val_texts, val_labels,
                                 test_texts, test_labels, technique_config):
    """Train model with advanced techniques"""
    logger.info(f"Training {model_name} with advanced techniques: {technique_config}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(np.unique(train_labels))
    )
    
    # Tokenize data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
    
    # Create datasets
    if technique_config.get('use_mixup', False):
        train_dataset = MixupDataset(train_encodings, train_labels, 
                                   alpha=technique_config.get('mixup_alpha', 0.2))
    else:
        train_dataset = JavaneseDataset(train_encodings, train_labels)
    
    val_dataset = JavaneseDataset(val_encodings, val_labels)
    test_dataset = JavaneseDataset(test_encodings, test_labels)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'./results/advanced_techniques_{model_name.replace("/", "_")}',
        num_train_epochs=technique_config.get('num_epochs', 5),
        per_device_train_batch_size=technique_config.get('batch_size', 16),
        per_device_eval_batch_size=technique_config.get('batch_size', 16),
        learning_rate=technique_config.get('learning_rate', 2e-5),
        weight_decay=technique_config.get('weight_decay', 0.01),
        warmup_ratio=technique_config.get('warmup_ratio', 0.1),
        gradient_accumulation_steps=technique_config.get('gradient_accumulation_steps', 1),
        logging_dir=f'./results/advanced_techniques_{model_name.replace("/", "_")}/logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=None,
        seed=42,
        lr_scheduler_type=technique_config.get('lr_scheduler_type', 'cosine'),
        dataloader_num_workers=0,
        remove_unused_columns=False
    )
    
    # Create custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        use_focal_loss=technique_config.get('use_focal_loss', False),
        focal_alpha=technique_config.get('focal_alpha', 1.0),
        focal_gamma=technique_config.get('focal_gamma', 2.0),
        label_smoothing=technique_config.get('label_smoothing', 0.0)
    )
    
    # Train model
    trainer.train()
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    
    # Get predictions for detailed analysis
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    
    # Calculate detailed metrics
    accuracy = accuracy_score(test_labels, predicted_labels)
    f1_macro = f1_score(test_labels, predicted_labels, average='macro')
    f1_weighted = f1_score(test_labels, predicted_labels, average='weighted')
    
    results = {
        'model_name': model_name,
        'technique_config': technique_config,
        'test_accuracy': accuracy,
        'test_f1_macro': f1_macro,
        'test_f1_weighted': f1_weighted,
        'test_results': test_results,
        'classification_report': classification_report(test_labels, predicted_labels, output_dict=True)
    }
    
    logger.info(f"Results for {model_name}: Accuracy={accuracy:.4f}, F1-Macro={f1_macro:.4f}")
    
    return results

def main():
    """Main function to run advanced training techniques experiments"""
    logger.info("Starting Advanced Training Techniques Experiments")
    
    # Load dataset
    logger.info("Loading dataset...")
    augmented_data_path = 'data/augmented/augmented_dataset.csv'
    df = pd.read_csv(augmented_data_path)
    
    texts = df['text'].tolist()
    labels = df['label_numeric'].tolist()
    
    logger.info(f"Dataset loaded: {len(texts)} samples, {len(set(labels))} classes")
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    logger.info(f"Data split - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Define models to test
    models = [
        "indobenchmark/indobert-base-p1",
        "flax-community/indonesian-roberta-base"
    ]
    
    # Define technique configurations
    technique_configs = [
        {
            'name': 'baseline',
            'num_epochs': 5,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1
        },
        {
            'name': 'focal_loss',
            'use_focal_loss': True,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'num_epochs': 5,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1
        },
        {
            'name': 'label_smoothing',
            'label_smoothing': 0.1,
            'num_epochs': 5,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1
        },
        {
            'name': 'mixup',
            'use_mixup': True,
            'mixup_alpha': 0.2,
            'num_epochs': 5,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1
        },
        {
            'name': 'combined_advanced',
            'use_focal_loss': True,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'label_smoothing': 0.05,
            'use_mixup': True,
            'mixup_alpha': 0.2,
            'num_epochs': 6,
            'batch_size': 16,
            'learning_rate': 1.5e-5,
            'weight_decay': 0.02,
            'warmup_ratio': 0.15,
            'lr_scheduler_type': 'cosine'
        }
    ]
    
    # Run experiments
    all_results = []
    
    for model_name in models:
        for config in technique_configs:
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"Training {model_name} with {config['name']} technique")
                logger.info(f"{'='*80}")
                
                results = train_with_advanced_techniques(
                    model_name, train_texts, train_labels, val_texts, val_labels,
                    test_texts, test_labels, config
                )
                
                all_results.append(results)
                
            except Exception as e:
                logger.error(f"Error training {model_name} with {config['name']}: {str(e)}")
                continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/advanced_training_techniques_results_{timestamp}.json'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("ADVANCED TRAINING TECHNIQUES - SUMMARY")
    logger.info("="*80)
    
    for result in all_results:
        model_name = result['model_name'].split('/')[-1]
        technique = result['technique_config']['name']
        accuracy = result['test_accuracy']
        f1_macro = result['test_f1_macro']
        
        logger.info(f"{model_name} + {technique}: Accuracy={accuracy:.4f}, F1-Macro={f1_macro:.4f}")
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info("="*80)

if __name__ == "__main__":
    main()