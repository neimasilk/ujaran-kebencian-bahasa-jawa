import os
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import Dataset
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HateSpeechDataset(Dataset):
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

class CrossValidationFramework:
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_and_prepare_data(self, data_path):
        """Load and prepare data with proper label mapping"""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Clean and prepare data
        df = df.dropna(subset=['text', 'final_label'])
        df['text'] = df['text'].astype(str)
        
        # Create label mapping
        unique_labels = sorted(df['final_label'].unique())
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        
        df['label_id'] = df['final_label'].map(label_to_id)
        
        logger.info(f"Dataset loaded: {len(df)} samples, {len(unique_labels)} classes")
        logger.info(f"Label distribution: {df['final_label'].value_counts().to_dict()}")
        
        return df, {'label_to_id': label_to_id, 'id_to_label': id_to_label}
    
    def train_fold(self, train_texts, train_labels, val_texts, val_labels, fold_num, model_name):
        """Train model for one fold with enhanced regularization"""
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=4,
            hidden_dropout_prob=0.3,  # Increased dropout
            attention_probs_dropout_prob=0.3
        )
        
        # Create datasets
        train_dataset = HateSpeechDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = HateSpeechDataset(val_texts, val_labels, self.tokenizer)
        
        # Enhanced training arguments for better generalization
        training_args = TrainingArguments(
            output_dir=f'./tmp_trainer/fold_{fold_num}',
            num_train_epochs=3,  # Reduced epochs
            per_device_train_batch_size=8,  # Smaller batch size
            per_device_eval_batch_size=16,
            learning_rate=1e-5,  # Lower learning rate
            weight_decay=0.1,  # Increased weight decay
            warmup_ratio=0.1,
            gradient_accumulation_steps=2,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            save_total_limit=2,
            dataloader_drop_last=True,
            fp16=torch.cuda.is_available(),
        )
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            f1_macro = f1_score(labels, predictions, average='macro')
            
            return {
                'accuracy': accuracy,
                'f1_macro': f1_macro
            }
        
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
        trainer.train()
        
        # Evaluate
        val_results = trainer.evaluate()
        
        return trainer, val_results
    
    def cross_validate_model(self, model_name, data_path):
        """Perform cross-validation for a model"""
        logger.info(f"Starting 5-fold cross-validation for {model_name}")
        
        # Load and prepare data
        df, label_info = self.load_and_prepare_data(data_path)
        
        # Setup stratified k-fold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        fold_results = []
        all_predictions = []
        all_true_labels = []
        
        # Run cross-validation
        for fold_num, (train_idx, val_idx) in enumerate(skf.split(df['text'], df['label_id'])):
            logger.info(f"Training fold {fold_num + 1}/{self.n_folds}")
            
            # Split data
            train_texts = df.iloc[train_idx]['text'].tolist()
            train_labels = df.iloc[train_idx]['label_id'].tolist()
            val_texts = df.iloc[val_idx]['text'].tolist()
            val_labels = df.iloc[val_idx]['label_id'].tolist()
            
            logger.info(f"Fold {fold_num + 1} - Train: {len(train_texts)}, Val: {len(val_texts)}")
            
            # Train fold
            trainer, val_results = self.train_fold(
                train_texts, train_labels, val_texts, val_labels, fold_num, model_name
            )
            
            # Get predictions for this fold
            val_dataset = HateSpeechDataset(val_texts, val_labels, self.tokenizer)
            predictions = trainer.predict(val_dataset)
            pred_labels = np.argmax(predictions.predictions, axis=1)
            
            # Store results
            fold_result = {
                'fold': fold_num + 1,
                'accuracy': val_results['eval_accuracy'],
                'f1_macro': val_results['eval_f1_macro'],
                'train_size': len(train_texts),
                'val_size': len(val_texts)
            }
            
            fold_results.append(fold_result)
            all_predictions.extend(pred_labels)
            all_true_labels.extend(val_labels)
            
            logger.info(f"Fold {fold_num + 1} Results - Acc: {val_results['eval_accuracy']:.4f}, F1: {val_results['eval_f1_macro']:.4f}")
        
        # Calculate overall metrics
        accuracies = [result['accuracy'] for result in fold_results]
        f1_macros = [result['f1_macro'] for result in fold_results]
        
        cv_results = {
            'model_name': model_name,
            'n_folds': self.n_folds,
            'fold_results': fold_results,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_f1_macro': np.mean(f1_macros),
            'std_f1_macro': np.std(f1_macros),
            'overall_accuracy': accuracy_score(all_true_labels, all_predictions),
            'overall_f1_macro': f1_score(all_true_labels, all_predictions, average='macro'),
            'classification_report': classification_report(
                all_true_labels, all_predictions,
                target_names=[label_info['id_to_label'][i] for i in range(len(label_info['id_to_label']))],
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(all_true_labels, all_predictions).tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return cv_results

def main():
    # Initialize framework
    cv_framework = CrossValidationFramework(n_folds=5)
    
    # Models to test
    models = [
        'indobenchmark/indobert-base-p1',
        'indolem/indobert-base-uncased',
        'cahya/roberta-base-indonesian-522M'
    ]
    
    data_path = 'data/augmented/augmented_dataset.csv'
    all_results = {}
    
    # Run cross-validation for each model
    for model_name in models:
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting cross-validation for {model_name}")
            logger.info(f"{'='*80}")
            
            results = cv_framework.cross_validate_model(model_name, data_path)
            all_results[model_name] = results
            
            logger.info(f"\nResults for {model_name}:")
            logger.info(f"Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
            logger.info(f"Mean F1-Macro: {results['mean_f1_macro']:.4f} ± {results['std_f1_macro']:.4f}")
            logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
            logger.info(f"Overall F1-Macro: {results['overall_f1_macro']:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing {model_name}: {e}")
            continue
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/cross_validation_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("\nAll results saved to results/cross_validation_summary.json")

if __name__ == "__main__":
    main()