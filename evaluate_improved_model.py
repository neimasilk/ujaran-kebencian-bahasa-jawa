#!/usr/bin/env python3
"""
Evaluate Improved Model Performance
Compare improved model with baseline and generate detailed metrics
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/improved_model_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('improved_evaluation')

class ImprovedModelEvaluator:
    def __init__(self, model_path="models/improved_model", data_path="data/standardized/balanced_dataset.csv"):
        self.model_path = model_path
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Class mapping
        self.label_map = {
            0: 'not_hate_speech',
            1: 'light_hate_speech', 
            2: 'medium_hate_speech',
            3: 'heavy_hate_speech'
        }
        
    def load_test_data(self):
        """Load and prepare test data"""
        logger.info("Loading test dataset...")
        df = pd.read_csv(self.data_path)
        
        # Use 20% as test set (stratified)
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            stratify=df['label_numeric'], 
            random_state=42
        )
        
        logger.info(f"Test set size: {len(test_df)}")
        logger.info(f"Test set distribution:\n{test_df['label_numeric'].value_counts().sort_index()}")
        
        return test_df
    
    def predict_batch(self, texts, batch_size=16):
        """Predict labels for batch of texts"""
        predictions = []
        probabilities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        logger.info("Starting model evaluation...")
        
        # Load test data
        test_df = self.load_test_data()
        texts = test_df['text'].tolist()
        true_labels = test_df['label_numeric'].values
        
        # Get predictions
        logger.info("Generating predictions...")
        predictions, probabilities = self.predict_batch(texts)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        
        # Detailed classification report
        class_report = classification_report(
            true_labels, 
            predictions, 
            target_names=list(self.label_map.values()),
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # Results summary
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'test_samples': len(test_df),
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        return results
    
    def compare_with_baseline(self, baseline_results_path="memory-bank/02-research-active/EXPERIMENTAL_RESULTS_FOR_PUBLICATION.md"):
        """Compare with baseline performance"""
        logger.info("Comparing with baseline results...")
        
        # Baseline performance (from previous experiments)
        baseline_f1_macro = 0.6075  # IndoBERT Large v1.2 best result
        baseline_accuracy = 0.6580  # Best accuracy achieved
        
        current_results = self.evaluate_model()
        
        improvement = {
            'f1_macro_improvement': current_results['f1_macro'] - baseline_f1_macro,
            'accuracy_improvement': current_results['accuracy'] - baseline_accuracy,
            'baseline_f1_macro': baseline_f1_macro,
            'baseline_accuracy': baseline_accuracy,
            'current_f1_macro': current_results['f1_macro'],
            'current_accuracy': current_results['accuracy']
        }
        
        return current_results, improvement
    
    def save_results(self, results, improvement, output_path="results/improved_model_evaluation.json"):
        """Save evaluation results"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        full_results = {
            'evaluation_results': results,
            'improvement_analysis': improvement
        }
        
        with open(output_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        
    def print_summary(self, results, improvement):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("IMPROVED MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"   F1-Macro: {results['f1_macro']:.4f} ({results['f1_macro']*100:.2f}%)")
        print(f"   F1-Weighted: {results['f1_weighted']:.4f} ({results['f1_weighted']*100:.2f}%)")
        
        print(f"\n📈 IMPROVEMENT vs BASELINE:")
        print(f"   F1-Macro: {improvement['baseline_f1_macro']:.4f} → {improvement['current_f1_macro']:.4f} (+{improvement['f1_macro_improvement']:.4f})")
        print(f"   Accuracy: {improvement['baseline_accuracy']:.4f} → {improvement['current_accuracy']:.4f} (+{improvement['accuracy_improvement']:.4f})")
        
        if improvement['f1_macro_improvement'] > 0:
            print(f"   ✅ IMPROVEMENT: +{improvement['f1_macro_improvement']*100:.2f}% F1-Macro")
        else:
            print(f"   ❌ REGRESSION: {improvement['f1_macro_improvement']*100:.2f}% F1-Macro")
        
        print(f"\n📋 PER-CLASS PERFORMANCE:")
        for class_name, metrics in results['classification_report'].items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                if isinstance(metrics, dict):
                    print(f"   {class_name}: F1={metrics['f1-score']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
        
        print(f"\n🎯 PROGRESS TOWARD 85% TARGET:")
        target_accuracy = 0.85
        current_progress = (results['accuracy'] / target_accuracy) * 100
        remaining = target_accuracy - results['accuracy']
        print(f"   Current: {results['accuracy']*100:.2f}% / Target: 85%")
        print(f"   Progress: {current_progress:.1f}% of target")
        print(f"   Remaining: {remaining*100:.2f}% to reach target")
        
        print("\n" + "="*60)

def main():
    """Main evaluation function"""
    try:
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Initialize evaluator
        evaluator = ImprovedModelEvaluator()
        
        # Run evaluation
        results, improvement = evaluator.compare_with_baseline()
        
        # Save results
        evaluator.save_results(results, improvement)
        
        # Print summary
        evaluator.print_summary(results, improvement)
        
        # Log completion
        logger.info("Evaluation completed successfully!")
        
        return results, improvement
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()