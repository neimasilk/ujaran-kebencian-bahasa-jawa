#!/usr/bin/env python3
"""
Advanced Threshold Tuning for Hate Speech Detection Model
Optimizes prediction thresholds to improve model performance
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import logging
from tqdm import tqdm
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThresholdTuner:
    def __init__(self, model_path, tokenizer_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping
        self.label_names = [
            "Bukan Ujaran Kebencian",
            "Ujaran Kebencian - Ringan", 
            "Ujaran Kebencian - Sedang",
            "Ujaran Kebencian - Berat"
        ]
        
        logger.info("Model and tokenizer loaded successfully")
    
    def predict_probabilities(self, texts, batch_size=32):
        """Get prediction probabilities for texts"""
        all_probs = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Getting probabilities"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_probs)
    
    def evaluate_with_thresholds(self, probs, true_labels, thresholds):
        """Evaluate model performance with given thresholds"""
        predictions = []
        
        for prob in probs:
            # Apply thresholds
            adjusted_probs = prob.copy()
            for i, threshold in enumerate(thresholds):
                if prob[i] < threshold:
                    adjusted_probs[i] = 0
            
            # Predict class with highest adjusted probability
            if np.sum(adjusted_probs) == 0:
                # If all probabilities are below threshold, use original
                pred = np.argmax(prob)
            else:
                pred = np.argmax(adjusted_probs)
            
            predictions.append(pred)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': predictions
        }
    
    def optimize_thresholds(self, probs, true_labels, metric='f1_macro'):
        """Optimize thresholds using scipy optimization"""
        def objective(thresholds):
            result = self.evaluate_with_thresholds(probs, true_labels, thresholds)
            return -result[metric]  # Negative because we want to maximize
        
        # Initial thresholds (0.25 for each class)
        initial_thresholds = [0.25] * len(self.label_names)
        
        # Bounds for thresholds (0.1 to 0.9)
        bounds = [(0.1, 0.9) for _ in range(len(self.label_names))]
        
        logger.info(f"Optimizing thresholds for {metric}...")
        result = minimize(
            objective,
            initial_thresholds,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        optimal_thresholds = result.x
        logger.info(f"Optimal thresholds: {optimal_thresholds}")
        
        return optimal_thresholds
    
    def grid_search_thresholds(self, probs, true_labels, metric='f1_macro'):
        """Grid search for optimal thresholds"""
        threshold_range = np.arange(0.1, 0.9, 0.1)
        best_score = 0
        best_thresholds = None
        
        logger.info("Starting grid search for thresholds...")
        
        # For computational efficiency, we'll do a coarse grid search
        for t1 in threshold_range[::2]:  # Skip every other value
            for t2 in threshold_range[::2]:
                for t3 in threshold_range[::2]:
                    for t4 in threshold_range[::2]:
                        thresholds = [t1, t2, t3, t4]
                        result = self.evaluate_with_thresholds(probs, true_labels, thresholds)
                        
                        if result[metric] > best_score:
                            best_score = result[metric]
                            best_thresholds = thresholds
        
        logger.info(f"Best {metric}: {best_score:.4f}")
        logger.info(f"Best thresholds: {best_thresholds}")
        
        return best_thresholds
    
    def evaluate_threshold_impact(self, probs, true_labels, thresholds):
        """Detailed evaluation of threshold impact"""
        # Baseline (no thresholds)
        baseline_preds = np.argmax(probs, axis=1)
        baseline_results = {
            'accuracy': accuracy_score(true_labels, baseline_preds),
            'f1_macro': f1_score(true_labels, baseline_preds, average='macro'),
            'f1_weighted': f1_score(true_labels, baseline_preds, average='weighted')
        }
        
        # With thresholds
        threshold_results = self.evaluate_with_thresholds(probs, true_labels, thresholds)
        
        # Per-class analysis
        baseline_report = classification_report(true_labels, baseline_preds, 
                                              target_names=self.label_names, output_dict=True)
        threshold_report = classification_report(true_labels, threshold_results['predictions'],
                                               target_names=self.label_names, output_dict=True)
        
        return {
            'baseline': baseline_results,
            'threshold': threshold_results,
            'baseline_report': baseline_report,
            'threshold_report': threshold_report,
            'improvement': {
                'accuracy': threshold_results['accuracy'] - baseline_results['accuracy'],
                'f1_macro': threshold_results['f1_macro'] - baseline_results['f1_macro'],
                'f1_weighted': threshold_results['f1_weighted'] - baseline_results['f1_weighted']
            }
        }
    
    def plot_threshold_analysis(self, analysis_results, output_path, true_labels=None):
        """Create visualization of threshold analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Metric comparison
        metrics = ['accuracy', 'f1_macro', 'f1_weighted']
        baseline_scores = [analysis_results['baseline'][m] for m in metrics]
        threshold_scores = [analysis_results['threshold'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, baseline_scores, width, label='Baseline', alpha=0.8)
        axes[0, 0].bar(x + width/2, threshold_scores, width, label='With Thresholds', alpha=0.8)
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Per-class F1 comparison
        baseline_f1 = [analysis_results['baseline_report'][label]['f1-score'] 
                      for label in self.label_names]
        threshold_f1 = [analysis_results['threshold_report'][label]['f1-score'] 
                       for label in self.label_names]
        
        x = np.arange(len(self.label_names))
        axes[0, 1].bar(x - width/2, baseline_f1, width, label='Baseline', alpha=0.8)
        axes[0, 1].bar(x + width/2, threshold_f1, width, label='With Thresholds', alpha=0.8)
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_title('Per-Class F1-Score Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([label.replace(' - ', '\n') for label in self.label_names], 
                                  rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confusion matrix for threshold predictions
        if true_labels is not None and 'predictions' in analysis_results['threshold']:
            threshold_cm = confusion_matrix(true_labels, analysis_results['threshold']['predictions'])
            sns.heatmap(threshold_cm, annot=True, fmt='d', ax=axes[1, 0], 
                       xticklabels=[label.replace(' - ', '\n') for label in self.label_names], 
                       yticklabels=[label.replace(' - ', '\n') for label in self.label_names])
            axes[1, 0].set_title('Confusion Matrix (With Thresholds)')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Actual')
        else:
            # Create improvement chart instead
            improvements = [analysis_results['improvement'][m] for m in metrics]
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            
            axes[1, 0].bar(metrics, improvements, color=colors, alpha=0.7)
            axes[1, 0].set_xlabel('Metrics')
            axes[1, 0].set_ylabel('Improvement')
            axes[1, 0].set_title('Performance Improvement')
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Summary text
        summary_text = f"""Threshold Tuning Results:
        
Accuracy: {analysis_results['baseline']['accuracy']:.3f} → {analysis_results['threshold']['accuracy']:.3f}
F1-Macro: {analysis_results['baseline']['f1_macro']:.3f} → {analysis_results['threshold']['f1_macro']:.3f}
F1-Weighted: {analysis_results['baseline']['f1_weighted']:.3f} → {analysis_results['threshold']['f1_weighted']:.3f}
        
Improvements:
Accuracy: {analysis_results['improvement']['accuracy']:+.3f}
F1-Macro: {analysis_results['improvement']['f1_macro']:+.3f}
F1-Weighted: {analysis_results['improvement']['f1_weighted']:+.3f}"""
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Threshold analysis plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Advanced Threshold Tuning')
    parser.add_argument('--model_path', required=True, help='Path to the trained model')
    parser.add_argument('--data_path', required=True, help='Path to evaluation dataset')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    parser.add_argument('--method', choices=['optimize', 'grid_search'], default='optimize',
                       help='Threshold tuning method')
    parser.add_argument('--metric', choices=['accuracy', 'f1_macro', 'f1_weighted'], 
                       default='f1_macro', help='Metric to optimize')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading evaluation data from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    # Prepare data
    texts = df['text'].tolist()
    
    # Map labels to numeric
    label_mapping = {
        "Bukan Ujaran Kebencian": 0,
        "Ujaran Kebencian - Ringan": 1,
        "Ujaran Kebencian - Sedang": 2,
        "Ujaran Kebencian - Berat": 3
    }
    
    true_labels = [label_mapping[label] for label in df['final_label'].tolist()]
    
    logger.info(f"Loaded {len(texts)} samples")
    
    # Initialize tuner
    tuner = ThresholdTuner(args.model_path)
    
    # Get probabilities
    logger.info("Getting model probabilities...")
    probs = tuner.predict_probabilities(texts, args.batch_size)
    
    # Optimize thresholds
    if args.method == 'optimize':
        optimal_thresholds = tuner.optimize_thresholds(probs, true_labels, args.metric)
    else:
        optimal_thresholds = tuner.grid_search_thresholds(probs, true_labels, args.metric)
    
    # Evaluate impact
    logger.info("Evaluating threshold impact...")
    analysis_results = tuner.evaluate_threshold_impact(probs, true_labels, optimal_thresholds)
    
    # Save results (convert numpy arrays to lists for JSON serialization)
    def convert_numpy(obj):
        """Convert numpy arrays and scalars to JSON serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, tuple):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results = {
        'optimal_thresholds': optimal_thresholds.tolist() if isinstance(optimal_thresholds, np.ndarray) else optimal_thresholds,
        'threshold_labels': tuner.label_names,
        'analysis': convert_numpy(analysis_results),
        'method': args.method,
        'optimized_metric': args.metric
    }
    
    output_file = os.path.join(args.output_dir, 'threshold_tuning_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Create visualization
    plot_file = os.path.join(args.output_dir, 'threshold_analysis.png')
    tuner.plot_threshold_analysis(analysis_results, plot_file, true_labels)
    
    # Print summary
    logger.info("\n=== THRESHOLD TUNING RESULTS ===")
    logger.info(f"Method: {args.method}")
    logger.info(f"Optimized metric: {args.metric}")
    logger.info(f"Optimal thresholds: {optimal_thresholds}")
    logger.info("\n=== PERFORMANCE COMPARISON ===")
    logger.info(f"Baseline Accuracy: {analysis_results['baseline']['accuracy']:.4f}")
    logger.info(f"Threshold Accuracy: {analysis_results['threshold']['accuracy']:.4f}")
    logger.info(f"Improvement: {analysis_results['improvement']['accuracy']:+.4f}")
    logger.info(f"\nBaseline F1-Macro: {analysis_results['baseline']['f1_macro']:.4f}")
    logger.info(f"Threshold F1-Macro: {analysis_results['threshold']['f1_macro']:.4f}")
    logger.info(f"Improvement: {analysis_results['improvement']['f1_macro']:+.4f}")
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Visualization saved to: {plot_file}")

if __name__ == "__main__":
    main()