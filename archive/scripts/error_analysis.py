#!/usr/bin/env python3
"""
Script untuk error analysis mendalam pada model hate speech classification
"""

import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import logging
import json
from collections import defaultdict, Counter
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorAnalyzer:
    def __init__(self, model_path, tokenizer_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Label mappings
        self.label_names = {
            0: "Bukan Ujaran Kebencian",
            1: "Ujaran Kebencian - Ringan", 
            2: "Ujaran Kebencian - Sedang",
            3: "Ujaran Kebencian - Berat"
        }
        
        self.reverse_label_mapping = {
            "Bukan Ujaran Kebencian": 0,
            "Ujaran Kebencian - Ringan": 1,
            "Ujaran Kebencian - Sedang": 2,
            "Ujaran Kebencian - Berat": 3
        }
    
    def predict_batch(self, texts, batch_size=32):
        """Predict labels for a batch of texts"""
        all_predictions = []
        all_probabilities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def analyze_text_features(self, text):
        """Extract features from text for analysis"""
        features = {}
        
        # Basic text statistics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Character patterns
        features['has_numbers'] = bool(re.search(r'\d', text))
        features['has_punctuation'] = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', text))
        features['has_uppercase'] = bool(re.search(r'[A-Z]', text))
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Language patterns (Javanese specific)
        javanese_words = ['wong', 'aku', 'kowe', 'iku', 'iki', 'ana', 'ora', 'lan', 'karo', 'ning']
        features['javanese_word_count'] = sum(1 for word in javanese_words if word in text.lower())
        
        # Potential hate speech indicators
        hate_indicators = ['benci', 'sial', 'bangsat', 'anjing', 'tolol', 'bodoh']
        features['hate_word_count'] = sum(1 for word in hate_indicators if word in text.lower())
        
        return features
    
    def analyze_errors(self, data_path, output_path, max_samples_per_class=200):
        """Perform comprehensive error analysis"""
        logger.info(f"Loading data from: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Map labels to numeric
        if 'label_numeric' in df.columns:
            true_labels = df['label_numeric'].values
        else:
            true_labels = df['final_label'].map(self.reverse_label_mapping).values
        
        texts = df['text'].tolist()
        
        # Get predictions
        logger.info("Running predictions...")
        predicted_labels, probabilities = self.predict_batch(texts)
        
        # Calculate confidence scores
        confidence_scores = np.max(probabilities, axis=1)
        
        # Create results dataframe
        results_df = df.copy()
        results_df['predicted_label'] = predicted_labels
        results_df['predicted_label_name'] = [self.label_names[pred] for pred in predicted_labels]
        results_df['confidence'] = confidence_scores
        results_df['is_correct'] = (true_labels == predicted_labels)
        
        # Add text features
        logger.info("Analyzing text features...")
        text_features = [self.analyze_text_features(text) for text in texts]
        for feature_name in text_features[0].keys():
            results_df[f'feature_{feature_name}'] = [features[feature_name] for features in text_features]
        
        # Error analysis
        error_analysis = self.perform_error_analysis(results_df, true_labels, predicted_labels, max_samples_per_class)
        
        # Save results
        logger.info(f"Saving results to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save detailed results
        results_df.to_csv(output_path.replace('.json', '_detailed.csv'), index=False)
        
        # Save error analysis
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(error_analysis, f, indent=2, ensure_ascii=False)
        
        return error_analysis
    
    def perform_error_analysis(self, results_df, true_labels, predicted_labels, max_samples_per_class):
        """Perform detailed error analysis"""
        analysis = {}
        
        # Overall metrics
        accuracy = np.mean(true_labels == predicted_labels)
        analysis['overall_accuracy'] = float(accuracy)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        analysis['confusion_matrix'] = cm.tolist()
        
        # Per-class analysis
        analysis['per_class_analysis'] = {}
        
        for class_id in range(4):
            class_name = self.label_names[class_id]
            class_mask = (true_labels == class_id)
            
            if not class_mask.any():
                continue
                
            class_data = results_df[class_mask]
            
            # Misclassified samples
            misclassified = class_data[~class_data['is_correct']]
            
            class_analysis = {
                'total_samples': int(class_mask.sum()),
                'correct_predictions': int((class_data['is_correct']).sum()),
                'accuracy': float((class_data['is_correct']).mean()),
                'misclassified_count': len(misclassified),
                'avg_confidence_correct': float(class_data[class_data['is_correct']]['confidence'].mean()) if len(class_data[class_data['is_correct']]) > 0 else 0,
                'avg_confidence_incorrect': float(misclassified['confidence'].mean()) if len(misclassified) > 0 else 0
            }
            
            # Analyze misclassification patterns
            if len(misclassified) > 0:
                # Most common misclassifications
                misclass_counts = misclassified['predicted_label_name'].value_counts()
                class_analysis['common_misclassifications'] = misclass_counts.to_dict()
                
                # Feature analysis for misclassified samples
                feature_cols = [col for col in misclassified.columns if col.startswith('feature_')]
                feature_analysis = {}
                
                for feature_col in feature_cols:
                    feature_name = feature_col.replace('feature_', '')
                    correct_mean = class_data[class_data['is_correct']][feature_col].mean()
                    incorrect_mean = misclassified[feature_col].mean()
                    
                    feature_analysis[feature_name] = {
                        'correct_mean': float(correct_mean) if not pd.isna(correct_mean) else 0,
                        'incorrect_mean': float(incorrect_mean) if not pd.isna(incorrect_mean) else 0,
                        'difference': float(incorrect_mean - correct_mean) if not pd.isna(correct_mean) and not pd.isna(incorrect_mean) else 0
                    }
                
                class_analysis['feature_analysis'] = feature_analysis
                
                # Sample misclassified examples
                sample_size = min(max_samples_per_class // 4, len(misclassified))
                sample_errors = misclassified.nlargest(sample_size, 'confidence')[['text', 'predicted_label_name', 'confidence']]
                class_analysis['sample_errors'] = sample_errors.to_dict('records')
            
            analysis['per_class_analysis'][class_name] = class_analysis
        
        # Low confidence predictions
        low_confidence = results_df[results_df['confidence'] < 0.6]
        analysis['low_confidence_analysis'] = {
            'count': len(low_confidence),
            'percentage': float(len(low_confidence) / len(results_df) * 100),
            'accuracy': float(low_confidence['is_correct'].mean()) if len(low_confidence) > 0 else 0,
            'label_distribution': low_confidence['predicted_label_name'].value_counts().to_dict() if len(low_confidence) > 0 else {}
        }
        
        return analysis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--data_path', required=True, help='Path to evaluation dataset')
    parser.add_argument('--output_path', default='results/error_analysis.json', help='Output path for analysis')
    parser.add_argument('--max_samples', type=int, default=200, help='Max samples per class for detailed analysis')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ErrorAnalyzer(args.model_path)
    
    # Run analysis
    analysis = analyzer.analyze_errors(args.data_path, args.output_path, args.max_samples)
    
    # Print summary
    logger.info("\n=== ERROR ANALYSIS SUMMARY ===")
    logger.info(f"Overall Accuracy: {analysis['overall_accuracy']:.4f}")
    
    for class_name, class_analysis in analysis['per_class_analysis'].items():
        logger.info(f"\n{class_name}:")
        logger.info(f"  Accuracy: {class_analysis['accuracy']:.4f}")
        logger.info(f"  Misclassified: {class_analysis['misclassified_count']}/{class_analysis['total_samples']}")
        if 'common_misclassifications' in class_analysis:
            logger.info(f"  Common errors: {class_analysis['common_misclassifications']}")
    
    logger.info(f"\nLow confidence predictions: {analysis['low_confidence_analysis']['count']} ({analysis['low_confidence_analysis']['percentage']:.2f}%)")
    logger.info("\nDetailed analysis saved to:", args.output_path)
    logger.info("Detailed CSV saved to:", args.output_path.replace('.json', '_detailed.csv'))

if __name__ == '__main__':
    main()