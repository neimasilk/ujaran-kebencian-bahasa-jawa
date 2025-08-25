#!/usr/bin/env python3
"""
Final 90% Optimization Strategy
Analisis bottleneck dan implementasi optimasi lanjutan untuk mencapai target 90% F1-macro

Current Status: 86.16% F1-macro (gap: 3.84%)
Target: 90% F1-macro
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class Final90PercentOptimizer:
    def __init__(self):
        self.results_file = "results/advanced_optimization_90_percent_results.json"
        self.current_results = None
        self.bottlenecks = {}
        self.optimization_strategies = []
        
    def load_current_results(self):
        """Load hasil eksperimen sebelumnya"""
        try:
            with open(self.results_file, 'r') as f:
                self.current_results = json.load(f)
            print(f"✅ Loaded current results: {self.current_results['best_f1_macro']:.4f} F1-macro")
            print(f"📊 Gap to target: {self.current_results['gap_to_target']:.4f}")
            return True
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return False
    
    def analyze_performance_bottlenecks(self):
        """Analisis mendalam bottleneck performa"""
        print("\n🔍 ANALYZING PERFORMANCE BOTTLENECKS")
        print("=" * 50)
        
        if not self.current_results:
            print("❌ No current results to analyze")
            return
        
        # Analisis per-class performance
        classification_report = self.current_results['advanced_meta_ensemble_results']['classification_report']
        
        print("\n📊 PER-CLASS PERFORMANCE ANALYSIS:")
        class_names = ['Bukan Ujaran Kebencian', 'Ujaran Kebencian Ringan', 'Ujaran Kebencian Sedang', 'Ujaran Kebencian Berat']
        
        for i, class_name in enumerate(class_names):
            class_data = classification_report[str(i)]
            f1 = class_data['f1-score']
            precision = class_data['precision']
            recall = class_data['recall']
            support = class_data['support']
            
            print(f"\nClass {i} ({class_name}):")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  Support: {support}")
            
            # Identifikasi bottleneck
            if f1 < 0.85:  # Threshold untuk improvement
                self.bottlenecks[i] = {
                    'class_name': class_name,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'support': support,
                    'improvement_needed': 0.90 - f1
                }
        
        # Analisis individual model performance
        print("\n📈 INDIVIDUAL MODEL ANALYSIS:")
        models = self.current_results['models']
        
        best_individual = max(models.items(), key=lambda x: x[1]['f1_macro'])
        worst_individual = min(models.items(), key=lambda x: x[1]['f1_macro'])
        
        print(f"Best Individual Model: {best_individual[0]} ({best_individual[1]['f1_macro']:.4f})")
        print(f"Worst Individual Model: {worst_individual[0]} ({worst_individual[1]['f1_macro']:.4f})")
        print(f"Individual Model Range: {best_individual[1]['f1_macro'] - worst_individual[1]['f1_macro']:.4f}")
        
        # Identifikasi strategi optimasi
        self.identify_optimization_strategies()
        
    def identify_optimization_strategies(self):
        """Identifikasi strategi optimasi berdasarkan analisis bottleneck"""
        print("\n🎯 OPTIMIZATION STRATEGIES IDENTIFIED:")
        print("=" * 50)
        
        # Strategy 1: Class-specific optimization
        if self.bottlenecks:
            worst_class = min(self.bottlenecks.items(), key=lambda x: x[1]['f1_score'])
            self.optimization_strategies.append({
                'name': 'Class-Specific Optimization',
                'priority': 'HIGH',
                'description': f"Focus on improving Class {worst_class[0]} ({worst_class[1]['class_name']}) with F1={worst_class[1]['f1_score']:.4f}",
                'techniques': [
                    'Focal Loss with increased gamma for hard examples',
                    'Class-specific data augmentation',
                    'Adjusted class weights',
                    'Hard negative mining'
                ]
            })
        
        # Strategy 2: Advanced ensemble techniques
        current_ensemble_f1 = self.current_results['best_f1_macro']
        if current_ensemble_f1 < 0.90:
            self.optimization_strategies.append({
                'name': 'Advanced Ensemble Techniques',
                'priority': 'HIGH',
                'description': f"Improve ensemble from {current_ensemble_f1:.4f} to 0.90+ F1-macro",
                'techniques': [
                    'Neural Network Meta-Learner (MLP)',
                    'Stacking with multiple meta-learners',
                    'Confidence-weighted voting',
                    'Dynamic ensemble selection'
                ]
            })
        
        # Strategy 3: Model diversity enhancement
        models = self.current_results['models']
        f1_scores = [model['f1_macro'] for model in models.values()]
        diversity = np.std(f1_scores)
        
        if diversity < 0.02:  # Low diversity threshold
            self.optimization_strategies.append({
                'name': 'Model Diversity Enhancement',
                'priority': 'MEDIUM',
                'description': f"Increase model diversity (current std: {diversity:.4f})",
                'techniques': [
                    'Add different architecture types',
                    'Vary training strategies significantly',
                    'Use different preprocessing approaches',
                    'Implement bagging with different subsets'
                ]
            })
        
        # Strategy 4: Data quality enhancement
        self.optimization_strategies.append({
            'name': 'Data Quality Enhancement',
            'priority': 'MEDIUM',
            'description': "Improve data quality and preprocessing",
            'techniques': [
                'Advanced text cleaning and normalization',
                'Noise detection and removal',
                'Selective high-quality augmentation',
                'Cross-validation data leakage check'
            ]
        })
        
        # Print strategies
        for i, strategy in enumerate(self.optimization_strategies, 1):
            print(f"\n{i}. {strategy['name']} ({strategy['priority']} PRIORITY)")
            print(f"   Description: {strategy['description']}")
            print(f"   Techniques:")
            for technique in strategy['techniques']:
                print(f"     - {technique}")
    
    def implement_class_specific_optimization(self):
        """Implementasi optimasi khusus per kelas"""
        print("\n🎯 IMPLEMENTING CLASS-SPECIFIC OPTIMIZATION")
        print("=" * 50)
        
        if not self.bottlenecks:
            print("✅ No significant class-specific bottlenecks found")
            return
        
        # Fokus pada kelas dengan performa terburuk
        worst_class = min(self.bottlenecks.items(), key=lambda x: x[1]['f1_score'])
        class_id, class_info = worst_class
        
        print(f"🎯 Targeting Class {class_id}: {class_info['class_name']}")
        print(f"   Current F1: {class_info['f1_score']:.4f}")
        print(f"   Improvement needed: {class_info['improvement_needed']:.4f}")
        
        # Strategi optimasi khusus
        optimization_config = {
            'focal_loss_gamma': 3.0,  # Increased for harder examples
            'class_weights': self.calculate_optimized_class_weights(),
            'augmentation_factor': {
                str(class_id): 2.0,  # Double augmentation for worst class
                'others': 1.0
            },
            'hard_negative_mining': True,
            'class_specific_threshold': True
        }
        
        print(f"\n📋 Optimization Configuration:")
        for key, value in optimization_config.items():
            print(f"   {key}: {value}")
        
        return optimization_config
    
    def calculate_optimized_class_weights(self):
        """Hitung class weights yang dioptimalkan"""
        if not self.current_results:
            return None
        
        # Ambil support untuk setiap kelas
        classification_report = self.current_results['advanced_meta_ensemble_results']['classification_report']
        supports = []
        f1_scores = []
        
        for i in range(4):  # 4 classes
            supports.append(classification_report[str(i)]['support'])
            f1_scores.append(classification_report[str(i)]['f1-score'])
        
        supports = np.array(supports)
        f1_scores = np.array(f1_scores)
        
        # Hitung weights berdasarkan inverse support dan inverse f1
        inverse_support_weights = 1.0 / (supports / np.sum(supports))
        inverse_f1_weights = 1.0 / f1_scores
        
        # Kombinasi weights
        combined_weights = 0.7 * inverse_support_weights + 0.3 * inverse_f1_weights
        
        # Normalisasi
        normalized_weights = combined_weights / np.sum(combined_weights) * len(combined_weights)
        
        return {i: weight for i, weight in enumerate(normalized_weights)}
    
    def implement_advanced_ensemble(self):
        """Implementasi ensemble techniques lanjutan"""
        print("\n🚀 IMPLEMENTING ADVANCED ENSEMBLE TECHNIQUES")
        print("=" * 50)
        
        # Konfigurasi meta-learners yang akan digunakan
        meta_learners = {
            'neural_network': {
                'model': MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=1000,
                    random_state=42
                ),
                'description': 'Neural Network Meta-Learner'
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    random_state=42
                ),
                'description': 'Gradient Boosting Meta-Learner'
            },
            'logistic_regression': {
                'model': LogisticRegression(
                    C=1.0,
                    solver='liblinear',
                    multi_class='ovr',
                    random_state=42
                ),
                'description': 'Logistic Regression Meta-Learner'
            }
        }
        
        print("📋 Meta-Learners Configuration:")
        for name, config in meta_learners.items():
            print(f"   - {config['description']}")
        
        # Konfigurasi ensemble strategies
        ensemble_strategies = {
            'stacking_ensemble': {
                'description': 'Multi-level stacking with cross-validation',
                'meta_learners': list(meta_learners.keys()),
                'cv_folds': 5
            },
            'confidence_weighted_voting': {
                'description': 'Voting weighted by prediction confidence',
                'weight_method': 'entropy_based'
            },
            'dynamic_selection': {
                'description': 'Dynamic ensemble selection based on local accuracy',
                'selection_method': 'local_class_accuracy'
            }
        }
        
        print("\n🎯 Ensemble Strategies:")
        for name, config in ensemble_strategies.items():
            print(f"   - {config['description']}")
        
        return {
            'meta_learners': meta_learners,
            'ensemble_strategies': ensemble_strategies
        }
    
    def generate_optimization_script(self):
        """Generate script untuk implementasi optimasi"""
        print("\n📝 GENERATING OPTIMIZATION IMPLEMENTATION SCRIPT")
        print("=" * 50)
        
        script_content = f'''
#!/usr/bin/env python3
"""
Ultimate 90% F1-Macro Optimization Implementation
Generated automatically based on bottleneck analysis

Current: {self.current_results['best_f1_macro']:.4f} F1-macro
Target: 0.9000 F1-macro
Gap: {self.current_results['gap_to_target']:.4f}
"""

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime

class UltimateOptimizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {{}}
        
    def load_data(self):
        """Load and preprocess data with enhanced quality"""
        # Implementation here
        pass
    
    def train_optimized_models(self):
        """Train models with class-specific optimizations"""
        # Implementation here
        pass
    
    def create_ultimate_ensemble(self):
        """Create ultimate ensemble with multiple meta-learners"""
        # Implementation here
        pass
    
    def run_optimization(self):
        """Run complete optimization pipeline"""
        print("🚀 Starting Ultimate 90% Optimization...")
        
        # Load and preprocess data
        self.load_data()
        
        # Train optimized individual models
        self.train_optimized_models()
        
        # Create ultimate ensemble
        self.create_ultimate_ensemble()
        
        # Evaluate and save results
        self.evaluate_and_save()
        
if __name__ == "__main__":
    optimizer = UltimateOptimizer()
    optimizer.run_optimization()
'''
        
        with open('ultimate_90_percent_optimization.py', 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print("✅ Generated: ultimate_90_percent_optimization.py")
        
    def run_analysis(self):
        """Run complete analysis and generate optimization plan"""
        print("🎯 FINAL 90% OPTIMIZATION ANALYSIS")
        print("=" * 60)
        
        # Load current results
        if not self.load_current_results():
            return
        
        # Analyze bottlenecks
        self.analyze_performance_bottlenecks()
        
        # Implement optimizations
        class_optimization = self.implement_class_specific_optimization()
        ensemble_config = self.implement_advanced_ensemble()
        
        # Generate implementation script
        self.generate_optimization_script()
        
        # Summary
        print("\n📊 OPTIMIZATION SUMMARY")
        print("=" * 50)
        print(f"Current F1-Macro: {self.current_results['best_f1_macro']:.4f}")
        print(f"Target F1-Macro: 0.9000")
        print(f"Gap to close: {self.current_results['gap_to_target']:.4f}")
        print(f"Bottlenecks identified: {len(self.bottlenecks)}")
        print(f"Optimization strategies: {len(self.optimization_strategies)}")
        
        print("\n🎯 NEXT STEPS:")
        print("1. Implement class-specific optimizations")
        print("2. Deploy advanced ensemble techniques")
        print("3. Enhance data quality and preprocessing")
        print("4. Run ultimate optimization experiment")
        print("5. Achieve 90% F1-Macro target! 🚀")
        
        return {
            'current_f1': self.current_results['best_f1_macro'],
            'gap_to_target': self.current_results['gap_to_target'],
            'bottlenecks': self.bottlenecks,
            'strategies': self.optimization_strategies,
            'class_optimization': class_optimization,
            'ensemble_config': ensemble_config
        }

if __name__ == "__main__":
    optimizer = Final90PercentOptimizer()
    results = optimizer.run_analysis()
    
    # Save analysis results
    with open('results/final_90_percent_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Analysis complete! Results saved to results/final_90_percent_analysis.json")