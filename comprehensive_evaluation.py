#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite
Analyze and compare all experimental results for 90%+ accuracy achievement
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from scipy import stats
import os
from datetime import datetime
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/comprehensive_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.class_names = [
            'Bukan Ujaran Kebencian',
            'Ujaran Kebencian - Ringan', 
            'Ujaran Kebencian - Sedang',
            'Ujaran Kebencian - Berat'
        ]
        self.target_accuracy = 0.90
        self.target_f1 = 0.90
        
    def load_all_results(self):
        """Load all experimental results"""
        logger.info("Loading all experimental results")
        
        results = {}
        
        # Define expected result files
        result_files = {
            'baseline': 'baseline_results.json',
            'data_augmentation': 'data_augmentation_results.json',
            'ensemble_advanced': 'ensemble_advanced_results.json',
            'multi_architecture': 'multi_architecture_ensemble_results.json',
            'hyperparameter_tuning': 'advanced_hyperparameter_tuning_results.json',
            'augmented_training': 'augmented_training_results.json'  # Expected from current training
        }
        
        for experiment, filename in result_files.items():
            filepath = self.results_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        results[experiment] = json.load(f)
                    logger.info(f"Loaded {experiment} results")
                except Exception as e:
                    logger.warning(f"Failed to load {experiment}: {e}")
            else:
                logger.warning(f"Result file not found: {filepath}")
        
        return results
    
    def extract_performance_metrics(self, results):
        """Extract standardized performance metrics from all experiments"""
        logger.info("Extracting performance metrics")
        
        metrics_summary = {}
        
        for experiment, data in results.items():
            try:
                if experiment == 'baseline':
                    # Extract from baseline format
                    metrics_summary[experiment] = {
                        'accuracy': data.get('test_accuracy', 0),
                        'f1_macro': data.get('test_f1_macro', 0),
                        'f1_weighted': data.get('test_f1_weighted', 0),
                        'precision_macro': data.get('test_precision_macro', 0),
                        'recall_macro': data.get('test_recall_macro', 0)
                    }
                
                elif experiment == 'data_augmentation':
                    # Extract augmentation statistics
                    metrics_summary[experiment] = {
                        'original_samples': data.get('original_dataset_size', 0),
                        'augmented_samples': data.get('final_dataset_size', 0),
                        'augmentation_ratio': data.get('augmentation_ratio', 0),
                        'techniques_used': len(data.get('augmentation_techniques', []))
                    }
                
                elif experiment == 'ensemble_advanced':
                    # Extract ensemble results
                    best_method = data.get('best_ensemble_method', {})
                    metrics_summary[experiment] = {
                        'accuracy': best_method.get('test_accuracy', 0),
                        'f1_macro': best_method.get('test_f1_macro', 0),
                        'f1_weighted': best_method.get('test_f1_weighted', 0),
                        'method': data.get('best_method_name', 'unknown'),
                        'improvement_over_single': data.get('improvement_over_single_model', {})
                    }
                
                elif experiment == 'multi_architecture':
                    # Extract multi-architecture ensemble results
                    optimized = data.get('ensemble_optimized_weights', {})
                    metrics_summary[experiment] = {
                        'accuracy': optimized.get('accuracy', 0),
                        'f1_macro': optimized.get('f1_macro', 0),
                        'f1_weighted': optimized.get('f1_weighted', 0),
                        'models_used': data.get('models_used', []),
                        'optimal_weights': data.get('optimal_weights', {})
                    }
                
                elif experiment == 'hyperparameter_tuning':
                    # Extract hyperparameter optimization results
                    test_results = data.get('test_results', {})
                    metrics_summary[experiment] = {
                        'accuracy': test_results.get('eval_accuracy', 0),
                        'f1_macro': test_results.get('eval_f1_macro', 0),
                        'f1_weighted': test_results.get('eval_f1_weighted', 0),
                        'optimization_score': data.get('optimization_score', 0),
                        'best_params': data.get('best_hyperparameters', {})
                    }
                
                elif experiment == 'augmented_training':
                    # Extract augmented training results (when available)
                    metrics_summary[experiment] = {
                        'accuracy': data.get('test_accuracy', 0),
                        'f1_macro': data.get('test_f1_macro', 0),
                        'f1_weighted': data.get('test_f1_weighted', 0),
                        'training_epochs': data.get('epochs_completed', 0),
                        'best_epoch': data.get('best_epoch', 0)
                    }
                
            except Exception as e:
                logger.error(f"Error extracting metrics for {experiment}: {e}")
                metrics_summary[experiment] = {'error': str(e)}
        
        return metrics_summary
    
    def calculate_progress_towards_target(self, metrics_summary):
        """Calculate progress towards 90% target"""
        logger.info("Calculating progress towards 90% target")
        
        progress_analysis = {
            'target_accuracy': self.target_accuracy,
            'target_f1_macro': self.target_f1,
            'experiments': {}
        }
        
        for experiment, metrics in metrics_summary.items():
            if 'accuracy' in metrics and 'f1_macro' in metrics:
                accuracy = metrics['accuracy']
                f1_macro = metrics['f1_macro']
                
                progress_analysis['experiments'][experiment] = {
                    'current_accuracy': accuracy,
                    'current_f1_macro': f1_macro,
                    'accuracy_gap': max(0, self.target_accuracy - accuracy),
                    'f1_gap': max(0, self.target_f1 - f1_macro),
                    'accuracy_progress': min(100, (accuracy / self.target_accuracy) * 100),
                    'f1_progress': min(100, (f1_macro / self.target_f1) * 100),
                    'target_achieved': accuracy >= self.target_accuracy and f1_macro >= self.target_f1
                }
        
        return progress_analysis
    
    def generate_comparison_table(self, metrics_summary):
        """Generate comparison table of all experiments"""
        logger.info("Generating comparison table")
        
        comparison_data = []
        
        for experiment, metrics in metrics_summary.items():
            if 'accuracy' in metrics and 'f1_macro' in metrics:
                row = {
                    'Experiment': experiment.replace('_', ' ').title(),
                    'Accuracy': f"{metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)",
                    'F1-Macro': f"{metrics['f1_macro']:.4f} ({metrics['f1_macro']*100:.2f}%)",
                    'F1-Weighted': f"{metrics.get('f1_weighted', 0):.4f} ({metrics.get('f1_weighted', 0)*100:.2f}%)",
                    'Target Achieved': '✅' if (metrics['accuracy'] >= self.target_accuracy and 
                                              metrics['f1_macro'] >= self.target_f1) else '❌'
                }
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def analyze_improvement_trajectory(self, metrics_summary):
        """Analyze improvement trajectory across experiments"""
        logger.info("Analyzing improvement trajectory")
        
        # Define logical experiment order
        experiment_order = [
            'baseline',
            'data_augmentation', 
            'augmented_training',
            'ensemble_advanced',
            'multi_architecture',
            'hyperparameter_tuning'
        ]
        
        trajectory = {
            'experiments': [],
            'accuracy_values': [],
            'f1_macro_values': [],
            'cumulative_improvement': []
        }
        
        baseline_accuracy = 0
        baseline_f1 = 0
        
        for i, experiment in enumerate(experiment_order):
            if experiment in metrics_summary:
                metrics = metrics_summary[experiment]
                if 'accuracy' in metrics and 'f1_macro' in metrics:
                    accuracy = metrics['accuracy']
                    f1_macro = metrics['f1_macro']
                    
                    if experiment == 'baseline':
                        baseline_accuracy = accuracy
                        baseline_f1 = f1_macro
                        improvement = 0
                    else:
                        improvement = ((accuracy + f1_macro) / 2) - ((baseline_accuracy + baseline_f1) / 2)
                    
                    trajectory['experiments'].append(experiment.replace('_', ' ').title())
                    trajectory['accuracy_values'].append(accuracy)
                    trajectory['f1_macro_values'].append(f1_macro)
                    trajectory['cumulative_improvement'].append(improvement)
        
        return trajectory
    
    def generate_statistical_analysis(self, metrics_summary):
        """Generate statistical analysis of results"""
        logger.info("Generating statistical analysis")
        
        # Extract accuracy and F1 values
        accuracy_values = []
        f1_values = []
        experiment_names = []
        
        for experiment, metrics in metrics_summary.items():
            if 'accuracy' in metrics and 'f1_macro' in metrics:
                accuracy_values.append(metrics['accuracy'])
                f1_values.append(metrics['f1_macro'])
                experiment_names.append(experiment)
        
        if len(accuracy_values) < 2:
            return {'error': 'Insufficient data for statistical analysis'}
        
        analysis = {
            'accuracy_stats': {
                'mean': np.mean(accuracy_values),
                'std': np.std(accuracy_values),
                'min': np.min(accuracy_values),
                'max': np.max(accuracy_values),
                'range': np.max(accuracy_values) - np.min(accuracy_values)
            },
            'f1_stats': {
                'mean': np.mean(f1_values),
                'std': np.std(f1_values),
                'min': np.min(f1_values),
                'max': np.max(f1_values),
                'range': np.max(f1_values) - np.min(f1_values)
            },
            'correlation': {
                'accuracy_f1_correlation': np.corrcoef(accuracy_values, f1_values)[0, 1]
            },
            'best_experiment': {
                'by_accuracy': experiment_names[np.argmax(accuracy_values)],
                'by_f1': experiment_names[np.argmax(f1_values)],
                'by_combined': experiment_names[np.argmax(np.array(accuracy_values) + np.array(f1_values))]
            }
        }
        
        return analysis
    
    def create_visualizations(self, metrics_summary, trajectory, output_dir='visualizations'):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Performance Comparison Bar Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        experiments = []
        accuracies = []
        f1_scores = []
        
        for exp, metrics in metrics_summary.items():
            if 'accuracy' in metrics and 'f1_macro' in metrics:
                experiments.append(exp.replace('_', ' ').title())
                accuracies.append(metrics['accuracy'] * 100)
                f1_scores.append(metrics['f1_macro'] * 100)
        
        x_pos = np.arange(len(experiments))
        
        ax1.bar(x_pos, accuracies, alpha=0.8, color='skyblue', edgecolor='navy')
        ax1.axhline(y=90, color='red', linestyle='--', label='90% Target')
        ax1.set_xlabel('Experiments')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy Comparison Across Experiments')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(experiments, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(x_pos, f1_scores, alpha=0.8, color='lightcoral', edgecolor='darkred')
        ax2.axhline(y=90, color='red', linestyle='--', label='90% Target')
        ax2.set_xlabel('Experiments')
        ax2.set_ylabel('F1-Macro (%)')
        ax2.set_title('F1-Macro Comparison Across Experiments')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(experiments, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Improvement Trajectory
        if trajectory['experiments']:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x_pos = np.arange(len(trajectory['experiments']))
            
            ax.plot(x_pos, np.array(trajectory['accuracy_values']) * 100, 
                   marker='o', linewidth=2, markersize=8, label='Accuracy', color='blue')
            ax.plot(x_pos, np.array(trajectory['f1_macro_values']) * 100, 
                   marker='s', linewidth=2, markersize=8, label='F1-Macro', color='red')
            
            ax.axhline(y=90, color='green', linestyle='--', linewidth=2, label='90% Target')
            
            ax.set_xlabel('Experiment Progression')
            ax.set_ylabel('Performance (%)')
            ax.set_title('Performance Improvement Trajectory')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(trajectory['experiments'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value annotations
            for i, (acc, f1) in enumerate(zip(trajectory['accuracy_values'], trajectory['f1_macro_values'])):
                ax.annotate(f'{acc*100:.1f}%', (i, acc*100), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
                ax.annotate(f'{f1*100:.1f}%', (i, f1*100), textcoords="offset points", 
                           xytext=(0,-15), ha='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/improvement_trajectory.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Target Achievement Progress
        fig, ax = plt.subplots(figsize=(10, 6))
        
        target_progress = []
        for exp, metrics in metrics_summary.items():
            if 'accuracy' in metrics and 'f1_macro' in metrics:
                combined_score = (metrics['accuracy'] + metrics['f1_macro']) / 2
                progress = min(100, (combined_score / 0.90) * 100)
                target_progress.append(progress)
        
        colors = ['green' if p >= 100 else 'orange' if p >= 85 else 'red' for p in target_progress]
        
        bars = ax.bar(experiments, target_progress, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Target Achieved')
        
        ax.set_xlabel('Experiments')
        ax.set_ylabel('Progress Towards 90% Target (%)')
        ax.set_title('Target Achievement Progress')
        ax.set_xticklabels(experiments, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value annotations
        for bar, progress in zip(bars, target_progress):
            height = bar.get_height()
            ax.annotate(f'{progress:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/target_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}/")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        logger.info("Generating comprehensive evaluation report")
        
        # Load all results
        results = self.load_all_results()
        
        if not results:
            logger.error("No results found to evaluate")
            return None
        
        # Extract metrics
        metrics_summary = self.extract_performance_metrics(results)
        
        # Calculate progress
        progress_analysis = self.calculate_progress_towards_target(metrics_summary)
        
        # Generate comparison table
        comparison_table = self.generate_comparison_table(metrics_summary)
        
        # Analyze trajectory
        trajectory = self.analyze_improvement_trajectory(metrics_summary)
        
        # Statistical analysis
        statistical_analysis = self.generate_statistical_analysis(metrics_summary)
        
        # Create visualizations
        self.create_visualizations(metrics_summary, trajectory)
        
        # Compile comprehensive report
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'experiments_evaluated': list(results.keys()),
            'metrics_summary': metrics_summary,
            'progress_analysis': progress_analysis,
            'comparison_table': comparison_table.to_dict('records'),
            'improvement_trajectory': trajectory,
            'statistical_analysis': statistical_analysis,
            'target_achievement_status': self._assess_target_achievement(progress_analysis),
            'recommendations': self._generate_recommendations(metrics_summary, progress_analysis)
        }
        
        # Save report
        os.makedirs('reports', exist_ok=True)
        with open('reports/comprehensive_evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _assess_target_achievement(self, progress_analysis):
        """Assess overall target achievement status"""
        achieved_experiments = []
        best_accuracy = 0
        best_f1 = 0
        
        for exp, data in progress_analysis['experiments'].items():
            if data['target_achieved']:
                achieved_experiments.append(exp)
            
            best_accuracy = max(best_accuracy, data['current_accuracy'])
            best_f1 = max(best_f1, data['current_f1_macro'])
        
        return {
            'target_achieved': len(achieved_experiments) > 0,
            'achieving_experiments': achieved_experiments,
            'best_overall_accuracy': best_accuracy,
            'best_overall_f1': best_f1,
            'remaining_accuracy_gap': max(0, 0.90 - best_accuracy),
            'remaining_f1_gap': max(0, 0.90 - best_f1)
        }
    
    def _generate_recommendations(self, metrics_summary, progress_analysis):
        """Generate recommendations for achieving 90% target"""
        recommendations = []
        
        # Find best performing experiment
        best_accuracy = 0
        best_f1 = 0
        best_experiment = None
        
        for exp, metrics in metrics_summary.items():
            if 'accuracy' in metrics and 'f1_macro' in metrics:
                combined_score = metrics['accuracy'] + metrics['f1_macro']
                if combined_score > best_accuracy + best_f1:
                    best_accuracy = metrics['accuracy']
                    best_f1 = metrics['f1_macro']
                    best_experiment = exp
        
        if best_accuracy < 0.90 or best_f1 < 0.90:
            accuracy_gap = 0.90 - best_accuracy
            f1_gap = 0.90 - best_f1
            
            recommendations.extend([
                f"Current best performance: {best_experiment} with {best_accuracy:.3f} accuracy and {best_f1:.3f} F1-macro",
                f"Remaining gap: {accuracy_gap:.3f} accuracy, {f1_gap:.3f} F1-macro",
                "Recommended next steps:"
            ])
            
            if accuracy_gap > 0.02:
                recommendations.append("- Implement advanced ensemble methods with more diverse models")
                recommendations.append("- Apply more aggressive data augmentation techniques")
            
            if f1_gap > 0.02:
                recommendations.append("- Focus on class-specific improvements using focal loss")
                recommendations.append("- Implement cost-sensitive learning approaches")
            
            recommendations.extend([
                "- Conduct extensive hyperparameter optimization with larger search space",
                "- Experiment with different pre-trained models (multilingual BERT, XLM-R)",
                "- Implement cross-validation ensemble for more robust predictions",
                "- Consider external data integration from related Indonesian datasets"
            ])
        else:
            recommendations.append(f"🎉 Target achieved with {best_experiment}!")
            recommendations.append("Focus on validation and reproducibility of results")
        
        return recommendations
    
    def print_summary_report(self, report):
        """Print formatted summary report"""
        print("\n" + "="*100)
        print("🔍 COMPREHENSIVE EVALUATION REPORT")
        print("="*100)
        
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"   Timestamp: {report['evaluation_timestamp']}")
        print(f"   Experiments Evaluated: {len(report['experiments_evaluated'])}")
        print(f"   Experiments: {', '.join(report['experiments_evaluated'])}")
        
        # Target achievement status
        status = report['target_achievement_status']
        print(f"\n🎯 TARGET ACHIEVEMENT STATUS:")
        if status['target_achieved']:
            print(f"   ✅ TARGET ACHIEVED!")
            print(f"   🏆 Achieving experiments: {', '.join(status['achieving_experiments'])}")
        else:
            print(f"   ⚠️ Target not yet achieved")
            print(f"   📈 Best accuracy: {status['best_overall_accuracy']:.4f} ({status['best_overall_accuracy']*100:.2f}%)")
            print(f"   📈 Best F1-macro: {status['best_overall_f1']:.4f} ({status['best_overall_f1']*100:.2f}%)")
            print(f"   📊 Remaining gaps: {status['remaining_accuracy_gap']:.3f} accuracy, {status['remaining_f1_gap']:.3f} F1")
        
        # Performance comparison
        print(f"\n📈 PERFORMANCE COMPARISON:")
        comparison_df = pd.DataFrame(report['comparison_table'])
        print(comparison_df.to_string(index=False))
        
        # Statistical insights
        stats = report['statistical_analysis']
        if 'error' not in stats:
            print(f"\n📊 STATISTICAL INSIGHTS:")
            print(f"   Accuracy - Mean: {stats['accuracy_stats']['mean']:.4f}, Std: {stats['accuracy_stats']['std']:.4f}")
            print(f"   F1-Macro - Mean: {stats['f1_stats']['mean']:.4f}, Std: {stats['f1_stats']['std']:.4f}")
            print(f"   Best by accuracy: {stats['best_experiment']['by_accuracy']}")
            print(f"   Best by F1: {stats['best_experiment']['by_f1']}")
            print(f"   Best combined: {stats['best_experiment']['by_combined']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n📁 OUTPUTS GENERATED:")
        print(f"   📊 Visualizations: visualizations/")
        print(f"   📋 Full Report: reports/comprehensive_evaluation_report.json")
        print(f"   📈 Performance Charts: visualizations/performance_comparison.png")
        print(f"   📉 Trajectory Plot: visualizations/improvement_trajectory.png")
        print(f"   🎯 Progress Chart: visualizations/target_progress.png")
        
        print("="*100)

def main():
    logger.info("Starting comprehensive evaluation")
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Generate comprehensive report
    report = evaluator.generate_comprehensive_report()
    
    if report:
        # Print summary
        evaluator.print_summary_report(report)
        
        logger.info("Comprehensive evaluation completed successfully")
        return report
    else:
        logger.error("Failed to generate evaluation report")
        return None

if __name__ == "__main__":
    main()