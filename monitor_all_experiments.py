#!/usr/bin/env python3
"""
Monitor All Running Experiments - Progress Towards 90% Target
Tracks: Multi-architecture ensemble, Hyperparameter optimization, 
Advanced training techniques, Cross-validation framework
"""

import os
import json
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_file_exists(filepath):
    """Check if file exists and return its modification time"""
    if os.path.exists(filepath):
        return os.path.getmtime(filepath)
    return None

def read_json_file(filepath):
    """Safely read JSON file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Error reading {filepath}: {e}")
    return None

def monitor_multi_architecture_ensemble():
    """Monitor multi-architecture ensemble results"""
    results_dir = 'results'
    ensemble_files = [
        'multi_architecture_ensemble_results.json',
        'ensemble_results.json',
        'advanced_ensemble_results.json'
    ]
    
    logger.info("\n=== Multi-Architecture Ensemble Status ===")
    
    for filename in ensemble_files:
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            data = read_json_file(filepath)
            if data:
                logger.info(f"Found results in {filename}:")
                if 'test_accuracy' in data:
                    logger.info(f"  Test Accuracy: {data['test_accuracy']:.4f}")
                if 'test_f1_macro' in data:
                    logger.info(f"  Test F1-Macro: {data['test_f1_macro']:.4f}")
                if 'validation_accuracy' in data:
                    logger.info(f"  Validation Accuracy: {data['validation_accuracy']:.4f}")
                if 'timestamp' in data:
                    logger.info(f"  Last Updated: {data['timestamp']}")
                
                # Check if 90% target achieved
                test_acc = data.get('test_accuracy', 0)
                if test_acc >= 0.90:
                    logger.info(f"  🎯 TARGET ACHIEVED! Test accuracy: {test_acc:.4f}")
                return data
    
    logger.info("  Status: Running (no results yet)")
    return None

def monitor_hyperparameter_optimization():
    """Monitor hyperparameter optimization progress"""
    logger.info("\n=== Hyperparameter Optimization Status ===")
    
    # Check Optuna database
    optuna_db = 'optuna_study.db'
    if os.path.exists(optuna_db):
        mod_time = datetime.fromtimestamp(os.path.getmtime(optuna_db))
        logger.info(f"  Optuna DB last modified: {mod_time}")
    
    # Check results files
    results_files = [
        'results/hyperparameter_optimization_results.json',
        'results/optuna_best_params.json',
        'HYPERPARAMETER_TUNING_RESULTS.md'
    ]
    
    best_result = None
    for filepath in results_files:
        if os.path.exists(filepath):
            if filepath.endswith('.json'):
                data = read_json_file(filepath)
                if data:
                    logger.info(f"Found results in {os.path.basename(filepath)}:")
                    if 'best_accuracy' in data:
                        logger.info(f"  Best Accuracy: {data['best_accuracy']:.4f}")
                        if data['best_accuracy'] >= 0.90:
                            logger.info(f"  🎯 TARGET ACHIEVED! Best accuracy: {data['best_accuracy']:.4f}")
                        best_result = data
                    if 'best_params' in data:
                        logger.info(f"  Best Parameters: {data['best_params']}")
            else:
                mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                logger.info(f"  {os.path.basename(filepath)} last modified: {mod_time}")
    
    if not best_result:
        logger.info("  Status: Running (optimization in progress)")
    
    return best_result

def monitor_advanced_training_techniques():
    """Monitor advanced training techniques results"""
    logger.info("\n=== Advanced Training Techniques Status ===")
    
    results_files = [
        'results/advanced_training_results.json',
        'results/training_techniques_comparison.json'
    ]
    
    best_result = None
    for filepath in results_files:
        if os.path.exists(filepath):
            data = read_json_file(filepath)
            if data:
                logger.info(f"Found results in {os.path.basename(filepath)}:")
                
                # Handle different result structures
                if isinstance(data, dict):
                    if 'results' in data:
                        # Multiple technique results
                        for technique, result in data['results'].items():
                            if 'test_accuracy' in result:
                                acc = result['test_accuracy']
                                logger.info(f"  {technique}: {acc:.4f}")
                                if acc >= 0.90:
                                    logger.info(f"  🎯 TARGET ACHIEVED! {technique}: {acc:.4f}")
                                    best_result = result
                    elif 'test_accuracy' in data:
                        acc = data['test_accuracy']
                        logger.info(f"  Test Accuracy: {acc:.4f}")
                        if acc >= 0.90:
                            logger.info(f"  🎯 TARGET ACHIEVED! Test accuracy: {acc:.4f}")
                            best_result = data
    
    if not best_result:
        logger.info("  Status: Running (training in progress)")
    
    return best_result

def monitor_cross_validation():
    """Monitor cross-validation framework results"""
    logger.info("\n=== Cross-Validation Framework Status ===")
    
    cv_results_file = 'results/cross_validation_summary.json'
    if os.path.exists(cv_results_file):
        data = read_json_file(cv_results_file)
        if data:
            logger.info("Cross-validation results found:")
            
            best_model = None
            best_accuracy = 0
            
            for model_name, results in data.items():
                if isinstance(results, dict) and 'mean_accuracy' in results:
                    mean_acc = results['mean_accuracy']
                    std_acc = results.get('std_accuracy', 0)
                    logger.info(f"  {model_name}: {mean_acc:.4f} ± {std_acc:.4f}")
                    
                    if mean_acc > best_accuracy:
                        best_accuracy = mean_acc
                        best_model = model_name
                    
                    if mean_acc >= 0.90:
                        logger.info(f"  🎯 TARGET ACHIEVED! {model_name}: {mean_acc:.4f}")
            
            if best_model:
                logger.info(f"  Best Model: {best_model} ({best_accuracy:.4f})")
                return data[best_model]
    
    logger.info("  Status: Running (cross-validation in progress)")
    return None

def check_target_achievement():
    """Check if 90% target has been achieved by any experiment"""
    logger.info("\n" + "="*60)
    logger.info("TARGET 90% ACCURACY ACHIEVEMENT CHECK")
    logger.info("="*60)
    
    achievements = []
    
    # Check all experiments
    ensemble_result = monitor_multi_architecture_ensemble()
    if ensemble_result and ensemble_result.get('test_accuracy', 0) >= 0.90:
        achievements.append(('Multi-Architecture Ensemble', ensemble_result['test_accuracy']))
    
    hyperparam_result = monitor_hyperparameter_optimization()
    if hyperparam_result and hyperparam_result.get('best_accuracy', 0) >= 0.90:
        achievements.append(('Hyperparameter Optimization', hyperparam_result['best_accuracy']))
    
    training_result = monitor_advanced_training_techniques()
    if training_result and training_result.get('test_accuracy', 0) >= 0.90:
        achievements.append(('Advanced Training Techniques', training_result['test_accuracy']))
    
    cv_result = monitor_cross_validation()
    if cv_result and cv_result.get('mean_accuracy', 0) >= 0.90:
        achievements.append(('Cross-Validation', cv_result['mean_accuracy']))
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    if achievements:
        logger.info(f"🎯 TARGET ACHIEVED! {len(achievements)} experiment(s) reached 90%:")
        for name, accuracy in achievements:
            logger.info(f"  ✅ {name}: {accuracy:.4f}")
    else:
        logger.info("⏳ Target not yet achieved. Experiments still running...")
        
        # Show current best results
        best_results = []
        if ensemble_result and 'test_accuracy' in ensemble_result:
            best_results.append(('Ensemble', ensemble_result['test_accuracy']))
        if hyperparam_result and 'best_accuracy' in hyperparam_result:
            best_results.append(('Hyperparameter', hyperparam_result['best_accuracy']))
        if training_result and 'test_accuracy' in training_result:
            best_results.append(('Training', training_result['test_accuracy']))
        if cv_result and 'mean_accuracy' in cv_result:
            best_results.append(('Cross-Validation', cv_result['mean_accuracy']))
        
        if best_results:
            best_results.sort(key=lambda x: x[1], reverse=True)
            logger.info("\nCurrent best results:")
            for name, accuracy in best_results:
                logger.info(f"  📊 {name}: {accuracy:.4f} ({(accuracy/0.90)*100:.1f}% of target)")
    
    return len(achievements) > 0

def main():
    """Main monitoring function"""
    logger.info("Starting comprehensive experiment monitoring...")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Check if target achieved
    target_achieved = check_target_achievement()
    
    # Save monitoring report
    report = {
        'timestamp': datetime.now().isoformat(),
        'target_achieved': target_achieved,
        'experiments_status': {
            'multi_architecture_ensemble': 'running',
            'hyperparameter_optimization': 'running', 
            'advanced_training_techniques': 'running',
            'cross_validation_framework': 'running'
        }
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/monitoring_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("\nMonitoring report saved to results/monitoring_report.json")
    
    if target_achieved:
        logger.info("\n🎉 CONGRATULATIONS! 90% accuracy target has been achieved!")
    else:
        logger.info("\n⏳ Continue monitoring. Target not yet achieved.")

if __name__ == "__main__":
    main()