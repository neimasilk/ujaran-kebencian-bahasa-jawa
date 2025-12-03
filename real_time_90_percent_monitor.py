#!/usr/bin/env python3
"""
Real-Time 90% Target Achievement Monitor
Continuously monitors all running experiments and alerts when 90% accuracy is achieved
"""

import os
import json
import time
from datetime import datetime
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/90_percent_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TargetAchievementMonitor:
    def __init__(self):
        self.target_accuracy = 0.90
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        self.achievement_file = self.results_dir / '90_percent_achievement.json'
        self.last_check_times = {}
        
    def check_file_updated(self, filepath):
        """Check if file has been updated since last check"""
        if not os.path.exists(filepath):
            return False
            
        current_mtime = os.path.getmtime(filepath)
        last_mtime = self.last_check_times.get(filepath, 0)
        
        if current_mtime > last_mtime:
            self.last_check_times[filepath] = current_mtime
            return True
        return False
    
    def read_json_safely(self, filepath):
        """Safely read JSON file with error handling"""
        try:
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Error reading {filepath}: {e}")
        return None
    
    def check_multi_architecture_ensemble(self):
        """Check multi-architecture ensemble results"""
        files_to_check = [
            'results/multi_architecture_ensemble_results.json',
            'results/ensemble_results.json',
            'results/advanced_ensemble_results.json'
        ]
        
        for filepath in files_to_check:
            if self.check_file_updated(filepath):
                data = self.read_json_safely(filepath)
                if data:
                    # Check various possible result structures
                    accuracy_keys = ['test_accuracy', 'accuracy', 'final_accuracy']
                    
                    for key in accuracy_keys:
                        if key in data and data[key] >= self.target_accuracy:
                            return {
                                'experiment': 'Multi-Architecture Ensemble',
                                'accuracy': data[key],
                                'file': filepath,
                                'timestamp': datetime.now().isoformat(),
                                'details': data
                            }
                    
                    # Check ensemble results
                    ensemble_keys = ['ensemble_equal_weights', 'ensemble_optimized_weights']
                    for ens_key in ensemble_keys:
                        if ens_key in data:
                            ens_data = data[ens_key]
                            if isinstance(ens_data, dict) and 'accuracy' in ens_data:
                                if ens_data['accuracy'] >= self.target_accuracy:
                                    return {
                                        'experiment': f'Multi-Architecture Ensemble ({ens_key})',
                                        'accuracy': ens_data['accuracy'],
                                        'file': filepath,
                                        'timestamp': datetime.now().isoformat(),
                                        'details': ens_data
                                    }
        return None
    
    def check_hyperparameter_optimization(self):
        """Check hyperparameter optimization results"""
        files_to_check = [
            'results/hyperparameter_optimization_results.json',
            'results/optuna_best_params.json',
            'results/best_hyperparameters.json'
        ]
        
        for filepath in files_to_check:
            if self.check_file_updated(filepath):
                data = self.read_json_safely(filepath)
                if data:
                    accuracy_keys = ['best_accuracy', 'test_accuracy', 'accuracy', 'best_test_accuracy']
                    
                    for key in accuracy_keys:
                        if key in data and data[key] >= self.target_accuracy:
                            return {
                                'experiment': 'Hyperparameter Optimization',
                                'accuracy': data[key],
                                'file': filepath,
                                'timestamp': datetime.now().isoformat(),
                                'details': data
                            }
        return None
    
    def check_advanced_training_techniques(self):
        """Check advanced training techniques results"""
        files_to_check = [
            'results/advanced_training_results.json',
            'results/training_techniques_comparison.json',
            'results/advanced_techniques_results.json'
        ]
        
        for filepath in files_to_check:
            if self.check_file_updated(filepath):
                data = self.read_json_safely(filepath)
                if data:
                    # Handle different result structures
                    if 'results' in data and isinstance(data['results'], dict):
                        # Multiple technique results
                        for technique, result in data['results'].items():
                            if isinstance(result, dict):
                                accuracy_keys = ['test_accuracy', 'accuracy', 'final_accuracy']
                                for key in accuracy_keys:
                                    if key in result and result[key] >= self.target_accuracy:
                                        return {
                                            'experiment': f'Advanced Training Techniques ({technique})',
                                            'accuracy': result[key],
                                            'file': filepath,
                                            'timestamp': datetime.now().isoformat(),
                                            'details': result
                                        }
                    else:
                        # Single result
                        accuracy_keys = ['test_accuracy', 'accuracy', 'final_accuracy']
                        for key in accuracy_keys:
                            if key in data and data[key] >= self.target_accuracy:
                                return {
                                    'experiment': 'Advanced Training Techniques',
                                    'accuracy': data[key],
                                    'file': filepath,
                                    'timestamp': datetime.now().isoformat(),
                                    'details': data
                                }
        return None
    
    def check_cross_validation(self):
        """Check cross-validation results"""
        filepath = 'results/cross_validation_summary.json'
        
        if self.check_file_updated(filepath):
            data = self.read_json_safely(filepath)
            if data:
                # Check each model's cross-validation results
                for model_name, results in data.items():
                    if isinstance(results, dict):
                        accuracy_keys = ['mean_accuracy', 'average_accuracy', 'best_fold_accuracy']
                        for key in accuracy_keys:
                            if key in results and results[key] >= self.target_accuracy:
                                return {
                                    'experiment': f'Cross-Validation ({model_name})',
                                    'accuracy': results[key],
                                    'file': filepath,
                                    'timestamp': datetime.now().isoformat(),
                                    'details': results
                                }
        return None
    
    def save_achievement(self, achievement):
        """Save achievement to file and log"""
        logger.info("\n" + "="*80)
        logger.info("🎯 TARGET 90% ACCURACY ACHIEVED! 🎯")
        logger.info("="*80)
        logger.info(f"Experiment: {achievement['experiment']}")
        logger.info(f"Accuracy: {achievement['accuracy']:.4f} ({achievement['accuracy']*100:.2f}%)")
        logger.info(f"File: {achievement['file']}")
        logger.info(f"Timestamp: {achievement['timestamp']}")
        logger.info("="*80)
        
        # Save to achievement file
        with open(self.achievement_file, 'w') as f:
            json.dump(achievement, f, indent=2)
        
        # Also save to a timestamped file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamped_file = self.results_dir / f'90_percent_achieved_{timestamp}.json'
        with open(timestamped_file, 'w') as f:
            json.dump(achievement, f, indent=2)
        
        return True
    
    def monitor_continuously(self, check_interval=30):
        """Continuously monitor for 90% achievement"""
        logger.info("Starting continuous monitoring for 90% accuracy target...")
        logger.info(f"Check interval: {check_interval} seconds")
        logger.info(f"Target accuracy: {self.target_accuracy*100}%")
        
        check_count = 0
        
        while True:
            check_count += 1
            logger.info(f"\n--- Check #{check_count} at {datetime.now().strftime('%H:%M:%S')} ---")
            
            # Check all experiments
            checks = [
                ('Multi-Architecture Ensemble', self.check_multi_architecture_ensemble),
                ('Hyperparameter Optimization', self.check_hyperparameter_optimization),
                ('Advanced Training Techniques', self.check_advanced_training_techniques),
                ('Cross-Validation', self.check_cross_validation)
            ]
            
            achievement_found = False
            
            for exp_name, check_func in checks:
                try:
                    result = check_func()
                    if result:
                        self.save_achievement(result)
                        achievement_found = True
                        break
                except Exception as e:
                    logger.error(f"Error checking {exp_name}: {e}")
            
            if achievement_found:
                logger.info("\n🎉 MONITORING COMPLETE - TARGET ACHIEVED! 🎉")
                break
            
            # Show current status
            if check_count % 5 == 0:  # Every 5th check
                logger.info("Current status: All experiments still running, target not yet achieved")
            
            time.sleep(check_interval)
    
    def run_single_check(self):
        """Run a single check and return result"""
        logger.info("Running single check for 90% accuracy achievement...")
        
        checks = [
            ('Multi-Architecture Ensemble', self.check_multi_architecture_ensemble),
            ('Hyperparameter Optimization', self.check_hyperparameter_optimization),
            ('Advanced Training Techniques', self.check_advanced_training_techniques),
            ('Cross-Validation', self.check_cross_validation)
        ]
        
        for exp_name, check_func in checks:
            try:
                result = check_func()
                if result:
                    self.save_achievement(result)
                    return result
            except Exception as e:
                logger.error(f"Error checking {exp_name}: {e}")
        
        logger.info("No experiment has achieved 90% accuracy yet.")
        return None

def main():
    import sys
    
    monitor = TargetAchievementMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        # Continuous monitoring mode
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        monitor.monitor_continuously(check_interval=interval)
    else:
        # Single check mode
        result = monitor.run_single_check()
        if result:
            print(f"\n🎯 SUCCESS: {result['experiment']} achieved {result['accuracy']:.4f} accuracy!")
        else:
            print("\n⏳ Target not yet achieved. Continue monitoring...")

if __name__ == "__main__":
    main()