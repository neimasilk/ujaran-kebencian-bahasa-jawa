#!/usr/bin/env python3
"""
Continuous 90% Achievement Monitor
Automatically monitors all experiments and alerts when 90% target is achieved
"""

import json
import os
import time
from datetime import datetime
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/90_percent_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_results_file(filepath, experiment_name):
    """Check if a results file contains 90%+ achievement"""
    if not os.path.exists(filepath):
        return None
        
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Different result structures for different experiments
        accuracy = None
        f1_macro = None
        
        # Check various possible structures
        if 'final_test_results' in data:
            # ensemble_advanced_results.json structure
            accuracy = data['final_test_results'].get('accuracy')
            f1_macro = data['final_test_results'].get('f1_macro')
        elif 'final_results' in data:
            # improved_meta_ensemble structure
            accuracy = data['final_results'].get('accuracy')
            f1_macro = data['final_results'].get('f1_macro')
        elif 'test_accuracy' in data:
            # Direct structure
            accuracy = data.get('test_accuracy')
            f1_macro = data.get('test_f1_macro')
        elif 'accuracy' in data:
            # Simple structure
            accuracy = data.get('accuracy')
            f1_macro = data.get('f1_macro')
            
        if accuracy is not None or f1_macro is not None:
            return {
                'experiment': experiment_name,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'accuracy_90_plus': accuracy >= 0.90 if accuracy else False,
                'f1_macro_90_plus': f1_macro >= 0.90 if f1_macro else False,
                'target_achieved': (accuracy >= 0.90 if accuracy else False) or (f1_macro >= 0.90 if f1_macro else False),
                'timestamp': data.get('experiment_timestamp', 'unknown'),
                'filepath': filepath
            }
            
    except Exception as e:
        logger.debug(f"Error reading {filepath}: {e}")
        
    return None

def monitor_single_check():
    """Single monitoring check for 90% achievement"""
    
    # Define all possible result files
    result_files = [
        ('results/improved_meta_ensemble_90_percent_results.json', 'Improved Meta-Ensemble'),
        ('results/advanced_ensemble_90_percent_results.json', 'Advanced Ensemble'),
        ('results/multi_architecture_ensemble_results.json', 'Multi-Architecture Ensemble'),
        ('results/hyperparameter_optimization_results.json', 'Hyperparameter Optimization'),
        ('results/advanced_training_results.json', 'Advanced Training Techniques'),
        ('results/cross_validation_results.json', 'Cross-Validation Framework'),
        ('results/ensemble_advanced_results.json', 'Original Advanced Ensemble'),
        ('results/improved_model_evaluation.json', 'Improved Model'),
        ('results/augmented_model_results.json', 'Augmented Model')
    ]
    
    achievements = []
    all_results = []
    
    for filepath, experiment_name in result_files:
        result = check_results_file(filepath, experiment_name)
        if result:
            all_results.append(result)
            if result['target_achieved']:
                achievements.append(result)
    
    return len(achievements) > 0, achievements, all_results

def continuous_monitor(check_interval=60, max_duration=7200):
    """Continuously monitor for 90% achievement"""
    logger.info("Starting Continuous 90% Achievement Monitor")
    logger.info(f"Check interval: {check_interval}s, Max duration: {max_duration}s")
    logger.info("="*80)
    
    os.makedirs('results', exist_ok=True)
    start_time = time.time()
    check_count = 0
    
    while time.time() - start_time < max_duration:
        check_count += 1
        current_time = datetime.now().strftime('%H:%M:%S')
        
        logger.info(f"Check #{check_count} at {current_time}")
        
        achieved, achievements, all_results = monitor_single_check()
        
        if achieved:
            logger.info("*** 90% TARGET ACHIEVED! ***")
            logger.info("="*80)
            
            for achievement in achievements:
                logger.info(f"SUCCESS: {achievement['experiment']}:")
                if achievement['accuracy']:
                    logger.info(f"   Accuracy: {achievement['accuracy']*100:.2f}%")
                if achievement['f1_macro']:
                    logger.info(f"   F1-Macro: {achievement['f1_macro']*100:.2f}%")
                logger.info(f"   File: {achievement['filepath']}")
                logger.info(f"   Timestamp: {achievement['timestamp']}")
            
            # Save achievement notification
            achievement_file = f"results/90_PERCENT_ACHIEVED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(achievement_file, 'w') as f:
                json.dump({
                    'achievement_timestamp': datetime.now().isoformat(),
                    'achievements': achievements,
                    'all_results': all_results,
                    'message': '90% TARGET ACHIEVED!',
                    'monitor_duration_seconds': time.time() - start_time,
                    'total_checks': check_count
                }, f, indent=2)
                
            logger.info(f"Achievement details saved to: {achievement_file}")
            logger.info("Monitoring completed successfully!")
            return True, achievements
        
        # Log current best performance
        if all_results:
            best_result = max(all_results, key=lambda x: max(x['accuracy'] or 0, x['f1_macro'] or 0))
            best_score = max(best_result['accuracy'] or 0, best_result['f1_macro'] or 0)
            logger.info(f"Current best: {best_result['experiment']} - {best_score*100:.2f}%")
        else:
            logger.info("No completed experiments yet")
            
        # Save monitoring status
        status = {
            'timestamp': datetime.now().isoformat(),
            'check_count': check_count,
            'elapsed_seconds': time.time() - start_time,
            'achievements': achievements,
            'all_results': all_results,
            'target_achieved': len(achievements) > 0,
            'best_performance': max([max(r['accuracy'] or 0, r['f1_macro'] or 0) for r in all_results]) if all_results else 0
        }
        
        with open('results/continuous_monitor_status.json', 'w') as f:
            json.dump(status, f, indent=2)
            
        logger.info(f"Waiting {check_interval}s before next check...")
        time.sleep(check_interval)
    
    logger.info(f"Monitoring duration completed after {check_count} checks")
    logger.info("No 90% achievement detected during monitoring period")
    return False, []

def main():
    """Main continuous monitoring function"""
    try:
        # First do a quick check
        logger.info("Initial 90% Achievement Check")
        achieved, achievements, all_results = monitor_single_check()
        
        if achieved:
            logger.info("90% TARGET ALREADY ACHIEVED!")
            for achievement in achievements:
                logger.info(f"SUCCESS: {achievement['experiment']}: {max(achievement['accuracy'] or 0, achievement['f1_macro'] or 0)*100:.2f}%")
            return True
        
        logger.info("90% target not yet achieved. Starting continuous monitoring...")
        
        # Start continuous monitoring
        success, achievements = continuous_monitor(
            check_interval=60,  # Check every minute
            max_duration=7200   # Monitor for 2 hours max
        )
        
        return success
        
    except KeyboardInterrupt:
        logger.info("\nMonitoring stopped by user")
        return False
    except Exception as e:
        logger.error(f"Error during monitoring: {e}")
        return False

if __name__ == "__main__":
    print("Continuous 90% Achievement Monitor")
    print("Press Ctrl+C to stop monitoring\n")
    
    success = main()
    
    if success:
        print("\nSUCCESS: 90% target achieved!")
    else:
        print("\nMonitoring completed without 90% achievement")
        print("Run again later or check individual experiment progress")