#!/usr/bin/env python3
"""
Comprehensive 90% Target Monitor
Monitors all running experiments and detects 90% achievement
"""

import json
import os
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
        logger.error(f"Error reading {filepath}: {e}")
        
    return None

def monitor_experiments():
    """Monitor all experiments for 90% achievement"""
    
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
    
    logger.info("Checking all experiment results...")
    
    for filepath, experiment_name in result_files:
        result = check_results_file(filepath, experiment_name)
        if result:
            all_results.append(result)
            if result['target_achieved']:
                achievements.append(result)
                logger.info(f"🎉 90%+ TARGET ACHIEVED: {experiment_name}")
                if result['accuracy'] and result['accuracy'] >= 0.90:
                    logger.info(f"   ✅ Accuracy: {result['accuracy']*100:.2f}%")
                if result['f1_macro'] and result['f1_macro'] >= 0.90:
                    logger.info(f"   ✅ F1-Macro: {result['f1_macro']*100:.2f}%")
            else:
                logger.info(f"📊 {experiment_name}: Acc={result['accuracy']*100:.2f}% F1={result['f1_macro']*100:.2f}%" if result['accuracy'] and result['f1_macro'] else f"📊 {experiment_name}: Results available")
    
    # Summary report
    print("\n" + "="*80)
    print("🎯 90% TARGET ACHIEVEMENT MONITOR")
    print("="*80)
    
    if achievements:
        print(f"\n🎉 SUCCESS! {len(achievements)} experiment(s) achieved 90%+ target:")
        for achievement in achievements:
            print(f"\n✅ {achievement['experiment']}:")
            if achievement['accuracy']:
                print(f"   Accuracy: {achievement['accuracy']*100:.2f}%")
            if achievement['f1_macro']:
                print(f"   F1-Macro: {achievement['f1_macro']*100:.2f}%")
            print(f"   Timestamp: {achievement['timestamp']}")
            print(f"   File: {achievement['filepath']}")
    else:
        print("\n⏳ No experiments have achieved 90%+ target yet.")
        
    if all_results:
        print(f"\n📊 All Available Results ({len(all_results)} experiments):")
        # Sort by best performance
        all_results.sort(key=lambda x: max(x['accuracy'] or 0, x['f1_macro'] or 0), reverse=True)
        
        for i, result in enumerate(all_results[:10]):  # Top 10
            best_metric = max(result['accuracy'] or 0, result['f1_macro'] or 0)
            print(f"   {i+1}. {result['experiment']}: {best_metric*100:.2f}%")
            
    print(f"\n🕒 Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save monitoring results
    monitoring_result = {
        'timestamp': datetime.now().isoformat(),
        'achievements': achievements,
        'all_results': all_results,
        'target_achieved': len(achievements) > 0,
        'best_performance': max([max(r['accuracy'] or 0, r['f1_macro'] or 0) for r in all_results]) if all_results else 0
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/90_percent_monitoring.json', 'w') as f:
        json.dump(monitoring_result, f, indent=2)
    
    return len(achievements) > 0, achievements

def continuous_monitor(check_interval=30, max_duration=3600):
    """Continuously monitor for 90% achievement"""
    logger.info(f"Starting continuous monitoring (check every {check_interval}s, max {max_duration}s)")
    
    start_time = time.time()
    
    while time.time() - start_time < max_duration:
        achieved, achievements = monitor_experiments()
        
        if achieved:
            logger.info("🎯 90% TARGET ACHIEVED! Stopping monitoring.")
            return True, achievements
            
        logger.info(f"Waiting {check_interval}s before next check...")
        time.sleep(check_interval)
    
    logger.info("Monitoring duration completed.")
    return False, []

def main():
    """Main monitoring function"""
    print("🔍 Comprehensive 90% Target Monitor")
    print("Checking all experiments for 90%+ accuracy or F1-macro achievement...\n")
    
    # Single check
    achieved, achievements = monitor_experiments()
    
    if achieved:
        print("\n🎉 SUCCESS: 90% target has been achieved!")
        
        # Save achievement notification
        achievement_file = f"results/90_PERCENT_ACHIEVED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(achievement_file, 'w') as f:
            json.dump({
                'achievement_timestamp': datetime.now().isoformat(),
                'achievements': achievements,
                'message': '90% target achieved!'
            }, f, indent=2)
            
        print(f"\n📁 Achievement details saved to: {achievement_file}")
        return True
    else:
        print("\n⏳ 90% target not yet achieved. Continue monitoring experiments.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n🔄 Run this script periodically to check for 90% achievement.")
        print("💡 Or use continuous_monitor() function for automated monitoring.")