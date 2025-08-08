#!/usr/bin/env python3
"""
Experiment Launcher
Script utama untuk menjalankan semua eksperimen ensemble secara otomatis

Usage:
  python run_all_experiments.py                    # Run all experiments
  python run_all_experiments.py --monitor          # Run with monitoring
  python run_all_experiments.py --quick            # Run quick experiments only
  python run_all_experiments.py --resume           # Resume from last checkpoint

Author: AI Assistant
Date: 2025-08-07
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from experiment_state_manager import ExperimentStateManager

# Setup logging with proper Unicode support
class UnicodeStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        if hasattr(self.stream, 'reconfigure'):
            try:
                self.stream.reconfigure(encoding='utf-8')
            except:
                pass
    
    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            # Fallback: remove emojis and special characters
            record.msg = str(record.msg).encode('ascii', 'ignore').decode('ascii')
            super().emit(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_experiments.log', encoding='utf-8'),
        UnicodeStreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ExperimentLauncher:
    def __init__(self, args):
        self.args = args
        self.start_time = datetime.now()
        self.state_manager = ExperimentStateManager()
        
        # Create directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Experiment configurations
        self.experiments = {
            'quick': [
                {
                    'name': 'baseline_ensemble',
                    'script': 'multi_architecture_ensemble.py',
                    'description': 'Standard 3-epoch ensemble (existing script)',
                    'time_minutes': 30
                }
            ],
            'full': [
                {
                    'name': 'baseline_ensemble',
                    'script': 'multi_architecture_ensemble.py',
                    'description': 'Standard 3-epoch ensemble',
                    'time_minutes': 30
                },
                {
                    'name': 'extended_ensemble',
                    'script': 'multi_architecture_ensemble_extended.py',
                    'description': '5-epoch ensemble with advanced optimization',
                    'time_minutes': 50
                },
                {
                    'name': 'augmented_ensemble',
                    'script': 'ensemble_with_augmentation.py',
                    'description': 'Ensemble with data augmentation',
                    'time_minutes': 40
                }
            ]
        }
        
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        logger.info("🔍 Checking prerequisites...")
        
        # Check Python environment
        logger.info(f"Python version: {sys.version}")
        
        # Check required files
        required_files = [
            'data/augmented/augmented_dataset.csv',
            'multi_architecture_ensemble.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
                
        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return False
            
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✅ GPU available: {gpu_name}")
            else:
                logger.warning("⚠️ No GPU available - experiments will be slow")
        except ImportError:
            logger.error("❌ PyTorch not installed")
            return False
            
        # Check disk space
        import psutil
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / 1024**3
        
        if free_gb < 5:
            logger.warning(f"⚠️ Low disk space: {free_gb:.1f} GB free")
        else:
            logger.info(f"✅ Disk space: {free_gb:.1f} GB free")
            
        logger.info("✅ Prerequisites check completed")
        return True
        
    def create_experiment_scripts(self):
        """Create additional experiment scripts if they don't exist"""
        logger.info("📝 Creating additional experiment scripts...")
        
        # Run the automated experiment creator
        try:
            from automated_ensemble_experiments import AutomatedExperimentRunner
            runner = AutomatedExperimentRunner()
            runner.create_extended_ensemble_script()
            runner.create_augmentation_ensemble_script()
            logger.info("✅ Additional scripts created")
        except Exception as e:
            logger.warning(f"⚠️ Could not create additional scripts: {str(e)}")
            
    def run_monitoring_in_background(self):
        """Start monitoring in a separate thread"""
        def monitor_thread():
            try:
                subprocess.run([
                    sys.executable, 'monitor_experiments.py', '--interval', '30'
                ], check=False)
            except Exception as e:
                logger.warning(f"Monitoring thread error: {str(e)}")
                
        if self.args.monitor:
            logger.info("🔍 Starting background monitoring...")
            monitor = threading.Thread(target=monitor_thread, daemon=True)
            monitor.start()
            time.sleep(2)  # Give monitor time to start
            
    def estimate_total_time(self, experiments):
        """Estimate total experiment time"""
        total_minutes = sum(exp['time_minutes'] for exp in experiments)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
            
    def run_experiments(self):
        """Run the selected experiments with resume support"""
        # Handle resume mode
        if self.args.resume:
            return self._handle_resume_mode()
        
        # Select experiment set
        if self.args.quick:
            experiments = self.experiments['quick']
            logger.info("🚀 Running QUICK experiments")
        else:
            experiments = self.experiments['full']
            logger.info("🚀 Running FULL experiment suite")
            
        total_time = self.estimate_total_time(experiments)
        logger.info(f"📊 {len(experiments)} experiments planned")
        logger.info(f"⏱️ Estimated total time: {total_time}")
        logger.info(f"🕐 Start time: {self.start_time.strftime('%H:%M:%S')}")
        
        # Register experiments in state manager
        self._register_experiments(experiments)
        
        # Start monitoring if requested
        self.run_monitoring_in_background()
        
        # Run automated experiments
        try:
            logger.info("\n" + "="*60)
            logger.info("STARTING AUTOMATED EXPERIMENT SUITE")
            logger.info("="*60)
            
            # Pass resume flag to automated experiments
            cmd = [sys.executable, 'automated_ensemble_experiments.py']
            if self.args.resume:
                cmd.append('--resume')
            
            result = subprocess.run(cmd, check=False)
            
            if result.returncode == 0:
                logger.info("All experiments completed successfully!")
                return True
            else:
                logger.error(f"Experiments failed with return code {result.returncode}")
                return False
                
        except KeyboardInterrupt:
            logger.info("\nExperiments interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Error running experiments: {str(e)}")
            return False
    
    def _handle_resume_mode(self):
        """Handle resume mode"""
        logger.info("\n" + "="*60)
        logger.info("RESUME MODE ACTIVATED")
        logger.info("="*60)
        
        # Analyze previous run
        analysis = self.state_manager.analyze_previous_run()
        
        if not analysis["has_previous_run"]:
            logger.warning("No previous experiment run detected")
            logger.info("Starting fresh experiment suite instead...")
            self.args.resume = False
            return self.run_experiments()
        
        # Show resume summary
        logger.info("Previous Run Analysis:")
        logger.info(f"   Completed: {len(analysis['completed_experiments'])}")
        logger.info(f"   Failed: {len(analysis['failed_experiments'])}")
        logger.info(f"   Result files: {len(analysis['result_files'])}")
        
        if analysis["completed_experiments"]:
            logger.info(f"   Completed experiments: {', '.join(analysis['completed_experiments'])}")
        
        if analysis["failed_experiments"]:
            logger.info(f"   Failed experiments to retry: {', '.join(analysis['failed_experiments'])}")
        
        # Show recommendations
        if analysis["recommendations"]:
            logger.info("\nRecommendations:")
            for rec in analysis["recommendations"]:
                logger.info(f"   - {rec}")
        
        # Update state manager with previous results
        self._update_state_from_analysis(analysis)
        
        # Start monitoring if requested
        self.run_monitoring_in_background()
        
        # Run automated experiments in resume mode
        try:
            logger.info("\n" + "="*60)
            logger.info("RESUMING AUTOMATED EXPERIMENT SUITE")
            logger.info("="*60)
            
            result = subprocess.run([
                sys.executable, 'automated_ensemble_experiments.py', '--resume'
            ], check=False)
            
            if result.returncode == 0:
                logger.info("Resume completed successfully!")
                return True
            else:
                logger.error(f"Resume failed with return code {result.returncode}")
                return False
                
        except KeyboardInterrupt:
            logger.info("\nResume interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Error during resume: {str(e)}")
            return False
    
    def _register_experiments(self, experiments):
        """Register experiments in state manager"""
        for exp in experiments:
            self.state_manager.register_experiment(exp['name'], exp)
    
    def _update_state_from_analysis(self, analysis):
        """Update state manager from previous run analysis"""
        # Register completed experiments
        for exp_name in analysis["completed_experiments"]:
            # Create a dummy config for completed experiments
            config = {
                'name': exp_name,
                'description': f'Previously completed experiment: {exp_name}',
                'estimated_time_minutes': 0
            }
            self.state_manager.register_experiment(exp_name, config)
            self.state_manager.complete_experiment(exp_name)
        
        # Register failed experiments
        for exp_name in analysis["failed_experiments"]:
            config = {
                'name': exp_name,
                'description': f'Previously failed experiment: {exp_name}',
                'estimated_time_minutes': 30  # Default estimate
            }
            self.state_manager.register_experiment(exp_name, config)
            self.state_manager.fail_experiment(exp_name, "Previous run failed")
            
    def show_results_summary(self):
        """Show final results summary"""
        logger.info("\n" + "="*60)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("="*60)
        
        # Look for results files
        results_dir = Path('results')
        if results_dir.exists():
            result_files = list(results_dir.glob('automated_experiments_*.json'))
            
            if result_files:
                # Get the most recent results file
                latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                
                try:
                    with open(latest_file, 'r') as f:
                        results = json.load(f)
                        
                    summary = results.get('summary', {})
                    total_duration = results.get('total_duration_hours', 0)
                    
                    logger.info(f"Results file: {latest_file.name}")
                    logger.info(f"Total duration: {total_duration:.1f} hours")
                    logger.info(f"Successful: {summary.get('successful', 0)}")
                    logger.info(f"Failed: {summary.get('failed', 0)}")
                    logger.info(f"Timeout: {summary.get('timeout', 0)}")
                    logger.info(f"Crashed: {summary.get('crashed', 0)}")
                    
                    # Show individual experiment results
                    experiments = results.get('experiments', {})
                    for exp_name, exp_result in experiments.items():
                        status = exp_result.get('status', 'unknown')
                        duration = exp_result.get('duration_minutes', 0)
                        
                        status_text = {
                            'success': 'SUCCESS',
                            'failed': 'FAILED',
                            'timeout': 'TIMEOUT',
                            'crashed': 'CRASHED'
                        }.get(status, 'UNKNOWN')
                        
                        logger.info(f"  {status_text} {exp_name}: {status} ({duration:.1f}m)")
                        
                except Exception as e:
                    logger.error(f"Error reading results file: {str(e)}")
            else:
                logger.warning("No automated experiment results found")
        else:
            logger.warning("Results directory not found")
            
        # Show model checkpoints
        models_dir = Path('models')
        if models_dir.exists():
            model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
            logger.info(f"\nModel checkpoints: {len(model_dirs)} directories saved")
            
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        logger.info(f"\nTotal launcher duration: {total_duration}")
        
def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Launch ensemble experiments automatically',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_experiments.py                    # Run all experiments
  python run_all_experiments.py --monitor          # Run with monitoring
  python run_all_experiments.py --quick            # Run quick experiments only
  python run_all_experiments.py --resume           # Resume from checkpoint
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick experiments only (faster)')
    parser.add_argument('--monitor', action='store_true',
                       help='Enable background monitoring')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = ExperimentLauncher(args)
    
    try:
        # Show banner
        print("\n" + "="*80)
        print("🚀 AUTOMATED ENSEMBLE EXPERIMENT LAUNCHER")
        print("="*80)
        print(f"Started: {launcher.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
        print(f"Monitoring: {'ENABLED' if args.monitor else 'DISABLED'}")
        print("="*80)
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No experiments will be executed")
            experiments = launcher.experiments['quick'] if args.quick else launcher.experiments['full']
            for exp in experiments:
                logger.info(f"  Would run: {exp['name']} ({exp['time_minutes']}m)")
            return 0
            
        # Check prerequisites
        if not launcher.check_prerequisites():
            logger.error("Prerequisites check failed")
            return 1
            
        # Create additional scripts
        launcher.create_experiment_scripts()
        
        # Run experiments
        success = launcher.run_experiments()
        
        # Show results
        launcher.show_results_summary()
        
        if success:
            logger.info("\nExperiment launcher completed successfully!")
            logger.info("You can now check the results in the 'results/' directory")
            return 0
        else:
            logger.error("\nSome experiments failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\nLauncher interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\nLauncher error: {str(e)}")
        return 1
        
if __name__ == "__main__":
    sys.exit(main())