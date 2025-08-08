#!/usr/bin/env python3
"""
Experiment Monitor
Script untuk memantau progress eksperimen secara real-time

Author: AI Assistant
Date: 2025-08-07
"""

import os
import sys
import time
import json
import psutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

class ExperimentMonitor:
    def __init__(self):
        self.start_time = datetime.now()
        self.log_files = [
            'logs/automated_experiments.log',
            'logs/multi_architecture_ensemble.log',
            'logs/extended_ensemble.log',
            'logs/augmentation_ensemble.log'
        ]
        
    def check_gpu_usage(self):
        """Check GPU usage if available"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    gpu_util, mem_used, mem_total = line.split(', ')
                    print(f"GPU {i}: {gpu_util}% utilization, {mem_used}/{mem_total} MB memory")
            return True
        except:
            return False
            
    def check_system_resources(self):
        """Monitor system resources"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / 1024**3
        memory_total_gb = memory.total / 1024**3
        
        # Disk usage
        disk = psutil.disk_usage('.')
        disk_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / 1024**3
        
        print(f"\n📊 System Resources:")
        print(f"CPU: {cpu_percent:.1f}%")
        print(f"Memory: {memory_percent:.1f}% ({memory_used_gb:.1f}/{memory_total_gb:.1f} GB)")
        print(f"Disk: {disk_percent:.1f}% used, {disk_free_gb:.1f} GB free")
        
        # Check GPU if available
        if not self.check_gpu_usage():
            print("GPU: Not available or nvidia-smi not found")
            
    def get_latest_log_entries(self, log_file, lines=5):
        """Get latest entries from log file"""
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    all_lines = f.readlines()
                    return all_lines[-lines:] if all_lines else []
            return []
        except Exception as e:
            return [f"Error reading {log_file}: {str(e)}"]
            
    def check_experiment_progress(self):
        """Check progress of all experiments"""
        print(f"\n🔍 Experiment Progress Check - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        for log_file in self.log_files:
            if os.path.exists(log_file):
                print(f"\n📄 {log_file}:")
                latest_entries = self.get_latest_log_entries(log_file, 3)
                for entry in latest_entries:
                    print(f"  {entry.strip()}")
            else:
                print(f"\n📄 {log_file}: Not found")
                
    def check_results_files(self):
        """Check for completed results"""
        results_dir = Path('results')
        if results_dir.exists():
            result_files = list(results_dir.glob('*.json'))
            if result_files:
                print(f"\n📋 Results Files Found: {len(result_files)}")
                for file in sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                    mod_time = datetime.fromtimestamp(file.stat().st_mtime)
                    print(f"  {file.name} (modified: {mod_time.strftime('%H:%M:%S')})")
            else:
                print("\n📋 No results files found yet")
        else:
            print("\n📋 Results directory not found")
            
    def check_model_checkpoints(self):
        """Check for saved model checkpoints"""
        models_dir = Path('models')
        if models_dir.exists():
            model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
            if model_dirs:
                print(f"\n🤖 Model Checkpoints: {len(model_dirs)} directories")
                for model_dir in sorted(model_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                    checkpoint_files = list(model_dir.glob('checkpoint-*'))
                    print(f"  {model_dir.name}: {len(checkpoint_files)} checkpoints")
            else:
                print("\n🤖 No model directories found yet")
        else:
            print("\n🤖 Models directory not found")
            
    def estimate_remaining_time(self):
        """Estimate remaining time based on progress"""
        elapsed = datetime.now() - self.start_time
        
        # Check if any experiments are running
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if any(script in cmdline for script in ['ensemble', 'experiment']):
                        python_processes.append(proc.info)
            except:
                continue
                
        print(f"\n⏱️ Time Elapsed: {elapsed}")
        
        if python_processes:
            print(f"🔄 Active Python processes: {len(python_processes)}")
            for proc in python_processes:
                print(f"  PID {proc['pid']}: {' '.join(proc['cmdline'][-2:])}")
        else:
            print("⏸️ No active experiment processes detected")
            
    def send_completion_notification(self):
        """Send notification when experiments complete"""
        try:
            # Simple Windows notification
            if os.name == 'nt':
                subprocess.run([
                    'powershell', '-Command',
                    'Add-Type -AssemblyName System.Windows.Forms; '
                    '[System.Windows.Forms.MessageBox]::Show("Ensemble experiments completed!", "Experiment Monitor")'
                ], check=False)
        except:
            pass
            
    def monitor_loop(self, check_interval=60):
        """Main monitoring loop"""
        print(f"🚀 Starting Experiment Monitor")
        print(f"Check interval: {check_interval} seconds")
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Press Ctrl+C to stop monitoring\n")
        
        last_notification_check = datetime.now()
        
        try:
            while True:
                # Clear screen (optional)
                if os.name == 'nt':
                    os.system('cls')
                else:
                    os.system('clear')
                    
                print(f"🔍 Experiment Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*80)
                
                # Check system resources
                self.check_system_resources()
                
                # Check experiment progress
                self.check_experiment_progress()
                
                # Check results and checkpoints
                self.check_results_files()
                self.check_model_checkpoints()
                
                # Estimate remaining time
                self.estimate_remaining_time()
                
                # Check for completion every 5 minutes
                if datetime.now() - last_notification_check > timedelta(minutes=5):
                    # Check if experiments are done
                    if os.path.exists('results/automated_experiments_*.json'):
                        print("\n🎉 Experiments appear to be completed!")
                        self.send_completion_notification()
                        break
                    last_notification_check = datetime.now()
                
                print(f"\n⏰ Next check in {check_interval} seconds...")
                print("Press Ctrl+C to stop monitoring")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
        except Exception as e:
            print(f"\n💥 Monitor error: {str(e)}")
            
def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor ensemble experiments')
    parser.add_argument('--interval', type=int, default=60, 
                       help='Check interval in seconds (default: 60)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (no continuous monitoring)')
    
    args = parser.parse_args()
    
    monitor = ExperimentMonitor()
    
    if args.once:
        # Run once and exit
        monitor.check_system_resources()
        monitor.check_experiment_progress()
        monitor.check_results_files()
        monitor.check_model_checkpoints()
        monitor.estimate_remaining_time()
    else:
        # Continuous monitoring
        monitor.monitor_loop(args.interval)
        
if __name__ == "__main__":
    main()