#!/usr/bin/env python3
"""
Experiment State Manager
Mengelola state dan resume functionality untuk eksperimen otomatis

Author: AI Assistant
Date: 2025-08-07
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class ExperimentStateManager:
    def __init__(self, state_file: str = "experiment_state.json"):
        self.state_file = Path(state_file)
        self.state = self._load_state()
        
    def _load_state(self) -> Dict[str, Any]:
        """Load experiment state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}")
                return self._create_empty_state()
        else:
            return self._create_empty_state()
    
    def _create_empty_state(self) -> Dict[str, Any]:
        """Create empty state structure"""
        return {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "experiments": {},
            "global_status": "not_started",
            "current_experiment": None,
            "completed_experiments": [],
            "failed_experiments": [],
            "skipped_experiments": []
        }
    
    def save_state(self):
        """Save current state to file"""
        self.state["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def register_experiment(self, name: str, config: Dict[str, Any]):
        """Register a new experiment"""
        self.state["experiments"][name] = {
            "name": name,
            "config": config,
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "duration_minutes": None,
            "result_files": [],
            "model_checkpoints": [],
            "error_message": None,
            "attempts": 0
        }
        self.save_state()
    
    def start_experiment(self, name: str):
        """Mark experiment as started"""
        if name in self.state["experiments"]:
            self.state["experiments"][name]["status"] = "running"
            self.state["experiments"][name]["start_time"] = datetime.now().isoformat()
            self.state["experiments"][name]["attempts"] += 1
            self.state["current_experiment"] = name
            self.state["global_status"] = "running"
            self.save_state()
            logger.info(f"Started experiment: {name}")
    
    def complete_experiment(self, name: str, result_files: List[str] = None, 
                          model_checkpoints: List[str] = None):
        """Mark experiment as completed"""
        if name in self.state["experiments"]:
            exp = self.state["experiments"][name]
            exp["status"] = "completed"
            exp["end_time"] = datetime.now().isoformat()
            
            if exp["start_time"]:
                start = datetime.fromisoformat(exp["start_time"])
                end = datetime.fromisoformat(exp["end_time"])
                exp["duration_minutes"] = (end - start).total_seconds() / 60
            
            if result_files:
                exp["result_files"] = result_files
            if model_checkpoints:
                exp["model_checkpoints"] = model_checkpoints
            
            if name not in self.state["completed_experiments"]:
                self.state["completed_experiments"].append(name)
            
            self.state["current_experiment"] = None
            self.save_state()
            logger.info(f"Completed experiment: {name}")
    
    def fail_experiment(self, name: str, error_message: str = None):
        """Mark experiment as failed"""
        if name in self.state["experiments"]:
            exp = self.state["experiments"][name]
            exp["status"] = "failed"
            exp["end_time"] = datetime.now().isoformat()
            exp["error_message"] = error_message
            
            if exp["start_time"]:
                start = datetime.fromisoformat(exp["start_time"])
                end = datetime.fromisoformat(exp["end_time"])
                exp["duration_minutes"] = (end - start).total_seconds() / 60
            
            if name not in self.state["failed_experiments"]:
                self.state["failed_experiments"].append(name)
            
            self.state["current_experiment"] = None
            self.save_state()
            logger.error(f"Failed experiment: {name} - {error_message}")
    
    def skip_experiment(self, name: str, reason: str = None):
        """Mark experiment as skipped"""
        if name in self.state["experiments"]:
            self.state["experiments"][name]["status"] = "skipped"
            self.state["experiments"][name]["error_message"] = reason
            
            if name not in self.state["skipped_experiments"]:
                self.state["skipped_experiments"].append(name)
            
            self.save_state()
            logger.info(f"Skipped experiment: {name} - {reason}")
    
    def get_experiment_status(self, name: str) -> str:
        """Get status of specific experiment"""
        if name in self.state["experiments"]:
            return self.state["experiments"][name]["status"]
        return "not_found"
    
    def is_experiment_completed(self, name: str) -> bool:
        """Check if experiment is completed"""
        return self.get_experiment_status(name) == "completed"
    
    def is_experiment_failed(self, name: str) -> bool:
        """Check if experiment failed"""
        return self.get_experiment_status(name) == "failed"
    
    def get_pending_experiments(self) -> List[str]:
        """Get list of pending experiments"""
        pending = []
        for name, exp in self.state["experiments"].items():
            if exp["status"] in ["pending", "failed"]:
                pending.append(name)
        return pending
    
    def should_skip_experiment(self, experiment_name: str) -> bool:
        """Check if experiment should be skipped (already completed)"""
        if experiment_name not in self.state['experiments']:
            return False
        
        exp_data = self.state['experiments'][experiment_name]
        return exp_data['status'] == 'completed'
    
    def get_experiment_result(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Get result of a completed experiment"""
        if experiment_name not in self.state['experiments']:
            return None
        
        exp_data = self.state['experiments'][experiment_name]
        if exp_data['status'] == 'completed':
            return exp_data.get('result', {})
        
        return None
    
    def get_completed_experiments(self) -> List[str]:
        """Get list of completed experiments"""
        return self.state["completed_experiments"].copy()
    
    def get_failed_experiments(self) -> List[str]:
        """Get list of failed experiments"""
        return self.state["failed_experiments"].copy()
    
    def should_resume(self) -> bool:
        """Check if there are experiments to resume"""
        pending = self.get_pending_experiments()
        return len(pending) > 0
    
    def get_resume_summary(self) -> Dict[str, Any]:
        """Get summary for resume operation"""
        return {
            "session_id": self.state["session_id"],
            "total_experiments": len(self.state["experiments"]),
            "completed": len(self.state["completed_experiments"]),
            "failed": len(self.state["failed_experiments"]),
            "pending": len(self.get_pending_experiments()),
            "last_updated": self.state["last_updated"],
            "current_experiment": self.state["current_experiment"]
        }
    
    def detect_existing_results(self) -> Dict[str, List[str]]:
        """Detect existing result files and model checkpoints"""
        results = {
            "result_files": [],
            "model_checkpoints": [],
            "log_files": []
        }
        
        # Check results folder
        results_dir = Path("results")
        if results_dir.exists():
            for file in results_dir.glob("*.json"):
                results["result_files"].append(str(file))
        
        # Check models folder
        models_dir = Path("models")
        if models_dir.exists():
            for folder in models_dir.iterdir():
                if folder.is_dir():
                    results["model_checkpoints"].append(str(folder))
        
        # Check logs folder
        logs_dir = Path("logs")
        if logs_dir.exists():
            for file in logs_dir.glob("*.log"):
                results["log_files"].append(str(file))
        
        return results
    
    def analyze_previous_run(self) -> Dict[str, Any]:
        """Analyze previous run from existing files"""
        analysis = {
            "has_previous_run": False,
            "completed_experiments": [],
            "failed_experiments": [],
            "result_files": [],
            "recommendations": []
        }
        
        # Check for automated experiment results
        results_dir = Path("results")
        if results_dir.exists():
            for file in results_dir.glob("automated_experiments_*.json"):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    
                    analysis["has_previous_run"] = True
                    analysis["result_files"].append(str(file))
                    
                    if "experiments" in data:
                        for exp_name, exp_data in data["experiments"].items():
                            if exp_data.get("status") == "success":
                                analysis["completed_experiments"].append(exp_name)
                            elif exp_data.get("status") == "failed":
                                analysis["failed_experiments"].append(exp_name)
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze {file}: {e}")
        
        # Generate recommendations
        if analysis["has_previous_run"]:
            if analysis["failed_experiments"]:
                analysis["recommendations"].append(
                    f"Resume failed experiments: {', '.join(analysis['failed_experiments'])}"
                )
            if analysis["completed_experiments"]:
                analysis["recommendations"].append(
                    f"Skip completed experiments: {', '.join(analysis['completed_experiments'])}"
                )
        
        return analysis
    
    def reset_state(self):
        """Reset experiment state"""
        self.state = self._create_empty_state()
        self.save_state()
        logger.info("Experiment state reset")
    
    def cleanup_old_states(self, days: int = 7):
        """Cleanup old state files"""
        # This could be implemented to clean up old state files
        pass