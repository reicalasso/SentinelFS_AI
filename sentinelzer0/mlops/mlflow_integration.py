"""
MLflow Integration

Provides integration with MLflow for experiment tracking, model logging,
and artifact management.
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path


class MLflowTracker:
    """
    MLflow integration for experiment tracking.
    
    Features:
    - Experiment tracking
    - Metric logging
    - Parameter logging
    - Model artifact logging
    - Run management
    
    Note: This is a lightweight wrapper. For full MLflow functionality,
    install mlflow package: pip install mlflow
    """
    
    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "SentinelZer0"):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking URI (optional)
            experiment_name: Experiment name
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)
        
        # Try to import mlflow
        self.mlflow_available = False
        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow_available = True
            
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            mlflow.set_experiment(experiment_name)
            self.logger.info(f"MLflow tracking enabled for experiment: {experiment_name}")
        
        except ImportError:
            self.logger.warning(
                "MLflow not available. Install with: pip install mlflow. "
                "Tracker will operate in no-op mode."
            )
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start an MLflow run.
        
        Args:
            run_name: Run name
            tags: Run tags
        """
        if not self.mlflow_available:
            self.logger.debug("MLflow not available, skipping start_run")
            return
        
        self.mlflow.start_run(run_name=run_name, tags=tags)
        self.logger.info(f"Started MLflow run: {run_name}")
    
    def end_run(self):
        """End the current MLflow run."""
        if not self.mlflow_available:
            return
        
        self.mlflow.end_run()
        self.logger.info("Ended MLflow run")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.
        
        Args:
            params: Parameters to log
        """
        if not self.mlflow_available:
            self.logger.debug(f"Would log params: {params}")
            return
        
        self.mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.
        
        Args:
            metrics: Metrics to log
            step: Step number (optional)
        """
        if not self.mlflow_available:
            self.logger.debug(f"Would log metrics: {metrics}")
            return
        
        self.mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model_path: str,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """
        Log a model artifact.
        
        Args:
            model_path: Path to model file
            artifact_path: Artifact path in MLflow
            registered_model_name: Name for model registry
        """
        if not self.mlflow_available:
            self.logger.debug(f"Would log model: {model_path}")
            return
        
        # Log as artifact
        self.mlflow.log_artifact(model_path, artifact_path)
        
        # Register model if name provided
        if registered_model_name:
            try:
                self.mlflow.register_model(
                    f"runs:/{self.mlflow.active_run().info.run_id}/{artifact_path}",
                    registered_model_name
                )
            except Exception as e:
                self.logger.warning(f"Could not register model: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_dir: Optional[str] = None):
        """
        Log an artifact file.
        
        Args:
            artifact_path: Path to artifact
            artifact_dir: Directory in MLflow to store artifact
        """
        if not self.mlflow_available:
            self.logger.debug(f"Would log artifact: {artifact_path}")
            return
        
        self.mlflow.log_artifact(artifact_path, artifact_dir)
    
    def set_tag(self, key: str, value: str):
        """Set a tag on the current run."""
        if not self.mlflow_available:
            return
        
        self.mlflow.set_tag(key, value)
    
    def get_run_id(self) -> Optional[str]:
        """Get the current run ID."""
        if not self.mlflow_available:
            return None
        
        active_run = self.mlflow.active_run()
        return active_run.info.run_id if active_run else None
    
    def is_available(self) -> bool:
        """Check if MLflow is available."""
        return self.mlflow_available
