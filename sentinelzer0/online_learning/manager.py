"""
Online Learning Manager

High-level orchestration of all online learning components.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from .incremental_learner import IncrementalLearner, LearningStrategy
from .drift_detector import ConceptDriftDetector, DriftDetectionMethod
from .feedback_collector import FeedbackCollector, FeedbackType
from .retraining_pipeline import RetrainingPipeline, RetrainingConfig
from .online_validator import OnlineValidator


class OnlineLearningManager:
    """
    Unified online learning system manager.
    
    Coordinates all online learning components:
    - Incremental learning
    - Drift detection
    - Feedback collection
    - Automated retraining
    - Online validation
    
    Features:
    - Automatic adaptation to concept drift
    - Continuous model improvement
    - Performance monitoring
    - Feedback-driven learning
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        learning_strategy: LearningStrategy = LearningStrategy.MINI_BATCH,
        drift_method: DriftDetectionMethod = DriftDetectionMethod.ADWIN,
        retraining_config: Optional[RetrainingConfig] = None,
        feedback_storage_path: Optional[str] = None,
        model_save_path: str = "online_model.pt"
    ):
        """
        Initialize online learning manager.
        
        Args:
            model: Model to manage
            learning_rate: Learning rate for incremental updates
            learning_strategy: Strategy for incremental learning
            drift_method: Method for drift detection
            retraining_config: Configuration for retraining
            feedback_storage_path: Path for feedback storage
            model_save_path: Path to save model
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        
        # Initialize components
        self.learner = IncrementalLearner(
            model=model,
            learning_rate=learning_rate,
            strategy=learning_strategy
        )
        
        self.drift_detector = ConceptDriftDetector(method=drift_method)
        
        self.feedback_collector = FeedbackCollector(
            storage_path=feedback_storage_path
        )
        
        self.retraining_pipeline = RetrainingPipeline(
            model=model,
            config=retraining_config or RetrainingConfig(),
            model_path=model_save_path
        )
        
        self.validator = OnlineValidator(model=model)
        
        # State
        self.enabled = True
        self.drift_detected = False
        
        self.logger.info("Initialized online learning manager")
    
    def process_sample(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        feedback_type: Optional[FeedbackType] = None,
        sample_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single sample through the online learning pipeline.
        
        Args:
            inputs: Input tensor
            labels: True labels (if available)
            feedback_type: Type of feedback (if any)
            sample_id: Sample identifier
        
        Returns:
            Processing results
        """
        if not self.enabled:
            return {'enabled': False}
        
        results = {}
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(inputs)
            _, predicted_class = prediction.max(1)
        
        results['prediction'] = predicted_class.item()
        
        # If we have labels, perform online learning
        if labels is not None:
            # Incremental learning update
            if self.enabled:
                update_stats = self.learner.update(inputs, labels)
                results['learning'] = update_stats
            
            # Drift detection
            is_correct = (predicted_class == labels).item()
            loss = torch.nn.functional.cross_entropy(prediction, labels).item()
            
            drift_result = self.drift_detector.add_sample(loss, is_correct)
            results['drift'] = drift_result
            
            if drift_result['drift_detected']:
                self.drift_detected = True
                self.logger.warning(f"Concept drift detected! Severity: {drift_result['severity']:.3f}")
                
                # Adapt learning rate when drift detected
                self.learner.adapt_learning_rate(factor=1.5)
            
            # Online validation
            val_result = self.validator.validate_sample(inputs, labels)
            results['validation'] = val_result
            
            # Collect feedback if specified
            if feedback_type and sample_id:
                feedback = self.feedback_collector.add_feedback(
                    sample_id=sample_id,
                    inputs=inputs,
                    prediction=prediction,
                    true_label=labels,
                    feedback_type=feedback_type
                )
                results['feedback'] = feedback
            
            # Check if retraining needed
            if self.retraining_pipeline.should_retrain(sample_count=1):
                results['retrain_triggered'] = True
                self.logger.info("Retraining triggered by sample count")
        
        return results
    
    def trigger_retraining(
        self,
        train_data: Optional[torch.utils.data.Dataset] = None
    ) -> Dict[str, Any]:
        """
        Manually trigger retraining.
        
        Args:
            train_data: Training dataset (uses feedback if None)
        
        Returns:
            Retraining results
        """
        if train_data is None:
            # Use collected feedback
            feedback_samples = self.feedback_collector.get_batch(batch_size=10000)
            
            if len(feedback_samples) < self.retraining_pipeline.config.min_samples:
                return {
                    'success': False,
                    'reason': 'insufficient_samples',
                    'available': len(feedback_samples),
                    'required': self.retraining_pipeline.config.min_samples
                }
            
            # Convert feedback to dataset
            train_data = self._feedback_to_dataset(feedback_samples)
        
        # Perform retraining
        retrain_results = self.retraining_pipeline.retrain(train_data)
        
        # Reset components after retraining
        if retrain_results.get('improved', False):
            self.drift_detector.reset()
            self.validator.reset()
            self.learner.reset_statistics()
            self.drift_detected = False
        
        return retrain_results
    
    def _feedback_to_dataset(self, feedback_samples: list) -> torch.utils.data.TensorDataset:
        """Convert feedback samples to PyTorch dataset."""
        inputs = []
        labels = []
        
        for sample in feedback_samples:
            inputs.append(torch.tensor(sample['inputs']))
            labels.append(torch.tensor(sample['true_label']))
        
        inputs_tensor = torch.stack(inputs)
        labels_tensor = torch.stack(labels)
        
        return torch.utils.data.TensorDataset(inputs_tensor, labels_tensor)
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get statistics from all components."""
        return {
            'learner': self.learner.get_statistics(),
            'drift_detector': self.drift_detector.get_statistics(),
            'feedback_collector': self.feedback_collector.get_statistics(),
            'retraining_pipeline': self.retraining_pipeline.get_statistics(),
            'validator': self.validator.get_statistics(),
            'system_state': {
                'enabled': self.enabled,
                'drift_detected': self.drift_detected
            }
        }
    
    def enable(self):
        """Enable online learning."""
        self.enabled = True
        self.logger.info("Online learning enabled")
    
    def disable(self):
        """Disable online learning."""
        self.enabled = False
        self.logger.info("Online learning disabled")
    
    def save_state(self, path: str):
        """Save complete online learning state."""
        self.learner.save_checkpoint(f"{path}_learner.pt")
        self.feedback_collector.save_feedback()
        
        # Save statistics
        import json
        with open(f"{path}_stats.json", 'w') as f:
            json.dump(self.get_comprehensive_statistics(), f, indent=2)
        
        self.logger.info(f"Saved online learning state to {path}")
    
    def load_state(self, path: str):
        """Load online learning state."""
        try:
            self.learner.load_checkpoint(f"{path}_learner.pt")
            self.logger.info(f"Loaded online learning state from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
