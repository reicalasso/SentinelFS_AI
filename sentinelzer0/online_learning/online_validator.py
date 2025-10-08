"""
Online Validator

Real-time model validation and performance monitoring.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from collections import deque
import logging
import numpy as np


@dataclass
class ValidationMetrics:
    """Validation metrics container."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float
    sample_count: int


class OnlineValidator:
    """
    Online model validation system.
    
    Features:
    - Real-time performance tracking
    - Rolling window metrics
    - Performance degradation detection
    - Multi-metric validation
    - Threshold-based alerting
    """
    
    def __init__(
        self,
        model: nn.Module,
        window_size: int = 1000,
        min_accuracy: float = 0.7,
        min_f1: float = 0.6
    ):
        """
        Initialize online validator.
        
        Args:
            model: Model to validate
            window_size: Size of rolling window
            min_accuracy: Minimum acceptable accuracy
            min_f1: Minimum acceptable F1 score
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.window_size = window_size
        self.min_accuracy = min_accuracy
        self.min_f1 = min_f1
        
        # Sliding windows
        self.predictions_window = deque(maxlen=window_size)
        self.labels_window = deque(maxlen=window_size)
        self.losses_window = deque(maxlen=window_size)
        
        # Statistics
        self.total_samples = 0
        self.alert_count = 0
        
        self.logger.info("Initialized online validator")
    
    def validate_sample(
        self,
        inputs: torch.Tensor,
        true_labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Validate on single sample/batch.
        
        Args:
            inputs: Input tensor
            true_labels: True labels
        
        Returns:
            Validation results
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)
            
            # Calculate loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, true_labels)
        
        # Add to windows
        for pred, label in zip(predicted.cpu().numpy(), true_labels.cpu().numpy()):
            self.predictions_window.append(pred)
            self.labels_window.append(label)
            self.losses_window.append(loss.item())
            self.total_samples += 1
        
        # Calculate metrics
        metrics = self.get_current_metrics()
        
        # Check for performance degradation
        alert = self._check_performance_degradation(metrics)
        
        return {
            'metrics': metrics,
            'alert': alert,
            'total_samples': self.total_samples
        }
    
    def get_current_metrics(self) -> ValidationMetrics:
        """Get current validation metrics."""
        if not self.predictions_window:
            return ValidationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        
        predictions = np.array(list(self.predictions_window))
        labels = np.array(list(self.labels_window))
        losses = np.array(list(self.losses_window))
        
        # Calculate metrics
        correct = (predictions == labels).sum()
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate precision, recall, F1 (binary classification)
        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_loss = losses.mean()
        
        return ValidationMetrics(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            loss=float(avg_loss),
            sample_count=total
        )
    
    def _check_performance_degradation(self, metrics: ValidationMetrics) -> bool:
        """Check if performance has degraded below thresholds."""
        alert = (
            metrics.accuracy < self.min_accuracy or
            metrics.f1_score < self.min_f1
        )
        
        if alert:
            self.alert_count += 1
            self.logger.warning(
                f"Performance degradation detected: "
                f"acc={metrics.accuracy:.3f}, f1={metrics.f1_score:.3f}"
            )
        
        return alert
    
    def reset(self):
        """Reset validation state."""
        self.predictions_window.clear()
        self.labels_window.clear()
        self.losses_window.clear()
        self.total_samples = 0
        
        self.logger.info("Reset online validator")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        metrics = self.get_current_metrics()
        
        return {
            'current_metrics': {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'loss': metrics.loss
            },
            'total_samples': self.total_samples,
            'window_size': len(self.predictions_window),
            'alert_count': self.alert_count
        }
