"""
Feedback Collection

Collects feedback from security events, user labels, and system outputs
for continuous model improvement.
"""

import torch
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import logging
from pathlib import Path


class FeedbackType(Enum):
    """Feedback type options."""
    USER_LABEL = "user_label"  # Manual user labeling
    SECURITY_EVENT = "security_event"  # Confirmed security event
    FALSE_POSITIVE = "false_positive"  # False alarm
    FALSE_NEGATIVE = "false_negative"  # Missed threat
    SYSTEM_VALIDATION = "system_validation"  # Automated validation


class FeedbackCollector:
    """
    Collect and manage feedback for online learning.
    
    Features:
    - Multiple feedback sources
    - Feedback validation
    - Priority scoring
    - Batch collection
    - Persistent storage
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize feedback collector.
        
        Args:
            storage_path: Path to store feedback data
        """
        self.logger = logging.getLogger(__name__)
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Feedback buffer
        self.feedback_buffer = []
        
        # Statistics
        self.feedback_count = {ft: 0 for ft in FeedbackType}
        self.total_feedback = 0
        
        # Create storage directory
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_feedback()
        
        self.logger.info("Initialized feedback collector")
    
    def add_feedback(
        self,
        sample_id: str,
        inputs: torch.Tensor,
        prediction: torch.Tensor,
        true_label: torch.Tensor,
        feedback_type: FeedbackType,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add feedback sample.
        
        Args:
            sample_id: Unique sample identifier
            inputs: Input tensor
            prediction: Model prediction
            true_label: True label
            feedback_type: Type of feedback
            confidence: Confidence in feedback
            metadata: Additional metadata
        
        Returns:
            Feedback record
        """
        feedback = {
            'id': sample_id,
            'timestamp': datetime.now().isoformat(),
            'inputs': inputs.cpu().tolist() if isinstance(inputs, torch.Tensor) else inputs,
            'prediction': prediction.cpu().tolist() if isinstance(prediction, torch.Tensor) else prediction,
            'true_label': true_label.cpu().tolist() if isinstance(true_label, torch.Tensor) else true_label,
            'feedback_type': feedback_type.value,
            'confidence': confidence,
            'metadata': metadata or {}
        }
        
        self.feedback_buffer.append(feedback)
        self.feedback_count[feedback_type] += 1
        self.total_feedback += 1
        
        return feedback
    
    def get_batch(
        self,
        batch_size: int = 32,
        feedback_types: Optional[List[FeedbackType]] = None,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Get batch of feedback samples.
        
        Args:
            batch_size: Number of samples to return
            feedback_types: Filter by feedback types
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of feedback samples
        """
        # Filter feedback
        filtered = self.feedback_buffer
        
        if feedback_types:
            filtered = [
                f for f in filtered
                if FeedbackType(f['feedback_type']) in feedback_types
            ]
        
        if min_confidence > 0:
            filtered = [
                f for f in filtered
                if f['confidence'] >= min_confidence
            ]
        
        # Return batch
        return filtered[:batch_size]
    
    def clear_buffer(self, count: Optional[int] = None):
        """
        Clear feedback buffer.
        
        Args:
            count: Number of samples to remove (None = all)
        """
        if count is None:
            self.feedback_buffer.clear()
        else:
            self.feedback_buffer = self.feedback_buffer[count:]
    
    def save_feedback(self):
        """Save feedback to disk."""
        if not self.storage_path:
            return
        
        with open(self.storage_path, 'w') as f:
            json.dump({
                'feedback': self.feedback_buffer,
                'statistics': {
                    'total': self.total_feedback,
                    'by_type': {ft.value: count for ft, count in self.feedback_count.items()}
                }
            }, f, indent=2)
        
        self.logger.info(f"Saved {len(self.feedback_buffer)} feedback samples")
    
    def _load_feedback(self):
        """Load feedback from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.feedback_buffer = data.get('feedback', [])
                stats = data.get('statistics', {})
                self.total_feedback = stats.get('total', 0)
            
            self.logger.info(f"Loaded {len(self.feedback_buffer)} feedback samples")
        except Exception as e:
            self.logger.error(f"Failed to load feedback: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        return {
            'total_feedback': self.total_feedback,
            'buffer_size': len(self.feedback_buffer),
            'by_type': {ft.value: count for ft, count in self.feedback_count.items()},
            'has_storage': self.storage_path is not None
        }
