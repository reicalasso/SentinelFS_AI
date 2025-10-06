"""
Feature extraction from file access patterns.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """
    Extract comprehensive features from file access patterns.
    
    Features extracted:
        - File size (MB)
        - Access hour (0-23)
        - Access type (0=read, 1=write, 2=delete, 3=rename)
        - Day of week (0-6)
        - Is weekend (0/1)
        - Access frequency (per hour)
        - File extension category (encoded)
        - User access velocity (files per minute)
    """
    
    FEATURE_NAMES = [
        'file_size_mb',
        'access_hour',
        'access_type',
        'day_of_week',
        'is_weekend',
        'access_frequency',
        'file_category',
        'access_velocity'
    ]
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def extract_features(self, access_events: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract features from a sequence of access events.
        
        Args:
            access_events: List of access event dictionaries
            
        Returns:
            Feature matrix of shape (seq_len, num_features)
        """
        features = []
        
        for event in access_events:
            feature_vector = [
                event.get('file_size_mb', 0),
                event.get('access_hour', 0),
                event.get('access_type', 0),
                event.get('day_of_week', 0),
                1 if event.get('day_of_week', 0) >= 5 else 0,  # is_weekend
                event.get('access_frequency', 0),
                event.get('file_category', 0),
                event.get('access_velocity', 0)
            ]
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data."""
        original_shape = data.shape
        # Reshape to 2D for scaling
        data_2d = data.reshape(-1, data.shape[-1])
        scaled = self.scaler.fit_transform(data_2d)
        self.fitted = True
        # Reshape back to original
        return scaled.reshape(original_shape)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        original_shape = data.shape
        data_2d = data.reshape(-1, data.shape[-1])
        scaled = self.scaler.transform(data_2d)
        return scaled.reshape(original_shape)
