"""
Feature extraction from file access patterns.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Dict, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings

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
    
    SCALER_TYPES = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler
    }
    
    def __init__(self, scaler_type: str = 'standard'):
        self.scaler_type = scaler_type
        if scaler_type not in self.SCALER_TYPES:
            raise ValueError(f"Invalid scaler type. Choose from {list(self.SCALER_TYPES.keys())}")
        
        self.scaler = self.SCALER_TYPES[scaler_type]()
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
        return self.scaler.transform(data_2d).reshape(original_shape)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data back to original scale."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        original_shape = data.shape
        data_2d = data.reshape(-1, data.shape[-1])
        return self.scaler.inverse_transform(data_2d).reshape(original_shape)
    
    def get_feature_importance(self, model_weights: np.ndarray) -> Dict[str, float]:
        """
        Get relative importance of each feature based on model weights.
        
        Args:
            model_weights: Weights from a trained model
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if len(model_weights) != len(self.FEATURE_NAMES):
            warnings.warn(f"Model weights length ({len(model_weights)}) doesn't match "
                         f"feature count ({len(self.FEATURE_NAMES)}). Using first weights.")
            # Use only the first features if weights don't match
            weights = model_weights[:len(self.FEATURE_NAMES)]
        else:
            weights = model_weights
        
        # Calculate absolute importance
        importance = np.abs(weights)
        importance = importance / np.sum(importance)  # Normalize to sum to 1
        
        return {
            name: importance[i] 
            for i, name in enumerate(self.FEATURE_NAMES[:len(importance)])
        }
    
    def get_feature_statistics(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Get statistical summary of features in the provided data.
        
        Args:
            data: Input data array of shape (batch, seq_len, num_features)
            
        Returns:
            Dictionary with statistical summary for each feature
        """
        stats = {}
        
        for i, feature_name in enumerate(self.FEATURE_NAMES):
            feature_data = data[:, :, i].flatten()
            stats[feature_name] = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'median': float(np.median(feature_data)),
                'q25': float(np.percentile(feature_data, 25)),
                'q75': float(np.percentile(feature_data, 75)),
                'skewness': float(self._calculate_skewness(feature_data)),
                'kurtosis': float(self._calculate_kurtosis(feature_data))
            }
        return stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        scaled = self.scaler.transform(data_2d)
        return scaled.reshape(original_shape)
