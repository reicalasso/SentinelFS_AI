"""
Generate synthetic file access data for training and testing.
"""

import numpy as np
from typing import Tuple, Optional

from ..data_types import AnomalyType
from ..utils.logger import get_logger

logger = get_logger(__name__)


def generate_sample_data(
    num_samples: int = 100, 
    seq_len: int = 10, 
    anomaly_ratio: float = 0.2, 
    include_anomaly_types: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Generate synthetic file access data for training and testing.
    
    Creates realistic access patterns with both normal and anomalous behavior:
    - Normal: Business hours access, typical file sizes, regular patterns
    - Anomalous: Off-hours access, large files, suspicious patterns
    
    Args:
        num_samples: Total number of sequences to generate
        seq_len: Length of each access sequence
        anomaly_ratio: Proportion of anomalous samples (default: 0.2)
        include_anomaly_types: Whether to return anomaly type labels
        
    Returns:
        Tuple of (data, labels, [anomaly_types]) as numpy arrays
        - data: shape (num_samples, seq_len, num_features)
        - labels: shape (num_samples, 1) with 0=normal, 1=anomaly
        - anomaly_types: shape (num_samples,) with specific anomaly type IDs (if requested)
    """
    data = []
    labels = []
    anomaly_types = [] if include_anomaly_types else None
    
    num_normal = int(num_samples * (1 - anomaly_ratio))
    num_anomaly = num_samples - num_normal
    
    logger.info(f"Generating {num_normal} normal and {num_anomaly} anomalous sequences")
    
    # Generate normal behavior sequences
    for _ in range(num_normal):
        seq = []
        for _ in range(seq_len):
            # Normal business hours access
            file_size = np.random.lognormal(2.3, 0.5)  # ~10 MB average
            hour = np.random.normal(13, 3)  # Peak around 1 PM
            hour = np.clip(hour, 8, 18)  # Business hours
            access_type = np.random.choice([0, 1, 2, 3], p=[0.6, 0.25, 0.1, 0.05])
            day_of_week = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.2, 0.2, 0.2, 0.2])
            access_frequency = np.random.normal(5, 2)  # Normal activity
            file_category = np.random.choice([0, 1, 2, 3, 4])  # Document types
            access_velocity = np.random.normal(2, 0.5)  # Normal speed
            
            seq.append([
                file_size, hour, access_type, day_of_week, 
                access_frequency, file_category, access_velocity
            ])
        data.append(seq)
        labels.append([0])  # Normal
        if include_anomaly_types:
            anomaly_types.append(AnomalyType.NORMAL)
    
    # Generate anomalous behavior sequences with specific types
    anomaly_patterns = {
        'data_exfiltration': AnomalyType.DATA_EXFILTRATION,
        'ransomware': AnomalyType.RANSOMWARE,
        'privilege_escalation': AnomalyType.PRIVILEGE_ESCALATION,
        'other': AnomalyType.OTHER
    }
    
    for _ in range(num_anomaly):
        seq = []
        anomaly_pattern = np.random.choice(list(anomaly_patterns.keys()))
        
        for _ in range(seq_len):
            if anomaly_pattern == 'data_exfiltration':
                # Large file transfers during off-hours
                file_size = np.random.lognormal(4.0, 0.6)  # Large files (50+ MB)
                hour = np.random.uniform(0, 6)  # Night access
                access_type = 0  # Read operations (exfiltrating)
                day_of_week = np.random.choice([5, 6], p=[0.6, 0.4])  # Weekend
                access_frequency = np.random.normal(15, 4)  # High frequency
                file_category = np.random.choice([0, 1, 2])  # Sensitive files
                access_velocity = np.random.normal(8, 2)  # High velocity
                
            elif anomaly_pattern == 'ransomware':
                # Rapid file modifications/encryptions
                file_size = np.random.lognormal(2.0, 0.5)  # Various sizes
                hour = np.random.uniform(0, 24)
                access_type = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])  # Write/Delete/Rename
                day_of_week = np.random.choice(range(7))
                access_frequency = np.random.normal(60, 15)  # Very high frequency
                file_category = np.random.choice([0, 1, 2, 3, 4])  # All file types
                access_velocity = np.random.normal(25, 8)  # Very high velocity
                
            elif anomaly_pattern == 'privilege_escalation':
                # Unusual administrative access patterns
                file_size = np.random.lognormal(1.8, 0.4)  # Small system files
                hour = np.random.uniform(0, 24)
                access_type = np.random.choice([1, 3], p=[0.6, 0.4])  # Write/Rename
                day_of_week = np.random.choice(range(7))
                access_frequency = np.random.normal(20, 5)
                file_category = np.random.choice([3, 4])  # System files
                access_velocity = np.random.normal(12, 3)
                
            else:  # other anomalies
                # Mixed unusual behaviors
                file_size = np.random.lognormal(3.0, 1.0)
                hour = np.random.uniform(0, 24)
                access_type = np.random.choice([0, 1, 2, 3])
                day_of_week = np.random.choice(range(7))
                access_frequency = np.random.normal(30, 10)
                file_category = np.random.choice([0, 1, 2, 3, 4])
                access_velocity = np.random.normal(15, 5)
            
            seq.append([
                file_size, hour, access_type, day_of_week,
                access_frequency, file_category, access_velocity
            ])
        data.append(seq)
        labels.append([1])  # Anomaly
        if include_anomaly_types:
            anomaly_types.append(anomaly_patterns[anomaly_pattern])
    
    # Shuffle the data
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]
    
    if include_anomaly_types:
        anomaly_types = np.array(anomaly_types)[indices]
    
    logger.info(f"Generated {len(data)} sequences with shape {data.shape}")
    
    if include_anomaly_types:
        return data, labels, anomaly_types
    return data, labels, None
