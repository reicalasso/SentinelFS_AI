"""
More realistic file access data generator with complex patterns for better model training.
This generator creates data with more nuanced patterns that are harder to distinguish.
"""

import numpy as np
from typing import Tuple, Optional
import random

from ..data_types import AnomalyType
from ..utils.logger import get_logger

logger = get_logger(__name__)


def generate_realistic_access_data(
    num_samples: int = 1000,
    seq_len: int = 20,
    anomaly_ratio: float = 0.15,
    complexity_level: str = 'medium'  # 'low', 'medium', 'high'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate more realistic file access data with complex, nuanced patterns.
    
    Args:
        num_samples: Total number of sequences to generate
        seq_len: Length of each access sequence
        anomaly_ratio: Proportion of anomalous samples (default: 0.15 for realistic ratio)
        complexity_level: How complex the patterns should be ('low', 'medium', 'high')
        
    Returns:
        Tuple of (data, labels, anomaly_types) as numpy arrays
    """
    data = []
    labels = []
    anomaly_types = []
    
    # Adjust ratio based on complexity
    if complexity_level == 'high':
        anomaly_ratio = min(anomaly_ratio, 0.25)  # Higher ratio for complex data
    elif complexity_level == 'low':
        anomaly_ratio = min(anomaly_ratio, 0.10)  # Lower ratio for simple data
    
    num_normal = int(num_samples * (1 - anomaly_ratio))
    num_anomaly = num_samples - num_normal
    
    logger.info(f"Generating {num_normal} normal and {num_anomaly} anomalous sequences with {complexity_level} complexity")
    
    # Complexity multipliers for generating more nuanced data
    complexity_multipliers = {
        'low': 0.3,
        'medium': 0.6,
        'high': 1.0
    }
    complexity_mult = complexity_multipliers[complexity_level]
    
    # Generate normal behavior sequences
    for i in range(num_normal):
        seq = generate_normal_sequence(seq_len, complexity_mult)
        data.append(seq)
        labels.append([0])  # Normal
        anomaly_types.append(AnomalyType.NORMAL)
    
    # Generate anomalous behavior sequences with varying degrees of subtlety
    anomaly_counts = {
        AnomalyType.DATA_EXFILTRATION: 0,
        AnomalyType.RANSOMWARE: 0,
        AnomalyType.PRIVILEGE_ESCALATION: 0,
        AnomalyType.OTHER: 0
    }
    
    for i in range(num_anomaly):
        # Randomly select anomaly type
        anomaly_type = random.choice([
            AnomalyType.DATA_EXFILTRATION,
            AnomalyType.RANSOMWARE,
            AnomalyType.PRIVILEGE_ESCALATION,
            AnomalyType.OTHER
        ])
        
        # Generate appropriate anomalous sequence
        if anomaly_type == AnomalyType.DATA_EXFILTRATION:
            seq = generate_data_exfiltration_sequence(seq_len, complexity_mult)
        elif anomaly_type == AnomalyType.RANSOMWARE:
            seq = generate_ransomware_sequence(seq_len, complexity_mult)
        elif anomaly_type == AnomalyType.PRIVILEGE_ESCALATION:
            seq = generate_privilege_escalation_sequence(seq_len, complexity_mult)
        else:  # OTHER
            seq = generate_other_anomaly_sequence(seq_len, complexity_mult)
        
        data.append(seq)
        labels.append([1])  # Anomaly
        anomaly_types.append(anomaly_type)
        anomaly_counts[anomaly_type] += 1
    
    logger.info(f"Anomaly distribution: {anomaly_counts}")
    
    # Add background noise to make patterns less obvious
    data = add_background_noise(data, complexity_mult)
    
    # Convert to numpy arrays
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    anomaly_types = np.array(anomaly_types)
    
    # Shuffle the data to avoid any sequence bias
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]
    anomaly_types = anomaly_types[indices]
    
    logger.info(f"Generated {len(data)} sequences with shape {data.shape}")
    
    return data, labels, anomaly_types


def generate_normal_sequence(seq_len: int, complexity_mult: float) -> list:
    """Generate a normal access sequence with realistic patterns."""
    seq = []
    
    # Base normal patterns
    for t in range(seq_len):
        # Simulate more realistic working patterns
        hour = np.random.normal(13.5, 3.2)  # Peak around 1:30 PM, std of 3.2 hours
        hour = np.clip(hour, 0, 23)  # 24-hour format
        
        # Add some weekend patterns
        if np.random.random() < 0.2:  # 20% chance of weekend
            day_of_week = np.random.choice([5, 6])
        else:
            day_of_week = np.random.choice([0, 1, 2, 3, 4])  # Weekdays
        
        # Work patterns: more activity during business hours
        hour_factor = 1.0
        if 8 <= hour <= 18:  # Business hours
            hour_factor = 1.0
        elif 6 <= hour < 8 or 18 <= hour < 21:  # Before/after hours
            hour_factor = 0.5
        else:  # Night
            hour_factor = 0.1
        
        # File size with business context (log normal distribution)
        if t == 0:  # First access
            base_file_size = np.random.lognormal(2.0, 0.8)  # ~7.4 MB mean
        else:
            # Correlated with previous access (more realistic)
            prev_size = seq[-1][0] if seq else np.random.lognormal(2.0, 0.8)
            # Small variation from previous access
            size_variation = np.random.normal(0, 0.2)
            base_file_size = max(0.1, prev_size * (1 + size_variation))
        
        # Access type with realistic distribution
        access_type = np.random.choice(
            [0, 1, 2, 3], 
            p=[0.55, 0.25, 0.15, 0.05]  # Read is most common
        )
        
        # File category based on access patterns
        if access_type == 0:  # Read
            file_category = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])  # Documents, code, media
        elif access_type == 1:  # Write
            file_category = np.random.choice([0, 1], p=[0.6, 0.4])  # Documents, code
        elif access_type == 2:  # Delete
            file_category = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.2, 0.2])  # Mix
        else:  # Rename
            file_category = np.random.choice([0, 1], p=[0.6, 0.4])  # Documents, code
        
        # Access frequency and velocity based on context
        base_freq = np.random.normal(5, 1.5) * hour_factor  # More frequent during business hours
        access_frequency = max(0.1, base_freq + np.random.normal(0, 0.5))
        
        base_velocity = np.random.normal(2, 0.5) * hour_factor
        access_velocity = max(0.1, base_velocity + np.random.normal(0, 0.3))
        
        # Add some correlation between features for realism
        access_velocity *= (file_category + 1) * 0.8  # Larger categories might have higher velocity
        
        seq.append([
            base_file_size, hour, access_type, day_of_week,
            access_frequency, file_category, access_velocity
        ])
    
    return seq


def generate_data_exfiltration_sequence(seq_len: int, complexity_mult: float) -> list:
    """Generate subtle data exfiltration patterns that may not be obvious."""
    seq = []
    
    # Vary the pattern based on complexity
    if complexity_mult < 0.5:  # Low complexity - obvious patterns
        for t in range(seq_len):
            hour = np.random.uniform(0, 6)  # Always night
            file_size = np.random.lognormal(4.0, 0.6)  # Always large
            access_type = 0  # Always read
            day_of_week = np.random.choice([5, 6], p=[0.7, 0.3])  # Weekend
            access_frequency = np.random.normal(20, 5)  # High
            file_category = np.random.choice([0, 1, 2])  # Sensitive
            access_velocity = np.random.normal(10, 2)  # High
            
            seq.append([
                file_size, hour, access_type, day_of_week,
                access_frequency, file_category, access_velocity
            ])
    else:  # Higher complexity - more subtle patterns
        # Create a sequence that starts normal and becomes suspicious
        normal_duration = max(1, int(seq_len * 0.3))  # 30% normal start
        
        for t in range(seq_len):
            if t < normal_duration:  # Start with normal patterns
                # Mix of normal and suspicious features
                hour = np.random.normal(13.5, 3.2)
                hour = np.clip(hour, 6, 20)  # Keep in reasonable business hours
                file_size = np.random.lognormal(2.2, 0.6)  # Similar to normal
                access_type = np.random.choice([0, 1], p=[0.6, 0.4])  # Mostly read/write
                day_of_week = np.random.choice([0, 1, 2, 3, 4])  # Usually weekday
                access_frequency = np.random.normal(5, 1.5)
                file_category = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
                access_velocity = np.random.normal(2, 0.5)
            else:  # Then suspicious patterns
                hour = np.random.uniform(0, 24)  # Any time
                if complexity_mult < 0.8:  # Medium - more obvious
                    file_size = np.random.lognormal(3.8, 0.5)  # Larger
                    access_frequency = np.random.normal(12, 4)  # Higher
                    access_velocity = np.random.normal(7, 2)  # Higher
                else:  # High complexity - subtle
                    # Mix of normal and suspicious features
                    hour_variation = np.random.uniform(0, 24)
                    file_size = np.random.lognormal(
                        2.4 + (complexity_mult - 0.5),  # Subtle increase
                        0.6
                    )
                    access_type = np.random.choice([0, 1], p=[0.7, 0.3])  # Still mainly read
                    day_of_week = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=[0.15]*7)
                    access_frequency = np.random.normal(6 + (complexity_mult * 4), 2)
                    file_category = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
                    access_velocity = np.random.normal(3 + (complexity_mult * 3), 1)
            
            seq.append([
                file_size, hour, access_type, day_of_week,
                access_frequency, file_category, access_velocity
            ])
    
    return seq


def generate_ransomware_sequence(seq_len: int, complexity_mult: float) -> list:
    """Generate subtle ransomware patterns that may not be obvious."""
    seq = []
    
    if complexity_mult < 0.5:  # Low complexity - obvious patterns
        for t in range(seq_len):
            hour = np.random.uniform(0, 24)  # Any time
            file_size = np.random.lognormal(1.8, 0.5)  # Various sizes
            access_type = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])  # Write/Delete/Rename
            day_of_week = np.random.choice(range(7))
            access_frequency = np.random.normal(50, 15)  # Very high
            file_category = np.random.choice([0, 1, 2, 3, 4])  # All types
            access_velocity = np.random.normal(20, 6)  # Very high
            
            seq.append([
                file_size, hour, access_type, day_of_week,
                access_frequency, file_category, access_velocity
            ])
    else:  # Higher complexity - more subtle patterns
        # Ransomware often starts with reconnaissance - generate accordingly
        for t in range(seq_len):
            if t < seq_len // 3:  # First third: reconnaissance phase
                hour = np.random.normal(13.5, 3.2)
                file_size = np.random.lognormal(2.0, 0.5)  # Normal looking
                access_type = 0  # Mainly reads
                day_of_week = np.random.choice([0, 1, 2, 3, 4])
                access_frequency = np.random.normal(8, 2)  # Slightly higher
                file_category = np.random.choice([0, 1, 2, 3, 4], p=[0.2]*5)
                access_velocity = np.random.normal(3, 0.8)  # Slightly higher
            else:  # Encryption phase
                if complexity_mult < 0.8:  # Medium complexity
                    hour = np.random.uniform(0, 24)
                    file_size = np.random.lognormal(1.8, 0.5)
                    access_type = np.random.choice([1, 2, 3], p=[0.4, 0.3, 0.3])
                    day_of_week = np.random.choice(range(7))
                    access_frequency = np.random.normal(35, 10)
                    file_category = np.random.choice([0, 1, 2, 3, 4], p=[0.2]*5)
                    access_velocity = np.random.normal(15, 4)
                else:  # High complexity - subtle
                    # Mix of read/write to make it harder to detect
                    hour = np.random.uniform(0, 24)
                    file_size = np.random.lognormal(2.0, 0.5)  # Normal looking
                    access_type = np.random.choice([0, 1, 2, 3], 
                                                 p=[0.3, 0.35, 0.2, 0.15])  # Still some reads
                    day_of_week = np.random.choice(range(7))
                    access_frequency = np.random.normal(15, 5)  # Higher but not extreme
                    file_category = np.random.choice([0, 1, 2, 3, 4], p=[0.2]*5)
                    access_velocity = np.random.normal(8, 2.5)  # Higher but not extreme
            
            seq.append([
                file_size, hour, access_type, day_of_week,
                access_frequency, file_category, access_velocity
            ])
    
    return seq


def generate_privilege_escalation_sequence(seq_len: int, complexity_mult: float) -> list:
    """Generate subtle privilege escalation patterns."""
    seq = []
    
    if complexity_mult < 0.5:  # Obvious patterns
        for t in range(seq_len):
            hour = np.random.uniform(0, 24)
            file_size = np.random.lognormal(1.5, 0.4)  # Often small system files
            access_type = np.random.choice([1, 3], p=[0.7, 0.3])  # Write/Rename
            day_of_week = np.random.choice(range(7))
            access_frequency = np.random.normal(25, 5)  # High
            file_category = np.random.choice([3, 4], p=[0.6, 0.4])  # System files
            access_velocity = np.random.normal(15, 3)  # High
            
            seq.append([
                file_size, hour, access_type, day_of_week,
                access_frequency, file_category, access_velocity
            ])
    else:  # Subtle patterns
        # Mimic legitimate admin activity that might be suspicious
        for t in range(seq_len):
            hour = np.random.uniform(0, 24)
            file_size = np.random.lognormal(1.8, 0.6)  # Mix of small/large
            access_type = np.random.choice([0, 1, 2, 3], 
                                         p=[0.2, 0.4, 0.2, 0.2])  # Various access types
            day_of_week = np.random.choice(range(7))
            
            if complexity_mult < 0.8:  # Medium complexity
                access_frequency = np.random.normal(20, 6)
                file_category = np.random.choice([2, 3, 4], p=[0.2, 0.5, 0.3])  # More system files
                access_velocity = np.random.normal(12, 4)
            else:  # High complexity - very subtle
                # Try to blend in with normal activity
                access_frequency = np.random.normal(8, 3)  # Not extremely high
                file_category = np.random.choice([0, 1, 2, 3, 4], 
                                               p=[0.15, 0.2, 0.2, 0.25, 0.2])  # Mix
                access_velocity = np.random.normal(4, 1.5)  # Closer to normal
            
            seq.append([
                file_size, hour, access_type, day_of_week,
                access_frequency, file_category, access_velocity
            ])
    
    return seq


def generate_other_anomaly_sequence(seq_len: int, complexity_mult: float) -> list:
    """Generate other types of anomalous patterns."""
    seq = []
    
    # Choose pattern type based on complexity
    pattern_type = np.random.choice(['resource_exhaustion', 'unusual_pattern', 'slow_anomaly'])
    
    for t in range(seq_len):
        if pattern_type == 'resource_exhaustion':
            # Gradually increasing resource usage
            if complexity_mult < 0.8:
                # Obvious pattern
                access_frequency = 5 + (t * 8)  # Increasing over time
                access_velocity = 2 + (t * 0.5)
            else:
                # Subtle pattern
                base_freq = 5 + (t * 2)  # Slower increase
                access_frequency = base_freq + np.random.normal(0, 2)
                access_velocity = 2 + (t * 0.2) + np.random.normal(0, 0.5)
                
        elif pattern_type == 'unusual_pattern':
            if complexity_mult < 0.8:
                # Obvious unusual pattern
                hour = np.random.choice([1, 2, 3, 4], p=[0.25]*4)  # Very specific hours
                access_type = np.random.choice([0, 1, 2, 3], p=[0.1, 0.1, 0.7, 0.1])  # Mostly delete
                access_frequency = np.random.normal(35, 8)
            else:
                # Subtle unusual pattern
                hour = np.random.choice([1, 2, 3, 4, 12, 13], p=[0.15, 0.15, 0.15, 0.15, 0.2, 0.2])
                access_type = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.3, 0.1])
                access_frequency = np.random.normal(12, 4)
                
        else:  # 'slow_anomaly'
            # Slowly developing anomaly
            hour = np.random.uniform(0, 24)
            if t > seq_len * 0.6:  # Anomaly develops in later part
                file_size = np.random.lognormal(3.0 + (complexity_mult * 0.5), 0.7)
                access_frequency = np.random.normal(15 + (complexity_mult * 5), 4)
                access_velocity = np.random.normal(6 + (complexity_mult * 2), 2)
            else:
                # Start with normal patterns
                file_size = np.random.lognormal(2.0, 0.6)
                access_frequency = np.random.normal(6, 2)
                access_velocity = np.random.normal(2.5, 0.8)
        
        # Fill in remaining features
        hour = getattr(locals(), 'hour', np.random.uniform(0, 24))
        access_frequency = max(0.1, getattr(locals(), 'access_frequency', 
                                          np.random.normal(5, 2)))
        access_velocity = max(0.1, getattr(locals(), 'access_velocity', 
                                         np.random.normal(2, 0.5)))
        
        file_size = np.random.lognormal(2.0, 0.6)
        access_type = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
        day_of_week = np.random.choice(range(7))
        file_category = np.random.choice([0, 1, 2, 3, 4], p=[0.2]*5)
        
        seq.append([
            file_size, hour, access_type, day_of_week,
            access_frequency, file_category, access_velocity
        ])
    
    return seq


def add_background_noise(data: list, complexity_mult: float) -> list:
    """Add subtle noise to make patterns less obvious."""
    noise_factor = 0.1 * complexity_mult  # Lower complexity gets more noise
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            # Add small random variations to each feature
            for k in range(len(data[i][j])):
                noise = np.random.normal(0, noise_factor)
                data[i][j][k] = max(0, data[i][j][k] + noise)
    
    return data