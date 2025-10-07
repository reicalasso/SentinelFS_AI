"""
Generate synthetic file access data for training and testing.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union
import random
from datetime import datetime, timedelta
import pandas as pd

from ..data_types import AnomalyType
from ..utils.logger import get_logger

logger = get_logger(__name__)


def generate_sample_data(
    num_samples: int = 1000, 
    seq_len: int = 20, 
    anomaly_ratio: float = 0.2, 
    include_anomaly_types: bool = True,
    seed: Optional[int] = None,
    complexity_level: str = 'medium',
    user_profile_distribution: Optional[Dict] = None,
    seasonal_variation: bool = True,
    temporal_dependencies: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Generate synthetic file access data for training and testing.
    
    Creates realistic access patterns with both normal and anomalous behavior:
    - Normal: Business hours access, typical file sizes, regular patterns
    - Anomalous: Off-hours access, large files, suspicious patterns
    
    Args:
        num_samples: Total number of sequences to generate
        seq_len: Length of each access sequence (default: 20)
        anomaly_ratio: Proportion of anomalous samples (default: 0.2)
        include_anomaly_types: Whether to return anomaly type labels
        seed: Random seed for reproducible results
        complexity_level: How complex the patterns should be ('low', 'medium', 'high')
        user_profile_distribution: Distribution of different user behavior profiles
        seasonal_variation: Whether to include seasonal patterns
        temporal_dependencies: Whether to include temporal dependencies
        
    Returns:
        Tuple of (data, labels, [anomaly_types]) as numpy arrays
        - data: shape (num_samples, seq_len, num_features)
        - labels: shape (num_samples, 1) with 0=normal, 1=anomaly
        - anomaly_types: shape (num_samples,) with specific anomaly type IDs (if requested)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Define user behavior profiles
    user_profiles = user_profile_distribution or {
        'developer': 0.25,
        'analyst': 0.25,
        'executive': 0.15,
        'admin': 0.15,
        'contractor': 0.15,
        'temporary': 0.05
    }
    
    complexity_multipliers = {
        'low': 0.3,
        'medium': 0.6,
        'high': 1.0
    }
    complexity_mult = complexity_multipliers.get(complexity_level, 0.6)
    
    data = []
    labels = []
    anomaly_types = [] if include_anomaly_types else None
    
    num_normal = int(num_samples * (1 - anomaly_ratio))
    num_anomaly = num_samples - num_normal
    
    logger.info(f"Generating {num_normal} normal and {num_anomaly} anomalous sequences "
                f"with sequence length {seq_len} at {complexity_level} complexity")
    
    # Generate normal behavior sequences
    for i in range(num_normal):
        # Select user profile based on distribution
        profile = random.choices(
            list(user_profiles.keys()),
            weights=list(user_profiles.values())
        )[0]
        
        seq = generate_normal_sequence(
            seq_len=seq_len,
            user_profile=profile,
            complexity_mult=complexity_mult,
            seasonal_variation=seasonal_variation,
            temporal_dependencies=temporal_dependencies
        )
        data.append(seq)
        labels.append([0])  # Normal
        if include_anomaly_types:
            anomaly_types.append(AnomalyType.NORMAL)
    
    # Generate anomalous behavior sequences with specific types
    anomaly_counts = {
        AnomalyType.DATA_EXFILTRATION: 0,
        AnomalyType.RANSOMWARE: 0,
        AnomalyType.PRIVILEGE_ESCALATION: 0,
        AnomalyType.OTHER: 0
    }
    
    for i in range(num_anomaly):
        # Randomly select anomaly type based on distribution
        anomaly_type = random.choice([
            AnomalyType.DATA_EXFILTRATION,
            AnomalyType.RANSOMWARE,
            AnomalyType.PRIVILEGE_ESCALATION,
            AnomalyType.OTHER
        ])
        
        # Select user profile for context
        profile = random.choices(
            list(user_profiles.keys()),
            weights=list(user_profiles.values())
        )[0]
        
        # Generate appropriate anomalous sequence
        if anomaly_type == AnomalyType.DATA_EXFILTRATION:
            seq = generate_data_exfiltration_sequence(
                seq_len=seq_len,
                user_profile=profile,
                complexity_mult=complexity_mult,
                seasonal_variation=seasonal_variation,
                temporal_dependencies=temporal_dependencies
            )
        elif anomaly_type == AnomalyType.RANSOMWARE:
            seq = generate_ransomware_sequence(
                seq_len=seq_len,
                user_profile=profile,
                complexity_mult=complexity_mult,
                seasonal_variation=seasonal_variation,
                temporal_dependencies=temporal_dependencies
            )
        elif anomaly_type == AnomalyType.PRIVILEGE_ESCALATION:
            seq = generate_privilege_escalation_sequence(
                seq_len=seq_len,
                user_profile=profile,
                complexity_mult=complexity_mult,
                seasonal_variation=seasonal_variation,
                temporal_dependencies=temporal_dependencies
            )
        else:  # OTHER
            seq = generate_other_anomaly_sequence(
                seq_len=seq_len,
                user_profile=profile,
                complexity_mult=complexity_mult,
                seasonal_variation=seasonal_variation,
                temporal_dependencies=temporal_dependencies
            )
        
        data.append(seq)
        labels.append([1])  # Anomaly
        if include_anomaly_types:
            anomaly_types.append(anomaly_type)
        anomaly_counts[anomaly_type] += 1
    
    logger.info(f"Anomaly type distribution: {dict(anomaly_counts)}")
    
    # Convert to numpy arrays
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    if include_anomaly_types:
        anomaly_types = np.array(anomaly_types)
    
    # Shuffle the data to avoid any sequence bias
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]
    if include_anomaly_types:
        anomaly_types = anomaly_types[indices]
    
    logger.info(f"Generated {len(data)} sequences with shape {data.shape}")
    
    return data, labels, anomaly_types


def generate_normal_sequence(
    seq_len: int,
    user_profile: str = 'general',
    complexity_mult: float = 0.6,
    seasonal_variation: bool = True,
    temporal_dependencies: bool = True
) -> List[List[float]]:
    """
    Generate a normal access sequence with realistic patterns.
    
    Args:
        seq_len: Length of the sequence
        user_profile: Type of user profile to simulate
        complexity_mult: Multiplier for pattern complexity
        seasonal_variation: Whether to include seasonal patterns
        temporal_dependencies: Whether to include temporal dependencies
        
    Returns:
        List of access events (each with 7 features)
    """
    seq = []
    
    # Profile-specific parameters
    profile_params = {
        'developer': {
            'business_hours_factor': 1.1,
            'weekend_factor': 0.3,
            'file_size_factor': 1.2,
            'access_freq_mean': 8,
            'access_freq_std': 3,
            'activity_distribution': [0.4, 0.4, 0.1, 0.1],  # [read, write, delete, execute]
            'preferred_hours': [9, 10, 11, 14, 15, 16, 17, 18]
        },
        'analyst': {
            'business_hours_factor': 1.0,
            'weekend_factor': 0.1,
            'file_size_factor': 1.0,
            'access_freq_mean': 6,
            'access_freq_std': 2,
            'activity_distribution': [0.6, 0.25, 0.1, 0.05],
            'preferred_hours': [9, 10, 11, 13, 14, 15, 16]
        },
        'executive': {
            'business_hours_factor': 0.9,
            'weekend_factor': 0.05,
            'file_size_factor': 0.8,
            'access_freq_mean': 4,
            'access_freq_std': 1.5,
            'activity_distribution': [0.7, 0.2, 0.08, 0.02],
            'preferred_hours': [8, 9, 10, 15, 16]
        },
        'admin': {
            'business_hours_factor': 0.8,
            'weekend_factor': 0.2,
            'file_size_factor': 0.9,
            'access_freq_mean': 12,
            'access_freq_std': 5,
            'activity_distribution': [0.3, 0.5, 0.15, 0.05],
            'preferred_hours': [2, 3, 4, 5, 6, 20, 21, 22, 23]  # Off-hours maintenance
        },
        'contractor': {
            'business_hours_factor': 1.2,
            'weekend_factor': 0.02,
            'file_size_factor': 0.7,
            'access_freq_mean': 3,
            'access_freq_std': 1,
            'activity_distribution': [0.65, 0.25, 0.08, 0.02],
            'preferred_hours': [10, 11, 12, 13, 14, 15]
        },
        'temporary': {
            'business_hours_factor': 1.0,
            'weekend_factor': 0.01,
            'file_size_factor': 0.6,
            'access_freq_mean': 2,
            'access_freq_std': 0.5,
            'activity_distribution': [0.8, 0.15, 0.04, 0.01],
            'preferred_hours': [9, 10, 14, 15]
        },
        'general': {
            'business_hours_factor': 1.0,
            'weekend_factor': 0.15,
            'file_size_factor': 1.0,
            'access_freq_mean': 5,
            'access_freq_std': 2,
            'activity_distribution': [0.55, 0.3, 0.1, 0.05],
            'preferred_hours': [9, 10, 11, 13, 14, 15, 16]
        }
    }
    
    params = profile_params.get(user_profile, profile_params['general'])
    
    # Base normal patterns
    for t in range(seq_len):
        # Simulate more realistic working patterns based on profile
        if np.random.random() < params['weekend_factor']:
            # Weekend access for this profile type
            day_of_week = np.random.choice([5, 6])
        else:
            day_of_week = np.random.choice([0, 1, 2, 3, 4])  # Weekdays
        
        # Determine hour based on profile preferences and business hours
        if day_of_week < 5:  # Weekday
            # More likely to access during profile's preferred hours
            if np.random.random() < params['business_hours_factor']:
                hour = np.random.choice(params['preferred_hours'])
            else:
                # Access outside preferred hours
                all_hours = [i for i in range(24) if i not in params['preferred_hours']]
                hour = np.random.choice(all_hours)
        else:  # Weekend
            # Weekend access based on profile's weekend behavior
            if np.random.random() < params['weekend_factor']:
                hour = np.random.choice(params['preferred_hours'])
            else:
                hour = np.random.uniform(10, 16)  # Normal weekend working hours
        
        # Apply random variation to hour
        hour = hour + np.random.normal(0, 1.0)  # Add some randomness
        hour = np.clip(hour, 0, 23)  # Keep in 24-hour range
        
        # File size with business context (log normal distribution)
        if t == 0:  # First access
            base_file_size = np.random.lognormal(2.0, 0.8)  # ~7.4 MB mean
        else:
            # Correlated with previous access (more realistic)
            prev_size = seq[-1][0] if seq else np.random.lognormal(2.0, 0.8)
            # Small variation from previous access
            size_variation = np.random.normal(0, 0.2 * complexity_mult)
            base_file_size = max(0.1, prev_size * (1 + size_variation))
        
        # Apply profile-specific file size factor
        base_file_size = base_file_size * params['file_size_factor']
        
        # Access type with realistic distribution for profile
        access_type = np.random.choice(
            [0, 1, 2, 3], 
            p=params['activity_distribution']
        )
        
        # File category based on access patterns and profile
        if access_type == 0:  # Read - more likely to read documents/media for analysts, code for developers
            if user_profile == 'developer':
                file_category = np.random.choice([0, 1], p=[0.3, 0.7])  # Documents, code
            elif user_profile in ['analyst', 'executive']:
                file_category = np.random.choice([0, 2], p=[0.6, 0.4])  # Documents, media
            else:
                file_category = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])  # Mix
        elif access_type == 1:  # Write
            if user_profile == 'admin':
                file_category = np.random.choice([3, 5, 7], p=[0.5, 0.4, 0.1])  # System, config, executable
            else:
                file_category = np.random.choice([0, 1], p=[0.5, 0.5])  # Documents, code
        else:  # Delete, Execute
            file_category = np.random.choice([3, 4, 5], p=[0.4, 0.3, 0.3])  # System, database, config
        
        # Access frequency and velocity based on context and profile
        base_freq = np.random.normal(params['access_freq_mean'], params['access_freq_std'])
        # Apply seasonal variation if enabled
        if seasonal_variation:
            seasonal_factor = 1.0 + (np.sin(2 * np.pi * (t % 365) / 365) * 0.1)
            base_freq = base_freq * seasonal_factor
        
        access_frequency = max(0.1, base_freq + np.random.normal(0, 0.5 * complexity_mult))
        
        base_velocity = np.random.normal(2, 0.5) * (base_freq / 5.0)  # Correlated with frequency
        access_velocity = max(0.1, base_velocity + np.random.normal(0, 0.3 * complexity_mult))
        
        # Add some correlation between features for realism
        access_velocity *= (file_category + 1) * 0.8  # Larger categories might have higher velocity
        
        seq.append([
            base_file_size, hour, access_type, day_of_week,
            access_frequency, file_category, access_velocity
        ])
    
    return seq


def generate_data_exfiltration_sequence(
    seq_len: int,
    user_profile: str = 'general',
    complexity_mult: float = 0.6,
    seasonal_variation: bool = True,
    temporal_dependencies: bool = True
) -> List[List[float]]:
    """
    Generate a data exfiltration pattern.
    
    Args:
        seq_len: Length of the sequence
        user_profile: Type of user profile for context
        complexity_mult: Multiplier for pattern complexity
        seasonal_variation: Whether to include seasonal patterns
        temporal_dependencies: Whether to include temporal dependencies
        
    Returns:
        List of access events with exfiltration patterns
    """
    seq = []
    
    # For exfiltration, start with normal patterns and become suspicious
    initial_normal_ratio = 0.3  # 30% normal behavior initially
    
    for t in range(seq_len):
        if t < seq_len * initial_normal_ratio:
            # Start with normal patterns for cover
            hour = np.random.normal(13.5, 2.5)  # Business hours
            hour = np.clip(hour, 8, 18)  # Business hours
            file_size = np.random.lognormal(2.2, 0.6)  # Normal size
            access_type = 0  # Read (normal)
            day_of_week = np.random.choice([0, 1, 2, 3, 4])  # Weekday
            access_frequency = np.random.normal(5, 1.5)  # Normal frequency
            file_category = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])  # Normal categories
            access_velocity = np.random.normal(2, 0.5)  # Normal velocity
        else:
            # Exfiltration patterns
            # High probability of off-hours access
            hour = np.random.uniform(0, 6)  # 12 AM to 6 AM
            file_size = np.random.lognormal(4.0, 0.6)  # Large files (50+ MB)
            access_type = 0  # Always read for exfiltration
            day_of_week = np.random.choice([5, 6], p=[0.7, 0.3])  # Weekend more likely
            access_frequency = np.random.normal(25, 8)  # High frequency
            file_category = np.random.choice([0, 1, 4], p=[0.3, 0.4, 0.3])  # Documents, code, database (sensitive)
            access_velocity = np.random.normal(15, 5)  # High velocity for data movement
        
        # Add escalation over time (subtle for high complexity)
        if t >= seq_len * initial_normal_ratio:
            escalation_factor = 1.0 + (t - seq_len * initial_normal_ratio) / (seq_len * (1 - initial_normal_ratio)) * complexity_mult
            access_frequency *= escalation_factor
            access_velocity *= escalation_factor
        
        # Add some correlation to make it realistic
        if t > 0 and temporal_dependencies:
            # Add small temporal dependencies
            prev_access = seq[-1]
            # Gradually increase suspiciousness
            if t >= seq_len * initial_normal_ratio:
                # Make patterns more consistent in suspicious phase
                file_size = max(file_size, prev_access[0] * 0.9)  # Don't decrease too much
        
        seq.append([
            file_size, hour, access_type, day_of_week,
            access_frequency, file_category, access_velocity
        ])
    
    return seq


def generate_ransomware_sequence(
    seq_len: int,
    user_profile: str = 'general',
    complexity_mult: float = 0.6,
    seasonal_variation: bool = True,
    temporal_dependencies: bool = True
) -> List[List[float]]:
    """
    Generate a ransomware pattern.
    
    Args:
        seq_len: Length of the sequence
        user_profile: Type of user profile for context
        complexity_mult: Multiplier for pattern complexity
        seasonal_variation: Whether to include seasonal patterns
        temporal_dependencies: Whether to include temporal dependencies
        
    Returns:
        List of access events with ransomware patterns
    """
    seq = []
    
    # Ransomware phases: reconnaissance (early) -> encryption (later)
    reconnaissance_phase = int(seq_len * (0.2 + 0.1 * complexity_mult))  # More reconnaissance for complex attacks
    
    for t in range(seq_len):
        if t < reconnaissance_phase:
            # Reconnaissance phase: mostly reads, normal hours
            hour = np.random.normal(13, 2.5)
            hour = np.clip(hour, 8, 18)  # Business hours
            access_type = 0  # Read (for reconnaissance)
            file_size = np.random.lognormal(2.0, 0.5)  # Normal size files
            access_frequency = np.random.normal(8, 2)
            access_velocity = np.random.normal(3, 1)
            day_of_week = np.random.choice([0, 1, 2, 3, 4])  # Weekday
            
            # Target various file types during recon
            file_category = np.random.choice(range(5), p=[0.25, 0.2, 0.15, 0.2, 0.2])  # Mix of all types
        else:
            # Encryption phase: writes/edits, any time
            hour = np.random.uniform(0, 24)  # Any time (more obvious for low complexity)
            if complexity_mult < 0.5:  # Low complexity - more obvious
                access_type = np.random.choice([1, 2], p=[0.7, 0.3])  # Write or delete
                file_size = np.random.lognormal(1.8, 0.6)  # Various sizes
                access_frequency = np.random.normal(50, 15)  # Very high frequency
                access_velocity = np.random.normal(25, 8)  # Very high velocity
            else:  # High complexity - more subtle
                access_type = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])  # Mix with more writes
                file_size = np.random.lognormal(2.0, 0.5)  # Still normal looking
                access_frequency = np.random.normal(20, 8)  # High but not extreme
                access_velocity = np.random.normal(12, 5)  # High but not extreme
            
            day_of_week = np.random.choice(range(7))
            
            # Target various file types during encryption
            file_category = np.random.choice(range(5))
        
        # Add some correlation for realism
        if t > 0 and temporal_dependencies:
            prev_access = seq[-1]
            # Maintain some consistency in suspicious behavior
            if t >= reconnaissance_phase:
                # Gradually escalate during encryption phase
                if access_frequency < prev_access[4]:
                    access_frequency = max(access_frequency, prev_access[4] * 0.95)
        
        seq.append([
            file_size, hour, access_type, day_of_week,
            access_frequency, file_category, access_velocity
        ])
    
    return seq


def generate_privilege_escalation_sequence(
    seq_len: int,
    user_profile: str = 'general',
    complexity_mult: float = 0.6,
    seasonal_variation: bool = True,
    temporal_dependencies: bool = True
) -> List[List[float]]:
    """
    Generate a privilege escalation pattern.
    
    Args:
        seq_len: Length of the sequence
        user_profile: Type of user profile for context
        complexity_mult: Multiplier for pattern complexity
        seasonal_variation: Whether to include seasonal patterns
        temporal_dependencies: Whether to include temporal dependencies
        
    Returns:
        List of access events with privilege escalation patterns
    """
    seq = []
    
    for t in range(seq_len):
        # Privilege escalation characteristics
        hour = np.random.uniform(0, 24)  # Any time
        file_size = np.random.lognormal(1.2, 0.4)  # Usually small system files
        
        # Different patterns based on complexity
        if complexity_mult < 0.5:  # Obvious patterns
            access_type = np.random.choice([1, 3], p=[0.8, 0.2])  # Mostly write, some execute
            access_frequency = np.random.normal(30, 8)  # High frequency
            access_velocity = np.random.normal(18, 6)  # High velocity
            day_of_week = np.random.choice(range(7))
        else:  # Subtle patterns
            access_type = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])  # Mix of operations
            access_frequency = np.random.normal(15, 5)  # Moderate but elevated
            access_velocity = np.random.normal(10, 4)  # Moderate but elevated
            day_of_week = np.random.choice(range(7))
        
        # Target system files
        system_categories = [3, 5, 7]  # System, config, executable
        file_category = np.random.choice(system_categories)
        
        # Add escalation over time if temporal dependencies enabled
        if temporal_dependencies and t > 0:
            prev_access = seq[-1]
            # Gradually increase suspiciousness
            if complexity_mult < 0.5:  # Obvious escalation
                access_frequency = max(access_frequency, prev_access[4] * 1.05)
                access_velocity = max(access_velocity, prev_access[6] * 1.05)
            else:  # Subtle escalation
                access_frequency = max(access_frequency, prev_access[4] * 1.02)
                access_velocity = max(access_velocity, prev_access[6] * 1.02)
        
        seq.append([
            file_size, hour, access_type, day_of_week,
            access_frequency, file_category, access_velocity
        ])
    
    return seq


def generate_other_anomaly_sequence(
    seq_len: int,
    user_profile: str = 'general',
    complexity_mult: float = 0.6,
    seasonal_variation: bool = True,
    temporal_dependencies: bool = True
) -> List[List[float]]:
    """
    Generate other types of anomalous patterns.
    
    Args:
        seq_len: Length of the sequence
        user_profile: Type of user profile for context
        complexity_mult: Multiplier for pattern complexity
        seasonal_variation: Whether to include seasonal patterns
        temporal_dependencies: Whether to include temporal dependencies
        
    Returns:
        List of access events with other anomaly patterns
    """
    seq = []
    
    # Choose anomaly subtype based on complexity
    if complexity_mult < 0.5:
        # Obvious anomaly types
        anomaly_subtype = np.random.choice([
            'resource_exhaustion', 'unusual_pattern', 'brute_force'
        ])
    else:
        # Subtle anomaly types
        anomaly_subtype = np.random.choice([
            'slow_anomaly', 'insider_threat', 'configuration_abuse'
        ], p=[0.3, 0.4, 0.3])
    
    phase_transition = int(seq_len * 0.6)  # When anomaly becomes more obvious
    
    for t in range(seq_len):
        hour = np.random.uniform(0, 24)
        day_of_week = np.random.randint(0, 7)
        
        if anomaly_subtype == 'resource_exhaustion':
            # Gradually increasing resource usage
            base_freq = 5 + (t * 8 * complexity_mult)  # Increasing over time
            access_frequency = base_freq + np.random.normal(0, 2)
            access_velocity = 2 + (t * 0.8 * complexity_mult) + np.random.normal(0, 1)
            file_size = np.random.lognormal(2.2, 0.7)
            access_type = np.random.choice([0, 1], p=[0.3, 0.7])  # More writes
            file_category = np.random.choice(range(5))
            
        elif anomaly_subtype == 'unusual_pattern':
            # Unusual access patterns
            if complexity_mult < 0.5:  # Obvious
                hour = np.random.choice([2, 3, 4], p=[0.5, 0.3, 0.2])  # Very specific hours
                access_type = np.random.choice([2, 3], p=[0.7, 0.3])  # Delete/execute
                access_frequency = np.random.normal(35, 8)
            else:  # Subtle
                hour = np.random.choice([2, 3, 4, 12, 13], p=[0.2, 0.2, 0.2, 0.2, 0.2])
                access_type = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
                access_frequency = np.random.normal(15, 5)
            
            file_size = np.random.lognormal(1.8, 0.8)
            file_category = np.random.choice([3, 4, 5])  # System/database files
            access_velocity = np.random.normal(12 if complexity_mult < 0.5 else 6, 3)
            
        elif anomaly_subtype == 'slow_anomaly':
            # Slowly developing anomaly
            access_frequency = np.random.normal(6, 2)
            access_velocity = np.random.normal(2.5, 0.8)
            file_size = np.random.lognormal(2.0, 0.6)
            access_type = np.random.choice([0, 1], p=[0.6, 0.4])
            
            if t > phase_transition:  # Develops later for more subtle detection
                access_frequency *= 3  # Spike in frequency
                access_velocity *= 2.5  # Spike in velocity
                access_type = 1  # More writes during anomaly phase
                file_size = np.random.lognormal(2.5, 0.7)  # Larger files during anomaly
                
            file_category = np.random.choice(range(5))
            
        else:  # insider_threat or configuration_abuse
            # Mix of normal and suspicious patterns
            if complexity_mult < 0.5:  # Obvious
                hour = np.random.uniform(0, 24)
                access_frequency = np.random.normal(20, 6)
                access_velocity = np.random.normal(12, 4)
            else:  # Subtle
                hour = np.random.normal(14, 2.5)  # Business hours but may be off
                access_frequency = np.random.normal(10, 3)  # Elevated but not extreme
                access_velocity = np.random.normal(6, 2)  # Elevated but not extreme
                
            file_size = np.random.lognormal(2.8 if complexity_mult < 0.5 else 2.4, 0.9)
            access_type = np.random.choice([0, 1], p=[0.4, 0.6])  # Read/write mix
            file_category = np.random.choice([1, 4, 5] if complexity_mult < 0.5 else [0, 1, 4, 5])  # Target sensitive data
        
        # Apply minimum bounds
        access_frequency = max(0.1, access_frequency)
        access_velocity = max(0.1, access_velocity)
        file_size = max(0.01, file_size)
        
        seq.append([
            file_size, hour, access_type, day_of_week,
            access_frequency, file_category, access_velocity
        ])
    
    return seq


def analyze_generated_data(data: np.ndarray, labels: np.ndarray, anomaly_types: Optional[np.ndarray] = None) -> Dict:
    """
    Analyze the generated dataset and return statistics.
    
    Args:
        data: Generated data array
        labels: Generated labels array
        anomaly_types: Optional anomaly type labels
        
    Returns:
        Dictionary with analysis statistics
    """
    analysis = {
        'total_samples': len(data),
        'sequence_length': data.shape[1],
        'num_features': data.shape[2],
        'anomaly_ratio': float(labels.mean()),
        'label_distribution': {
            'normal': int((labels == 0).sum()),
            'anomaly': int((labels == 1).sum())
        }
    }
    
    # Feature statistics
    feature_names = ['file_size', 'hour', 'access_type', 'day_of_week', 'freq', 'category', 'velocity']
    feature_stats = {}
    
    for i, name in enumerate(feature_names):
        feature_data = data[:, :, i].flatten()
        feature_stats[name] = {
            'mean': float(feature_data.mean()),
            'std': float(feature_data.std()),
            'min': float(feature_data.min()),
            'max': float(feature_data.max()),
            'median': float(np.median(feature_data)),
            'q25': float(np.percentile(feature_data, 25)),
            'q75': float(np.percentile(feature_data, 75))
        }
    
    analysis['feature_statistics'] = feature_stats
    
    if anomaly_types is not None:
        unique, counts = np.unique(anomaly_types, return_counts=True)
        analysis['anomaly_type_distribution'] = dict(zip(unique, counts))
    
    return analysis
