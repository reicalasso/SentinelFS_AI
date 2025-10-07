"""
Advanced dataset generator for SentinelFS AI with sophisticated patterns and realistic scenarios.

This module provides comprehensive data generation capabilities for training AI models
to detect file access anomalies. It includes realistic user behavior modeling,
complex attack scenarios, temporal dependencies, and multi-dimensional feature generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from datetime import datetime, timedelta
import random
from dataclasses import dataclass
from enum import Enum
import warnings
from pathlib import Path
import json

from ..data_types import AnomalyType
from ..utils.logger import get_logger
import torch

logger = get_logger(__name__)


class UserBehaviorProfile(Enum):
    """Different user behavior profiles for realistic access patterns."""
    DEVELOPER = "developer"
    ANALYST = "analyst" 
    EXECUTIVE = "executive"
    ADMIN = "admin"
    CONTRACTOR = "contractor"
    TEMPORARY = "temporary"


@dataclass
class AccessPatternConfig:
    """Configuration for access pattern generation."""
    # Normal behavior parameters
    business_hours_start: int = 8
    business_hours_end: int = 18
    weekend_access_probability: float = 0.15
    after_hours_probability: float = 0.1
    file_size_mean: float = 2.0  # lognormal mean for normal files
    file_size_std: float = 0.8   # lognormal std for normal files
    
    # Anomaly parameters
    anomaly_complexity_level: str = 'high'  # 'low', 'medium', 'high'
    anomaly_subtlety_factor: float = 0.7    # How subtle anomalies should be
    
    # Temporal parameters
    sequence_length: int = 20
    time_step_minutes: int = 10             # Time resolution in minutes
    behavior_drift_probability: float = 0.05 # Chance of behavior change over time
    seasonal_factor: float = 0.1            # Seasonal variation factor


class AdvancedDatasetGenerator:
    """
    Advanced dataset generator with sophisticated modeling capabilities.
    
    Features:
    - Realistic user behavior profiles
    - Complex attack scenarios
    - Temporal dependencies and seasonality
    - Organizational context modeling
    - Sophisticated anomaly injection
    - Multi-dimensional feature generation
    - Behavioral pattern evolution
    """
    
    def __init__(
        self,
        config: Optional[AccessPatternConfig] = None,
        random_seed: Optional[int] = None
    ):
        self.config = config or AccessPatternConfig()
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Define user behavior characteristics
        self.user_profiles = self._init_user_profiles()
        
        # Define file category mappings
        self.file_categories = {
            0: {'name': 'document', 'extensions': ['pdf', 'doc', 'docx', 'txt', 'rtf'], 'sensitivity': 0.3},
            1: {'name': 'code', 'extensions': ['py', 'js', 'java', 'cpp', 'h', 'cs'], 'sensitivity': 0.7},
            2: {'name': 'media', 'extensions': ['jpg', 'png', 'mp4', 'mp3', 'avi'], 'sensitivity': 0.2},
            3: {'name': 'system', 'extensions': ['exe', 'dll', 'sys', 'so'], 'sensitivity': 0.9},
            4: {'name': 'database', 'extensions': ['db', 'sql', 'sqlite', 'mdb'], 'sensitivity': 0.8},
            5: {'name': 'configuration', 'extensions': ['conf', 'cfg', 'ini', 'yaml', 'json'], 'sensitivity': 0.6},
            6: {'name': 'backup', 'extensions': ['bak', 'zip', 'tar', 'gz'], 'sensitivity': 0.4},
            7: {'name': 'executable', 'extensions': ['bat', 'sh', 'cmd'], 'sensitivity': 0.7}
        }
        
        # Define access types
        self.access_types = {
            0: {'name': 'read', 'weight': 0.6},
            1: {'name': 'write', 'weight': 0.25},
            2: {'name': 'delete', 'weight': 0.1},
            3: {'name': 'execute', 'weight': 0.05}
        }
        
        # Complexity multipliers
        self.complexity_multipliers = {
            'low': 0.3,
            'medium': 0.6,
            'high': 1.0
        }
    
    def _init_user_profiles(self) -> Dict:
        """Initialize user behavior profiles with realistic characteristics."""
        return {
            UserBehaviorProfile.DEVELOPER: {
                'business_hours_factor': 1.0,
                'weekend_probability': 0.3,
                'file_size_factor': 1.2,
                'access_frequency_mean': 8,
                'access_frequency_std': 3,
                'file_categories': [0, 1, 5, 6],  # Documents, code, config, backup
                'preferred_hours': [9, 10, 11, 14, 15, 16],
                'access_type_weights': [0.5, 0.4, 0.05, 0.05]
            },
            UserBehaviorProfile.ANALYST: {
                'business_hours_factor': 1.1,
                'weekend_probability': 0.1,
                'file_size_factor': 1.0,
                'access_frequency_mean': 6,
                'access_frequency_std': 2,
                'file_categories': [0, 2, 5],  # Documents, media, config
                'preferred_hours': [10, 11, 13, 14, 15],
                'access_type_weights': [0.7, 0.2, 0.08, 0.02]
            },
            UserBehaviorProfile.EXECUTIVE: {
                'business_hours_factor': 0.9,
                'weekend_probability': 0.05,
                'file_size_factor': 0.8,
                'access_frequency_mean': 4,
                'access_frequency_std': 1.5,
                'file_categories': [0, 2],  # Documents, media
                'preferred_hours': [8, 9, 14, 15, 16],
                'access_type_weights': [0.8, 0.15, 0.04, 0.01]
            },
            UserBehaviorProfile.ADMIN: {
                'business_hours_factor': 0.8,
                'weekend_probability': 0.2,
                'file_size_factor': 0.9,
                'access_frequency_mean': 12,
                'access_frequency_std': 5,
                'file_categories': [3, 5, 7],  # System, config, executable
                'preferred_hours': [6, 7, 8, 20, 21, 22],  # Off-hours for maintenance
                'access_type_weights': [0.4, 0.4, 0.15, 0.05]
            },
            UserBehaviorProfile.CONTRACTOR: {
                'business_hours_factor': 1.2,
                'weekend_probability': 0.02,
                'file_size_factor': 0.7,
                'access_frequency_mean': 3,
                'access_frequency_std': 1,
                'file_categories': [0, 1],  # Limited to documents and code
                'preferred_hours': [9, 10, 11, 15, 16],
                'access_type_weights': [0.65, 0.25, 0.08, 0.02]
            },
            UserBehaviorProfile.TEMPORARY: {
                'business_hours_factor': 1.0,
                'weekend_probability': 0.01,
                'file_size_factor': 0.6,
                'access_frequency_mean': 2,
                'access_frequency_std': 0.5,
                'file_categories': [0],  # Very limited access
                'preferred_hours': [10, 11, 14, 15],
                'access_type_weights': [0.8, 0.15, 0.04, 0.01]
            }
        }
    
    def generate_dataset(
        self,
        num_samples: int = 5000,
        sequence_length: Optional[int] = None,
        anomaly_ratio: float = 0.15,
        user_profile_ratio: Optional[Dict[UserBehaviorProfile, float]] = None,
        include_timestamps: bool = True,
        seasonal_variation: bool = True,
        temporal_dependencies: bool = True
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Generate comprehensive dataset with realistic access patterns.
        
        Args:
            num_samples: Number of sequences to generate
            sequence_length: Length of each sequence (if None, uses config)
            anomaly_ratio: Proportion of anomalous sequences
            user_profile_ratio: Distribution of user profiles
            include_timestamps: Whether to include timestamps
            seasonal_variation: Whether to include seasonal patterns
            temporal_dependencies: Whether to include temporal dependencies
            
        Returns:
            Dictionary containing data, labels, timestamps, and metadata
        """
        seq_len = sequence_length or self.config.sequence_length
        
        # Determine normal vs anomaly distribution
        num_normal = int(num_samples * (1 - anomaly_ratio))
        num_anomaly = num_samples - num_normal
        
        logger.info(f"Generating {num_samples} samples ({num_normal} normal, {num_anomaly} anomalous)")
        
        # Set default user profile distribution
        if user_profile_ratio is None:
            user_profile_ratio = {
                UserBehaviorProfile.DEVELOPER: 0.25,
                UserBehaviorProfile.ANALYST: 0.25,
                UserBehaviorProfile.EXECUTIVE: 0.15,
                UserBehaviorProfile.ADMIN: 0.15,
                UserBehaviorProfile.CONTRACTOR: 0.15,
                UserBehaviorProfile.TEMPORARY: 0.05
            }
        
        # Generate normal sequences
        normal_data, normal_labels, normal_timestamps = self._generate_normal_sequences(
            num_normal, seq_len, user_profile_ratio, seasonal_variation, temporal_dependencies
        )
        
        # Generate anomalous sequences
        anomaly_data, anomaly_labels, anomaly_timestamps = self._generate_anomalous_sequences(
            num_anomaly, seq_len, user_profile_ratio, seasonal_variation, temporal_dependencies
        )
        
        # Combine and shuffle
        all_data = np.vstack([normal_data, anomaly_data])
        all_labels = np.vstack([normal_labels, anomaly_labels])
        all_timestamps = normal_timestamps + anomaly_timestamps
        
        # Shuffle the data
        shuffle_indices = np.random.permutation(len(all_data))
        all_data = all_data[shuffle_indices]
        all_labels = all_labels[shuffle_indices]
        all_timestamps = [all_timestamps[i] for i in shuffle_indices]
        
        # Create feature names
        feature_names = [
            'file_size_mb', 'access_hour', 'access_type', 'day_of_week',
            'access_frequency', 'file_category', 'access_velocity',
            'user_department', 'access_context', 'sensitivity_score',
            'temporal_pattern', 'access_correlation'
        ]
        
        # Create DataFrame for additional metadata if needed
        metadata_df = pd.DataFrame({
            'timestamp': all_timestamps,
            'label': all_labels.flatten(),
            'sequence_id': range(len(all_data))
        })
        
        logger.info(f"Dataset generation completed: {all_data.shape}")
        
        return {
            'data': all_data,
            'labels': all_labels,
            'timestamps': all_timestamps,
            'metadata': metadata_df,
            'feature_names': feature_names,
            'config': self.config.__dict__
        }
    
    def _generate_normal_sequences(
        self,
        num_sequences: int,
        seq_len: int,
        user_profile_ratio: Dict,
        seasonal_variation: bool,
        temporal_dependencies: bool
    ) -> Tuple[np.ndarray, np.ndarray, List[List[datetime]]]:
        """Generate normal access sequences based on user profiles."""
        sequences = []
        labels = []
        timestamps = []
        
        # Determine user assignments
        user_profile_list = list(user_profile_ratio.keys())
        user_weights = list(user_profile_ratio.values())
        
        for i in range(num_sequences):
            # Select user profile
            user_profile = np.random.choice(user_profile_list, p=user_weights)
            
            # Generate sequence based on profile
            seq, seq_timestamps = self._generate_normal_sequence_for_profile(
                seq_len, user_profile, seasonal_variation, temporal_dependencies
            )
            
            sequences.append(seq)
            labels.append([0])  # Normal
            timestamps.append(seq_timestamps)
        
        return (
            np.array(sequences, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            timestamps
        )
    
    def _generate_normal_sequence_for_profile(
        self,
        seq_len: int,
        user_profile: UserBehaviorProfile,
        seasonal_variation: bool,
        temporal_dependencies: bool
    ) -> Tuple[List[List[float]], List[datetime]]:
        """Generate a single normal sequence for a specific user profile."""
        profile_config = self.user_profiles[user_profile]
        sequence = []
        timestamps = []
        
        # Start time for sequence (random date in recent period)
        start_time = datetime.now() - timedelta(days=random.randint(1, 365))
        
        for t in range(seq_len):
            # Calculate temporal context
            current_time = start_time + timedelta(
                minutes=t * self.config.time_step_minutes
            )
            
            # Apply seasonal variation
            seasonal_factor = 1.0
            if seasonal_variation:
                seasonal_factor = self._calculate_seasonal_factor(current_time)
            
            # Apply business hour patterns
            business_hour_factor = self._calculate_business_hour_factor(
                current_time.hour, profile_config['business_hours_factor']
            )
            
            # Generate access characteristics based on profile
            hour = self._generate_access_hour(
                current_time, profile_config, business_hour_factor, seasonal_factor
            )
            
            day_of_week = current_time.weekday()
            
            # File size with user-specific characteristics
            file_size = self._generate_file_size(
                profile_config['file_size_factor'], 
                current_time,
                seasonal_variation
            )
            
            # Access type based on profile
            access_type = np.random.choice(
                list(self.access_types.keys()),
                p=profile_config['access_type_weights']
            )
            
            # Access frequency and velocity
            access_frequency = self._generate_access_frequency(
                profile_config, business_hour_factor, seasonal_factor
            )
            
            access_velocity = self._generate_access_velocity(
                access_frequency, profile_config['file_size_factor']
            )
            
            # File category based on user profile
            file_category = np.random.choice(profile_config['file_categories'])
            
            # Additional features
            user_department = self._map_user_profile_to_department(user_profile)
            access_context = self._determine_access_context(day_of_week, hour)
            sensitivity_score = self.file_categories[file_category]['sensitivity']
            
            # Temporal pattern features
            temporal_pattern = self._calculate_temporal_pattern(t, seq_len, current_time)
            access_correlation = self._calculate_access_correlation(sequence, access_type)
            
            sequence.append([
                file_size, hour, float(access_type), float(day_of_week),
                access_frequency, float(file_category), access_velocity,
                float(user_department), float(access_context), sensitivity_score,
                temporal_pattern, access_correlation
            ])
            
            timestamps.append(current_time)
        
        return sequence, timestamps
    
    def _generate_anomalous_sequences(
        self,
        num_sequences: int,
        seq_len: int,
        user_profile_ratio: Dict,
        seasonal_variation: bool,
        temporal_dependencies: bool
    ) -> Tuple[np.ndarray, np.ndarray, List[List[datetime]]]:
        """Generate anomalous access sequences with various attack types."""
        sequences = []
        labels = []
        timestamps = []
        
        # Anomaly type distribution
        anomaly_types = [
            AnomalyType.DATA_EXFILTRATION,
            AnomalyType.RANSOMWARE,
            AnomalyType.PRIVILEGE_ESCALATION,
            AnomalyType.OTHER
        ]
        
        # Distribution of anomaly types
        anomaly_weights = [0.25, 0.25, 0.25, 0.25]  # Equal distribution
        
        for i in range(num_sequences):
            # Select anomaly type
            anomaly_type = np.random.choice(anomaly_types, p=anomaly_weights)
            
            # Select a base user profile for context
            user_profile_list = list(user_profile_ratio.keys())
            user_weights = list(user_profile_ratio.values())
            base_profile = np.random.choice(user_profile_list, p=user_weights)
            
            # Generate anomalous sequence based on type
            if anomaly_type == AnomalyType.DATA_EXFILTRATION:
                seq, seq_timestamps = self._generate_data_exfiltration_sequence(
                    seq_len, base_profile, seasonal_variation, temporal_dependencies
                )
            elif anomaly_type == AnomalyType.RANSOMWARE:
                seq, seq_timestamps = self._generate_ransomware_sequence(
                    seq_len, base_profile, seasonal_variation, temporal_dependencies
                )
            elif anomaly_type == AnomalyType.PRIVILEGE_ESCALATION:
                seq, seq_timestamps = self._generate_privilege_escalation_sequence(
                    seq_len, base_profile, seasonal_variation, temporal_dependencies
                )
            else:  # OTHER
                seq, seq_timestamps = self._generate_other_anomaly_sequence(
                    seq_len, base_profile, seasonal_variation, temporal_dependencies
                )
            
            sequences.append(seq)
            labels.append([1])  # Anomaly
            timestamps.append(seq_timestamps)
        
        return (
            np.array(sequences, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            timestamps
        )
    
    def _generate_data_exfiltration_sequence(
        self,
        seq_len: int,
        base_profile: UserBehaviorProfile,
        seasonal_variation: bool,
        temporal_dependencies: bool
    ) -> Tuple[List[List[float]], List[datetime]]:
        """Generate data exfiltration attack pattern."""
        sequence = []
        timestamps = []
        
        start_time = datetime.now() - timedelta(days=random.randint(1, 365))
        
        # Exfiltration typically happens during off-hours
        exfil_hour = np.random.choice([0, 1, 2, 3, 4, 22, 23])  # Very early/late
        
        for t in range(seq_len):
            current_time = start_time + timedelta(minutes=t * self.config.time_step_minutes)
            
            # Exfiltration characteristics
            hour = exfil_hour
            day_of_week = np.random.choice([5, 6]) if np.random.random() < 0.7 else current_time.weekday()
            
            # Large file sizes, especially as sequence progresses
            size_factor = 1.0 + (t / seq_len) * 2.0  # Gradually increase
            file_size = np.random.lognormal(3.5 + np.log(size_factor), 0.7)
            
            access_type = 0  # Read (for exfiltration)
            
            # High access frequency and velocity
            access_frequency = np.random.normal(25, 8)
            access_velocity = np.random.normal(15, 5)
            
            # Target sensitive files
            sensitive_categories = [1, 3, 4, 5]  # Code, system, database, config
            file_category = np.random.choice(sensitive_categories)
            
            # Additional features
            user_department = self._map_user_profile_to_department(base_profile)
            access_context = 2  # Suspicious context
            sensitivity_score = self.file_categories[file_category]['sensitivity']
            
            temporal_pattern = self._calculate_temporal_pattern(t, seq_len, current_time)
            access_correlation = self._calculate_access_correlation(sequence, access_type)
            
            sequence.append([
                file_size, hour, float(access_type), float(day_of_week),
                access_frequency, float(file_category), access_velocity,
                float(user_department), float(access_context), sensitivity_score,
                temporal_pattern, access_correlation
            ])
            
            timestamps.append(current_time)
        
        return sequence, timestamps
    
    def _generate_ransomware_sequence(
        self,
        seq_len: int,
        base_profile: UserBehaviorProfile,
        seasonal_variation: bool,
        temporal_dependencies: bool
    ) -> Tuple[List[List[float]], List[datetime]]:
        """Generate ransomware attack pattern."""
        sequence = []
        timestamps = []
        
        start_time = datetime.now() - timedelta(days=random.randint(1, 365))
        
        # Ransomware phases: reconnaissance (early) -> encryption (later)
        reconnaissance_phase = int(seq_len * 0.3)
        
        for t in range(seq_len):
            current_time = start_time + timedelta(minutes=t * self.config.time_step_minutes)
            
            # Ransomware characteristics
            if t < reconnaissance_phase:
                # Reconnaissance phase: mostly reads, normal hours
                hour = np.random.normal(13, 2.5)
                hour = np.clip(hour, 8, 18)  # Business hours
                access_type = 0  # Read
                file_size = np.random.lognormal(2.0, 0.5)  # Normal size
                access_frequency = np.random.normal(8, 2)
                access_velocity = np.random.normal(3, 1)
            else:
                # Encryption phase: writes/edits, any time
                hour = np.random.uniform(0, 24)
                access_type = np.random.choice([1, 2], p=[0.7, 0.3])  # Write or delete
                file_size = np.random.lognormal(1.8, 0.6)  # Various sizes
                access_frequency = np.random.normal(40, 12)  # High frequency
                access_velocity = np.random.normal(25, 8)  # High velocity
            
            day_of_week = current_time.weekday()
            
            # Target various file types during encryption
            file_category = np.random.choice(range(len(self.file_categories)))
            
            # Additional features
            user_department = self._map_user_profile_to_department(base_profile)
            access_context = 3  # Very suspicious context
            sensitivity_score = self.file_categories[file_category]['sensitivity']
            
            temporal_pattern = self._calculate_temporal_pattern(t, seq_len, current_time)
            access_correlation = self._calculate_access_correlation(sequence, access_type)
            
            sequence.append([
                file_size, hour, float(access_type), float(day_of_week),
                access_frequency, float(file_category), access_velocity,
                float(user_department), float(access_context), sensitivity_score,
                temporal_pattern, access_correlation
            ])
            
            timestamps.append(current_time)
        
        return sequence, timestamps
    
    def _generate_privilege_escalation_sequence(
        self,
        seq_len: int,
        base_profile: UserBehaviorProfile,
        seasonal_variation: bool,
        temporal_dependencies: bool
    ) -> Tuple[List[List[float]], List[datetime]]:
        """Generate privilege escalation attack pattern."""
        sequence = []
        timestamps = []
        
        start_time = datetime.now() - timedelta(days=random.randint(1, 365))
        
        for t in range(seq_len):
            current_time = start_time + timedelta(minutes=t * self.config.time_step_minutes)
            
            # Privilege escalation characteristics
            hour = np.random.uniform(0, 24)  # Any time
            file_size = np.random.lognormal(1.2, 0.4)  # Usually small system files
            access_type = np.random.choice([1, 3], p=[0.6, 0.4])  # Write or execute
            
            # High frequency and velocity for admin activities
            access_frequency = np.random.normal(30, 8)
            access_velocity = np.random.normal(18, 6)
            
            day_of_week = current_time.weekday()
            
            # Target system files
            system_categories = [3, 5, 7]  # System, config, executable
            file_category = np.random.choice(system_categories)
            
            # Additional features
            user_department = 4  # Security/IT department (for escalation)
            access_context = 3  # Suspicious context
            sensitivity_score = self.file_categories[file_category]['sensitivity']
            
            temporal_pattern = self._calculate_temporal_pattern(t, seq_len, current_time)
            access_correlation = self._calculate_access_correlation(sequence, access_type)
            
            sequence.append([
                file_size, hour, float(access_type), float(day_of_week),
                access_frequency, float(file_category), access_velocity,
                float(user_department), float(access_context), sensitivity_score,
                temporal_pattern, access_correlation
            ])
            
            timestamps.append(current_time)
        
        return sequence, timestamps
    
    def _generate_other_anomaly_sequence(
        self,
        seq_len: int,
        base_profile: UserBehaviorProfile,
        seasonal_variation: bool,
        temporal_dependencies: bool
    ) -> Tuple[List[List[float]], List[datetime]]:
        """Generate other types of anomalous patterns."""
        sequence = []
        timestamps = []
        
        start_time = datetime.now() - timedelta(days=random.randint(1, 365))
        
        # Choose anomaly subtype
        anomaly_subtype = np.random.choice([
            'resource_exhaustion', 'unusual_pattern', 'slow_anomaly', 'insider_threat'
        ])
        
        for t in range(seq_len):
            current_time = start_time + timedelta(minutes=t * self.config.time_step_minutes)
            
            hour = np.random.uniform(0, 24)
            day_of_week = current_time.weekday()
            
            if anomaly_subtype == 'resource_exhaustion':
                # Gradually increasing resource usage
                access_frequency = 5 + (t * 10)  # Increasing over time
                access_velocity = 2 + (t * 0.8)
                file_size = np.random.lognormal(2.2, 0.7)
                access_type = np.random.choice([0, 1], p=[0.3, 0.7])  # More writes
                file_category = np.random.choice(range(len(self.file_categories)))
                
            elif anomaly_subtype == 'unusual_pattern':
                # Unusual access patterns
                hour = np.random.choice([2, 3, 4], p=[0.5, 0.3, 0.2])  # Very specific hours
                access_type = np.random.choice([2, 3], p=[0.7, 0.3])  # Delete/execute
                access_frequency = np.random.normal(30, 10)
                access_velocity = np.random.normal(12, 4)  # Define velocity here
                file_size = np.random.lognormal(1.8, 0.8)
                file_category = np.random.choice([3, 4, 5])  # System/database files
                
            elif anomaly_subtype == 'slow_anomaly':
                # Slowly developing anomaly
                access_frequency = np.random.normal(6, 2)
                access_velocity = np.random.normal(2.5, 0.8)
                file_size = np.random.lognormal(2.0, 0.6)
                access_type = np.random.choice([0, 1], p=[0.6, 0.4])
                
                if t > seq_len * 0.7:  # Develops later
                    access_frequency *= 3  # Spike in frequency
                    access_velocity *= 2.5  # Spike in velocity
                    access_type = 1  # More writes
                    
                file_category = np.random.choice(range(len(self.file_categories)))
                
            else:  # insider_threat
                # Mix of normal and suspicious patterns
                hour = np.random.normal(14, 2.5)  # Business hours but off
                access_frequency = np.random.normal(15, 4)  # Higher than normal
                access_velocity = np.random.normal(8, 2)  # Higher than normal
                file_size = np.random.lognormal(2.8, 0.9)  # Larger files
                access_type = np.random.choice([0, 1], p=[0.4, 0.6])  # Read/write
                file_category = np.random.choice([1, 4, 5])  # Code, database, config
            
            # Calculate velocity based on pattern
            access_velocity = max(0.1, access_velocity + np.random.normal(0, 1))
            access_frequency = max(0.1, access_frequency + np.random.normal(0, 2))
            
            # Additional features
            user_department = self._map_user_profile_to_department(base_profile)
            access_context = 2  # Suspicious context
            sensitivity_score = self.file_categories[
                file_category
            ]['sensitivity']
            
            temporal_pattern = self._calculate_temporal_pattern(t, seq_len, current_time)
            access_correlation = self._calculate_access_correlation(sequence, access_type)
            
            sequence.append([
                file_size, hour, float(access_type), float(day_of_week),
                access_frequency, float(file_category), access_velocity,
                float(user_department), float(access_context), sensitivity_score,
                temporal_pattern, access_correlation
            ])
            
            timestamps.append(current_time)
        
        return sequence, timestamps
    
    def _generate_access_hour(
        self, 
        current_time: datetime, 
        profile_config: Dict, 
        business_factor: float, 
        seasonal_factor: float
    ) -> float:
        """Generate access hour based on profile and temporal factors."""
        # Base hour generation based on profile preferences
        if current_time.weekday() in [5, 6]:  # Weekend
            preferred_hours = profile_config['preferred_hours']
            if np.random.random() < profile_config['weekend_probability']:
                hour = np.random.choice(preferred_hours)
            else:
                hour = np.random.uniform(10, 16)  # Normal weekend hours
        else:  # Weekday
            # Check if it's business hours or after hours
            if np.random.random() < self.config.after_hours_probability:
                # After hours access
                hour = np.random.choice([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7])
            else:
                # Business hours with profile preferences
                hour = np.random.choice(profile_config['preferred_hours'])
        
        # Apply factors
        hour = hour + np.random.normal(0, 0.5)  # Add slight variation
        hour = np.clip(hour, 0, 23)  # Keep in valid range
        
        return hour
    
    def _generate_file_size(
        self, 
        size_factor: float, 
        current_time: datetime, 
        seasonal: bool
    ) -> float:
        """Generate file size based on user profile and context."""
        base_mean = self.config.file_size_mean
        base_std = self.config.file_size_std
        
        # Apply user-specific factor
        mean = base_mean + np.log(size_factor)
        
        # Apply seasonal variation if enabled
        if seasonal:
            seasonal_mult = 1.0 + (np.sin(2 * np.pi * current_time.timetuple().tm_yday / 365) * 0.1)
            mean = mean * seasonal_mult
        
        file_size = np.random.lognormal(mean, base_std)
        return max(0.01, file_size)  # Minimum file size
    
    def _generate_access_frequency(
        self, 
        profile_config: Dict, 
        business_factor: float, 
        seasonal_factor: float
    ) -> float:
        """Generate access frequency based on profile and context."""
        base_freq = np.random.normal(
            profile_config['access_frequency_mean'],
            profile_config['access_frequency_std']
        )
        
        # Apply temporal factors
        freq = base_freq * business_factor * seasonal_factor
        return max(0.1, freq)  # Minimum frequency
    
    def _generate_access_velocity(
        self, 
        access_frequency: float, 
        size_factor: float
    ) -> float:
        """Generate access velocity based on frequency and size."""
        velocity = access_frequency * size_factor * 0.3  # Correlation factor
        velocity = velocity + np.random.normal(0, 0.5)  # Add some variation
        return max(0.1, velocity)
    
    def _calculate_seasonal_factor(self, current_time: datetime) -> float:
        """Calculate seasonal factor based on time of year."""
        day_of_year = current_time.timetuple().tm_yday
        # Sinusoidal seasonal variation
        seasonal_var = np.sin(2 * np.pi * day_of_year / 365.25)
        return 1.0 + seasonal_var * self.config.seasonal_factor
    
    def _calculate_business_hour_factor(self, hour: int, profile_factor: float) -> float:
        """Calculate factor based on business hours."""
        if 8 <= hour <= 18:  # Business hours
            return 1.0 * profile_factor
        else:  # After hours
            return 0.3 * profile_factor  # Reduced factor for after hours
    
    def _map_user_profile_to_department(self, profile: UserBehaviorProfile) -> int:
        """Map user profile to department code."""
        dept_mapping = {
            UserBehaviorProfile.DEVELOPER: 1,
            UserBehaviorProfile.ANALYST: 2,
            UserBehaviorProfile.EXECUTIVE: 3,
            UserBehaviorProfile.ADMIN: 4,
            UserBehaviorProfile.CONTRACTOR: 5,
            UserBehaviorProfile.TEMPORARY: 6
        }
        return dept_mapping.get(profile, 0)
    
    def _determine_access_context(self, day_of_week: int, hour: float) -> int:
        """Determine access context based on time."""
        if day_of_week >= 5:  # Weekend
            if 8 <= hour <= 18:
                return 1  # Weekend business hours (suspicious)
            else:
                return 2  # Weekend off-hours (more suspicious)
        else:  # Weekday
            if 8 <= hour <= 18:
                return 0  # Normal business hours
            else:
                return 1  # Off-hours (suspicious)
    
    def _calculate_temporal_pattern(self, t: int, seq_len: int, current_time: datetime) -> float:
        """Calculate temporal pattern feature."""
        # Pattern based on sequence position and time
        position_factor = t / seq_len
        time_factor = (current_time.hour + current_time.minute / 60.0) / 24.0
        return (position_factor + time_factor) / 2.0
    
    def _calculate_access_correlation(self, sequence: List, current_access_type: int) -> float:
        """Calculate correlation with previous accesses."""
        if not sequence:
            return 0.0
        
        # Calculate correlation with previous access types
        prev_access_types = [access[2] for access in sequence[-5:]]  # Last 5 accesses
        if not prev_access_types:
            return 0.0
        
        # Calculate similarity (higher if similar access types)
        similarity = sum(1 for at in prev_access_types if at == current_access_type)
        return similarity / len(prev_access_types)
    
    def generate_multi_scenario_dataset(
        self,
        scenarios: List[Dict],
        base_config: Optional[AccessPatternConfig] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate dataset with multiple scenarios mixed together.
        
        Args:
            scenarios: List of scenario configurations
            base_config: Base configuration to use
            
        Returns:
            Combined dataset with labels
        """
        all_data = []
        all_labels = []
        all_scenario_labels = []
        
        for i, scenario in enumerate(scenarios):
            # Create config for this scenario
            scenario_config = base_config or AccessPatternConfig()
            if 'config_overrides' in scenario:
                for attr, value in scenario['config_overrides'].items():
                    setattr(scenario_config, attr, value)
            
            # Create generator with scenario config
            scenario_gen = AdvancedDatasetGenerator(config=scenario_config)
            
            # Generate data for this scenario
            result = scenario_gen.generate_dataset(
                num_samples=scenario['num_samples'],
                anomaly_ratio=scenario['anomaly_ratio'],
                user_profile_ratio=scenario.get('user_profile_ratio', None)
            )
            
            all_data.append(result['data'])
            all_labels.append(result['labels'])
            all_scenario_labels.extend([i] * len(result['data']))
        
        # Combine all scenarios - handle different sequence lengths by padding or truncating
        # First, make sure all have the same sequence length by using the first scenario's length as reference
        ref_seq_len = all_data[0].shape[1] 
        ref_num_features = all_data[0].shape[2]
        
        # Pad or truncate all datasets to match the reference
        for idx, data in enumerate(all_data):
            if data.shape[1] != ref_seq_len:
                current_seq_len = data.shape[1]
                if current_seq_len < ref_seq_len:
                    # Pad with zeros
                    padding_needed = ref_seq_len - current_seq_len
                    padding = np.zeros((data.shape[0], padding_needed, data.shape[2]))
                    all_data[idx] = np.concatenate([data, padding], axis=1)
                else:
                    # Truncate
                    all_data[idx] = data[:, :ref_seq_len, :]
        
        # Now combine all scenarios
        combined_data = np.vstack(all_data)
        combined_labels = np.vstack(all_labels)
        
        return {
            'data': combined_data,
            'labels': combined_labels,
            'scenario_labels': np.array(all_scenario_labels)
        }
    
    def save_dataset(
        self,
        dataset: Dict[str, Union[np.ndarray, pd.DataFrame]],
        save_path: Union[str, Path],
        include_metadata: bool = True
    ):
        """Save generated dataset to disk."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save main data and labels as numpy files
        np.save(save_path / 'data.npy', dataset['data'])
        np.save(save_path / 'labels.npy', dataset['labels'])
        
        # Save metadata
        if include_metadata and 'metadata' in dataset:
            dataset['metadata'].to_csv(save_path / 'metadata.csv', index=False)
        
        # Save config
        config_path = save_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(dataset.get('config', self.config.__dict__), f, indent=2, default=str)
        
        # Save feature names
        with open(save_path / 'feature_names.txt', 'w') as f:
            for name in dataset.get('feature_names', []):
                f.write(f"{name}\n")
        
        logger.info(f"Dataset saved to {save_path}")


# Additional utility functions for dataset analysis
def analyze_dataset_distribution(dataset: Dict[str, np.ndarray]) -> Dict:
    """Analyze the distribution of generated dataset."""
    data = dataset['data']
    labels = dataset['labels']
    
    analysis = {
        'total_samples': len(data),
        'feature_count': data.shape[2] if len(data.shape) > 2 else data.shape[1],
        'sequence_length': data.shape[1] if len(data.shape) > 2 else 1,
        'anomaly_ratio': float(labels.mean()),
        'label_distribution': {
            'normal': int((labels == 0).sum()),
            'anomaly': int((labels == 1).sum())
        }
    }
    
    # Feature statistics
    feature_stats = {}
    for i in range(analysis['feature_count']):
        if len(data.shape) > 2:  # 3D array (seq, time, features)
            feature_data = data[:, :, i].flatten()
        else:  # 2D array (seq, features)
            feature_data = data[:, i]
        
        feature_stats[f'feature_{i}'] = {
            'mean': float(feature_data.mean()),
            'std': float(feature_data.std()),
            'min': float(feature_data.min()),
            'max': float(feature_data.max()),
            'median': float(np.median(feature_data))
        }
    
    analysis['feature_statistics'] = feature_stats
    return analysis


def visualize_dataset_patterns(dataset: Dict[str, np.ndarray], feature_indices: List[int] = [0, 1, 2]):
    """Visualize patterns in the generated dataset."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        data = dataset['data']
        labels = dataset['labels'].flatten()
        
        n_features = len(feature_indices)
        fig, axes = plt.subplots(n_features, 2, figsize=(15, 5*n_features))
        
        if n_features == 1:
            axes = axes.reshape(-1, 2)
        
        for i, feat_idx in enumerate(feature_indices):
            # Flatten feature data across sequences and time steps if 3D
            if len(data.shape) > 2:
                feat_data = data[:, :, feat_idx]
                normal_data = feat_data[labels == 0].flatten()
                anomaly_data = feat_data[labels == 1].flatten()
            else:
                feat_data = data[:, feat_idx]
                normal_data = feat_data[labels == 0]
                anomaly_data = feat_data[labels == 1]
            
            # Histograms
            axes[i, 0].hist(normal_data, bins=50, alpha=0.7, label='Normal', color='blue')
            axes[i, 0].hist(anomaly_data, bins=50, alpha=0.7, label='Anomaly', color='red')
            axes[i, 0].set_title(f'Feature {feat_idx} Distribution')
            axes[i, 0].legend()
            
            # Time series plot for first few sequences
            if len(data.shape) > 2:
                for j in range(min(3, len(data))):
                    axes[i, 1].plot(data[j, :, feat_idx], 
                                   alpha=0.7, 
                                   label=f'Seq {j} ({"A" if labels[j] else "N"})')
            axes[i, 1].set_title(f'Feature {feat_idx} Over Time')
            axes[i, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        logger.warning("Matplotlib/seaborn not available for visualization")


# Example usage and testing
def create_example_dataset():
    """Create and return an example dataset."""
    logger.info("Creating example dataset...")
    
    generator = AdvancedDatasetGenerator(
        config=AccessPatternConfig(
            sequence_length=20,
            anomaly_complexity_level='high',
            anomaly_subtlety_factor=0.7
        ),
        random_seed=42
    )
    
    dataset = generator.generate_dataset(
        num_samples=2000,
        anomaly_ratio=0.2,
        seasonal_variation=True,
        temporal_dependencies=True
    )
    
    logger.info("Example dataset created successfully")
    
    # Analyze the generated dataset
    analysis = analyze_dataset_distribution(dataset)
    logger.info(f"Dataset analysis: {analysis['label_distribution']}")
    
    return dataset


if __name__ == "__main__":
    # Create and examine example dataset
    example_dataset = create_example_dataset()
    
    print("Dataset shape:", example_dataset['data'].shape)
    print("Labels shape:", example_dataset['labels'].shape)
    print("Feature names:", len(example_dataset['feature_names']))
    
    # Show sample of first sequence
    print("\nSample of first sequence:")
    for t, access in enumerate(example_dataset['data'][0][:3]):  # First 3 time steps
        print(f"  Time {t}: {access[:7]}")  # Show first 7 features
    
    # Analyze distribution
    analysis = analyze_dataset_distribution(example_dataset)
    print(f"\nAnomaly ratio: {analysis['anomaly_ratio']:.3f}")
    print(f"Normal samples: {analysis['label_distribution']['normal']}")
    print(f"Anomaly samples: {analysis['label_distribution']['anomaly']}")