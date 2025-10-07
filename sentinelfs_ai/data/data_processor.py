"""
Data preprocessing, splitting, and batch creation.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils import shuffle

from .feature_extractor import FeatureExtractor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """Handle data preprocessing, splitting, and batch creation."""
    
    def __init__(
        self, 
        batch_size: int = 32, 
        val_split: float = 0.2, 
        test_split: float = 0.1, 
        random_state: int = 42,
        scaler_type: str = 'standard',  # 'standard', 'minmax', 'robust'
        normalize_features: bool = True,
        shuffle_data: bool = True
    ):
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        self.shuffle_data = shuffle_data
        
        # Initialize feature extractor with specified scaler
        self.scaler_type = scaler_type
        self.feature_extractor = FeatureExtractor(scaler_type=scaler_type)
        self.normalize_features = normalize_features
    
    def prepare_data(
        self, 
        data: np.ndarray, 
        labels: np.ndarray,
        anomaly_types: Optional[np.ndarray] = None
    ) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Prepare data for training: normalize, split, and create DataLoaders.
        
        Args:
            data: Raw data array of shape (num_samples, seq_len, num_features)
            labels: Labels array of shape (num_samples, 1)
            anomaly_types: Optional anomaly type labels for stratification
            
        Returns:
            Dictionary with 'train', 'val', and 'test' DataLoaders
        """
        logger.info("Preparing data for training...")
        
        # Optionally shuffle data
        if self.shuffle_data:
            if anomaly_types is not None:
                data, labels, anomaly_types = shuffle(
                    data, labels, anomaly_types, random_state=self.random_state
                )
            else:
                data, labels = shuffle(data, labels, random_state=self.random_state)
        
        # First split: separate test set
        stratify_arg = labels if anomaly_types is None else anomaly_types
        X_temp, X_test, y_temp, y_test = train_test_split(
            data, labels, test_size=self.test_split, 
            random_state=self.random_state, stratify=stratify_arg
        )
        
        # Handle anomaly_types if provided
        if anomaly_types is not None:
            _, _, anomaly_types_temp, anomaly_types_test = train_test_split(
                data, anomaly_types, test_size=self.test_split,
                random_state=self.random_state, stratify=stratify_arg
            )
        else:
            anomaly_types_temp = anomaly_types_test = None
        
        # Second split: separate train and validation
        val_size = self.val_split / (1 - self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size,
            random_state=self.random_state, stratify=y_temp
        )
        
        if anomaly_types_temp is not None:
            _, _, anomaly_types_train, anomaly_types_val = train_test_split(
                X_temp, anomaly_types_temp, test_size=val_size,
                random_state=self.random_state, stratify=y_temp
            )
        
        # Apply feature normalization if enabled
        if self.normalize_features:
            # Fit scaler on training data and transform all sets
            X_train = self.feature_extractor.fit_transform(X_train)
            X_val = self.feature_extractor.transform(X_val)
            X_test = self.feature_extractor.transform(X_test)
        else:
            logger.info("Feature normalization skipped as requested")
        
        logger.info(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples, Test: {len(X_test)} samples")
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)
        )
        
        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    def prepare_data_with_anomaly_types(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        anomaly_types: np.ndarray
    ) -> Tuple[Dict[str, torch.utils.data.DataLoader], Dict[str, torch.nn.Module]]:
        """
        Prepare data with anomaly type labels for multi-task learning.
        
        Args:
            data: Raw data array (num_samples, seq_len, num_features)
            labels: Binary labels (0=normal, 1=anomaly)
            anomaly_types: Anomaly type labels (0=normal, 1-4=anomaly types)
            
        Returns:
            Tuple of (data_loaders, feature_extractors)
        """
        logger.info("Preparing data with anomaly type labels...")
        
        # Prepare regular data loaders
        data_loaders = self.prepare_data(data, labels, anomaly_types)
        
        # Create a separate dataset for anomaly type prediction
        if self.normalize_features:
            normalized_data = self.feature_extractor.transform(data)
        else:
            normalized_data = data
            
        # Convert anomaly types to tensor format (one-hot might be needed later)
        anomaly_type_tensor = torch.LongTensor(anomaly_types.flatten())
        
        # Create special datasets that include both binary and type labels
        # For multi-task learning
        full_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(normalized_data),
            torch.FloatTensor(labels),
            anomaly_type_tensor
        )
        
        # Split the full dataset
        train_size = int((1 - self.test_split - self.val_split) * len(full_dataset))
        val_size = int(self.val_split * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_full, val_full, test_full = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_state)
        )
        
        # Create additional loaders for multi-task learning
        train_anomaly_loader = torch.utils.data.DataLoader(
            train_full, batch_size=self.batch_size, shuffle=True
        )
        val_anomaly_loader = torch.utils.data.DataLoader(
            val_full, batch_size=self.batch_size, shuffle=False
        )
        test_anomaly_loader = torch.utils.data.DataLoader(
            test_full, batch_size=self.batch_size, shuffle=False
        )
        
        return data_loaders, {
            'train_multi': train_anomaly_loader,
            'val_multi': val_anomaly_loader,
            'test_multi': test_anomaly_loader
        }
    
    def get_data_statistics(self, data: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Get comprehensive statistics about the dataset.
        
        Args:
            data: Raw data array
            labels: Labels array
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_samples': len(data),
            'sequence_length': data.shape[1],
            'num_features': data.shape[2],
            'feature_means': data.mean(axis=(0, 1)).tolist(),
            'feature_stds': data.std(axis=(0, 1)).tolist(),
            'label_distribution': {
                'normal': int((labels == 0).sum()),
                'anomaly': int((labels == 1).sum())
            }
        }
        
        # Add per-feature statistics
        feature_stats = {}
        for i, feature_name in enumerate(self.feature_extractor.FEATURE_NAMES):
            feature_stats[feature_name] = {
                'mean': float(data[:, :, i].mean()),
                'std': float(data[:, :, i].std()),
                'min': float(data[:, :, i].min()),
                'max': float(data[:, :, i].max()),
                'median': float(np.median(data[:, :, i]))
            }
        
        stats['feature_statistics'] = feature_stats
        return stats
    
    def augment_data(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        anomaly_types: Optional[np.ndarray] = None,
        augmentation_factor: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Perform data augmentation to increase dataset size.
        
        Args:
            data: Input data array
            labels: Labels array
            anomaly_types: Optional anomaly type labels
            augmentation_factor: Proportion of data to augment
            
        Returns:
            Augmented data, labels, and anomaly_types
        """
        logger.info(f"Performing data augmentation with factor {augmentation_factor}")
        
        num_samples = len(data)
        num_augment = int(num_samples * augmentation_factor)
        
        if num_augment <= 0:
            return data, labels, anomaly_types
        
        # Randomly select samples to augment
        indices_to_augment = np.random.choice(num_samples, num_augment, replace=False)
        
        augmented_data = []
        augmented_labels = []
        augmented_anomaly_types = [] if anomaly_types is not None else None
        
        for idx in indices_to_augment:
            # Add small noise to each sequence
            noise = np.random.normal(0, 0.05, data[idx].shape)  # Small Gaussian noise
            augmented_seq = data[idx] + noise
            # Ensure values stay within reasonable bounds
            augmented_seq = np.clip(augmented_seq, 0, None)
            
            augmented_data.append(augmented_seq)
            augmented_labels.append(labels[idx])
            if anomaly_types is not None:
                augmented_anomaly_types.append(anomaly_types[idx])
        
        # Combine original and augmented data
        all_data = np.vstack([data, np.array(augmented_data)])
        all_labels = np.vstack([labels, np.array(augmented_labels)])
        
        if anomaly_types is not None:
            all_anomaly_types = np.hstack([anomaly_types, np.array(augmented_anomaly_types)])
        else:
            all_anomaly_types = None
        
        logger.info(f"Data augmentation: {num_samples} -> {len(all_data)} samples")
        
        return all_data, all_labels, all_anomaly_types
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
