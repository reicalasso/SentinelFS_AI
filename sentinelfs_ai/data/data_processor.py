"""
Data preprocessing, splitting, and batch creation.
"""

import torch
import numpy as np
from typing import Dict
from sklearn.model_selection import train_test_split

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
        random_state: int = 42
    ):
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        self.feature_extractor = FeatureExtractor()
    
    def prepare_data(
        self, 
        data: np.ndarray, 
        labels: np.ndarray
    ) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Prepare data for training: normalize, split, and create DataLoaders.
        
        Args:
            data: Raw data array of shape (num_samples, seq_len, num_features)
            labels: Labels array of shape (num_samples, 1)
            
        Returns:
            Dictionary with 'train', 'val', and 'test' DataLoaders
        """
        logger.info("Preparing data for training...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            data, labels, test_size=self.test_split, 
            random_state=self.random_state, stratify=labels
        )
        
        # Second split: separate train and validation
        val_size = self.val_split / (1 - self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size,
            random_state=self.random_state, stratify=y_temp
        )
        
        # Fit scaler on training data and transform all sets
        X_train = self.feature_extractor.fit_transform(X_train)
        X_val = self.feature_extractor.transform(X_val)
        X_test = self.feature_extractor.transform(X_test)
        
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
