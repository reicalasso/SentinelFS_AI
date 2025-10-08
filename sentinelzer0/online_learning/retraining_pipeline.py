"""
Retraining Pipeline

Automat retraining and deployment pipeline for online learning.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
import logging
from pathlib import Path
import time


@dataclass
class RetrainingConfig:
    """Retraining configuration."""
    min_samples: int = 100
    retrain_frequency: int = 1000  # samples
    validation_split: float = 0.2
    max_epochs: int = 10
    early_stopping_patience: int = 3
    min_improvement: float = 0.001
    save_best_model: bool = True
    backup_old_model: bool = True


class RetrainingPipeline:
    """
    Automated retraining pipeline.
    
    Features:
    - Scheduled retraining
    - Validation-based model selection
    - Automatic deployment
    - Model backup and versioning
    - Performance tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: RetrainingConfig,
        model_path: str,
        train_fn: Optional[Callable] = None,
        validate_fn: Optional[Callable] = None
    ):
        """
        Initialize retraining pipeline.
        
        Args:
            model: Model to retrain
            config: Retraining configuration
            model_path: Path to save model
            train_fn: Custom training function
            validate_fn: Custom validation function
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.config = config
        self.model_path = Path(model_path)
        self.train_fn = train_fn
        self.validate_fn = validate_fn
        
        # Statistics
        self.retrain_count = 0
        self.samples_since_retrain = 0
        self.last_retrain_time = None
        self.best_validation_score = 0.0
        
        self.logger.info("Initialized retraining pipeline")
    
    def should_retrain(self, sample_count: int) -> bool:
        """
        Check if retraining should be triggered.
        
        Args:
            sample_count: Number of new samples
        
        Returns:
            Whether to retrain
        """
        self.samples_since_retrain += sample_count
        
        return (
            self.samples_since_retrain >= self.config.min_samples and
            self.samples_since_retrain >= self.config.retrain_frequency
        )
    
    def retrain(
        self,
        train_data: torch.utils.data.Dataset,
        validation_data: Optional[torch.utils.data.Dataset] = None
    ) -> Dict[str, Any]:
        """
        Perform retraining.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
        
        Returns:
            Retraining results
        """
        self.logger.info(f"Starting retraining #{self.retrain_count + 1}")
        start_time = time.time()
        
        # Backup old model
        if self.config.backup_old_model:
            self._backup_model()
        
        # Use custom train function if provided
        if self.train_fn:
            train_results = self.train_fn(
                self.model,
                train_data,
                self.config.max_epochs
            )
        else:
            train_results = self._default_train(train_data)
        
        # Validate
        if validation_data:
            val_score = self._validate(validation_data)
            
            # Save if improved
            if self.config.save_best_model and val_score > self.best_validation_score:
                self.best_validation_score = val_score
                self._save_model()
                improved = True
            else:
                improved = False
        else:
            val_score = None
            improved = True
            self._save_model()
        
        # Update statistics
        self.retrain_count += 1
        self.samples_since_retrain = 0
        self.last_retrain_time = time.time()
        
        duration = time.time() - start_time
        
        return {
            'retrain_count': self.retrain_count,
            'duration_seconds': duration,
            'validation_score': val_score,
            'improved': improved,
            'train_samples': len(train_data),
            **train_results
        }
    
    def _default_train(self, train_data: torch.utils.data.Dataset) -> Dict[str, Any]:
        """Default training loop."""
        self.model.train()
        
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=32,
            shuffle=True
        )
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            epoch_loss = 0.0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            
            # Early stopping
            if avg_loss < best_loss - self.config.min_improvement:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        return {
            'final_loss': best_loss,
            'epochs_trained': epoch + 1
        }
    
    def _validate(self, validation_data: torch.utils.data.Dataset) -> float:
        """Validate model."""
        if self.validate_fn:
            return self.validate_fn(self.model, validation_data)
        
        self.model.eval()
        correct = 0
        total = 0
        
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=32,
            shuffle=False
        )
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def _save_model(self):
        """Save model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'retrain_count': self.retrain_count,
            'best_validation_score': self.best_validation_score
        }, self.model_path)
        
        self.logger.info(f"Saved model to {self.model_path}")
    
    def _backup_model(self):
        """Backup current model."""
        if not self.model_path.exists():
            return
        
        backup_path = self.model_path.parent / f"{self.model_path.stem}_backup_{self.retrain_count}{self.model_path.suffix}"
        import shutil
        shutil.copy(self.model_path, backup_path)
        
        self.logger.info(f"Backed up model to {backup_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retraining statistics."""
        return {
            'retrain_count': self.retrain_count,
            'samples_since_retrain': self.samples_since_retrain,
            'last_retrain_time': self.last_retrain_time,
            'best_validation_score': self.best_validation_score
        }
