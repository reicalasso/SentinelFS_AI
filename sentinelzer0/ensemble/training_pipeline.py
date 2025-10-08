"""
Ensemble Training Pipeline

Trains multiple diverse models for ensemble.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging
from dataclasses import dataclass
from pathlib import Path
import json

from .model_architectures import CNNDetector, LSTMDetector, TransformerDetector, DeepMLPDetector


@dataclass
class TrainingConfig:
    """Configuration for ensemble training."""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    diversity_weight: float = 0.1  # Weight for diversity loss
    use_bagging: bool = True  # Use bagging (bootstrap sampling)
    bag_fraction: float = 0.8  # Fraction of data for each model
    save_dir: Optional[Path] = None


class EnsembleTrainer:
    """
    Trains ensemble of diverse models.
    
    Features:
    - Multiple architecture support
    - Diversity promotion
    - Bagging (bootstrap aggregating)
    - Early stopping
    - Checkpointing
    - Training metrics tracking
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        device: Optional[torch.device] = None
    ):
        """
        Initialize ensemble trainer.
        
        Args:
            config: Training configuration
            device: Device to train on
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.models = []
        self.optimizers = []
        self.training_history = []
        
        self.logger.info(f"Initialized ensemble trainer on {self.device}")
    
    def create_ensemble(
        self,
        input_dim: int,
        num_classes: int = 2,
        architectures: Optional[List[str]] = None
    ) -> List[nn.Module]:
        """
        Create ensemble of diverse models.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            architectures: List of architecture names
        
        Returns:
            List of models
        """
        if architectures is None:
            architectures = ['cnn', 'lstm', 'transformer', 'deep_mlp']
        
        models = []
        
        for arch in architectures:
            if arch.lower() == 'cnn':
                model = CNNDetector(input_dim, num_classes=num_classes)
            elif arch.lower() == 'lstm':
                model = LSTMDetector(input_dim, num_classes=num_classes)
            elif arch.lower() == 'transformer':
                model = TransformerDetector(input_dim, num_classes=num_classes)
            elif arch.lower() == 'deep_mlp':
                model = DeepMLPDetector(input_dim, num_classes=num_classes)
            else:
                raise ValueError(f"Unknown architecture: {arch}")
            
            model = model.to(self.device)
            models.append(model)
            
            self.logger.info(f"Created {arch} model")
        
        self.models = models
        
        # Create optimizers
        self.optimizers = [
            optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            for model in models
        ]
        
        return models
    
    def train(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Train ensemble of models.
        
        Args:
            train_data: Training data
            train_labels: Training labels
            val_data: Validation data (optional)
            val_labels: Validation labels (optional)
        
        Returns:
            Training history
        """
        if len(self.models) == 0:
            raise ValueError("No models created. Call create_ensemble() first.")
        
        self.logger.info(f"Training ensemble of {len(self.models)} models")
        
        # Prepare data
        if self.config.use_bagging:
            # Create bootstrap samples for each model
            loaders = self._create_bagging_loaders(train_data, train_labels)
        else:
            # All models use same data
            dataset = TensorDataset(train_data, train_labels)
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            loaders = [loader] * len(self.models)
        
        # Validation loader
        val_loader = None
        if val_data is not None and val_labels is not None:
            val_dataset = TensorDataset(val_data, val_labels)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        
        # Training loop
        history = {f'model_{i}': [] for i in range(len(self.models))}
        
        for epoch in range(self.config.epochs):
            epoch_metrics = self._train_epoch(loaders, val_loader)
            
            # Store history
            for i, metrics in enumerate(epoch_metrics):
                history[f'model_{i}'].append(metrics)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                avg_loss = np.mean([m['train_loss'] for m in epoch_metrics])
                avg_acc = np.mean([m['train_acc'] for m in epoch_metrics])
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs}: "
                    f"Avg Loss={avg_loss:.4f}, Avg Acc={avg_acc:.4f}"
                )
            
            # Early stopping (if validation available)
            if val_loader is not None and self._should_stop_early(history, epoch):
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        self.training_history = history
        
        # Save models
        if self.config.save_dir:
            self._save_models(self.config.save_dir)
        
        return history
    
    def _train_epoch(
        self,
        loaders: List[DataLoader],
        val_loader: Optional[DataLoader]
    ) -> List[Dict[str, float]]:
        """Train one epoch for all models."""
        epoch_metrics = []
        
        for model_idx, (model, optimizer, loader) in enumerate(
            zip(self.models, self.optimizers, loaders)
        ):
            model.train()
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_data, batch_labels in loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_data)
                
                # Classification loss
                loss = F.cross_entropy(outputs, batch_labels)
                
                # Diversity loss (if multiple models)
                if self.config.diversity_weight > 0 and len(self.models) > 1:
                    diversity_loss = self._compute_diversity_loss(
                        model, batch_data, model_idx
                    )
                    loss = loss + self.config.diversity_weight * diversity_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Metrics
                total_loss += loss.item()
                predictions = outputs.argmax(dim=-1)
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)
            
            # Epoch metrics
            train_loss = total_loss / len(loader)
            train_acc = correct / total
            
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc
            }
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._validate(model, val_loader)
                metrics['val_loss'] = val_loss
                metrics['val_acc'] = val_acc
            
            epoch_metrics.append(metrics)
        
        return epoch_metrics
    
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """Validate model."""
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = model(batch_data)
                loss = F.cross_entropy(outputs, batch_labels)
                
                total_loss += loss.item()
                predictions = outputs.argmax(dim=-1)
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)
        
        val_loss = total_loss / len(val_loader)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def _compute_diversity_loss(
        self,
        model: nn.Module,
        batch_data: torch.Tensor,
        model_idx: int
    ) -> torch.Tensor:
        """
        Compute diversity loss to encourage model disagreement.
        
        Uses negative correlation between predictions.
        """
        if len(self.models) <= 1:
            return torch.tensor(0.0, device=self.device)
        
        # Get current model predictions
        current_outputs = model(batch_data)
        current_probs = F.softmax(current_outputs, dim=-1)
        
        # Get predictions from other models
        diversity_loss = 0.0
        
        for i, other_model in enumerate(self.models):
            if i == model_idx:
                continue
            
            other_model.eval()
            with torch.no_grad():
                other_outputs = other_model(batch_data)
                other_probs = F.softmax(other_outputs, dim=-1)
            
            # Negative correlation (encourage diversity)
            correlation = (current_probs * other_probs).sum(dim=-1).mean()
            diversity_loss += correlation
        
        diversity_loss /= (len(self.models) - 1)
        
        return diversity_loss
    
    def _create_bagging_loaders(
        self,
        data: torch.Tensor,
        labels: torch.Tensor
    ) -> List[DataLoader]:
        """Create bootstrap sampling loaders for bagging."""
        loaders = []
        n_samples = len(data)
        bag_size = int(n_samples * self.config.bag_fraction)
        
        for _ in range(len(self.models)):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, bag_size, replace=True)
            sampler = SubsetRandomSampler(indices)
            
            dataset = TensorDataset(data, labels)
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=sampler
            )
            loaders.append(loader)
        
        return loaders
    
    def _should_stop_early(
        self,
        history: Dict[str, List[Dict]],
        current_epoch: int
    ) -> bool:
        """Check early stopping condition."""
        if current_epoch < self.config.early_stopping_patience:
            return False
        
        # Check if all models haven't improved
        for model_key, model_history in history.items():
            if 'val_loss' not in model_history[0]:
                continue
            
            recent_losses = [
                h['val_loss']
                for h in model_history[-self.config.early_stopping_patience:]
            ]
            
            if len(recent_losses) < self.config.early_stopping_patience:
                continue
            
            # If any model is still improving, don't stop
            if recent_losses[-1] < min(recent_losses[:-1]):
                return False
        
        return True
    
    def _save_models(self, save_dir: Path):
        """Save all models."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = save_dir / f"model_{i}.pt"
            torch.save(model.state_dict(), model_path)
            self.logger.info(f"Saved model {i} to {model_path}")
        
        # Save training history
        history_path = save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_models(self, load_dir: Path, model_configs: List[Dict]):
        """
        Load saved models.
        
        Args:
            load_dir: Directory with saved models
            model_configs: List of model configurations
        """
        for i, config in enumerate(model_configs):
            model_path = load_dir / f"model_{i}.pt"
            
            # Create model based on config
            arch = config['architecture']
            if arch == 'cnn':
                model = CNNDetector(**config.get('params', {}))
            elif arch == 'lstm':
                model = LSTMDetector(**config.get('params', {}))
            elif arch == 'transformer':
                model = TransformerDetector(**config.get('params', {}))
            elif arch == 'deep_mlp':
                model = DeepMLPDetector(**config.get('params', {}))
            else:
                raise ValueError(f"Unknown architecture: {arch}")
            
            # Load weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            
            self.models.append(model)
            self.logger.info(f"Loaded model {i} from {model_path}")


import torch.nn.functional as F  # Import at top level for use in methods
