"""
Incremental Learning

Implements online learning algorithms that allow models to learn from
new data without full retraining.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import logging
import numpy as np


class LearningStrategy(Enum):
    """Learning strategy options."""
    SGD = "sgd"  # Stochastic gradient descent
    MINI_BATCH = "mini_batch"  # Mini-batch updates
    REPLAY_BUFFER = "replay_buffer"  # Experience replay
    EWMA = "ewma"  # Exponentially weighted moving average


class IncrementalLearner:
    """
    Incremental learning system for online model updates.
    
    Features:
    - Stochastic gradient descent (SGD)
    - Mini-batch learning
    - Experience replay buffer
    - Learning rate scheduling
    - Catastrophic forgetting mitigation
    - Memory-efficient updates
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        strategy: LearningStrategy = LearningStrategy.MINI_BATCH,
        batch_size: int = 32,
        buffer_size: int = 1000,
        ewma_alpha: float = 0.9
    ):
        """
        Initialize incremental learner.
        
        Args:
            model: Model to train incrementally
            learning_rate: Learning rate for updates
            strategy: Learning strategy to use
            batch_size: Batch size for mini-batch learning
            buffer_size: Size of replay buffer
            ewma_alpha: Alpha for EWMA updates
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.strategy = strategy
        self.batch_size = batch_size
        self.ewma_alpha = ewma_alpha
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Replay buffer for experience replay
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Mini-batch accumulator
        self.batch_accumulator = []
        
        # Statistics
        self.update_count = 0
        self.total_samples = 0
        self.recent_losses = deque(maxlen=100)
        
        self.logger.info(f"Initialized incremental learner with {strategy.value} strategy")
    
    def update(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform incremental update with new samples.
        
        Args:
            inputs: Input tensor
            labels: Label tensor
            weights: Optional sample weights
        
        Returns:
            Update statistics
        """
        if self.strategy == LearningStrategy.SGD:
            return self._update_sgd(inputs, labels, weights)
        elif self.strategy == LearningStrategy.MINI_BATCH:
            return self._update_mini_batch(inputs, labels, weights)
        elif self.strategy == LearningStrategy.REPLAY_BUFFER:
            return self._update_replay_buffer(inputs, labels, weights)
        elif self.strategy == LearningStrategy.EWMA:
            return self._update_ewma(inputs, labels, weights)
        else:
            raise ValueError(f"Unknown learning strategy: {self.strategy}")
    
    def _update_sgd(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Perform SGD update (one sample at a time)."""
        self.model.train()
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Compute loss
        if weights is not None:
            loss = (self.criterion(outputs, labels) * weights).mean()
        else:
            loss = self.criterion(outputs, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update statistics
        self.update_count += 1
        self.total_samples += inputs.size(0)
        self.recent_losses.append(loss.item())
        
        return {
            'loss': loss.item(),
            'samples': inputs.size(0),
            'updates': 1
        }
    
    def _update_mini_batch(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Accumulate samples and update when batch is full."""
        # Add to accumulator
        self.batch_accumulator.append((inputs, labels, weights))
        
        # Check if batch is ready
        if len(self.batch_accumulator) >= self.batch_size:
            return self._flush_batch()
        
        return {
            'loss': 0.0,
            'samples': inputs.size(0),
            'updates': 0,
            'accumulated': len(self.batch_accumulator)
        }
    
    def _flush_batch(self) -> Dict[str, float]:
        """Process accumulated mini-batch."""
        if not self.batch_accumulator:
            return {'loss': 0.0, 'samples': 0, 'updates': 0}
        
        # Combine batch
        all_inputs = []
        all_labels = []
        all_weights = []
        
        for inputs, labels, weights in self.batch_accumulator:
            all_inputs.append(inputs)
            all_labels.append(labels)
            if weights is not None:
                all_weights.append(weights)
        
        batch_inputs = torch.cat(all_inputs, dim=0)
        batch_labels = torch.cat(all_labels, dim=0)
        batch_weights = torch.cat(all_weights, dim=0) if all_weights else None
        
        # Clear accumulator
        batch_size = len(self.batch_accumulator)
        self.batch_accumulator = []
        
        # Perform update
        stats = self._update_sgd(batch_inputs, batch_labels, batch_weights)
        stats['batch_size'] = batch_size
        
        return stats
    
    def _update_replay_buffer(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Update with experience replay to prevent catastrophic forgetting."""
        # Add new samples to replay buffer
        for i in range(inputs.size(0)):
            sample = (
                inputs[i:i+1],
                labels[i:i+1],
                weights[i:i+1] if weights is not None else None
            )
            self.replay_buffer.append(sample)
        
        # Sample from replay buffer
        if len(self.replay_buffer) < self.batch_size:
            # Not enough samples yet
            return {
                'loss': 0.0,
                'samples': inputs.size(0),
                'updates': 0,
                'buffer_size': len(self.replay_buffer)
            }
        
        # Random sample from buffer
        indices = np.random.choice(
            len(self.replay_buffer),
            size=self.batch_size,
            replace=False
        )
        
        replay_inputs = []
        replay_labels = []
        replay_weights = []
        
        for idx in indices:
            inp, lab, wei = self.replay_buffer[idx]
            replay_inputs.append(inp)
            replay_labels.append(lab)
            if wei is not None:
                replay_weights.append(wei)
        
        batch_inputs = torch.cat(replay_inputs, dim=0)
        batch_labels = torch.cat(replay_labels, dim=0)
        batch_weights = torch.cat(replay_weights, dim=0) if replay_weights else None
        
        # Perform update
        stats = self._update_sgd(batch_inputs, batch_labels, batch_weights)
        stats['buffer_size'] = len(self.replay_buffer)
        
        return stats
    
    def _update_ewma(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Update using exponentially weighted moving average.
        
        Gradually adapts model parameters using EWMA to balance
        old and new knowledge.
        """
        # Save old parameters
        old_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }
        
        # Perform regular SGD update
        stats = self._update_sgd(inputs, labels, weights)
        
        # Apply EWMA to parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = (
                    self.ewma_alpha * old_params[name] +
                    (1 - self.ewma_alpha) * param.data
                )
        
        stats['ewma_alpha'] = self.ewma_alpha
        
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'update_count': self.update_count,
            'total_samples': self.total_samples,
            'recent_avg_loss': np.mean(list(self.recent_losses)) if self.recent_losses else 0.0,
            'buffer_size': len(self.replay_buffer) if self.strategy == LearningStrategy.REPLAY_BUFFER else 0,
            'accumulated_samples': len(self.batch_accumulator) if self.strategy == LearningStrategy.MINI_BATCH else 0,
            'strategy': self.strategy.value
        }
    
    def reset_statistics(self):
        """Reset learning statistics."""
        self.update_count = 0
        self.total_samples = 0
        self.recent_losses.clear()
    
    def save_checkpoint(self, path: str):
        """Save learner checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'total_samples': self.total_samples,
            'strategy': self.strategy.value,
            'replay_buffer': list(self.replay_buffer) if self.strategy == LearningStrategy.REPLAY_BUFFER else None
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load learner checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)
        self.total_samples = checkpoint.get('total_samples', 0)
        
        if checkpoint.get('replay_buffer') and self.strategy == LearningStrategy.REPLAY_BUFFER:
            self.replay_buffer = deque(checkpoint['replay_buffer'], maxlen=self.replay_buffer.maxlen)
        
        self.logger.info(f"Loaded checkpoint from {path}")
    
    def adapt_learning_rate(self, factor: float):
        """Adapt learning rate by multiplication factor."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] *= factor
            new_lr = param_group['lr']
        
        self.logger.info(f"Adapted learning rate: {old_lr:.6f} -> {new_lr:.6f}")
    
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
