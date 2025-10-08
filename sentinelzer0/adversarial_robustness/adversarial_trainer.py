"""
Adversarial Training Framework
==============================

Implements adversarial training to improve model robustness.
Uses on-the-fly adversarial example generation during training.

Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import time

from .attack_generator import AttackGenerator, AttackConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for adversarial training"""
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Adversarial training parameters
    adversarial_ratio: float = 0.5  # Ratio of adversarial examples
    attack_type: str = 'pgd'  # Attack method to use
    warmup_epochs: int = 5  # Epochs before adversarial training
    
    # Attack configuration
    attack_config: AttackConfig = field(default_factory=AttackConfig)
    
    # Learning rate scheduling
    use_lr_schedule: bool = True
    lr_decay_epochs: list = field(default_factory=lambda: [75, 90])
    lr_decay_rate: float = 0.1
    
    # Regularization
    use_label_smoothing: bool = True
    label_smoothing: float = 0.1
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 1
    save_interval: int = 10
    checkpoint_dir: str = "checkpoints/adversarial"
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_delta: float = 0.001


class AdversarialTrainer:
    """
    Adversarial Training Manager
    
    Trains models with adversarial examples to improve robustness.
    Supports multiple training strategies and attack methods.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize adversarial trainer
        
        Args:
            model: Model to train
            config: Training configuration
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize attack generator
        self.attack_generator = AttackGenerator(model, self.config.attack_config)
        
        # Initialize optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Initialize loss function
        if self.config.use_label_smoothing:
            self.criterion = self._label_smoothing_loss
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Training statistics
        self.train_stats = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'clean_acc': [],
            'adv_acc': [],
            'learning_rate': []
        }
        
        self.best_acc = 0.0
        self.early_stopping_counter = 0
        
    def _label_smoothing_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate loss with label smoothing
        
        Args:
            outputs: Model predictions [batch_size, num_classes]
            targets: True labels [batch_size]
            
        Returns:
            Smoothed loss
        """
        n_classes = outputs.size(1)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        
        # One-hot encoding with smoothing
        targets_one_hot = torch.zeros_like(outputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * (1 - self.config.label_smoothing) + \
                        self.config.label_smoothing / n_classes
        
        loss = -(targets_smooth * log_probs).sum(dim=1).mean()
        return loss
    
    def _adjust_learning_rate(self, epoch: int):
        """Adjust learning rate based on schedule"""
        lr = self.config.learning_rate
        for decay_epoch in self.config.lr_decay_epochs:
            if epoch >= decay_epoch:
                lr *= self.config.lr_decay_rate
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Determine if using adversarial training this epoch
        use_adversarial = epoch >= self.config.warmup_epochs
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Generate adversarial examples
            if use_adversarial and torch.rand(1).item() < self.config.adversarial_ratio:
                # Generate adversarial examples
                with torch.enable_grad():
                    data_adv = self.attack_generator.generate(
                        data, target, self.config.attack_type
                    )
                data = data_adv
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            # Handle tuple outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Calculate loss
            loss = self.criterion(outputs, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                    'adv': 'Yes' if use_adversarial else 'No'
                })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(
        self,
        test_loader: DataLoader,
        use_adversarial: bool = False
    ) -> Tuple[float, float]:
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            use_adversarial: Whether to use adversarial examples
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Generate adversarial examples if requested
                if use_adversarial:
                    data = self.attack_generator.generate(
                        data, target, self.config.attack_type
                    )
                
                # Forward pass
                outputs = self.model(data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Calculate loss
                loss = self.criterion(outputs, target)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader for evaluation
            
        Returns:
            Training statistics
        """
        logger.info("Starting adversarial training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Attack: {self.config.attack_type}")
        logger.info(f"Adversarial ratio: {self.config.adversarial_ratio}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            # Adjust learning rate
            if self.config.use_lr_schedule:
                self._adjust_learning_rate(epoch)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Evaluation
            if test_loader and epoch % self.config.eval_interval == 0:
                clean_loss, clean_acc = self.evaluate(test_loader, use_adversarial=False)
                adv_loss, adv_acc = self.evaluate(test_loader, use_adversarial=True)
                
                # Log statistics
                current_lr = self.optimizer.param_groups[0]['lr']
                self.train_stats['epoch'].append(epoch)
                self.train_stats['train_loss'].append(train_loss)
                self.train_stats['train_acc'].append(train_acc)
                self.train_stats['clean_acc'].append(clean_acc)
                self.train_stats['adv_acc'].append(adv_acc)
                self.train_stats['learning_rate'].append(current_lr)
                
                logger.info(
                    f"Epoch {epoch}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Clean Acc: {clean_acc:.2f}% | Adv Acc: {adv_acc:.2f}% | "
                    f"LR: {current_lr:.6f}"
                )
                
                # Save best model
                if adv_acc > self.best_acc:
                    self.best_acc = adv_acc
                    self.save_checkpoint(f"{self.config.checkpoint_dir}/best_model.pt")
                    self.early_stopping_counter = 0
                    logger.info(f"New best adversarial accuracy: {adv_acc:.2f}%")
                else:
                    self.early_stopping_counter += 1
                
                # Early stopping
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(
                    f"{self.config.checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
                )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Best adversarial accuracy: {self.best_acc:.2f}%")
        
        return {
            'training_time': training_time,
            'best_accuracy': self.best_acc,
            'final_epoch': epoch,
            'stats': self.train_stats
        }
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'stats': self.train_stats,
            'best_acc': self.best_acc
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_stats = checkpoint['stats']
        self.best_acc = checkpoint['best_acc']
        
        logger.info(f"Loaded checkpoint from {path}")
        logger.info(f"Best accuracy: {self.best_acc:.2f}%")
