"""
Adversarial training functionality for improving model robustness.
Includes FGSM, PGD, and other adversarial attack methods for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


def fgsm_attack(
    model: nn.Module, 
    data: torch.Tensor, 
    target: torch.Tensor, 
    epsilon: float = 0.01
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.
    
    Args:
        model: The model to attack
        data: Input data
        target: Target labels
        epsilon: Perturbation magnitude
        
    Returns:
        Adversarial examples
    """
    was_training = model.training  # Remember if model was in training mode
    model.train()  # Ensure gradients can flow through RNN layers
    
    data = data.clone().detach().requires_grad_(True)
    
    output = model(data)
    loss = F.binary_cross_entropy(output, target)
    
    # Compute gradients
    model.zero_grad()
    loss.backward(retain_graph=True)
    
    # Generate adversarial example
    sign_data_grad = data.grad.data.sign()
    perturbed_data = data + epsilon * sign_data_grad
    perturbed_data = torch.clamp(perturbed_data, 0, 1)  # Adjust bounds as needed
    
    # Restore original training state
    if not was_training:
        model.eval()
    
    return perturbed_data


def pgd_attack(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 0.01,
    alpha: float = 0.001,
    num_iter: int = 10
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) adversarial attack.
    
    Args:
        model: The model to attack
        data: Input data
        target: Target labels
        epsilon: Maximum perturbation
        alpha: Step size
        num_iter: Number of iterations
        
    Returns:
        Adversarial examples
    """
    was_training = model.training  # Remember if model was in training mode
    model.eval()  # Set model to evaluation mode
    
    perturbed_data = data.clone().detach().requires_grad_(True)
    
    for _ in range(num_iter):
        if perturbed_data.grad is not None:
            perturbed_data.grad.zero_()
        
        output = model(perturbed_data)
        loss = F.binary_cross_entropy(output, target)
        
        loss.backward(retain_graph=True)
        
        # Update perturbed data
        data_grad = perturbed_data.grad.data
        perturbed_data = perturbed_data.detach() + alpha * data_grad.sign()
        
        # Project to epsilon ball
        delta = torch.clamp(perturbed_data - data, min=-epsilon, max=epsilon)
        perturbed_data = torch.clamp(data + delta, 0, 1)  # Adjust bounds as needed

    # Restore original training state
    if was_training:
        model.train()
    
    return perturbed_data


def adversarial_training_step(
    model: nn.Module,
    batch_data: torch.Tensor,
    batch_labels: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    adversarial_ratio: float = 0.3,
    epsilon: float = 0.01
) -> Tuple[torch.Tensor, float]:
    """
    Perform one training step with adversarial examples.
    
    Args:
        model: Model to train
        batch_data: Input batch data
        batch_labels: Input batch labels
        criterion: Loss function
        optimizer: Optimizer
        device: Training device
        adversarial_ratio: Ratio of adversarial examples in batch
        epsilon: Adversarial perturbation magnitude
        
    Returns:
        (loss, accuracy) tuple
    """
    model.train()
    
    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
    
    # Determine how many samples should be adversarial
    num_adversarial = int(batch_data.size(0) * adversarial_ratio)
    
    if num_adversarial > 0:
        # Split batch into clean and adversarial
        clean_data = batch_data[:batch_data.size(0) - num_adversarial]
        clean_labels = batch_labels[:batch_labels.size(0) - num_adversarial]
        
        adversarial_data = batch_data[batch_data.size(0) - num_adversarial:]
        adversarial_labels = batch_labels[batch_labels.size(0) - num_adversarial:]
        
        # Generate adversarial examples; enable gradients so FGSM can compute input grads
        with torch.enable_grad():
            adversarial_examples = fgsm_attack(
                model, adversarial_data, adversarial_labels, epsilon
            )
        adversarial_examples = adversarial_examples.detach()
        
        # Combine clean and adversarial data
        combined_data = torch.cat([clean_data, adversarial_examples], dim=0)
        combined_labels = torch.cat([clean_labels, adversarial_labels], dim=0)
    else:
        combined_data = batch_data
        combined_labels = batch_labels
    
    # Forward pass
    optimizer.zero_grad()
    outputs = model(combined_data)
    loss = criterion(outputs, combined_labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    predicted = (outputs > 0.5).float()
    accuracy = (predicted == combined_labels).float().mean().item()
    
    return loss, accuracy


class AdversarialTrainer:
    """
    Enhanced trainer with adversarial training capabilities.
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        epochs: int = 50,
        lr: float = 0.001,
        patience: int = 10,
        adversarial_ratio: float = 0.3,
        epsilon: float = 0.01,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.dataloaders = dataloaders
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.adversarial_ratio = adversarial_ratio
        self.epsilon = epsilon
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup model and optimizer
        self.model = self.model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
    def train(self) -> Dict[str, List[float]]:
        """Train the model with adversarial examples."""
        from ..training.early_stopping import EarlyStopping
        from ..training.metrics import calculate_metrics
        from ..management.checkpoint import save_checkpoint
        from pathlib import Path
        
        early_stopping = EarlyStopping(patience=self.patience, mode='min')
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        
        logger.info(f"Starting adversarial training for {self.epochs} epochs")
        logger.info(f"Adversarial ratio: {self.adversarial_ratio}, Epsilon: {self.epsilon}")
        
        for epoch in range(self.epochs):
            # Training phase with adversarial examples
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            
            for batch_data, batch_labels in self.dataloaders['train']:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                loss, acc = adversarial_training_step(
                    self.model, batch_data, batch_labels, self.criterion,
                    self.optimizer, self.device, self.adversarial_ratio, self.epsilon
                )
                
                train_loss += loss.item()
                # Get predictions for metrics
                with torch.no_grad():
                    outputs = self.model(batch_data)
                    train_preds.extend(outputs.detach().cpu().numpy())
                    train_labels.extend(batch_labels.detach().cpu().numpy())
            
            train_loss /= len(self.dataloaders['train'])
            train_metrics = calculate_metrics(
                np.array(train_preds), np.array(train_labels)
            )
            
            # Validation phase (without adversarial training to measure true performance)
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch_data, batch_labels in self.dataloaders['val']:
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_data)
                    loss = self.criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(batch_labels.cpu().numpy())
            
            val_loss /= len(self.dataloaders['val'])
            val_metrics = calculate_metrics(
                np.array(val_preds), np.array(val_labels)
            )
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['val_f1'].append(val_metrics['f1_score'])
            
            # Log progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Val F1: {val_metrics['f1_score']:.4f}"
                )
            
            # Check early stopping
            if early_stopping(val_loss, self.model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                self.model.load_state_dict(early_stopping.best_model_state)
                break
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = Path('./checkpoints') / f"adv_checkpoint_epoch_{epoch+1}.pt"
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, checkpoint_path)
        
        logger.info("Adversarial training completed!")
        return history


def generate_adversarial_examples(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    method: str = 'fgsm',
    epsilon: float = 0.01
) -> torch.Tensor:
    """
    Generate adversarial examples using specified method.
    
    Args:
        model: Model to attack
        data: Input data
        labels: True labels
        method: Attack method ('fgsm', 'pgd')
        epsilon: Perturbation magnitude
        
    Returns:
        Adversarial examples
    """
    if method == 'fgsm':
        return fgsm_attack(model, data, labels, epsilon)
    elif method == 'pgd':
        return pgd_attack(model, data, labels, epsilon)
    else:
        raise ValueError(f"Unknown attack method: {method}")


class RobustnessEvaluator:
    """
    Evaluate model robustness against adversarial attacks.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def evaluate_robustness(
        self, 
        test_data: torch.Tensor, 
        test_labels: torch.Tensor,
        epsilons: List[float] = [0.001, 0.005, 0.01, 0.02, 0.05]
    ) -> Dict[str, List[float]]:
        """
        Evaluate model performance under different adversarial attack strengths.
        
        Args:
            test_data: Test dataset
            test_labels: Test labels
            epsilons: List of epsilon values to test
            
        Returns:
            Dictionary with accuracy under different attack strengths
        """
        results = {
            'epsilon': epsilons,
            'clean_accuracy': [],
            'adversarial_accuracy': []
        }
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            # Test clean accuracy
            clean_data = test_data.to(device)
            clean_labels = test_labels.to(device)
            clean_outputs = self.model(clean_data)
            clean_preds = (clean_outputs > 0.5).float()
            clean_acc = (clean_preds == clean_labels).float().mean().item()
            
            results['clean_accuracy'] = [clean_acc] * len(epsilons)
            
            # Test adversarial accuracy for different epsilons
            for eps in epsilons:
                adversarial_data = fgsm_attack(
                    self.model, clean_data, clean_labels, eps
                )
                adversarial_outputs = self.model(adversarial_data)
                adversarial_preds = (adversarial_outputs > 0.5).float()
                adversarial_acc = (adversarial_preds == clean_labels).float().mean().item()
                
                results['adversarial_accuracy'].append(adversarial_acc)
                
        return results