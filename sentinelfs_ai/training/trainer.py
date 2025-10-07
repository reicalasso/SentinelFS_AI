"""
Model training functionality with advanced techniques.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .metrics import calculate_metrics
from .early_stopping import EarlyStopping
from ..utils.logger import get_logger
from ..utils.device import get_device

logger = get_logger(__name__)


def train_model(
    model: nn.Module, 
    dataloaders: Dict[str, torch.utils.data.DataLoader], 
    epochs: int = 50, 
    lr: float = 0.001, 
    patience: int = 10,
    checkpoint_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    weight_decay: float = 1e-5,
    gradient_clipping: float = 1.0,
    use_scheduler: bool = True,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    log_interval: int = 5,
    enable_progress_bar: bool = True,
    loss_function: str = 'bce'  # 'bce', 'focal', 'weighted_bce'
) -> Dict[str, List[float]]:
    """
    Train the behavioral analyzer model with validation and early stopping.
    
    Args:
        model: The model to train
        dataloaders: Dictionary with 'train' and 'val' DataLoaders
        epochs: Maximum number of training epochs
        lr: Learning rate
        patience: Early stopping patience
        checkpoint_dir: Directory to save checkpoints
        device: Device to train on (auto-detected if None)
        weight_decay: Weight decay for optimizer
        gradient_clipping: Gradient clipping threshold (0 to disable)
        use_scheduler: Whether to use learning rate scheduler
        scheduler_patience: Patience for learning rate scheduler
        scheduler_factor: Factor to reduce learning rate
        log_interval: How often to log training progress
        enable_progress_bar: Whether to show training progress bar
        loss_function: Type of loss function to use ('bce', 'focal', 'weighted_bce')
        
    Returns:
        Dictionary containing training history
    """
    from ..management.checkpoint import save_checkpoint
    
    logger.info(f"Starting training for {epochs} epochs with lr={lr}")
    
    if device is None:
        device = get_device()
    
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # Initialize loss function
    if loss_function == 'bce':
        criterion = nn.BCELoss()
    elif loss_function == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    elif loss_function == 'weighted_bce':
        # Calculate class weights based on dataset
        if 'train' in dataloaders:
            # Estimate class distribution from training data
            total_samples = 0
            positive_samples = 0
            for _, labels in dataloaders['train']:
                total_samples += len(labels)
                positive_samples += labels.sum().item()
            
            pos_weight = (total_samples - positive_samples) / positive_samples if positive_samples > 0 else 1.0
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            criterion = nn.BCEWithLogitsLoss()
            logger.warning("Could not calculate class weights, using default BCEWithLogitsLoss")
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
        )
    
    early_stopping = EarlyStopping(patience=patience, mode='min')
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'learning_rate': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        # Initialize progress bar if enabled
        train_loader = dataloaders['train']
        if enable_progress_bar:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")
        
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Get model outputs
            outputs = model(batch_data)
            
            # Apply sigmoid if using BCE (not needed for BCEWithLogitsLoss)
            if loss_function == 'bce':
                outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)  # Avoid log(0) in BCE
                loss = criterion(outputs, batch_labels)
            else:
                # For BCEWithLogitsLoss, we expect raw logits
                if hasattr(model, 'get_raw_output'):
                    raw_outputs = model.get_raw_output(batch_data)
                else:
                    # Assume the model returns probabilities, need to convert to logits
                    outputs_clamped = torch.clamp(outputs, 1e-7, 1 - 1e-7)
                    raw_outputs = torch.log(outputs_clamped / (1 - outputs_clamped))
                loss = criterion(raw_outputs, batch_labels)
            
            loss.backward()
            
            # Gradient clipping
            if gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(batch_labels.detach().cpu().numpy())
        
        train_loss /= len(dataloaders['train'])
        train_metrics = calculate_metrics(
            np.array(train_preds).flatten(), np.array(train_labels).flatten()
        )
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in dataloaders['val']:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_data)
                
                # Apply same loss logic as training
                if loss_function == 'bce':
                    outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)
                    loss = criterion(outputs, batch_labels)
                else:
                    # For BCEWithLogitsLoss, we expect raw logits
                    outputs_clamped = torch.clamp(outputs, 1e-7, 1 - 1e-7)
                    raw_outputs = torch.log(outputs_clamped / (1 - outputs_clamped))
                    loss = criterion(raw_outputs, batch_labels)
                
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())
        
        val_loss /= len(dataloaders['val'])
        val_metrics = calculate_metrics(
            np.array(val_preds).flatten(), np.array(val_labels).flatten()
        )
        
        # Update learning rate
        if scheduler:
            scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Log progress
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1_score']:.4f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
        
        # Check early stopping
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(early_stopping.best_model_state)
            break
        
        # Save checkpoint
        if checkpoint_dir and (epoch + 1) % 10 == 0:
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
    
    logger.info("Training completed!")
    return history


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute BCE loss
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Compute focal weight
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()


def train_with_augmentation(
    model: nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 10,
    checkpoint_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    augmentation_factor: float = 0.2
) -> Dict[str, List[float]]:
    """
    Train model with data augmentation techniques.
    
    Args:
        model: Model to train
        dataloaders: Training and validation dataloaders
        epochs: Number of epochs
        lr: Learning rate
        patience: Early stopping patience
        checkpoint_dir: Directory to save checkpoints
        device: Training device
        augmentation_factor: Proportion of augmented samples per epoch
        
    Returns:
        Training history
    """
    # This is a simplified version - in a full implementation, we would
    # create augmented versions of the training data on the fly
    logger.info(f"Starting training with augmentation (factor: {augmentation_factor})")
    
    # For now, we just call the regular training function
    # In a full implementation, we would modify the dataloader to include augmented data
    return train_model(
        model=model,
        dataloaders=dataloaders,
        epochs=epochs,
        lr=lr,
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        device=device
    )


def train_robust(
    model: nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 10,
    checkpoint_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    adversarial_training: bool = False,
    adversarial_ratio: float = 0.3,
    epsilon: float = 0.01
) -> Dict[str, List[float]]:
    """
    Train model with robustness considerations.
    
    Args:
        model: Model to train
        dataloaders: Training and validation dataloaders
        epochs: Number of epochs
        lr: Learning rate
        patience: Early stopping patience
        checkpoint_dir: Directory to save checkpoints
        device: Training device
        adversarial_training: Whether to use adversarial training
        adversarial_ratio: Ratio of adversarial examples in each batch
        epsilon: Epsilon for adversarial perturbation
        
    Returns:
        Training history
    """
    from .adversarial_training import adversarial_training_step
    
    logger.info("Starting robust training (adversarial training enabled)" 
                if adversarial_training else "Starting robust training")
    
    if device is None:
        device = get_device()
    
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=patience, mode='min')
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch_data, batch_labels in dataloaders['train']:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            if adversarial_training:
                # Perform adversarial training step
                loss, accuracy = adversarial_training_step(
                    model, batch_data, batch_labels, criterion, 
                    optimizer, device, adversarial_ratio, epsilon
                )
                
                # Get predictions for metrics
                with torch.no_grad():
                    outputs = model(batch_data)
                    train_preds.extend(outputs.detach().cpu().numpy())
                    train_labels.extend(batch_labels.detach().cpu().numpy())
            else:
                # Regular training step
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(outputs.detach().cpu().numpy())
                train_labels.extend(batch_labels.detach().cpu().numpy())
        
        if not adversarial_training:
            train_loss /= len(dataloaders['train'])
        
        train_metrics = calculate_metrics(
            np.array(train_preds).flatten(), np.array(train_labels).flatten()
        )
        
        # Validation phase (always without adversarial training)
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in dataloaders['val']:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())
        
        val_loss /= len(dataloaders['val'])
        val_metrics = calculate_metrics(
            np.array(val_preds).flatten(), np.array(val_labels).flatten()
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss if adversarial_training else train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1_score'])
        
        # Log progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss if adversarial_training else train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1_score']:.4f}"
            )
        
        # Check early stopping
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(early_stopping.best_model_state)
            break
        
        # Save checkpoint
        if checkpoint_dir and (epoch + 1) % 10 == 0:
            checkpoint_path = Path(checkpoint_dir) / f"robust_checkpoint_epoch_{epoch+1}.pt"
            from ..management.checkpoint import save_checkpoint
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
    
    logger.info("Robust training completed!")
    return history
