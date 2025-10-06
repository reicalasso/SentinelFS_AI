"""
Model training functionality.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

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
    device: Optional[torch.device] = None
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
        
    Returns:
        Dictionary containing training history
    """
    from ..management.checkpoint import save_checkpoint
    
    logger.info(f"Starting training for {epochs} epochs with lr={lr}")
    
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
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(batch_labels.detach().cpu().numpy())
        
        train_loss /= len(dataloaders['train'])
        train_metrics = calculate_metrics(
            np.array(train_preds), np.array(train_labels)
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
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())
        
        val_loss /= len(dataloaders['val'])
        val_metrics = calculate_metrics(
            np.array(val_preds), np.array(val_labels)
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
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
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
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
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
    
    logger.info("Training completed!")
    return history
