"""
Checkpoint saving and loading functionality.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    loss: float, 
    filepath: Path,
    metadata: Optional[Dict] = None
):
    """
    Save model checkpoint with optimizer state and metadata.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        loss: Current loss value
        filepath: Path to save checkpoint
        metadata: Additional metadata to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: nn.Module, 
    filepath: Path, 
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load state into
        filepath: Path to checkpoint file
        optimizer: Optional optimizer to load state into
        device: Device to load model onto
        
    Returns:
        Checkpoint dictionary with metadata
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {filepath} (epoch {checkpoint['epoch']})")
    return checkpoint
