"""
Early stopping to prevent overfitting.
"""

import torch.nn as nn
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self, 
        patience: int = 7, 
        min_delta: float = 0.0001, 
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to save if score improves
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered")
                return True
        
        return False
