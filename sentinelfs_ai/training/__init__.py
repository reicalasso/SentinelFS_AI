"""Training package initialization."""

from .trainer import train_model
from .metrics import calculate_metrics, evaluate_model
from .early_stopping import EarlyStopping
from .real_trainer import RealWorldTrainer

__all__ = [
    'train_model', 
    'calculate_metrics', 
    'evaluate_model', 
    'EarlyStopping',
    'RealWorldTrainer'
]
