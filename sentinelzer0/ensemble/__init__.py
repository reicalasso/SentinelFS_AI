"""
Ensemble Management Framework

Provides robust multi-model ensemble capabilities for improved detection accuracy.
"""

from .voting_system import EnsembleVoter, VotingResult
from .model_architectures import CNNDetector, LSTMDetector, TransformerDetector
from .training_pipeline import EnsembleTrainer, TrainingConfig
from .diversity_metrics import DiversityAnalyzer, DiversityMetrics
from .manager import EnsembleManager

__all__ = [
    'EnsembleVoter', 'VotingResult',
    'CNNDetector', 'LSTMDetector', 'TransformerDetector',
    'EnsembleTrainer', 'TrainingConfig',
    'DiversityAnalyzer', 'DiversityMetrics',
    'EnsembleManager',
]

__version__ = '3.7.0'
