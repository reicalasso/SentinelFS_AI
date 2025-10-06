"""
SentinelFS AI Behavioral Analyzer
Modular AI-powered anomaly detection for distributed file system security.
"""

__version__ = '1.0.0'

# Core types
from .data_types import (
    AnalysisResult,
    AnomalyType,
    TrainingConfig
)

# Models
from .models.behavioral_analyzer import BehavioralAnalyzer
from .models.attention import AttentionLayer
from .models.advanced_models import (
    TransformerBehavioralAnalyzer,
    CNNLSTMAnalyzer,
    EnsembleAnalyzer,
    AdaptiveAnalyzer
)

# Data processing
from .data.feature_extractor import FeatureExtractor
from .data.data_processor import DataProcessor
from .data.data_generator import generate_sample_data
from .data.realistic_data_generator import generate_realistic_access_data

# Training
from .training.trainer import train_model
from .training.metrics import calculate_metrics, evaluate_model
from .training.early_stopping import EarlyStopping
from .training.adversarial_training import (
    AdversarialTrainer, 
    RobustnessEvaluator, 
    generate_adversarial_examples
)
from .training.ensemble_training import EnsembleManager, create_weighted_ensemble

# Evaluation
from .evaluation.advanced_evaluation import (
    AdvancedEvaluator, 
    plot_roc_curve, 
    plot_precision_recall_curve,
    plot_confusion_matrix,
    calibration_plot
)

# Inference
from .inference.engine import InferenceEngine

# Management
from .management.model_manager import ModelManager
from .management.checkpoint import save_checkpoint, load_checkpoint

# Utils
from .utils.logger import get_logger

__all__ = [
    # Version
    '__version__',
    
    # Types
    'AnalysisResult',
    'AnomalyType',
    'TrainingConfig',
    
    # Models
    'BehavioralAnalyzer',
    'AttentionLayer',
    'TransformerBehavioralAnalyzer',
    'CNNLSTMAnalyzer',
    'EnsembleAnalyzer',
    'AdaptiveAnalyzer',
    
    # Data
    'FeatureExtractor',
    'DataProcessor',
    'generate_sample_data',
    
    # New data generators
    'generate_realistic_access_data',
    
    # Training
    'train_model',
    'calculate_metrics',
    'evaluate_model',
    'EarlyStopping',
    'AdversarialTrainer',
    'RobustnessEvaluator',
    'generate_adversarial_examples',
    'EnsembleManager',
    'create_weighted_ensemble',

# Evaluation
    'AdvancedEvaluator',
    'plot_roc_curve',
    'plot_precision_recall_curve', 
    'plot_confusion_matrix',
    'calibration_plot',
    
    # Inference
    'InferenceEngine',
    
    # Management
    'ModelManager',
    'save_checkpoint',
    'load_checkpoint',
    
    # Utils
    'get_logger',
]
