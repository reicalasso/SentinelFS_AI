"""
SentinelFS AI Behavioral Analyzer
Modular AI-powered anomaly detection for distributed file system security.

This package provides advanced deep learning models for detecting anomalous file access patterns
in real-time. It includes LSTM, Transformer, and CNN-LSTM architectures with attention
mechanisms, ensemble methods, and adversarial training capabilities.

Key Features:
- Multi-architecture neural networks (LSTM, Transformer, CNN-LSTM)
- Attention mechanisms for temporal pattern focus
- Ensemble methods for improved robustness
- Adversarial training and robustness evaluation
- Explainable AI with attention weights and feature importance
- Production-ready with checkpoint management and export formats
- Advanced dataset generation with realistic user behaviors
"""

__version__ = '2.1.0'

# Core types and data structures
from .data_types import (
    AnalysisResult,
    AnomalyType,
    TrainingConfig
)

# Neural network models
from .models.behavioral_analyzer import BehavioralAnalyzer
from .models.attention import AttentionLayer
from .models.advanced_models import (
    TransformerBehavioralAnalyzer,
    CNNLSTMAnalyzer,
    EnsembleAnalyzer,
    AdaptiveAnalyzer
)

# Data processing and generation
from .data.feature_extractor import FeatureExtractor
from .data.data_processor import DataProcessor
from .data.data_generator import (
    generate_sample_data,
    analyze_generated_data
)
from .data.realistic_data_generator import generate_realistic_access_data
from .data.advanced_dataset_generator import (
    AdvancedDatasetGenerator,
    AccessPatternConfig,
    UserBehaviorProfile,
    analyze_dataset_distribution,
    visualize_dataset_patterns,
    create_example_dataset
)

# Training functionality
from .training.trainer import train_model
from .training.metrics import calculate_metrics, evaluate_model
from .training.early_stopping import EarlyStopping
from .training.adversarial_training import (
    AdversarialTrainer, 
    RobustnessEvaluator, 
    generate_adversarial_examples,
    fgsm_attack,
    pgd_attack
)
from .training.ensemble_training import EnsembleManager, create_weighted_ensemble

# Advanced evaluation
from .evaluation.advanced_evaluation import (
    AdvancedEvaluator, 
    plot_roc_curve, 
    plot_precision_recall_curve,
    plot_confusion_matrix,
    calibration_plot
)

# Inference engine
from .inference.engine import InferenceEngine

# Model management
from .management.model_manager import ModelManager
from .management.checkpoint import save_checkpoint, load_checkpoint

# Utilities
from .utils.logger import get_logger

__all__ = [
    # Version
    '__version__',
    
    # Core Types
    'AnalysisResult',           # Result of AI behavioral analysis
    'AnomalyType',              # Enumeration of anomaly types
    'TrainingConfig',           # Training configuration
    
    # Neural Network Models
    'BehavioralAnalyzer',       # LSTM-based behavioral analyzer
    'AttentionLayer',           # Self-attention mechanism
    'TransformerBehavioralAnalyzer',  # Transformer-based model
    'CNNLSTMAnalyzer',          # CNN-LSTM hybrid model
    'EnsembleAnalyzer',         # Ensemble of multiple models
    'AdaptiveAnalyzer',         # Adaptive architecture model
    
    # Data Processing and Generation
    'FeatureExtractor',         # Feature extraction and normalization
    'DataProcessor',            # Data preprocessing and loaders
    'generate_sample_data',     # Generate synthetic sample data
    'analyze_generated_data',   # Analyze generated dataset
    'generate_realistic_access_data',  # Generate realistic access patterns
    'AdvancedDatasetGenerator', # Advanced dataset generation with user profiles
    'AccessPatternConfig',      # Configuration for access pattern generation
    'UserBehaviorProfile',      # User behavior profile types
    'analyze_dataset_distribution',  # Analyze dataset characteristics
    'visualize_dataset_patterns',    # Visualize dataset patterns
    'create_example_dataset',        # Create example dataset
    
    # Training Components
    'train_model',              # Core training function
    'calculate_metrics',        # Calculate performance metrics
    'evaluate_model',           # Model evaluation
    'EarlyStopping',            # Early stopping implementation
    'AdversarialTrainer',       # Adversarial training
    'RobustnessEvaluator',      # Robustness evaluation
    'generate_adversarial_examples',  # Generate adversarial examples
    'fgsm_attack',              # FGSM adversarial attack
    'pgd_attack',               # PGD adversarial attack
    'EnsembleManager',          # Ensemble training management
    'create_weighted_ensemble', # Create weighted ensemble
    
    # Evaluation Tools
    'AdvancedEvaluator',        # Comprehensive evaluation
    'plot_roc_curve',           # ROC curve visualization
    'plot_precision_recall_curve',  # Precision-recall curve
    'plot_confusion_matrix',    # Confusion matrix visualization
    'calibration_plot',         # Model calibration visualization
    
    # Inference Engine
    'InferenceEngine',          # Production inference
    
    # Model Management
    'ModelManager',             # Model lifecycle management
    'save_checkpoint',          # Save model checkpoint
    'load_checkpoint',          # Load model checkpoint
    
    # Utilities
    'get_logger',               # Get configured logger
]

# Friendly aliases for common use cases
Model = BehavioralAnalyzer
Analyzer = InferenceEngine
