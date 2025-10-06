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

# Data processing
from .data.feature_extractor import FeatureExtractor
from .data.data_processor import DataProcessor
from .data.data_generator import generate_sample_data

# Training
from .training.trainer import train_model
from .training.metrics import calculate_metrics, evaluate_model
from .training.early_stopping import EarlyStopping

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
    
    # Data
    'FeatureExtractor',
    'DataProcessor',
    'generate_sample_data',
    
    # Training
    'train_model',
    'calculate_metrics',
    'evaluate_model',
    'EarlyStopping',
    
    # Inference
    'InferenceEngine',
    
    # Management
    'ModelManager',
    'save_checkpoint',
    'load_checkpoint',
    
    # Utils
    'get_logger',
]
