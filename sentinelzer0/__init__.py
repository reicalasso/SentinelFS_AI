"""
SentinelFS AI - Real-World Threat Detection System
Production-ready AI-powered behavioral analysis for distributed file system security.

This package provides a hybrid threat detection system combining deep learning,
anomaly detection, and heuristic rules for real-time file system threat detection.

Key Features:
- Hybrid detection (GRU + Isolation Forest + Heuristic Rules)
- Real feature extraction from file system events (30 features)
- Sub-25ms inference latency
- Incremental learning capability
- Production monitoring and evaluation
- Explainable AI with attention weights
- High accuracy with low false positives
"""

__version__ = '3.0.0'

# Core types and data structures
from .data_types import (
    AnalysisResult,
    AnomalyType
)

# Neural network models
from .models.hybrid_detector import HybridThreatDetector

# Data processing and feature extraction
from .data.real_feature_extractor import RealFeatureExtractor

# Training functionality
from .training.real_trainer import RealWorldTrainer

# Evaluation
from .evaluation.production_evaluator import ProductionEvaluator

# Inference engines
from .inference.real_engine import RealTimeInferenceEngine

# Utilities
from .utils.logger import get_logger

__all__ = [
    # Version
    '__version__',
    
    # Core Types
    'AnalysisResult',               # Result of AI behavioral analysis
    'AnomalyType',                  # Enumeration of anomaly types
    
    # Neural Network Models
    'BehavioralAnalyzer',           # LSTM-based behavioral analyzer (legacy)
    'AttentionLayer',               # Self-attention mechanism
    'HybridThreatDetector',         # Production hybrid threat detector
    'LightweightThreatDetector',    # Lightweight variant for ultra-low latency
    
    # Data Processing and Feature Extraction
    'FeatureExtractor',             # Legacy feature extraction
    'RealFeatureExtractor',         # Real-world feature extraction (30 features)
    'DataProcessor',                # Data preprocessing and loaders
    
    # Training Components
    'train_model',                  # Legacy training function
    'RealWorldTrainer',             # Production training system
    'calculate_metrics',            # Calculate performance metrics
    'evaluate_model',               # Model evaluation
    'EarlyStopping',                # Early stopping implementation
    
    # Evaluation Tools
    'ProductionEvaluator',          # Production monitoring and evaluation
    
    # Inference Engines
    'InferenceEngine',              # Legacy inference engine
    'RealTimeInferenceEngine',      # Production real-time inference (<25ms)
    
    # Utilities
    'get_logger',                   # Get configured logger
]

# Friendly aliases for production use
ThreatDetector = HybridThreatDetector
RealTimeEngine = RealTimeInferenceEngine
