"""
SentinelZer0 MLOps Module

This module provides comprehensive MLOps capabilities including:
- Model versioning and metadata tracking
- Model registry with approval workflows
- A/B testing framework
- Automated rollback mechanisms
- MLflow integration
- CI/CD pipeline support
"""

from .version_manager import ModelVersionManager, ModelVersion, ModelMetadata, VersionStatus
from .model_registry import ModelRegistry, ModelStage, RegistryEntry
from .ab_testing import ABTestManager, ABTest, TestMetrics
from .rollback import RollbackManager, RollbackStrategy
from .mlflow_integration import MLflowTracker

__all__ = [
    'ModelVersionManager',
    'ModelVersion',
    'ModelMetadata',
    'VersionStatus',
    'ModelRegistry',
    'ModelStage',
    'RegistryEntry',
    'ABTestManager',
    'ABTest',
    'TestMetrics',
    'RollbackManager',
    'RollbackStrategy',
    'MLflowTracker',
]

__version__ = '1.0.0'
