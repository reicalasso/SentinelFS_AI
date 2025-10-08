"""
Explainability & Interpretability Framework

Provides transparency into model decisions through multiple explanation methods:
- SHAP: SHapley Additive exPlanations for feature attribution
- LIME: Local Interpretable Model-agnostic Explanations
- Feature Importance: Global and local feature importance analysis
- Decision Reasoning: Human-readable explanation generation
- Audit Trail: Comprehensive decision logging
- Confidence Scoring: Calibrated uncertainty quantification
"""

from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
from .feature_importance import FeatureImportanceAnalyzer
from .decision_reasoning import DecisionReasoningEngine
from .audit_trail import AuditTrailSystem
from .confidence_scorer import ConfidenceScorer
from .manager import ExplainabilityManager

__all__ = [
    'SHAPExplainer',
    'LIMEExplainer',
    'FeatureImportanceAnalyzer',
    'DecisionReasoningEngine',
    'AuditTrailSystem',
    'ConfidenceScorer',
    'ExplainabilityManager'
]

__version__ = '3.2.0'
