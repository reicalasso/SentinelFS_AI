"""
Explainability Manager - Unified Interface

Orchestrates all explainability components for comprehensive model interpretation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

from .shap_explainer import SHAPExplainer, SHAPValues
from .lime_explainer import LIMEExplainer, LIMEExplanation
from .feature_importance import FeatureImportanceAnalyzer, FeatureImportance
from .decision_reasoning import DecisionReasoningEngine, DecisionExplanation
from .audit_trail import AuditTrailSystem, AuditEntry
from .confidence_scorer import ConfidenceScorer, ConfidenceScore


class ExplainabilityManager:
    """
    Unified explainability interface.
    
    Coordinates all explainability components:
    - SHAP explanations
    - LIME explanations
    - Feature importance analysis
    - Decision reasoning (natural language)
    - Confidence scoring
    - Audit trail logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        audit_db_path: Optional[Path] = None,
        enable_caching: bool = True
    ):
        """
        Initialize explainability manager.
        
        Args:
            model: PyTorch model to explain
            feature_names: Names of input features
            class_names: Names of output classes
            audit_db_path: Path for audit database
            enable_caching: Enable explanation caching
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(100)]
        self.class_names = class_names or ['benign', 'malicious']
        
        # Initialize all components
        self.shap_explainer = SHAPExplainer(model)
        self.lime_explainer = LIMEExplainer(model, feature_names=self.feature_names)
        self.feature_analyzer = FeatureImportanceAnalyzer(model, feature_names=self.feature_names)
        self.reasoning_engine = DecisionReasoningEngine(
            feature_names=self.feature_names,
            class_names=self.class_names
        )
        self.confidence_scorer = ConfidenceScorer(model)
        
        # Audit trail
        if audit_db_path is None:
            audit_db_path = Path("audit_trail.db")
        self.audit_trail = AuditTrailSystem(db_path=audit_db_path)
        
        # Caching
        self.enable_caching = enable_caching
        self.cache = {} if enable_caching else None
        
        self.logger.info("Initialized explainability manager with all components")
    
    def explain(
        self,
        inputs: torch.Tensor,
        methods: Optional[List[str]] = None,
        log_to_audit: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation.
        
        Args:
            inputs: Input tensor
            methods: Explanation methods to use (None = all)
            log_to_audit: Log to audit trail
            use_cache: Use cached explanations if available
        
        Returns:
            Dictionary with all explanations
        """
        # Check cache
        if use_cache and self.enable_caching:
            cache_key = self._get_cache_key(inputs)
            if cache_key in self.cache:
                self.logger.info("Using cached explanation")
                return self.cache[cache_key]
        
        # Available methods
        available_methods = ['shap', 'lime', 'feature_importance', 'reasoning', 'confidence']
        
        if methods is None:
            methods = available_methods
        
        # Validate methods
        invalid = set(methods) - set(available_methods)
        if invalid:
            raise ValueError(f"Invalid methods: {invalid}")
        
        explanation = {
            'timestamp': datetime.now().isoformat(),
            'input_shape': list(inputs.shape),
            'methods_used': methods
        }
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(inputs)
            prediction = output.argmax(dim=-1).item()
            explanation['prediction'] = prediction
            explanation['prediction_label'] = self.class_names[prediction]
        
        # SHAP
        if 'shap' in methods:
            try:
                shap_values = self.shap_explainer.explain(inputs, method='kernel')
                explanation['shap'] = {
                    'values': shap_values.values.tolist() if hasattr(shap_values.values, 'tolist') else shap_values.values,
                    'base_value': float(shap_values.base_value),
                    'top_features': [
                        {'name': name, 'value': float(val)}
                        for name, val in shap_values.top_features(k=5)
                    ]
                }
            except Exception as e:
                self.logger.warning(f"SHAP explanation failed: {e}")
                explanation['shap'] = {'error': str(e)}
        
        # LIME
        if 'lime' in methods:
            try:
                lime_exp = self.lime_explainer.explain(inputs)
                explanation['lime'] = {
                    'coefficients': {name: float(val) for name, val in lime_exp.coefficients.items()},
                    'intercept': float(lime_exp.intercept),
                    'score': float(lime_exp.score),
                    'top_features': [
                        {'name': name, 'value': float(val)}
                        for name, val in lime_exp.top_features(k=5)
                    ]
                }
            except Exception as e:
                self.logger.warning(f"LIME explanation failed: {e}")
                explanation['lime'] = {'error': str(e)}
        
        # Feature Importance
        if 'feature_importance' in methods:
            try:
                importance = self.feature_analyzer.compute_importance(
                    inputs,
                    method='permutation',
                    n_repeats=5
                )
                explanation['feature_importance'] = {
                    'scores': {name: float(score) for name, score in importance.scores.items()},
                    'method': importance.method,
                    'top_features': [
                        {'name': name, 'score': float(score)}
                        for name, score in importance.top_features(k=5)
                    ]
                }
            except Exception as e:
                self.logger.warning(f"Feature importance failed: {e}")
                explanation['feature_importance'] = {'error': str(e)}
        
        # Decision Reasoning
        if 'reasoning' in methods:
            try:
                reasoning = self.reasoning_engine.explain_decision(
                    inputs,
                    self.model,
                    explanation.get('feature_importance', {}).get('scores', {})
                )
                explanation['reasoning'] = {
                    'prediction_label': reasoning.prediction_label,
                    'confidence': float(reasoning.confidence),
                    'explanation_text': reasoning.explanation_text,
                    'key_factors': reasoning.key_factors,
                    'risk_level': reasoning.risk_level
                }
            except Exception as e:
                self.logger.warning(f"Decision reasoning failed: {e}")
                explanation['reasoning'] = {'error': str(e)}
        
        # Confidence Score
        if 'confidence' in methods:
            try:
                conf_score = self.confidence_scorer.score(inputs, use_dropout=False)
                explanation['confidence'] = {
                    'raw_confidence': float(conf_score.raw_confidence),
                    'calibrated_confidence': float(conf_score.calibrated_confidence),
                    'uncertainty': float(conf_score.uncertainty),
                    'entropy': float(conf_score.entropy),
                    'margin': float(conf_score.margin),
                    'method': conf_score.method
                }
            except Exception as e:
                self.logger.warning(f"Confidence scoring failed: {e}")
                explanation['confidence'] = {'error': str(e)}
        
        # Log to audit trail
        if log_to_audit:
            try:
                self._log_to_audit(inputs, explanation)
            except Exception as e:
                self.logger.warning(f"Audit logging failed: {e}")
        
        # Cache explanation
        if use_cache and self.enable_caching:
            self.cache[cache_key] = explanation
        
        return explanation
    
    def explain_batch(
        self,
        inputs: torch.Tensor,
        methods: Optional[List[str]] = None,
        log_to_audit: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Explain batch of inputs.
        
        Args:
            inputs: Batch of inputs (B, ...)
            methods: Explanation methods
            log_to_audit: Log to audit trail
        
        Returns:
            List of explanations
        """
        explanations = []
        
        for i in range(inputs.shape[0]):
            single_input = inputs[i:i+1]
            exp = self.explain(
                single_input,
                methods=methods,
                log_to_audit=log_to_audit
            )
            explanations.append(exp)
        
        return explanations
    
    def calibrate(
        self,
        val_inputs: torch.Tensor,
        val_labels: torch.Tensor
    ):
        """
        Calibrate confidence scorer on validation set.
        
        Args:
            val_inputs: Validation inputs
            val_labels: Validation labels
        """
        self.logger.info("Calibrating confidence scorer...")
        self.confidence_scorer.calibrate_temperature(val_inputs, val_labels)
        self.logger.info("Calibration complete")
    
    def assess_calibration(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Assess calibration quality.
        
        Args:
            inputs: Input tensor
            labels: True labels
        
        Returns:
            Calibration metrics
        """
        return self.confidence_scorer.assess_calibration(inputs, labels)
    
    def compare_methods(
        self,
        inputs: torch.Tensor,
        methods: List[str] = ['shap', 'lime', 'feature_importance']
    ) -> Dict[str, Any]:
        """
        Compare different explanation methods.
        
        Args:
            inputs: Input tensor
            methods: Methods to compare
        
        Returns:
            Comparison results
        """
        explanations = self.explain(inputs, methods=methods, log_to_audit=False)
        
        # Extract top features from each method
        comparison = {
            'methods': methods,
            'top_features': {},
            'agreement_score': 0.0
        }
        
        # Get top features from each method
        for method in methods:
            if method in explanations and 'top_features' in explanations[method]:
                top_feats = explanations[method]['top_features']
                comparison['top_features'][method] = [f['name'] for f in top_feats]
        
        # Compute agreement (Jaccard similarity)
        if len(comparison['top_features']) >= 2:
            sets = [set(feats) for feats in comparison['top_features'].values()]
            intersection = set.intersection(*sets)
            union = set.union(*sets)
            comparison['agreement_score'] = len(intersection) / len(union) if union else 0.0
        
        return comparison
    
    def generate_report(
        self,
        inputs: torch.Tensor,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate comprehensive explanation report.
        
        Args:
            inputs: Input tensor
            output_path: Path to save report (optional)
        
        Returns:
            Report text
        """
        explanation = self.explain(inputs, methods=None)
        
        lines = []
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE EXPLANATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {explanation['timestamp']}")
        lines.append(f"Input Shape: {explanation['input_shape']}")
        lines.append(f"Prediction: {explanation['prediction_label']} (class {explanation['prediction']})")
        lines.append("")
        
        # Confidence
        if 'confidence' in explanation:
            conf = explanation['confidence']
            lines.append("CONFIDENCE ASSESSMENT")
            lines.append("-" * 80)
            lines.append(f"  Calibrated Confidence: {conf['calibrated_confidence']:.2%}")
            lines.append(f"  Uncertainty: {conf['uncertainty']:.4f}")
            lines.append(f"  Entropy: {conf['entropy']:.4f}")
            lines.append(f"  Margin: {conf['margin']:.4f}")
            lines.append("")
        
        # Decision Reasoning
        if 'reasoning' in explanation:
            reasoning = explanation['reasoning']
            lines.append("DECISION REASONING")
            lines.append("-" * 80)
            lines.append(f"  {reasoning['explanation_text']}")
            lines.append(f"  Risk Level: {reasoning['risk_level']}")
            lines.append("")
        
        # Top Features (from multiple methods)
        lines.append("TOP INFLUENTIAL FEATURES")
        lines.append("-" * 80)
        
        if 'shap' in explanation and 'top_features' in explanation['shap']:
            lines.append("  SHAP:")
            for feat in explanation['shap']['top_features'][:5]:
                lines.append(f"    - {feat['name']}: {feat['value']:.4f}")
        
        if 'lime' in explanation and 'top_features' in explanation['lime']:
            lines.append("  LIME:")
            for feat in explanation['lime']['top_features'][:5]:
                lines.append(f"    - {feat['name']}: {feat['value']:.4f}")
        
        if 'feature_importance' in explanation and 'top_features' in explanation['feature_importance']:
            lines.append("  Feature Importance:")
            for feat in explanation['feature_importance']['top_features'][:5]:
                lines.append(f"    - {feat['name']}: {feat['score']:.4f}")
        
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        # Save if requested
        if output_path:
            output_path.write_text(report)
            self.logger.info(f"Report saved to {output_path}")
        
        return report
    
    def _log_to_audit(self, inputs: torch.Tensor, explanation: Dict[str, Any]):
        """Log explanation to audit trail."""
        # Extract key information
        prediction = explanation.get('prediction', -1)
        confidence = explanation.get('confidence', {}).get('calibrated_confidence', 0.0)
        
        # Create audit entry
        self.audit_trail.log_decision(
            input_data=inputs.cpu().numpy().tolist(),
            prediction=prediction,
            confidence=float(confidence),
            explanation=explanation,
            metadata={'methods': explanation.get('methods_used', [])}
        )
    
    def _get_cache_key(self, inputs: torch.Tensor) -> str:
        """Generate cache key from inputs."""
        import hashlib
        data_bytes = inputs.cpu().numpy().tobytes()
        return hashlib.sha256(data_bytes).hexdigest()[:16]
    
    def clear_cache(self):
        """Clear explanation cache."""
        if self.cache is not None:
            self.cache.clear()
            self.logger.info("Explanation cache cleared")
    
    def export_audit_trail(self, output_path: Path, format: str = 'csv'):
        """
        Export audit trail.
        
        Args:
            output_path: Output file path
            format: Export format ('csv' or 'json')
        """
        if format == 'csv':
            self.audit_trail.export_to_csv(output_path)
        elif format == 'json':
            self.audit_trail.export_to_json(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Audit trail exported to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get explainability statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'cache_size': len(self.cache) if self.cache else 0,
            'audit_stats': self.audit_trail.get_statistics(),
            'components': {
                'shap': 'active',
                'lime': 'active',
                'feature_importance': 'active',
                'reasoning': 'active',
                'confidence': 'active',
                'audit_trail': 'active'
            }
        }
