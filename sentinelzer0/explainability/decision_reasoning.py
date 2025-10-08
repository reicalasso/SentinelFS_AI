"""
Decision Reasoning Engine

Generates human-readable explanations for model decisions.
Combines multiple explanation methods into coherent narratives.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DecisionExplanation:
    """Container for decision explanation."""
    prediction: float  # Model prediction
    confidence: float  # Prediction confidence
    reasoning: str  # Human-readable explanation
    key_factors: List[Tuple[str, float, str]]  # (feature, impact, description)
    timestamp: datetime
    metadata: Dict[str, Any]


class DecisionReasoningEngine:
    """
    Generates natural language explanations for model decisions.
    
    Features:
    - Template-based explanation generation
    - Feature impact narratives
    - Confidence calibration explanations
    - Decision justification chains
    - Counterfactual explanations
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[List[str]] = None,
        feature_descriptions: Optional[Dict[str, str]] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize decision reasoning engine.
        
        Args:
            model: PyTorch model
            feature_names: Names of input features
            feature_descriptions: Human-readable feature descriptions
            class_names: Names of output classes
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.model.eval()
        self.feature_names = feature_names or []
        self.feature_descriptions = feature_descriptions or {}
        self.class_names = class_names or []
        
        self.logger.info("Initialized decision reasoning engine")
    
    def explain_decision(
        self,
        inputs: torch.Tensor,
        feature_importance: Dict[str, float],
        confidence: float,
        predicted_class: Optional[int] = None
    ) -> DecisionExplanation:
        """
        Generate comprehensive decision explanation.
        
        Args:
            inputs: Input features
            feature_importance: Feature importance scores
            confidence: Prediction confidence
            predicted_class: Predicted class index
        
        Returns:
            Decision explanation object
        """
        # Get prediction if not provided
        if predicted_class is None:
            with torch.no_grad():
                output = self.model(inputs)
                predicted_class = output.argmax(dim=-1).item()
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Generate explanation components
        key_factors = self._generate_key_factors(sorted_features[:5], inputs)
        reasoning = self._generate_reasoning(
            predicted_class,
            confidence,
            key_factors
        )
        
        return DecisionExplanation(
            prediction=float(predicted_class),
            confidence=float(confidence),
            reasoning=reasoning,
            key_factors=key_factors,
            timestamp=datetime.now(),
            metadata={
                'n_features_used': len(feature_importance),
                'top_feature': sorted_features[0][0] if sorted_features else None
            }
        )
    
    def _generate_key_factors(
        self,
        top_features: List[Tuple[str, float]],
        inputs: torch.Tensor
    ) -> List[Tuple[str, float, str]]:
        """
        Generate key factor descriptions.
        
        Args:
            top_features: List of (feature_name, importance)
            inputs: Input tensor
        
        Returns:
            List of (feature, impact, description) tuples
        """
        key_factors = []
        
        for feature_name, importance in top_features:
            # Get feature value if possible
            if self.feature_names:
                try:
                    feature_idx = self.feature_names.index(feature_name)
                    value = inputs[0, feature_idx].item()
                except (ValueError, IndexError):
                    value = None
            else:
                value = None
            
            # Generate description
            description = self._describe_feature_impact(
                feature_name,
                importance,
                value
            )
            
            key_factors.append((feature_name, importance, description))
        
        return key_factors
    
    def _describe_feature_impact(
        self,
        feature_name: str,
        importance: float,
        value: Optional[float]
    ) -> str:
        """
        Generate natural language description of feature impact.
        
        Args:
            feature_name: Name of feature
            importance: Importance score
            value: Feature value
        
        Returns:
            Human-readable description
        """
        # Get feature description if available
        feature_desc = self.feature_descriptions.get(
            feature_name,
            feature_name.replace('_', ' ').title()
        )
        
        # Determine direction
        if importance > 0:
            direction = "increases"
            strength = "strongly" if abs(importance) > 0.5 else "moderately"
        else:
            direction = "decreases"
            strength = "strongly" if abs(importance) > 0.5 else "moderately"
        
        # Build description
        if value is not None:
            desc = f"{feature_desc} (value: {value:.3f}) {strength} {direction} the prediction"
        else:
            desc = f"{feature_desc} {strength} {direction} the prediction"
        
        return desc
    
    def _generate_reasoning(
        self,
        predicted_class: int,
        confidence: float,
        key_factors: List[Tuple[str, float, str]]
    ) -> str:
        """
        Generate complete reasoning narrative.
        
        Args:
            predicted_class: Predicted class
            confidence: Prediction confidence
            key_factors: Key contributing factors
        
        Returns:
            Human-readable reasoning
        """
        lines = []
        
        # Main prediction statement
        class_name = self.class_names[predicted_class] if self.class_names else f"Class {predicted_class}"
        confidence_level = self._confidence_level_text(confidence)
        
        lines.append(f"Prediction: {class_name}")
        lines.append(f"Confidence: {confidence_level} ({confidence:.1%})")
        lines.append("")
        lines.append("Reasoning:")
        
        # Key factors
        if key_factors:
            lines.append("The decision is primarily based on:")
            for i, (feature, importance, description) in enumerate(key_factors, 1):
                lines.append(f"  {i}. {description}")
        
        # Overall assessment
        lines.append("")
        if confidence > 0.8:
            lines.append("This is a high-confidence prediction with clear supporting evidence.")
        elif confidence > 0.6:
            lines.append("This is a moderate-confidence prediction with reasonable supporting evidence.")
        else:
            lines.append("This is a low-confidence prediction; additional evidence may be needed.")
        
        return "\n".join(lines)
    
    def _confidence_level_text(self, confidence: float) -> str:
        """Convert confidence score to descriptive text."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.75:
            return "High"
        elif confidence >= 0.6:
            return "Moderate"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def generate_counterfactual(
        self,
        inputs: torch.Tensor,
        target_class: int,
        feature_importance: Dict[str, float],
        max_changes: int = 3
    ) -> str:
        """
        Generate counterfactual explanation.
        
        "To get prediction X instead, you would need to change..."
        
        Args:
            inputs: Original inputs
            target_class: Desired target class
            feature_importance: Feature importance scores
            max_changes: Maximum features to suggest changing
        
        Returns:
            Counterfactual explanation text
        """
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:max_changes]
        
        target_name = self.class_names[target_class] if self.class_names else f"Class {target_class}"
        
        lines = [f"To achieve prediction '{target_name}', consider:"]
        
        for feature_name, importance in sorted_features:
            feature_desc = self.feature_descriptions.get(
                feature_name,
                feature_name.replace('_', ' ').title()
            )
            
            if importance > 0:
                suggestion = f"Decreasing {feature_desc}"
            else:
                suggestion = f"Increasing {feature_desc}"
            
            lines.append(f"  • {suggestion}")
        
        return "\n".join(lines)
    
    def compare_predictions(
        self,
        inputs_a: torch.Tensor,
        inputs_b: torch.Tensor,
        feature_importance_a: Dict[str, float],
        feature_importance_b: Dict[str, float]
    ) -> str:
        """
        Compare two predictions and explain differences.
        
        Args:
            inputs_a: First input
            inputs_b: Second input
            feature_importance_a: Importance for first input
            feature_importance_b: Importance for second input
        
        Returns:
            Comparison explanation
        """
        with torch.no_grad():
            pred_a = self.model(inputs_a).argmax(dim=-1).item()
            pred_b = self.model(inputs_b).argmax(dim=-1).item()
        
        class_a = self.class_names[pred_a] if self.class_names else f"Class {pred_a}"
        class_b = self.class_names[pred_b] if self.class_names else f"Class {pred_b}"
        
        lines = [
            "Prediction Comparison:",
            f"  Input A: {class_a}",
            f"  Input B: {class_b}",
            ""
        ]
        
        if pred_a == pred_b:
            lines.append("Both inputs yield the same prediction.")
        else:
            lines.append("The predictions differ due to:")
            
            # Find features with largest difference
            all_features = set(feature_importance_a.keys()) | set(feature_importance_b.keys())
            diffs = []
            
            for feature in all_features:
                imp_a = feature_importance_a.get(feature, 0.0)
                imp_b = feature_importance_b.get(feature, 0.0)
                diff = abs(imp_a - imp_b)
                diffs.append((feature, diff, imp_a, imp_b))
            
            diffs.sort(key=lambda x: x[1], reverse=True)
            
            for feature, diff, imp_a, imp_b in diffs[:5]:
                feature_desc = self.feature_descriptions.get(
                    feature,
                    feature.replace('_', ' ').title()
                )
                lines.append(f"  • {feature_desc}: {imp_a:+.3f} vs {imp_b:+.3f}")
        
        return "\n".join(lines)
    
    def generate_summary(
        self,
        explanation: DecisionExplanation
    ) -> str:
        """
        Generate one-sentence summary of decision.
        
        Args:
            explanation: Decision explanation
        
        Returns:
            One-sentence summary
        """
        class_name = self.class_names[int(explanation.prediction)] if self.class_names else f"Class {int(explanation.prediction)}"
        confidence_text = self._confidence_level_text(explanation.confidence)
        
        if explanation.key_factors:
            top_factor = explanation.key_factors[0][0]
            feature_desc = self.feature_descriptions.get(
                top_factor,
                top_factor.replace('_', ' ').title()
            )
            return f"{confidence_text} confidence prediction of {class_name}, primarily due to {feature_desc}."
        else:
            return f"{confidence_text} confidence prediction of {class_name}."
