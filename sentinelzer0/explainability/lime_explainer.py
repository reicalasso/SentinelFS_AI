"""
LIME (Local Interpretable Model-agnostic Explanations)

Explains individual predictions by fitting interpretable models locally.
LIME perturbs inputs and trains simple models to approximate complex model behavior.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from dataclasses import dataclass
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class LIMEExplanation:
    """Container for LIME explanation."""
    feature_weights: Dict[str, float]  # Feature contributions
    intercept: float  # Model intercept
    score: float  # R² or accuracy of local model
    prediction: float  # Original prediction
    local_prediction: float  # Local model prediction
    feature_names: Optional[List[str]] = None


class LIMEExplainer:
    """
    LIME-based local explainer.
    
    Creates interpretable explanations by:
    1. Perturbing input around the instance
    2. Getting model predictions for perturbations
    3. Fitting simple model (linear) to perturbed data
    4. Using simple model coefficients as explanations
    
    Features:
    - Model-agnostic (works with any model)
    - Local fidelity (accurate for single instances)
    - Interpretable weights
    - Confidence intervals
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 1000,
        feature_names: Optional[List[str]] = None,
        kernel_width: float = 0.25,
        random_state: int = 42
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: PyTorch model to explain
            n_samples: Number of perturbation samples
            feature_names: Names of input features
            kernel_width: Width of exponential kernel
            random_state: Random seed for reproducibility
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.model.eval()
        self.n_samples = n_samples
        self.feature_names = feature_names
        self.kernel_width = kernel_width
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        self.logger.info(f"Initialized LIME explainer with {n_samples} samples")
    
    def explain(
        self,
        instance: torch.Tensor,
        perturbation_std: float = 0.1,
        num_features: Optional[int] = None
    ) -> LIMEExplanation:
        """
        Explain a single prediction.
        
        Args:
            instance: Input instance to explain (single sample)
            perturbation_std: Standard deviation for perturbations
            num_features: Number of top features to return (None = all)
        
        Returns:
            LIME explanation with feature weights
        """
        if len(instance.shape) == 1:
            instance = instance.unsqueeze(0)
        
        # Get original prediction
        with torch.no_grad():
            original_pred = self.model(instance).squeeze().item()
        
        # Generate perturbations
        perturbed_data = self._generate_perturbations(
            instance, 
            perturbation_std
        )
        
        # Get model predictions for perturbations
        with torch.no_grad():
            predictions = self.model(perturbed_data).squeeze()
            if len(predictions.shape) == 0:
                predictions = predictions.unsqueeze(0)
            predictions = predictions.cpu().numpy()
        
        # Calculate distances and weights
        distances = self._calculate_distances(instance, perturbed_data)
        weights = self._kernel_fn(distances)
        
        # Fit interpretable model
        feature_weights, intercept, score, local_pred = self._fit_linear_model(
            perturbed_data.cpu().numpy(),
            predictions,
            weights,
            instance.cpu().numpy()
        )
        
        # Create explanation
        if self.feature_names:
            feature_dict = {
                name: float(weight) 
                for name, weight in zip(self.feature_names, feature_weights)
            }
        else:
            feature_dict = {
                f"feature_{i}": float(weight) 
                for i, weight in enumerate(feature_weights)
            }
        
        # Select top features if specified
        if num_features is not None:
            sorted_features = sorted(
                feature_dict.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            feature_dict = dict(sorted_features[:num_features])
        
        return LIMEExplanation(
            feature_weights=feature_dict,
            intercept=float(intercept),
            score=float(score),
            prediction=float(original_pred),
            local_prediction=float(local_pred),
            feature_names=self.feature_names
        )
    
    def _generate_perturbations(
        self,
        instance: torch.Tensor,
        std: float
    ) -> torch.Tensor:
        """
        Generate perturbed samples around instance.
        
        Args:
            instance: Original instance
            std: Standard deviation for noise
        
        Returns:
            Tensor of perturbed samples
        """
        n_features = instance.shape[1]
        
        # Generate Gaussian perturbations
        noise = torch.randn(
            self.n_samples, 
            n_features,
            device=instance.device
        ) * std
        
        # Add noise to instance
        perturbed = instance.repeat(self.n_samples, 1) + noise
        
        return perturbed
    
    def _calculate_distances(
        self,
        instance: torch.Tensor,
        perturbed_data: torch.Tensor
    ) -> np.ndarray:
        """
        Calculate L2 distances from instance to perturbations.
        
        Args:
            instance: Original instance
            perturbed_data: Perturbed samples
        
        Returns:
            Array of distances
        """
        distances = torch.norm(
            perturbed_data - instance, 
            dim=1, 
            p=2
        ).cpu().numpy()
        
        return distances
    
    def _kernel_fn(self, distances: np.ndarray) -> np.ndarray:
        """
        Exponential kernel for weighting samples.
        
        Closer samples get higher weights.
        
        Args:
            distances: Distances from original instance
        
        Returns:
            Sample weights
        """
        return np.exp(-(distances ** 2) / (self.kernel_width ** 2))
    
    def _fit_linear_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
        instance: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Fit weighted linear regression model.
        
        Args:
            X: Perturbed features
            y: Model predictions
            sample_weights: Sample weights from kernel
            instance: Original instance
        
        Returns:
            Tuple of (coefficients, intercept, R², local_prediction)
        """
        # Fit Ridge regression with weights
        model = Ridge(alpha=1.0, random_state=self.random_state)
        model.fit(X, y, sample_weight=sample_weights)
        
        # Get coefficients and score
        coefficients = model.coef_
        intercept = model.intercept_
        score = model.score(X, y, sample_weight=sample_weights)
        
        # Predict for original instance
        local_pred = model.predict(instance.reshape(1, -1))[0]
        
        return coefficients, intercept, score, local_pred
    
    def explain_batch(
        self,
        instances: torch.Tensor,
        perturbation_std: float = 0.1,
        num_features: Optional[int] = None
    ) -> List[LIMEExplanation]:
        """
        Explain multiple instances.
        
        Args:
            instances: Batch of instances
            perturbation_std: Standard deviation for perturbations
            num_features: Number of top features
        
        Returns:
            List of LIME explanations
        """
        explanations = []
        
        for i in range(len(instances)):
            explanation = self.explain(
                instances[i:i+1],
                perturbation_std,
                num_features
            )
            explanations.append(explanation)
        
        return explanations
    
    def get_top_features(
        self,
        explanation: LIMEExplanation,
        n: int = 5,
        positive_only: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            explanation: LIME explanation
            n: Number of features to return
            positive_only: Only return positive contributions
        
        Returns:
            List of (feature, weight) tuples
        """
        features = explanation.feature_weights.items()
        
        if positive_only:
            features = [(k, v) for k, v in features if v > 0]
        
        sorted_features = sorted(
            features,
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return sorted_features[:n]
    
    def visualize_explanation(
        self,
        explanation: LIMEExplanation,
        top_n: int = 10
    ) -> str:
        """
        Create text visualization of explanation.
        
        Args:
            explanation: LIME explanation
            top_n: Number of features to show
        
        Returns:
            Formatted string representation
        """
        lines = []
        lines.append("=" * 60)
        lines.append("LIME Explanation")
        lines.append("=" * 60)
        lines.append(f"Prediction: {explanation.prediction:.4f}")
        lines.append(f"Local Model Prediction: {explanation.local_prediction:.4f}")
        lines.append(f"Local Model R²: {explanation.score:.4f}")
        lines.append(f"Intercept: {explanation.intercept:.4f}")
        lines.append("\nTop Feature Contributions:")
        lines.append("-" * 60)
        
        top_features = self.get_top_features(explanation, top_n)
        
        for feature, weight in top_features:
            direction = "increases" if weight > 0 else "decreases"
            lines.append(f"  {feature:30s}: {weight:+.4f} ({direction} prediction)")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def aggregate_explanations(
        self,
        explanations: List[LIMEExplanation]
    ) -> Dict[str, Any]:
        """
        Aggregate multiple explanations for global insights.
        
        Args:
            explanations: List of LIME explanations
        
        Returns:
            Aggregated statistics
        """
        if not explanations:
            return {}
        
        # Collect all feature weights
        all_features = set()
        for exp in explanations:
            all_features.update(exp.feature_weights.keys())
        
        # Calculate statistics per feature
        feature_stats = {}
        
        for feature in all_features:
            weights = [
                exp.feature_weights.get(feature, 0.0)
                for exp in explanations
            ]
            
            feature_stats[feature] = {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'min': float(np.min(weights)),
                'max': float(np.max(weights)),
                'abs_mean': float(np.mean(np.abs(weights)))
            }
        
        # Overall statistics
        all_scores = [exp.score for exp in explanations]
        all_pred_diffs = [
            abs(exp.prediction - exp.local_prediction)
            for exp in explanations
        ]
        
        return {
            'n_explanations': len(explanations),
            'feature_stats': feature_stats,
            'mean_local_fidelity': float(np.mean(all_scores)),
            'mean_prediction_diff': float(np.mean(all_pred_diffs)),
            'top_global_features': sorted(
                feature_stats.items(),
                key=lambda x: x[1]['abs_mean'],
                reverse=True
            )[:10]
        }
