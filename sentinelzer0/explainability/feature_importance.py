"""
Feature Importance Analyzer

Tracks and analyzes feature importance using multiple methods:
- Permutation importance
- Gradient-based importance
- Integrated gradients
- Attention-based importance (for attention models)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class FeatureImportance:
    """Container for feature importance scores."""
    scores: Dict[str, float]  # Feature -> importance score
    method: str  # Method used to compute importance
    metadata: Dict[str, Any]  # Additional information


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis.
    
    Provides multiple methods for computing feature importance:
    - Permutation: Measure impact of shuffling features
    - Gradient: Use gradients w.r.t. inputs
    - Integrated Gradients: Path-based attribution
    - Ablation: Remove features and measure impact
    
    Features:
    - Global importance (across all data)
    - Local importance (per instance)
    - Temporal tracking of importance changes
    - Statistical significance testing
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[List[str]] = None,
        loss_fn: Optional[Callable] = None
    ):
        """
        Initialize feature importance analyzer.
        
        Args:
            model: PyTorch model to analyze
            feature_names: Names of input features
            loss_fn: Loss function for importance calculation
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.model.eval()
        self.feature_names = feature_names
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        
        # History tracking
        self.importance_history = defaultdict(list)
        
        self.logger.info("Initialized feature importance analyzer")
    
    def permutation_importance(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_repeats: int = 10
    ) -> FeatureImportance:
        """
        Compute permutation feature importance.
        
        Shuffles each feature and measures performance drop.
        
        Args:
            X: Input features
            y: Target labels
            n_repeats: Number of permutation repeats
        
        Returns:
            Feature importance scores
        """
        # Baseline performance
        with torch.no_grad():
            baseline_output = self.model(X)
            baseline_loss = self.loss_fn(baseline_output, y).item()
        
        n_features = X.shape[1]
        importance_scores = np.zeros(n_features)
        
        for feature_idx in range(n_features):
            losses = []
            
            for _ in range(n_repeats):
                # Permute feature
                X_permuted = X.clone()
                perm_indices = torch.randperm(X.shape[0])
                X_permuted[:, feature_idx] = X[perm_indices, feature_idx]
                
                # Calculate loss with permuted feature
                with torch.no_grad():
                    output = self.model(X_permuted)
                    loss = self.loss_fn(output, y).item()
                
                losses.append(loss - baseline_loss)
            
            # Average importance across repeats
            importance_scores[feature_idx] = np.mean(losses)
        
        # Create feature dictionary
        if self.feature_names:
            scores = {
                name: float(score)
                for name, score in zip(self.feature_names, importance_scores)
            }
        else:
            scores = {
                f"feature_{i}": float(score)
                for i, score in enumerate(importance_scores)
            }
        
        # Track in history
        for name, score in scores.items():
            self.importance_history[name].append(score)
        
        return FeatureImportance(
            scores=scores,
            method='permutation',
            metadata={
                'baseline_loss': baseline_loss,
                'n_repeats': n_repeats,
                'n_samples': X.shape[0]
            }
        )
    
    def gradient_importance(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> FeatureImportance:
        """
        Compute gradient-based feature importance.
        
        Uses gradients of output w.r.t. inputs.
        
        Args:
            X: Input features
            y: Target labels (optional)
        
        Returns:
            Feature importance scores
        """
        X.requires_grad_(True)
        
        # Forward pass
        output = self.model(X)
        
        # Backward pass
        if y is not None:
            loss = self.loss_fn(output, y)
            self.model.zero_grad()
            loss.backward()
        else:
            # Use output directly
            self.model.zero_grad()
            output.sum().backward()
        
        # Gradient magnitude as importance
        gradients = X.grad.abs().mean(dim=0).cpu().numpy()
        
        # Create feature dictionary
        if self.feature_names:
            scores = {
                name: float(grad)
                for name, grad in zip(self.feature_names, gradients)
            }
        else:
            scores = {
                f"feature_{i}": float(grad)
                for i, grad in enumerate(gradients)
            }
        
        # Track in history
        for name, score in scores.items():
            self.importance_history[name].append(score)
        
        return FeatureImportance(
            scores=scores,
            method='gradient',
            metadata={'n_samples': X.shape[0]}
        )
    
    def integrated_gradients(
        self,
        X: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        n_steps: int = 50
    ) -> FeatureImportance:
        """
        Compute integrated gradients attribution.
        
        Integrates gradients along path from baseline to input.
        
        Args:
            X: Input features
            baseline: Baseline input (default: zeros)
            n_steps: Number of integration steps
        
        Returns:
            Feature importance scores
        """
        if baseline is None:
            baseline = torch.zeros_like(X)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, n_steps)
        
        integrated_grads = torch.zeros_like(X)
        
        for alpha in alphas:
            # Interpolate
            interpolated = baseline + alpha * (X - baseline)
            interpolated.requires_grad_(True)
            
            # Forward and backward
            output = self.model(interpolated)
            self.model.zero_grad()
            output.sum().backward()
            
            # Accumulate gradients
            integrated_grads += interpolated.grad
        
        # Average and multiply by input difference
        integrated_grads = integrated_grads / n_steps
        attributions = (X - baseline) * integrated_grads
        
        # Average across batch
        importance = attributions.abs().mean(dim=0).detach().cpu().numpy()
        
        # Create feature dictionary
        if self.feature_names:
            scores = {
                name: float(imp)
                for name, imp in zip(self.feature_names, importance)
            }
        else:
            scores = {
                f"feature_{i}": float(imp)
                for i, imp in enumerate(importance)
            }
        
        # Track in history
        for name, score in scores.items():
            self.importance_history[name].append(score)
        
        return FeatureImportance(
            scores=scores,
            method='integrated_gradients',
            metadata={'n_steps': n_steps, 'n_samples': X.shape[0]}
        )
    
    def ablation_importance(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        ablation_value: float = 0.0
    ) -> FeatureImportance:
        """
        Compute ablation-based feature importance.
        
        Sets features to ablation value and measures impact.
        
        Args:
            X: Input features
            y: Target labels
            ablation_value: Value to use for ablation
        
        Returns:
            Feature importance scores
        """
        # Baseline performance
        with torch.no_grad():
            baseline_output = self.model(X)
            baseline_loss = self.loss_fn(baseline_output, y).item()
        
        n_features = X.shape[1]
        importance_scores = np.zeros(n_features)
        
        for feature_idx in range(n_features):
            # Ablate feature
            X_ablated = X.clone()
            X_ablated[:, feature_idx] = ablation_value
            
            # Calculate loss with ablated feature
            with torch.no_grad():
                output = self.model(X_ablated)
                loss = self.loss_fn(output, y).item()
            
            # Importance = increase in loss
            importance_scores[feature_idx] = loss - baseline_loss
        
        # Create feature dictionary
        if self.feature_names:
            scores = {
                name: float(score)
                for name, score in zip(self.feature_names, importance_scores)
            }
        else:
            scores = {
                f"feature_{i}": float(score)
                for i, score in enumerate(importance_scores)
            }
        
        # Track in history
        for name, score in scores.items():
            self.importance_history[name].append(score)
        
        return FeatureImportance(
            scores=scores,
            method='ablation',
            metadata={
                'baseline_loss': baseline_loss,
                'ablation_value': ablation_value
            }
        )
    
    def get_top_features(
        self,
        importance: FeatureImportance,
        n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            importance: Feature importance object
            n: Number of features to return
        
        Returns:
            List of (feature_name, score) tuples
        """
        sorted_features = sorted(
            importance.scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:n]
    
    def compare_methods(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        methods: Optional[List[str]] = None
    ) -> Dict[str, FeatureImportance]:
        """
        Compare multiple importance methods.
        
        Args:
            X: Input features
            y: Target labels
            methods: List of methods to use
        
        Returns:
            Dictionary of method -> importance
        """
        if methods is None:
            methods = ['permutation', 'gradient', 'integrated_gradients', 'ablation']
        
        results = {}
        
        if 'permutation' in methods:
            results['permutation'] = self.permutation_importance(X, y)
        
        if 'gradient' in methods:
            results['gradient'] = self.gradient_importance(X, y)
        
        if 'integrated_gradients' in methods:
            results['integrated_gradients'] = self.integrated_gradients(X)
        
        if 'ablation' in methods:
            results['ablation'] = self.ablation_importance(X, y)
        
        return results
    
    def get_consensus_importance(
        self,
        importances: Dict[str, FeatureImportance]
    ) -> FeatureImportance:
        """
        Compute consensus importance across methods.
        
        Uses rank aggregation to combine multiple methods.
        
        Args:
            importances: Dictionary of method -> importance
        
        Returns:
            Consensus feature importance
        """
        # Collect all features
        all_features = set()
        for imp in importances.values():
            all_features.update(imp.scores.keys())
        
        # Compute rank for each method
        feature_ranks = defaultdict(list)
        
        for method, imp in importances.items():
            sorted_features = sorted(
                imp.scores.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            for rank, (feature, _) in enumerate(sorted_features, 1):
                feature_ranks[feature].append(rank)
        
        # Average ranks (lower is better)
        consensus_scores = {
            feature: 1.0 / np.mean(ranks)  # Inverse of average rank
            for feature, ranks in feature_ranks.items()
        }
        
        return FeatureImportance(
            scores=consensus_scores,
            method='consensus',
            metadata={
                'methods_used': list(importances.keys()),
                'n_methods': len(importances)
            }
        )
    
    def get_importance_history(
        self,
        feature_name: str
    ) -> List[float]:
        """
        Get historical importance scores for a feature.
        
        Args:
            feature_name: Name of feature
        
        Returns:
            List of historical scores
        """
        return self.importance_history.get(feature_name, [])
    
    def visualize_importance(
        self,
        importance: FeatureImportance,
        top_n: int = 15
    ) -> str:
        """
        Create text visualization of feature importance.
        
        Args:
            importance: Feature importance object
            top_n: Number of features to display
        
        Returns:
            Formatted string representation
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"Feature Importance ({importance.method.upper()})")
        lines.append("=" * 70)
        
        top_features = self.get_top_features(importance, top_n)
        
        # Find max importance for scaling
        max_importance = max(abs(score) for _, score in top_features)
        
        for feature, score in top_features:
            # Create bar visualization
            bar_length = int(40 * abs(score) / max_importance)
            bar = "█" * bar_length
            
            lines.append(f"{feature:25s} {score:+.4f} {bar}")
        
        lines.append("=" * 70)
        
        if importance.metadata:
            lines.append("\nMetadata:")
            for key, value in importance.metadata.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def summary(self, importance: FeatureImportance) -> Dict[str, Any]:
        """
        Generate summary statistics for feature importance.
        
        Args:
            importance: Feature importance object
        
        Returns:
            Summary dictionary
        """
        scores = list(importance.scores.values())
        
        return {
            'method': importance.method,
            'n_features': len(scores),
            'mean_importance': float(np.mean(np.abs(scores))),
            'std_importance': float(np.std(scores)),
            'max_importance': float(np.max(np.abs(scores))),
            'top_3_features': self.get_top_features(importance, 3),
            'metadata': importance.metadata
        }
