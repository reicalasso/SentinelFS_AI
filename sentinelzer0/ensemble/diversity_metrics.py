"""
Model Diversity Metrics

Measures and optimizes ensemble diversity for improved performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy.stats import entropy


@dataclass
class DiversityMetrics:
    """Container for diversity metrics."""
    disagreement: float  # Fraction of disagreements
    q_statistic: float  # Pairwise Q-statistic
    correlation: float  # Average correlation
    kappa: float  # Cohen's kappa
    entropy: float  # Prediction entropy
    diversity_score: float  # Overall diversity (0-1)


class DiversityAnalyzer:
    """
    Analyzes and optimizes ensemble diversity.
    
    Features:
    - Multiple diversity metrics
    - Pairwise analysis
    - Diversity optimization
    - Visualization support
    """
    
    def __init__(self):
        """Initialize diversity analyzer."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized diversity analyzer")
    
    def compute_diversity(
        self,
        models: List[nn.Module],
        data: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> DiversityMetrics:
        """
        Compute comprehensive diversity metrics.
        
        Args:
            models: List of models
            data: Input data
            labels: True labels (optional)
        
        Returns:
            Diversity metrics
        """
        # Get predictions from all models
        predictions = self._get_predictions(models, data)
        
        # Compute metrics
        disagreement = self._compute_disagreement(predictions)
        q_stat = self._compute_q_statistic(predictions, labels)
        correlation = self._compute_correlation(predictions)
        kappa = self._compute_kappa(predictions)
        pred_entropy = self._compute_entropy(predictions)
        
        # Overall diversity score (weighted combination)
        diversity_score = (
            disagreement * 0.3 +
            (1 - abs(q_stat)) * 0.2 +
            (1 - correlation) * 0.2 +
            (1 - kappa) * 0.15 +
            pred_entropy * 0.15
        )
        
        return DiversityMetrics(
            disagreement=float(disagreement),
            q_statistic=float(q_stat),
            correlation=float(correlation),
            kappa=float(kappa),
            entropy=float(pred_entropy),
            diversity_score=float(diversity_score)
        )
    
    def _get_predictions(
        self,
        models: List[nn.Module],
        data: torch.Tensor
    ) -> np.ndarray:
        """
        Get predictions from all models.
        
        Returns:
            Array of shape (n_models, n_samples)
        """
        predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                outputs = model(data)
                preds = outputs.argmax(dim=-1).cpu().numpy()
                predictions.append(preds)
        
        return np.array(predictions)
    
    def _compute_disagreement(self, predictions: np.ndarray) -> float:
        """
        Compute disagreement rate.
        
        Fraction of samples where models disagree.
        """
        n_models, n_samples = predictions.shape
        
        # Count disagreements for each sample
        disagreements = 0
        for i in range(n_samples):
            sample_preds = predictions[:, i]
            if len(np.unique(sample_preds)) > 1:
                disagreements += 1
        
        return disagreements / n_samples
    
    def _compute_q_statistic(
        self,
        predictions: np.ndarray,
        labels: Optional[np.ndarray]
    ) -> float:
        """
        Compute Q-statistic (Yule's Q).
        
        Measures pairwise agreement between models.
        Q = 1: perfect agreement
        Q = 0: independent
        Q = -1: perfect disagreement
        """
        if labels is None:
            return 0.0
        
        n_models = predictions.shape[0]
        q_values = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pred_i = predictions[i]
                pred_j = predictions[j]
                
                # Contingency table
                n11 = np.sum((pred_i == labels) & (pred_j == labels))
                n00 = np.sum((pred_i != labels) & (pred_j != labels))
                n10 = np.sum((pred_i == labels) & (pred_j != labels))
                n01 = np.sum((pred_i != labels) & (pred_j == labels))
                
                # Q-statistic
                numerator = n11 * n00 - n01 * n10
                denominator = n11 * n00 + n01 * n10
                
                if denominator > 0:
                    q = numerator / denominator
                    q_values.append(q)
        
        return np.mean(q_values) if q_values else 0.0
    
    def _compute_correlation(self, predictions: np.ndarray) -> float:
        """
        Compute average pairwise correlation.
        
        High correlation = low diversity.
        """
        n_models = predictions.shape[0]
        correlations = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Correlation between predictions
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _compute_kappa(self, predictions: np.ndarray) -> float:
        """
        Compute Cohen's kappa (average pairwise).
        
        Measures agreement beyond chance.
        """
        n_models = predictions.shape[0]
        kappa_values = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pred_i = predictions[i]
                pred_j = predictions[j]
                
                # Observed agreement
                po = np.mean(pred_i == pred_j)
                
                # Expected agreement (by chance)
                unique_classes = np.unique(np.concatenate([pred_i, pred_j]))
                pe = 0
                for cls in unique_classes:
                    pi = np.mean(pred_i == cls)
                    pj = np.mean(pred_j == cls)
                    pe += pi * pj
                
                # Kappa
                if pe < 1:
                    kappa = (po - pe) / (1 - pe)
                    kappa_values.append(kappa)
        
        return np.mean(kappa_values) if kappa_values else 0.0
    
    def _compute_entropy(self, predictions: np.ndarray) -> float:
        """
        Compute prediction entropy.
        
        Measures diversity of ensemble predictions.
        """
        n_models, n_samples = predictions.shape
        entropies = []
        
        for i in range(n_samples):
            sample_preds = predictions[:, i]
            
            # Distribution of predictions
            unique, counts = np.unique(sample_preds, return_counts=True)
            probs = counts / n_models
            
            # Entropy
            ent = entropy(probs, base=2)
            entropies.append(ent)
        
        # Normalize by maximum entropy
        max_entropy = np.log2(n_models)
        avg_entropy = np.mean(entropies)
        
        return avg_entropy / max_entropy if max_entropy > 0 else 0.0
    
    def analyze_pairwise(
        self,
        models: List[nn.Module],
        data: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Analyze pairwise diversity between models.
        
        Args:
            models: List of models
            data: Input data
            labels: True labels (optional)
        
        Returns:
            Dictionary of pairwise metrics
        """
        predictions = self._get_predictions(models, data)
        n_models = len(models)
        
        pairwise_metrics = {}
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pred_i = predictions[i]
                pred_j = predictions[j]
                
                # Disagreement
                disagreement = np.mean(pred_i != pred_j)
                
                # Correlation
                corr = np.corrcoef(pred_i, pred_j)[0, 1]
                
                # Q-statistic (if labels available)
                q_stat = 0.0
                if labels is not None:
                    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
                    
                    n11 = np.sum((pred_i == labels_np) & (pred_j == labels_np))
                    n00 = np.sum((pred_i != labels_np) & (pred_j != labels_np))
                    n10 = np.sum((pred_i == labels_np) & (pred_j != labels_np))
                    n01 = np.sum((pred_i != labels_np) & (pred_j == labels_np))
                    
                    numerator = n11 * n00 - n01 * n10
                    denominator = n11 * n00 + n01 * n10
                    
                    if denominator > 0:
                        q_stat = numerator / denominator
                
                pairwise_metrics[(i, j)] = {
                    'disagreement': float(disagreement),
                    'correlation': float(corr),
                    'q_statistic': float(q_stat)
                }
        
        return pairwise_metrics
    
    def optimize_subset(
        self,
        models: List[nn.Module],
        data: torch.Tensor,
        labels: torch.Tensor,
        target_size: int,
        metric: str = 'diversity_score'
    ) -> List[int]:
        """
        Select optimal subset of models for maximum diversity.
        
        Args:
            models: List of all models
            data: Input data
            labels: True labels
            target_size: Desired ensemble size
            metric: Optimization metric
        
        Returns:
            Indices of selected models
        """
        if target_size >= len(models):
            return list(range(len(models)))
        
        # Greedy selection
        selected = []
        remaining = list(range(len(models)))
        
        # Start with best individual model
        predictions = self._get_predictions(models, data)
        labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        
        accuracies = [
            np.mean(predictions[i] == labels_np)
            for i in range(len(models))
        ]
        
        best_idx = np.argmax(accuracies)
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        # Iteratively add most diverse model
        while len(selected) < target_size and remaining:
            best_score = -np.inf
            best_candidate = None
            
            for candidate in remaining:
                # Evaluate subset + candidate
                test_subset = selected + [candidate]
                subset_models = [models[i] for i in test_subset]
                
                metrics = self.compute_diversity(subset_models, data, labels)
                
                if metric == 'diversity_score':
                    score = metrics.diversity_score
                elif metric == 'disagreement':
                    score = metrics.disagreement
                elif metric == 'entropy':
                    score = metrics.entropy
                else:
                    score = metrics.diversity_score
                
                # Also consider accuracy
                subset_preds = predictions[test_subset]
                ensemble_pred = np.round(subset_preds.mean(axis=0))
                accuracy = np.mean(ensemble_pred == labels_np)
                
                # Combined score
                combined_score = score * 0.7 + accuracy * 0.3
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        self.logger.info(f"Selected {len(selected)} models: {selected}")
        return selected
    
    def visualize_diversity(
        self,
        metrics: DiversityMetrics
    ) -> str:
        """
        Create text visualization of diversity metrics.
        
        Args:
            metrics: Diversity metrics
        
        Returns:
            Formatted string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Ensemble Diversity Analysis")
        lines.append("=" * 60)
        lines.append(f"Overall Diversity Score: {metrics.diversity_score:.4f}")
        lines.append("")
        lines.append("Individual Metrics:")
        lines.append("-" * 60)
        lines.append(f"  Disagreement:     {metrics.disagreement:.4f}")
        lines.append(f"  Q-Statistic:      {metrics.q_statistic:.4f}")
        lines.append(f"  Correlation:      {metrics.correlation:.4f}")
        lines.append(f"  Kappa:            {metrics.kappa:.4f}")
        lines.append(f"  Entropy:          {metrics.entropy:.4f}")
        lines.append("")
        lines.append("Interpretation:")
        lines.append("-" * 60)
        
        if metrics.diversity_score > 0.7:
            lines.append("  ✅ Excellent diversity - models complement each other well")
        elif metrics.diversity_score > 0.5:
            lines.append("  ✓ Good diversity - reasonable model variation")
        elif metrics.diversity_score > 0.3:
            lines.append("  ⚠ Moderate diversity - consider more diverse architectures")
        else:
            lines.append("  ❌ Low diversity - models too similar")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
