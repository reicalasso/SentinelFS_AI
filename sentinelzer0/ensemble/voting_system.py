"""
Ensemble Voting System

Implements multiple voting strategies for combining model predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum


class VotingStrategy(Enum):
    """Supported voting strategies."""
    HARD = "hard"  # Majority voting
    SOFT = "soft"  # Average probabilities
    WEIGHTED = "weighted"  # Weighted average by model performance
    STACKING = "stacking"  # Meta-learner on predictions


@dataclass
class VotingResult:
    """Container for voting results."""
    prediction: int  # Final prediction
    confidence: float  # Confidence in prediction
    probabilities: np.ndarray  # Class probabilities
    individual_predictions: List[int]  # Predictions from each model
    individual_confidences: List[float]  # Confidences from each model
    strategy: str  # Voting strategy used
    weights: Optional[List[float]] = None  # Model weights (if weighted)


class EnsembleVoter:
    """
    Ensemble voting system for combining multiple model predictions.
    
    Features:
    - Hard voting (majority)
    - Soft voting (average probabilities)
    - Weighted voting (by model performance)
    - Stacking with meta-learner
    - Confidence calibration
    - Uncertainty quantification
    """
    
    def __init__(
        self,
        strategy: VotingStrategy = VotingStrategy.SOFT,
        weights: Optional[List[float]] = None,
        meta_learner: Optional[nn.Module] = None
    ):
        """
        Initialize ensemble voter.
        
        Args:
            strategy: Voting strategy to use
            weights: Model weights for weighted voting
            meta_learner: Meta-learner for stacking
        """
        self.logger = logging.getLogger(__name__)
        self.strategy = strategy
        self.weights = weights
        self.meta_learner = meta_learner
        
        if self.strategy == VotingStrategy.WEIGHTED and weights is None:
            raise ValueError("Weights required for weighted voting")
        
        if self.strategy == VotingStrategy.STACKING and meta_learner is None:
            raise ValueError("Meta-learner required for stacking")
        
        self.logger.info(f"Initialized ensemble voter with {strategy.value} strategy")
    
    def vote(
        self,
        predictions: List[torch.Tensor],
        return_details: bool = False
    ) -> VotingResult:
        """
        Combine predictions from multiple models.
        
        Args:
            predictions: List of prediction tensors from models
            return_details: Return detailed voting information
        
        Returns:
            Voting result
        """
        if len(predictions) == 0:
            raise ValueError("No predictions provided")
        
        # Route to appropriate voting method
        if self.strategy == VotingStrategy.HARD:
            result = self._hard_voting(predictions)
        elif self.strategy == VotingStrategy.SOFT:
            result = self._soft_voting(predictions)
        elif self.strategy == VotingStrategy.WEIGHTED:
            result = self._weighted_voting(predictions)
        elif self.strategy == VotingStrategy.STACKING:
            result = self._stacking_voting(predictions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return result
    
    def _hard_voting(self, predictions: List[torch.Tensor]) -> VotingResult:
        """
        Hard voting (majority).
        
        Args:
            predictions: List of logits or probabilities
        
        Returns:
            Voting result
        """
        # Get class predictions from each model
        class_preds = []
        confidences = []
        
        for pred in predictions:
            probs = F.softmax(pred, dim=-1) if pred.dim() > 1 else pred
            class_pred = probs.argmax(dim=-1).item()
            conf = probs.max(dim=-1)[0].item()
            
            class_preds.append(class_pred)
            confidences.append(conf)
        
        # Majority vote
        from collections import Counter
        vote_counts = Counter(class_preds)
        final_pred = vote_counts.most_common(1)[0][0]
        
        # Confidence is average confidence of models that voted for winner
        winner_confidences = [
            conf for pred, conf in zip(class_preds, confidences)
            if pred == final_pred
        ]
        final_confidence = np.mean(winner_confidences)
        
        # Compute ensemble probabilities (simple average)
        all_probs = [F.softmax(p, dim=-1).cpu().numpy() for p in predictions]
        ensemble_probs = np.mean(all_probs, axis=0)
        
        return VotingResult(
            prediction=final_pred,
            confidence=float(final_confidence),
            probabilities=ensemble_probs,
            individual_predictions=class_preds,
            individual_confidences=confidences,
            strategy="hard"
        )
    
    def _soft_voting(self, predictions: List[torch.Tensor]) -> VotingResult:
        """
        Soft voting (average probabilities).
        
        Args:
            predictions: List of logits or probabilities
        
        Returns:
            Voting result
        """
        # Convert to probabilities
        all_probs = [F.softmax(p, dim=-1) for p in predictions]
        
        # Average probabilities
        ensemble_probs = torch.stack(all_probs).mean(dim=0)
        
        # Get final prediction
        final_pred = ensemble_probs.argmax(dim=-1).item()
        final_confidence = ensemble_probs.max(dim=-1)[0].item()
        
        # Individual predictions
        class_preds = [p.argmax(dim=-1).item() for p in all_probs]
        confidences = [p.max(dim=-1)[0].item() for p in all_probs]
        
        return VotingResult(
            prediction=final_pred,
            confidence=float(final_confidence),
            probabilities=ensemble_probs.cpu().numpy(),
            individual_predictions=class_preds,
            individual_confidences=confidences,
            strategy="soft"
        )
    
    def _weighted_voting(self, predictions: List[torch.Tensor]) -> VotingResult:
        """
        Weighted voting (weighted average by model performance).
        
        Args:
            predictions: List of logits or probabilities
        
        Returns:
            Voting result
        """
        if self.weights is None:
            raise ValueError("Weights not set")
        
        if len(predictions) != len(self.weights):
            raise ValueError("Number of predictions must match number of weights")
        
        # Normalize weights
        weights = np.array(self.weights)
        weights = weights / weights.sum()
        
        # Convert to probabilities
        all_probs = [F.softmax(p, dim=-1) for p in predictions]
        
        # Weighted average
        weighted_probs = torch.zeros_like(all_probs[0])
        for prob, weight in zip(all_probs, weights):
            weighted_probs += prob * weight
        
        # Get final prediction
        final_pred = weighted_probs.argmax(dim=-1).item()
        final_confidence = weighted_probs.max(dim=-1)[0].item()
        
        # Individual predictions
        class_preds = [p.argmax(dim=-1).item() for p in all_probs]
        confidences = [p.max(dim=-1)[0].item() for p in all_probs]
        
        return VotingResult(
            prediction=final_pred,
            confidence=float(final_confidence),
            probabilities=weighted_probs.cpu().numpy(),
            individual_predictions=class_preds,
            individual_confidences=confidences,
            strategy="weighted",
            weights=weights.tolist()
        )
    
    def _stacking_voting(self, predictions: List[torch.Tensor]) -> VotingResult:
        """
        Stacking with meta-learner.
        
        Args:
            predictions: List of logits or probabilities
        
        Returns:
            Voting result
        """
        if self.meta_learner is None:
            raise ValueError("Meta-learner not set")
        
        # Convert to probabilities
        all_probs = [F.softmax(p, dim=-1) for p in predictions]
        
        # Stack predictions as input to meta-learner
        stacked = torch.cat(all_probs, dim=-1)
        
        # Meta-learner prediction
        with torch.no_grad():
            meta_logits = self.meta_learner(stacked)
            meta_probs = F.softmax(meta_logits, dim=-1)
        
        # Get final prediction
        final_pred = meta_probs.argmax(dim=-1).item()
        final_confidence = meta_probs.max(dim=-1)[0].item()
        
        # Individual predictions
        class_preds = [p.argmax(dim=-1).item() for p in all_probs]
        confidences = [p.max(dim=-1)[0].item() for p in all_probs]
        
        return VotingResult(
            prediction=final_pred,
            confidence=float(final_confidence),
            probabilities=meta_probs.cpu().numpy(),
            individual_predictions=class_preds,
            individual_confidences=confidences,
            strategy="stacking"
        )
    
    def set_weights(self, weights: List[float]):
        """
        Set model weights for weighted voting.
        
        Args:
            weights: List of model weights
        """
        self.weights = weights
        self.logger.info(f"Updated weights: {weights}")
    
    def compute_optimal_weights(
        self,
        val_predictions: List[List[torch.Tensor]],
        val_labels: torch.Tensor
    ) -> List[float]:
        """
        Compute optimal weights based on validation performance.
        
        Args:
            val_predictions: List of prediction lists (one per model)
            val_labels: Validation labels
        
        Returns:
            Optimal weights
        """
        n_models = len(val_predictions)
        
        # Compute accuracy for each model
        accuracies = []
        for model_preds in val_predictions:
            correct = 0
            total = len(val_labels)
            
            for pred, label in zip(model_preds, val_labels):
                pred_class = F.softmax(pred, dim=-1).argmax(dim=-1).item()
                if pred_class == label.item():
                    correct += 1
            
            accuracy = correct / total
            accuracies.append(accuracy)
        
        # Normalize to weights
        weights = np.array(accuracies)
        weights = weights / weights.sum()
        
        self.logger.info(f"Computed optimal weights: {weights}")
        return weights.tolist()
    
    def get_uncertainty(self, predictions: List[torch.Tensor]) -> float:
        """
        Compute ensemble uncertainty.
        
        Args:
            predictions: List of predictions
        
        Returns:
            Uncertainty score (0-1)
        """
        # Convert to probabilities
        all_probs = [F.softmax(p, dim=-1).cpu().numpy() for p in predictions]
        
        # Compute variance across models
        variance = np.var(all_probs, axis=0).mean()
        
        # Also compute entropy of mean prediction
        mean_probs = np.mean(all_probs, axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        
        # Combine (normalized)
        uncertainty = (variance + entropy / np.log(len(mean_probs))) / 2
        
        return float(uncertainty)
    
    def get_agreement(self, predictions: List[torch.Tensor]) -> float:
        """
        Compute agreement among models.
        
        Args:
            predictions: List of predictions
        
        Returns:
            Agreement score (0-1)
        """
        # Get class predictions
        class_preds = [
            F.softmax(p, dim=-1).argmax(dim=-1).item()
            for p in predictions
        ]
        
        # Count most common prediction
        from collections import Counter
        counts = Counter(class_preds)
        max_count = counts.most_common(1)[0][1]
        
        # Agreement is fraction agreeing with majority
        agreement = max_count / len(class_preds)
        
        return float(agreement)
