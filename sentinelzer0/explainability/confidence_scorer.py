"""
Confidence Scoring Module

Provides calibrated confidence scores and uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
from scipy.special import softmax


@dataclass
class ConfidenceScore:
    """Container for confidence metrics."""
    raw_confidence: float  # Raw model output
    calibrated_confidence: float  # Calibrated score
    uncertainty: float  # Uncertainty estimate
    entropy: float  # Prediction entropy
    margin: float  # Margin between top predictions
    method: str  # Calibration method used


class ConfidenceScorer:
    """
    Confidence calibration and uncertainty quantification.
    
    Features:
    - Temperature scaling calibration
    - Platt scaling
    - Monte Carlo dropout uncertainty
    - Ensemble-based confidence
    - Entropy-based metrics
    - Margin-based confidence
    """
    
    def __init__(
        self,
        model: nn.Module,
        calibration_method: str = 'temperature',
        temperature: float = 1.0
    ):
        """
        Initialize confidence scorer.
        
        Args:
            model: PyTorch model
            calibration_method: Calibration method ('temperature', 'platt', 'none')
            temperature: Temperature parameter for scaling
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.calibration_method = calibration_method
        self.temperature = temperature
        
        # Calibration parameters
        self.calibration_params = {'temperature': temperature}
        
        self.logger.info(f"Initialized confidence scorer with {calibration_method} calibration")
    
    def score(
        self,
        inputs: torch.Tensor,
        use_dropout: bool = False,
        n_samples: int = 10
    ) -> ConfidenceScore:
        """
        Compute calibrated confidence score.
        
        Args:
            inputs: Input tensor
            use_dropout: Use MC dropout for uncertainty
            n_samples: Number of MC samples
        
        Returns:
            Confidence score object
        """
        if use_dropout:
            return self._mc_dropout_confidence(inputs, n_samples)
        else:
            return self._standard_confidence(inputs)
    
    def _standard_confidence(self, inputs: torch.Tensor) -> ConfidenceScore:
        """
        Compute standard confidence metrics.
        
        Args:
            inputs: Input tensor
        
        Returns:
            Confidence score
        """
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(inputs)
            
            # Raw probabilities
            probs = F.softmax(logits, dim=-1)
            raw_conf = probs.max(dim=-1)[0].item()
            
            # Calibrate
            if self.calibration_method == 'temperature':
                calibrated_probs = F.softmax(logits / self.temperature, dim=-1)
            elif self.calibration_method == 'platt':
                calibrated_probs = self._platt_scaling(logits)
            else:
                calibrated_probs = probs
            
            calib_conf = calibrated_probs.max(dim=-1)[0].item()
            
            # Entropy
            entropy = -(calibrated_probs * torch.log(calibrated_probs + 1e-10)).sum(dim=-1).item()
            
            # Margin (difference between top 2)
            top2 = calibrated_probs.topk(2, dim=-1)[0]
            margin = (top2[0, 0] - top2[0, 1]).item()
            
            # Uncertainty (1 - confidence or entropy-based)
            uncertainty = 1.0 - calib_conf
        
        return ConfidenceScore(
            raw_confidence=float(raw_conf),
            calibrated_confidence=float(calib_conf),
            uncertainty=float(uncertainty),
            entropy=float(entropy),
            margin=float(margin),
            method=self.calibration_method
        )
    
    def _mc_dropout_confidence(
        self,
        inputs: torch.Tensor,
        n_samples: int
    ) -> ConfidenceScore:
        """
        Compute confidence using Monte Carlo dropout.
        
        Args:
            inputs: Input tensor
            n_samples: Number of MC samples
        
        Returns:
            Confidence score with epistemic uncertainty
        """
        # Enable dropout
        self.model.train()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.model(inputs)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs.cpu().numpy())
        
        # Aggregate predictions
        predictions = np.array(predictions)  # (n_samples, batch, classes)
        mean_probs = predictions.mean(axis=0)  # (batch, classes)
        
        # Raw confidence
        raw_conf = mean_probs.max()
        
        # Epistemic uncertainty (variance across samples)
        epistemic_uncertainty = predictions.std(axis=0).mean()
        
        # Predictive entropy
        entropy = -(mean_probs * np.log(mean_probs + 1e-10)).sum()
        
        # Margin
        top2 = np.sort(mean_probs[0])[-2:]
        margin = top2[1] - top2[0]
        
        # Set model back to eval
        self.model.eval()
        
        return ConfidenceScore(
            raw_confidence=float(raw_conf),
            calibrated_confidence=float(raw_conf),  # Already averaged
            uncertainty=float(epistemic_uncertainty),
            entropy=float(entropy),
            margin=float(margin),
            method='mc_dropout'
        )
    
    def _platt_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply Platt scaling for calibration.
        
        Simplified version using temperature scaling as approximation.
        
        Args:
            logits: Model logits
        
        Returns:
            Calibrated probabilities
        """
        # In practice, would fit parameters on validation set
        # Here we use temperature scaling as approximation
        return F.softmax(logits / self.temperature, dim=-1)
    
    def calibrate_temperature(
        self,
        val_inputs: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter: int = 50
    ):
        """
        Calibrate temperature parameter on validation set.
        
        Args:
            val_inputs: Validation inputs
            val_labels: Validation labels
            max_iter: Maximum optimization iterations
        """
        self.logger.info("Calibrating temperature parameter...")
        
        # Get logits
        with torch.no_grad():
            logits = self.model(val_inputs)
        
        # Optimize temperature
        temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
        
        nll_criterion = nn.CrossEntropyLoss()
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(logits / temperature, val_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        self.temperature = temperature.item()
        self.calibration_params['temperature'] = self.temperature
        
        self.logger.info(f"Calibrated temperature: {self.temperature:.4f}")
    
    def get_confidence_intervals(
        self,
        inputs: torch.Tensor,
        n_samples: int = 100,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute confidence intervals using bootstrap.
        
        Args:
            inputs: Input tensor
            n_samples: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95%)
        
        Returns:
            Tuple of (lower_bound, mean, upper_bound)
        """
        predictions = []
        
        self.model.eval()
        
        # Bootstrap predictions
        for _ in range(n_samples):
            # Add small noise for bootstrap effect
            noisy_inputs = inputs + torch.randn_like(inputs) * 0.01
            
            with torch.no_grad():
                logits = self.model(noisy_inputs)
                probs = F.softmax(logits, dim=-1)
                pred = probs.max(dim=-1)[0].item()
                predictions.append(pred)
        
        # Compute intervals
        predictions = np.array(predictions)
        alpha = 1 - confidence_level
        lower = np.percentile(predictions, alpha/2 * 100)
        upper = np.percentile(predictions, (1 - alpha/2) * 100)
        mean = np.mean(predictions)
        
        return float(lower), float(mean), float(upper)
    
    def assess_calibration(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Assess calibration quality.
        
        Computes Expected Calibration Error (ECE).
        
        Args:
            inputs: Input tensor
            labels: True labels
            n_bins: Number of bins for calibration plot
        
        Returns:
            Calibration metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(inputs)
            probs = F.softmax(logits / self.temperature, dim=-1)
            confidences = probs.max(dim=-1)[0]
            predictions = probs.argmax(dim=-1)
            accuracies = (predictions == labels).float()
        
        # Compute ECE
        ece = 0.0
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_metrics = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean().item()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean().item()
                avg_confidence_in_bin = confidences[in_bin].mean().item()
                
                ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                bin_metrics.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'proportion': prop_in_bin
                })
        
        return {
            'expected_calibration_error': float(ece),
            'n_bins': n_bins,
            'bin_metrics': bin_metrics,
            'overall_accuracy': accuracies.mean().item(),
            'avg_confidence': confidences.mean().item()
        }
    
    def visualize_calibration(
        self,
        calibration_metrics: Dict[str, Any]
    ) -> str:
        """
        Create text visualization of calibration.
        
        Args:
            calibration_metrics: Output from assess_calibration()
        
        Returns:
            Formatted string representation
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Calibration Assessment")
        lines.append("=" * 60)
        lines.append(f"Expected Calibration Error: {calibration_metrics['expected_calibration_error']:.4f}")
        lines.append(f"Overall Accuracy: {calibration_metrics['overall_accuracy']:.2%}")
        lines.append(f"Average Confidence: {calibration_metrics['avg_confidence']:.2%}")
        lines.append("\nBin-wise Calibration:")
        lines.append("-" * 60)
        lines.append(f"{'Bin':15s} {'Accuracy':>12s} {'Confidence':>12s} {'Gap':>10s}")
        lines.append("-" * 60)
        
        for bin_metric in calibration_metrics['bin_metrics']:
            bin_range = f"{bin_metric['bin_lower']:.1f}-{bin_metric['bin_upper']:.1f}"
            acc = bin_metric['accuracy']
            conf = bin_metric['confidence']
            gap = abs(acc - conf)
            
            lines.append(f"{bin_range:15s} {acc:12.2%} {conf:12.2%} {gap:10.4f}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
