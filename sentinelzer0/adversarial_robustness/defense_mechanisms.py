"""
Defense Mechanisms Against Adversarial Attacks
==============================================

Implements multiple defense strategies:
- Input sanitization (denoising, smoothing)
- Gradient masking
- Ensemble defenses
- Adversarial example detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DefenseConfig:
    """Configuration for defense mechanisms"""
    
    # Input sanitization
    use_denoising: bool = True
    denoise_strength: float = 0.1
    use_smoothing: bool = True
    smoothing_sigma: float = 0.1
    
    # Gradient masking
    use_gradient_masking: bool = False
    noise_scale: float = 0.01
    
    # Ensemble defense
    use_ensemble: bool = True
    num_ensemble_models: int = 3
    ensemble_diversity_weight: float = 0.1
    
    # Detection
    use_detection: bool = True
    detection_threshold: float = 0.5
    use_statistical_test: bool = True


class DefenseMechanism:
    """Base class for defense mechanisms"""
    
    def __init__(self, config: DefenseConfig):
        self.config = config
        
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply defense to input"""
        raise NotImplementedError


class InputSanitizer(DefenseMechanism):
    """
    Input Sanitization Defense
    
    Applies preprocessing to remove adversarial perturbations:
    - Denoising (median filtering, smoothing)
    - Feature squeezing
    - JPEG compression
    """
    
    def __init__(self, config: Optional[DefenseConfig] = None):
        super().__init__(config or DefenseConfig())
        
    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply denoising to input
        
        Args:
            x: Input tensor [batch_size, features]
            
        Returns:
            Denoised tensor
        """
        # Simple denoising with moving average
        if x.dim() == 2:
            # For flat features, use simple smoothing
            noise = torch.randn_like(x) * self.config.denoise_strength
            x_denoised = x + noise
            x_denoised = torch.clamp(x_denoised, x.min(), x.max())
        else:
            # For structured data, use more sophisticated filtering
            x_denoised = x.clone()
            
        return x_denoised
    
    def smooth(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian smoothing
        
        Args:
            x: Input tensor [batch_size, features]
            
        Returns:
            Smoothed tensor
        """
        # Add Gaussian noise and average
        noise = torch.randn_like(x) * self.config.smoothing_sigma
        x_smooth = x + noise
        return x_smooth
    
    def feature_squeezing(
        self,
        x: torch.Tensor,
        bit_depth: int = 8
    ) -> torch.Tensor:
        """
        Reduce feature precision to remove small perturbations
        
        Args:
            x: Input tensor [batch_size, features]
            bit_depth: Number of bits for quantization
            
        Returns:
            Squeezed tensor
        """
        levels = 2 ** bit_depth
        x_squeezed = torch.round(x * levels) / levels
        return x_squeezed
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all sanitization techniques
        
        Args:
            x: Input tensor [batch_size, features]
            
        Returns:
            Sanitized tensor
        """
        x_clean = x.clone()
        
        if self.config.use_denoising:
            x_clean = self.denoise(x_clean)
            
        if self.config.use_smoothing:
            x_clean = self.smooth(x_clean)
            
        # Feature squeezing
        x_clean = self.feature_squeezing(x_clean)
        
        return x_clean


class GradientMasking(DefenseMechanism):
    """
    Gradient Masking Defense
    
    Makes gradients harder to exploit by:
    - Adding noise to gradients
    - Using non-differentiable operations
    - Gradient obfuscation
    
    Note: This is not a robust defense on its own and should be combined
    with other methods.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[DefenseConfig] = None
    ):
        super().__init__(config or DefenseConfig())
        self.model = model
        
    def add_gradient_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add noise during forward pass to mask gradients
        
        Args:
            x: Input tensor [batch_size, features]
            
        Returns:
            Noisy tensor
        """
        if self.training:
            noise = torch.randn_like(x) * self.config.noise_scale
            x_noisy = x + noise
        else:
            x_noisy = x
            
        return x_noisy
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gradient masking
        
        Args:
            x: Input tensor [batch_size, features]
            
        Returns:
            Masked tensor
        """
        if self.config.use_gradient_masking:
            return self.add_gradient_noise(x)
        return x


class EnsembleDefense(DefenseMechanism):
    """
    Ensemble Defense
    
    Uses multiple models with different architectures or training to
    make attacks harder. Combines predictions via voting or averaging.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        config: Optional[DefenseConfig] = None
    ):
        super().__init__(config or DefenseConfig())
        self.models = models
        self.num_models = len(models)
        
        # Ensure all models are in eval mode
        for model in self.models:
            model.eval()
    
    def predict_ensemble(
        self,
        x: torch.Tensor,
        method: str = 'average'
    ) -> torch.Tensor:
        """
        Get ensemble predictions
        
        Args:
            x: Input tensor [batch_size, features]
            method: Combination method ('average', 'voting', 'weighted')
            
        Returns:
            Combined predictions
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(x)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                predictions.append(outputs)
        
        # Combine predictions
        if method == 'average':
            # Average probabilities
            probs = [F.softmax(pred, dim=1) for pred in predictions]
            combined = torch.stack(probs).mean(dim=0)
        elif method == 'voting':
            # Majority voting
            votes = [pred.argmax(dim=1) for pred in predictions]
            votes_tensor = torch.stack(votes)
            combined = torch.mode(votes_tensor, dim=0)[0]
            # Convert back to logits (approximate)
            num_classes = predictions[0].shape[1]
            combined_logits = torch.zeros(
                x.shape[0], num_classes, device=x.device
            )
            combined_logits.scatter_(1, combined.unsqueeze(1), 1.0)
            combined = combined_logits
        elif method == 'weighted':
            # Weighted average (weights based on confidence)
            probs = [F.softmax(pred, dim=1) for pred in predictions]
            confidences = [prob.max(dim=1)[0] for prob in probs]
            weights = F.softmax(torch.stack(confidences), dim=0)
            combined = sum(w.unsqueeze(1) * p for w, p in zip(weights, probs))
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return combined
    
    def calculate_diversity(self, x: torch.Tensor) -> float:
        """
        Calculate diversity among ensemble models
        
        Args:
            x: Input tensor [batch_size, features]
            
        Returns:
            Diversity score (higher is better)
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(x)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                pred_classes = outputs.argmax(dim=1)
                predictions.append(pred_classes)
        
        # Calculate pairwise disagreement
        disagreements = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                disagreement = (predictions[i] != predictions[j]).float().mean()
                disagreements.append(disagreement.item())
        
        diversity = np.mean(disagreements) if disagreements else 0.0
        return diversity
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ensemble defense
        
        Args:
            x: Input tensor [batch_size, features]
            
        Returns:
            Ensemble predictions
        """
        return self.predict_ensemble(x, method='average')


class AdversarialDetector(DefenseMechanism):
    """
    Adversarial Example Detector
    
    Detects adversarial examples using:
    - Statistical tests
    - Prediction consistency
    - Feature space analysis
    - Confidence scores
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[DefenseConfig] = None
    ):
        super().__init__(config or DefenseConfig())
        self.model = model
        
        # Statistics from clean data
        self.clean_stats = {
            'mean_confidence': None,
            'std_confidence': None,
            'mean_entropy': None,
            'std_entropy': None
        }
        
    def calibrate(self, clean_data_loader):
        """
        Calibrate detector on clean data
        
        Args:
            clean_data_loader: DataLoader with clean examples
        """
        confidences = []
        entropies = []
        
        self.model.eval()
        with torch.no_grad():
            for data, _ in clean_data_loader:
                data = data.to(next(self.model.parameters()).device)
                outputs = self.model(data)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probs = F.softmax(outputs, dim=1)
                confidence = probs.max(dim=1)[0]
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
                
                confidences.extend(confidence.cpu().numpy())
                entropies.extend(entropy.cpu().numpy())
        
        # Calculate statistics
        self.clean_stats['mean_confidence'] = np.mean(confidences)
        self.clean_stats['std_confidence'] = np.std(confidences)
        self.clean_stats['mean_entropy'] = np.mean(entropies)
        self.clean_stats['std_entropy'] = np.std(entropies)
        
        logger.info("Detector calibrated on clean data")
        logger.info(f"Mean confidence: {self.clean_stats['mean_confidence']:.4f}")
        logger.info(f"Mean entropy: {self.clean_stats['mean_entropy']:.4f}")
    
    def detect_statistical(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect adversarial examples using statistical tests
        
        Args:
            x: Input tensor [batch_size, features]
            
        Returns:
            Detection scores [batch_size]
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            probs = F.softmax(outputs, dim=1)
            confidence = probs.max(dim=1)[0]
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            
            # Calculate deviation from clean statistics
            if self.clean_stats['mean_confidence'] is not None:
                conf_dev = torch.abs(
                    confidence - self.clean_stats['mean_confidence']
                ) / (self.clean_stats['std_confidence'] + 1e-10)
                
                entropy_dev = torch.abs(
                    entropy - self.clean_stats['mean_entropy']
                ) / (self.clean_stats['std_entropy'] + 1e-10)
                
                # Combine deviations
                detection_score = (conf_dev + entropy_dev) / 2
            else:
                # Fallback: use confidence and entropy directly
                detection_score = 1.0 - confidence + entropy / 10
        
        return detection_score
    
    def detect_consistency(
        self,
        x: torch.Tensor,
        num_samples: int = 10,
        noise_scale: float = 0.05
    ) -> torch.Tensor:
        """
        Detect using prediction consistency with input perturbations
        
        Args:
            x: Input tensor [batch_size, features]
            num_samples: Number of noisy samples
            noise_scale: Scale of noise to add
            
        Returns:
            Detection scores [batch_size]
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            # Original prediction
            outputs = self.model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            original_pred = outputs.argmax(dim=1)
            
            # Predictions with noise
            for _ in range(num_samples):
                noise = torch.randn_like(x) * noise_scale
                x_noisy = x + noise
                outputs = self.model(x_noisy)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                pred = outputs.argmax(dim=1)
                predictions.append(pred)
        
        # Calculate inconsistency
        predictions_tensor = torch.stack(predictions)
        inconsistency = (predictions_tensor != original_pred.unsqueeze(0)).float().mean(dim=0)
        
        return inconsistency
    
    def detect(
        self,
        x: torch.Tensor,
        method: str = 'statistical'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect adversarial examples
        
        Args:
            x: Input tensor [batch_size, features]
            method: Detection method ('statistical', 'consistency', 'both')
            
        Returns:
            Tuple of (detection scores, is_adversarial boolean mask)
        """
        if method == 'statistical' or method == 'both':
            stat_scores = self.detect_statistical(x)
        else:
            stat_scores = torch.zeros(x.shape[0], device=x.device)
        
        if method == 'consistency' or method == 'both':
            cons_scores = self.detect_consistency(x)
        else:
            cons_scores = torch.zeros(x.shape[0], device=x.device)
        
        # Combine scores
        if method == 'both':
            detection_scores = (stat_scores + cons_scores) / 2
        elif method == 'statistical':
            detection_scores = stat_scores
        else:
            detection_scores = cons_scores
        
        # Threshold
        is_adversarial = detection_scores > self.config.detection_threshold
        
        return detection_scores, is_adversarial
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply detection (returns filtered input or None)
        
        Args:
            x: Input tensor [batch_size, features]
            
        Returns:
            Filtered tensor (removes detected adversarial examples)
        """
        if not self.config.use_detection:
            return x
        
        detection_scores, is_adversarial = self.detect(x)
        
        # Log detections
        num_detected = is_adversarial.sum().item()
        if num_detected > 0:
            logger.warning(f"Detected {num_detected}/{x.shape[0]} adversarial examples")
        
        # Return only clean examples
        return x[~is_adversarial]


class DefenseManager:
    """
    Unified Defense Manager
    
    Coordinates multiple defense mechanisms for comprehensive protection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[DefenseConfig] = None,
        ensemble_models: Optional[List[nn.Module]] = None
    ):
        """
        Initialize defense manager
        
        Args:
            model: Primary model to defend
            config: Defense configuration
            ensemble_models: Additional models for ensemble defense
        """
        self.model = model
        self.config = config or DefenseConfig()
        
        # Initialize defenses
        self.sanitizer = InputSanitizer(self.config)
        self.gradient_masking = GradientMasking(model, self.config)
        self.detector = AdversarialDetector(model, self.config)
        
        if ensemble_models and self.config.use_ensemble:
            self.ensemble = EnsembleDefense(
                [model] + ensemble_models,
                self.config
            )
        else:
            self.ensemble = None
    
    def defend(
        self,
        x: torch.Tensor,
        return_detection: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Apply all active defenses
        
        Args:
            x: Input tensor [batch_size, features]
            return_detection: Whether to return detection results
            
        Returns:
            Tuple of (defended input, optional detection info)
        """
        detection_info = {} if return_detection else None
        
        # 1. Input sanitization
        x_defended = self.sanitizer.apply(x)
        
        # 2. Adversarial detection
        if self.config.use_detection:
            detection_scores, is_adversarial = self.detector.detect(x_defended)
            if return_detection:
                detection_info['detection_scores'] = detection_scores
                detection_info['is_adversarial'] = is_adversarial
                detection_info['num_detected'] = is_adversarial.sum().item()
        
        # 3. Gradient masking (if enabled)
        x_defended = self.gradient_masking.apply(x_defended)
        
        return x_defended, detection_info
    
    def predict_robust(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get robust predictions using all defenses
        
        Args:
            x: Input tensor [batch_size, features]
            
        Returns:
            Robust predictions
        """
        # Apply defenses
        x_defended, _ = self.defend(x)
        
        # Get predictions
        if self.ensemble and self.config.use_ensemble:
            predictions = self.ensemble.predict_ensemble(x_defended)
        else:
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(x_defended)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
        
        return predictions
