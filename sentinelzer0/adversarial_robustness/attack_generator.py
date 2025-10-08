"""
Adversarial Attack Generation Framework
========================================

Implements multiple adversarial attack methods to test model robustness:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- C&W (Carlini & Wagner)
- Boundary Attack

These attacks help identify vulnerabilities and train robust models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks"""
    epsilon: float = 0.3  # Perturbation magnitude
    alpha: float = 0.01  # Step size for iterative attacks
    num_steps: int = 40  # Number of iterations for PGD
    targeted: bool = False  # Targeted vs untargeted attack
    clip_min: float = 0.0  # Minimum value for clipping
    clip_max: float = 1.0  # Maximum value for clipping
    confidence: float = 0.0  # Confidence for C&W attack
    binary_search_steps: int = 9  # Binary search steps for C&W
    learning_rate: float = 0.01  # Learning rate for C&W
    initial_const: float = 0.001  # Initial constant for C&W


class BaseAttack:
    """Base class for adversarial attacks"""
    
    def __init__(self, model: nn.Module, config: AttackConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples"""
        raise NotImplementedError
        
    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        """Clip values to valid range"""
        return torch.clamp(x, self.config.clip_min, self.config.clip_max)


class FGSMAttack(BaseAttack):
    """
    Fast Gradient Sign Method (FGSM)
    
    Single-step attack that perturbs input in the direction of gradient.
    Fast but less powerful than iterative methods.
    
    Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2015)
    """
    
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate FGSM adversarial examples
        
        Args:
            x: Input tensor [batch_size, features]
            y: True labels [batch_size]
            
        Returns:
            Adversarial examples [batch_size, features]
        """
        x = x.clone().detach().to(self.device).requires_grad_(True)
        y = y.to(self.device)
        
        # Forward pass
        self.model.eval()
        outputs = self.model(x)
        
        # Calculate loss
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = F.cross_entropy(outputs, y)
        
        # Backward pass to get gradient
        self.model.zero_grad()
        loss.backward()
        
        # Generate perturbation
        grad = x.grad.data
        perturbation = self.config.epsilon * grad.sign()
        
        # Apply perturbation
        if self.config.targeted:
            x_adv = x - perturbation
        else:
            x_adv = x + perturbation
            
        # Clip to valid range
        x_adv = self._clip(x_adv)
        
        return x_adv.detach()


class PGDAttack(BaseAttack):
    """
    Projected Gradient Descent (PGD)
    
    Multi-step iterative attack. Stronger than FGSM, considered one of the
    strongest first-order attacks.
    
    Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018)
    """
    
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate PGD adversarial examples
        
        Args:
            x: Input tensor [batch_size, features]
            y: True labels [batch_size]
            
        Returns:
            Adversarial examples [batch_size, features]
        """
        x = x.clone().detach().to(self.device)
        y = y.to(self.device)
        
        # Random initialization within epsilon ball
        x_adv = x + torch.empty_like(x).uniform_(
            -self.config.epsilon, self.config.epsilon
        )
        x_adv = self._clip(x_adv)
        
        # Iterative attack
        for i in range(self.config.num_steps):
            x_adv.requires_grad_(True)
            
            # Forward pass
            self.model.eval()
            outputs = self.model(x_adv)
            
            # Calculate loss
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = F.cross_entropy(outputs, y)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update perturbation
            grad = x_adv.grad.data
            if self.config.targeted:
                x_adv = x_adv - self.config.alpha * grad.sign()
            else:
                x_adv = x_adv + self.config.alpha * grad.sign()
            
            # Project back to epsilon ball
            delta = torch.clamp(
                x_adv - x,
                -self.config.epsilon,
                self.config.epsilon
            )
            x_adv = self._clip(x + delta)
            x_adv = x_adv.detach()
            
        return x_adv


class CarliniWagnerAttack(BaseAttack):
    """
    Carlini & Wagner (C&W) Attack
    
    Optimization-based attack that finds minimal perturbations.
    Very powerful but computationally expensive.
    
    Reference: Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks" (2017)
    """
    
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate C&W adversarial examples
        
        Args:
            x: Input tensor [batch_size, features]
            y: True labels [batch_size]
            
        Returns:
            Adversarial examples [batch_size, features]
        """
        x = x.clone().detach().to(self.device)
        y = y.to(self.device)
        batch_size = x.shape[0]
        
        # Initialize best perturbations
        best_adv = x.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=self.device)
        
        # Binary search for optimal constant
        lower_bound = torch.zeros(batch_size, device=self.device)
        upper_bound = torch.full((batch_size,), 1e10, device=self.device)
        const = torch.full((batch_size,), self.config.initial_const, device=self.device)
        
        for binary_step in range(self.config.binary_search_steps):
            # Initialize perturbation in tanh space
            w = torch.zeros_like(x, requires_grad=True)
            optimizer = torch.optim.Adam([w], lr=self.config.learning_rate)
            
            # Optimize perturbation
            for step in range(self.config.num_steps):
                optimizer.zero_grad()
                
                # Transform from tanh space to input space
                x_adv = 0.5 * (torch.tanh(w) + 1) * (
                    self.config.clip_max - self.config.clip_min
                ) + self.config.clip_min
                
                # Forward pass
                self.model.eval()
                outputs = self.model(x_adv)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Calculate losses
                l2_loss = torch.sum((x_adv - x) ** 2, dim=1)
                
                # Classification loss
                real = torch.sum(outputs * F.one_hot(y, outputs.shape[1]), dim=1)
                other = torch.max(
                    outputs * (1 - F.one_hot(y, outputs.shape[1])) - 1e10 * F.one_hot(y, outputs.shape[1]),
                    dim=1
                )[0]
                
                if self.config.targeted:
                    loss_adv = torch.clamp(other - real + self.config.confidence, min=0)
                else:
                    loss_adv = torch.clamp(real - other + self.config.confidence, min=0)
                
                # Total loss
                loss = torch.sum(l2_loss + const * loss_adv)
                
                loss.backward()
                optimizer.step()
                
            # Update best results
            with torch.no_grad():
                x_adv = 0.5 * (torch.tanh(w) + 1) * (
                    self.config.clip_max - self.config.clip_min
                ) + self.config.clip_min
                
                outputs = self.model(x_adv)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                pred = outputs.argmax(dim=1)
                
                l2_dist = torch.sum((x_adv - x) ** 2, dim=1).sqrt()
                
                if self.config.targeted:
                    success = (pred == y)
                else:
                    success = (pred != y)
                
                # Update best adversarial examples
                mask = success & (l2_dist < best_l2)
                best_adv[mask] = x_adv[mask]
                best_l2[mask] = l2_dist[mask]
                
            # Update const with binary search
            mask = success
            upper_bound[mask] = const[mask]
            const[mask] = (lower_bound[mask] + upper_bound[mask]) / 2
            
            mask = ~success
            lower_bound[mask] = const[mask]
            const[mask] = (lower_bound[mask] + upper_bound[mask]) / 2
            const[upper_bound == 1e10] *= 10
            
        return best_adv


class BoundaryAttack(BaseAttack):
    """
    Boundary Attack
    
    Decision-based attack that only requires model predictions (no gradients).
    Useful for black-box scenarios.
    
    Reference: Brendel et al., "Decision-Based Adversarial Attacks" (2018)
    """
    
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate Boundary adversarial examples
        
        Args:
            x: Input tensor [batch_size, features]
            y: True labels [batch_size]
            
        Returns:
            Adversarial examples [batch_size, features]
        """
        x = x.clone().detach().to(self.device)
        y = y.to(self.device)
        batch_size = x.shape[0]
        
        # Initialize with random noise
        x_adv = torch.rand_like(x)
        x_adv = self._clip(x_adv)
        
        # Verify it's adversarial
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_adv)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            pred = outputs.argmax(dim=1)
            
            # If not adversarial, try different initializations
            for _ in range(10):
                mask = (pred == y)
                if not mask.any():
                    break
                x_adv[mask] = torch.rand_like(x_adv[mask])
                x_adv = self._clip(x_adv)
                
                outputs = self.model(x_adv)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                pred = outputs.argmax(dim=1)
        
        # Iterative boundary walk
        delta = 0.1
        for step in range(self.config.num_steps):
            # Random direction
            direction = torch.randn_like(x)
            direction = direction / torch.norm(direction, dim=1, keepdim=True)
            
            # Orthogonal step
            perturbed = x_adv + delta * direction
            perturbed = self._clip(perturbed)
            
            # Check if still adversarial
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(perturbed)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                pred = outputs.argmax(dim=1)
                
                # Move towards original if adversarial
                mask = (pred != y)
                proposal = mask.unsqueeze(1) * (
                    perturbed + (x - perturbed) * self.config.alpha
                ) + (~mask).unsqueeze(1) * x_adv
                proposal = self._clip(proposal)
                
                # Check proposal
                outputs = self.model(proposal)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                pred = outputs.argmax(dim=1)
                
                # Accept if adversarial and closer
                mask = (pred != y)
                x_adv = mask.unsqueeze(1) * proposal + (~mask).unsqueeze(1) * x_adv
            
            # Adapt delta
            if step % 10 == 0:
                delta *= 0.9
                
        return x_adv


class AttackGenerator:
    """
    Unified interface for generating adversarial attacks
    """
    
    def __init__(self, model: nn.Module, config: Optional[AttackConfig] = None):
        """
        Initialize attack generator
        
        Args:
            model: Model to attack
            config: Attack configuration
        """
        self.model = model
        self.config = config or AttackConfig()
        
        # Initialize attack methods
        self.attacks = {
            'fgsm': FGSMAttack(model, self.config),
            'pgd': PGDAttack(model, self.config),
            'cw': CarliniWagnerAttack(model, self.config),
            'boundary': BoundaryAttack(model, self.config)
        }
        
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attack_type: str = 'pgd'
    ) -> torch.Tensor:
        """
        Generate adversarial examples
        
        Args:
            x: Input tensor [batch_size, features]
            y: True labels [batch_size]
            attack_type: Type of attack ('fgsm', 'pgd', 'cw', 'boundary')
            
        Returns:
            Adversarial examples [batch_size, features]
        """
        if attack_type not in self.attacks:
            raise ValueError(f"Unknown attack type: {attack_type}")
            
        logger.info(f"Generating {attack_type.upper()} attack...")
        return self.attacks[attack_type].generate(x, y)
    
    def evaluate_robustness(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attack_types: Optional[list] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model robustness against multiple attacks
        
        Args:
            x: Input tensor [batch_size, features]
            y: True labels [batch_size]
            attack_types: List of attack types to test
            
        Returns:
            Dictionary of metrics for each attack type
        """
        if attack_types is None:
            attack_types = ['fgsm', 'pgd', 'cw', 'boundary']
            
        results = {}
        self.model.eval()
        
        # Clean accuracy
        with torch.no_grad():
            outputs = self.model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            clean_pred = outputs.argmax(dim=1)
            clean_acc = (clean_pred == y).float().mean().item()
            
        results['clean'] = {'accuracy': clean_acc}
        
        # Test each attack
        for attack_type in attack_types:
            try:
                x_adv = self.generate(x, y, attack_type)
                
                with torch.no_grad():
                    outputs = self.model(x_adv)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    adv_pred = outputs.argmax(dim=1)
                    adv_acc = (adv_pred == y).float().mean().item()
                    
                    # Calculate perturbation size
                    l2_dist = torch.norm(x_adv - x, p=2, dim=1).mean().item()
                    linf_dist = torch.norm(x_adv - x, p=float('inf'), dim=1).mean().item()
                    
                results[attack_type] = {
                    'accuracy': adv_acc,
                    'l2_distance': l2_dist,
                    'linf_distance': linf_dist,
                    'success_rate': 1.0 - adv_acc / clean_acc if clean_acc > 0 else 0.0
                }
                
                logger.info(f"{attack_type.upper()} - Accuracy: {adv_acc:.4f}, Success Rate: {results[attack_type]['success_rate']:.4f}")
                
            except Exception as e:
                logger.error(f"Error in {attack_type} attack: {e}")
                results[attack_type] = {'error': str(e)}
                
        return results
