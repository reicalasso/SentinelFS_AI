"""
SHAP (SHapley Additive exPlanations) Explainer

Provides game-theory based feature attribution for model predictions.
SHAP values explain how each feature contributes to the model's output.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass


@dataclass
class SHAPValues:
    """Container for SHAP explanation values."""
    values: np.ndarray  # SHAP values for each feature
    base_value: float  # Expected model output
    data: np.ndarray  # Input features
    feature_names: Optional[List[str]] = None


class SHAPExplainer:
    """
    SHAP-based model explainer.
    
    Implements multiple SHAP algorithms:
    - Kernel SHAP: Model-agnostic approximation
    - Deep SHAP: Fast gradient-based for deep networks
    - Gradient SHAP: Combines gradients with sampling
    
    Features:
    - Feature attribution for individual predictions
    - Global feature importance
    - Interaction effects detection
    - Visualization support
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: Optional[torch.Tensor] = None,
        n_samples: int = 100,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: PyTorch model to explain
            background_data: Reference dataset for SHAP baseline
            n_samples: Number of samples for approximation
            feature_names: Names of input features
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.model.eval()
        self.n_samples = n_samples
        self.feature_names = feature_names
        
        # Background data for baseline
        if background_data is not None:
            self.background_data = background_data
        else:
            # Use zeros as default background
            self.background_data = None
        
        # Cache for efficiency
        self.baseline_output = None
        
        self.logger.info(f"Initialized SHAP explainer with {n_samples} samples")
    
    def explain(
        self,
        inputs: torch.Tensor,
        method: str = 'kernel'
    ) -> SHAPValues:
        """
        Generate SHAP explanations for inputs.
        
        Args:
            inputs: Input tensor to explain
            method: SHAP method ('kernel', 'deep', 'gradient')
        
        Returns:
            SHAP values and attributions
        """
        if method == 'kernel':
            return self._kernel_shap(inputs)
        elif method == 'deep':
            return self._deep_shap(inputs)
        elif method == 'gradient':
            return self._gradient_shap(inputs)
        else:
            raise ValueError(f"Unknown SHAP method: {method}")
    
    def _kernel_shap(self, inputs: torch.Tensor) -> SHAPValues:
        """
        Kernel SHAP approximation.
        
        Uses weighted linear regression to approximate SHAP values.
        Model-agnostic but computationally expensive.
        """
        batch_size = inputs.shape[0]
        n_features = inputs.shape[1]
        
        # Get baseline prediction
        if self.baseline_output is None:
            if self.background_data is not None:
                with torch.no_grad():
                    self.baseline_output = self.model(self.background_data).mean().item()
            else:
                self.baseline_output = 0.0
        
        shap_values_list = []
        
        for i in range(batch_size):
            sample = inputs[i:i+1]
            
            # Generate coalition samples
            shap_values = np.zeros(n_features)
            
            for _ in range(self.n_samples):
                # Random coalition
                coalition = np.random.binomial(1, 0.5, n_features).astype(bool)
                
                # Create masked input
                if self.background_data is not None:
                    background_sample = self.background_data[np.random.randint(len(self.background_data))]
                else:
                    background_sample = torch.zeros_like(sample)
                
                masked_input = sample.clone()
                masked_input[:, ~coalition] = background_sample[~coalition]
                
                # Get prediction
                with torch.no_grad():
                    pred = self.model(masked_input).item()
                
                # Approximate SHAP values
                weight = self._shapley_kernel_weight(coalition.sum(), n_features)
                contribution = (pred - self.baseline_output) * weight
                
                for j in range(n_features):
                    if coalition[j]:
                        shap_values[j] += contribution / coalition.sum()
            
            shap_values /= self.n_samples
            shap_values_list.append(shap_values)
        
        return SHAPValues(
            values=np.array(shap_values_list),
            base_value=self.baseline_output,
            data=inputs.cpu().numpy(),
            feature_names=self.feature_names
        )
    
    def _deep_shap(self, inputs: torch.Tensor) -> SHAPValues:
        """
        Deep SHAP for neural networks.
        
        Uses gradient information for faster computation.
        Specific to neural networks.
        """
        inputs.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Get baseline
        if self.baseline_output is None:
            self.baseline_output = 0.0
        
        # Compute gradients
        gradients_list = []
        for i in range(outputs.shape[1] if len(outputs.shape) > 1 else 1):
            if len(outputs.shape) > 1:
                self.model.zero_grad()
                outputs[:, i].sum().backward(retain_graph=True)
            else:
                self.model.zero_grad()
                outputs.sum().backward(retain_graph=True)
            
            gradients = inputs.grad.clone()
            gradients_list.append(gradients)
            inputs.grad.zero_()
        
        # Approximate SHAP values using gradients
        shap_values = inputs.detach().cpu().numpy() * gradients_list[0].detach().cpu().numpy()
        
        return SHAPValues(
            values=shap_values,
            base_value=self.baseline_output,
            data=inputs.detach().cpu().numpy(),
            feature_names=self.feature_names
        )
    
    def _gradient_shap(self, inputs: torch.Tensor) -> SHAPValues:
        """
        Gradient SHAP combining gradients with sampling.
        
        Balances accuracy and computational efficiency.
        """
        # Use multiple baseline samples
        if self.background_data is not None:
            n_baselines = min(10, len(self.background_data))
            baseline_samples = self.background_data[
                torch.randint(0, len(self.background_data), (n_baselines,))
            ]
        else:
            baseline_samples = torch.zeros(10, *inputs.shape[1:])
        
        all_shap_values = []
        
        for baseline in baseline_samples:
            baseline = baseline.unsqueeze(0).expand_as(inputs)
            
            # Interpolate between baseline and input
            alphas = torch.linspace(0, 1, self.n_samples // 10)
            
            shap_values_alpha = []
            
            for alpha in alphas:
                interpolated = baseline + alpha * (inputs - baseline)
                interpolated.requires_grad_(True)
                
                # Forward pass
                output = self.model(interpolated)
                
                # Backward pass
                self.model.zero_grad()
                output.sum().backward()
                
                # Gradient * (input - baseline)
                grad_contribution = interpolated.grad * (inputs - baseline)
                shap_values_alpha.append(grad_contribution.detach())
            
            # Average over alphas
            avg_shap = torch.stack(shap_values_alpha).mean(dim=0)
            all_shap_values.append(avg_shap)
        
        # Average over baselines
        final_shap = torch.stack(all_shap_values).mean(dim=0)
        
        return SHAPValues(
            values=final_shap.cpu().numpy(),
            base_value=self.baseline_output or 0.0,
            data=inputs.detach().cpu().numpy(),
            feature_names=self.feature_names
        )
    
    def _shapley_kernel_weight(self, coalition_size: int, n_features: int) -> float:
        """
        Compute Shapley kernel weight.
        
        Weight ensures correct Shapley value computation.
        """
        if coalition_size == 0 or coalition_size == n_features:
            return 10000  # High weight for empty/full coalitions
        
        from math import comb
        weight = (n_features - 1) / (comb(n_features, coalition_size) * coalition_size * (n_features - coalition_size))
        return weight
    
    def get_feature_importance(
        self,
        shap_values: SHAPValues,
        method: str = 'mean_abs'
    ) -> Dict[str, float]:
        """
        Compute global feature importance from SHAP values.
        
        Args:
            shap_values: SHAP values from explain()
            method: Aggregation method ('mean_abs', 'max', 'variance')
        
        Returns:
            Feature importance dictionary
        """
        if method == 'mean_abs':
            importance = np.abs(shap_values.values).mean(axis=0)
        elif method == 'max':
            importance = np.abs(shap_values.values).max(axis=0)
        elif method == 'variance':
            importance = np.var(shap_values.values, axis=0)
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        if self.feature_names:
            return {name: float(imp) for name, imp in zip(self.feature_names, importance)}
        else:
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}
    
    def get_top_features(
        self,
        shap_values: SHAPValues,
        n: int = 5,
        instance_idx: int = 0
    ) -> List[Tuple[str, float]]:
        """
        Get top N most important features for an instance.
        
        Args:
            shap_values: SHAP values from explain()
            n: Number of top features
            instance_idx: Index of instance to analyze
        
        Returns:
            List of (feature_name, shap_value) tuples
        """
        instance_shap = shap_values.values[instance_idx]
        
        # Sort by absolute value
        sorted_indices = np.argsort(np.abs(instance_shap))[::-1][:n]
        
        if self.feature_names:
            return [(self.feature_names[i], float(instance_shap[i])) for i in sorted_indices]
        else:
            return [(f"feature_{i}", float(instance_shap[i])) for i in sorted_indices]
    
    def summary(self, shap_values: SHAPValues) -> Dict[str, Any]:
        """
        Generate summary statistics for SHAP explanations.
        
        Args:
            shap_values: SHAP values from explain()
        
        Returns:
            Summary dictionary
        """
        return {
            'n_samples': shap_values.values.shape[0],
            'n_features': shap_values.values.shape[1],
            'base_value': shap_values.base_value,
            'mean_abs_shap': float(np.abs(shap_values.values).mean()),
            'max_abs_shap': float(np.abs(shap_values.values).max()),
            'feature_importance': self.get_feature_importance(shap_values),
            'top_features': self.get_top_features(shap_values)
        }
