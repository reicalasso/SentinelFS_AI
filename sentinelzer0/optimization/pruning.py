"""
Model Pruning

Remove unnecessary weights to reduce model size and improve inference speed.
Supports structured and unstructured pruning strategies.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import logging
import copy


class PruningStrategy:
    """Pruning strategy enumeration."""
    MAGNITUDE = "magnitude"  # Remove weights with smallest magnitude
    RANDOM = "random"  # Random weight removal
    L1_STRUCTURED = "l1_structured"  # Structured pruning by L1 norm
    LN_STRUCTURED = "ln_structured"  # Structured pruning by Ln norm


class ModelPruner:
    """
    Model pruning for size and speed optimization.
    
    Features:
    - Unstructured pruning (magnitude-based, random)
    - Structured pruning (channel/filter pruning)
    - Iterative pruning with fine-tuning
    - Automatic sparsity analysis
    - Pruning mask management
    """
    
    def __init__(self):
        """Initialize model pruner."""
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prune_model(
        self,
        model: nn.Module,
        amount: float = 0.3,
        strategy: str = PruningStrategy.MAGNITUDE,
        layers_to_prune: Optional[List[str]] = None
    ) -> nn.Module:
        """
        Prune model weights.
        
        Args:
            model: Model to prune
            amount: Fraction of weights to prune (0.0 to 1.0)
            strategy: Pruning strategy
            layers_to_prune: Specific layer names to prune (None = all)
        
        Returns:
            Pruned model
        """
        self.logger.info(f"Pruning model with {strategy} strategy, amount={amount}")
        
        # Get layers to prune
        if layers_to_prune is None:
            parameters_to_prune = self._get_prunable_layers(model)
        else:
            parameters_to_prune = self._get_named_layers(model, layers_to_prune)
        
        # Apply pruning based on strategy
        if strategy == PruningStrategy.MAGNITUDE:
            self._prune_magnitude(parameters_to_prune, amount)
        elif strategy == PruningStrategy.RANDOM:
            self._prune_random(parameters_to_prune, amount)
        elif strategy == PruningStrategy.L1_STRUCTURED:
            self._prune_l1_structured(parameters_to_prune, amount)
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}")
        
        # Make pruning permanent
        self._make_pruning_permanent(model)
        
        self.logger.info("Pruning completed")
        return model
    
    def _get_prunable_layers(self, model: nn.Module) -> List[Tuple[nn.Module, str]]:
        """
        Get all prunable layers from model.
        
        Returns list of (module, parameter_name) tuples.
        """
        parameters_to_prune = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                parameters_to_prune.append((module, 'weight'))
        
        return parameters_to_prune
    
    def _get_named_layers(
        self,
        model: nn.Module,
        layer_names: List[str]
    ) -> List[Tuple[nn.Module, str]]:
        """Get specific named layers for pruning."""
        parameters_to_prune = []
        
        for name, module in model.named_modules():
            if name in layer_names:
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    parameters_to_prune.append((module, 'weight'))
        
        return parameters_to_prune
    
    def _prune_magnitude(
        self,
        parameters_to_prune: List[Tuple[nn.Module, str]],
        amount: float
    ):
        """Apply global magnitude-based pruning."""
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        self.logger.info(f"Applied magnitude pruning with amount={amount}")
    
    def _prune_random(
        self,
        parameters_to_prune: List[Tuple[nn.Module, str]],
        amount: float
    ):
        """Apply random unstructured pruning."""
        for module, param_name in parameters_to_prune:
            prune.random_unstructured(module, name=param_name, amount=amount)
        
        self.logger.info(f"Applied random pruning with amount={amount}")
    
    def _prune_l1_structured(
        self,
        parameters_to_prune: List[Tuple[nn.Module, str]],
        amount: float
    ):
        """Apply L1 structured pruning (prune entire filters/channels)."""
        for module, param_name in parameters_to_prune:
            if isinstance(module, nn.Conv2d):
                # Prune output channels for Conv2d
                prune.ln_structured(
                    module,
                    name=param_name,
                    amount=amount,
                    n=1,  # L1 norm
                    dim=0  # Output channel dimension
                )
            elif isinstance(module, nn.Linear):
                # Prune output features for Linear
                prune.ln_structured(
                    module,
                    name=param_name,
                    amount=amount,
                    n=1,
                    dim=0
                )
        
        self.logger.info(f"Applied L1 structured pruning with amount={amount}")
    
    def _make_pruning_permanent(self, model: nn.Module):
        """Make pruning masks permanent by removing them."""
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    # No pruning mask to remove
                    pass
    
    def iterative_pruning(
        self,
        model: nn.Module,
        train_fn: callable,
        val_fn: callable,
        initial_amount: float = 0.1,
        final_amount: float = 0.5,
        num_iterations: int = 5,
        strategy: str = PruningStrategy.MAGNITUDE
    ) -> nn.Module:
        """
        Iteratively prune and fine-tune model.
        
        Args:
            model: Model to prune
            train_fn: Training function (takes model, returns trained model)
            val_fn: Validation function (takes model, returns metrics dict)
            initial_amount: Initial pruning amount
            final_amount: Final pruning amount
            num_iterations: Number of pruning iterations
            strategy: Pruning strategy
        
        Returns:
            Pruned and fine-tuned model
        """
        self.logger.info(
            f"Starting iterative pruning: {num_iterations} iterations, "
            f"{initial_amount:.2f} -> {final_amount:.2f}"
        )
        
        # Calculate pruning schedule
        amounts = torch.linspace(initial_amount, final_amount, num_iterations)
        
        best_model = None
        best_metric = 0.0
        
        for i, amount in enumerate(amounts):
            self.logger.info(f"Iteration {i+1}/{num_iterations}: pruning amount={amount:.3f}")
            
            # Prune model
            model = self.prune_model(model, amount=amount, strategy=strategy)
            
            # Fine-tune
            model = train_fn(model)
            
            # Validate
            metrics = val_fn(model)
            current_metric = metrics.get('accuracy', 0.0)
            
            self.logger.info(f"Iteration {i+1} metrics: {metrics}")
            
            # Save best model
            if current_metric > best_metric:
                best_metric = current_metric
                best_model = copy.deepcopy(model)
        
        self.logger.info(f"Iterative pruning completed. Best metric: {best_metric:.4f}")
        return best_model if best_model is not None else model
    
    def analyze_sparsity(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze model sparsity.
        
        Args:
            model: Model to analyze
        
        Returns:
            Sparsity statistics
        """
        total_params = 0
        zero_params = 0
        layer_sparsity = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                # Get weight tensor
                if hasattr(module, 'weight_orig'):
                    # Pruning mask still attached
                    weight = module.weight_orig
                else:
                    weight = module.weight
                
                # Count parameters
                num_params = weight.numel()
                num_zeros = (weight == 0).sum().item()
                
                total_params += num_params
                zero_params += num_zeros
                
                layer_sparsity[name] = {
                    'total_params': num_params,
                    'zero_params': num_zeros,
                    'sparsity': num_zeros / num_params if num_params > 0 else 0.0
                }
        
        global_sparsity = zero_params / total_params if total_params > 0 else 0.0
        
        analysis = {
            'global_sparsity': global_sparsity,
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'layer_sparsity': layer_sparsity
        }
        
        self.logger.info(f"Model sparsity analysis: {global_sparsity:.2%} sparse")
        return analysis
    
    def estimate_speedup(
        self,
        original_model: nn.Module,
        pruned_model: nn.Module,
        test_input: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Estimate speedup from pruning.
        
        Args:
            original_model: Original unpruned model
            pruned_model: Pruned model
            test_input: Test input tensor
            num_runs: Number of inference runs
        
        Returns:
            Performance comparison
        """
        import time
        
        # Move to device
        original_model = original_model.to(self.device)
        pruned_model = pruned_model.to(self.device)
        test_input = test_input.to(self.device)
        
        # Set to eval mode
        original_model.eval()
        pruned_model.eval()
        
        # Benchmark original model
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = original_model(test_input)
            
            # Measure
            start = time.time()
            for _ in range(num_runs):
                _ = original_model(test_input)
            original_time = (time.time() - start) / num_runs
        
        # Benchmark pruned model
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = pruned_model(test_input)
            
            # Measure
            start = time.time()
            for _ in range(num_runs):
                _ = pruned_model(test_input)
            pruned_time = (time.time() - start) / num_runs
        
        # Calculate speedup
        speedup = original_time / pruned_time
        
        # Get model sizes
        original_size = sum(p.numel() for p in original_model.parameters())
        pruned_size = sum(p.numel() for p in pruned_model.parameters())
        size_reduction = (original_size - pruned_size) / original_size
        
        comparison = {
            'original_latency_ms': original_time * 1000,
            'pruned_latency_ms': pruned_time * 1000,
            'speedup': speedup,
            'original_params': original_size,
            'pruned_params': pruned_size,
            'param_reduction': size_reduction
        }
        
        self.logger.info(f"Pruning speedup: {speedup:.2f}x")
        return comparison
    
    def save_pruned_model(
        self,
        model: nn.Module,
        save_path: str,
        pruning_info: Optional[Dict[str, Any]] = None
    ):
        """
        Save pruned model with metadata.
        
        Args:
            model: Pruned model
            save_path: Path to save model
            pruning_info: Pruning metadata
        """
        # Analyze sparsity
        sparsity_info = self.analyze_sparsity(model)
        
        # Combine with user info
        metadata = {
            'sparsity': sparsity_info,
            'pruning_info': pruning_info or {}
        }
        
        # Save
        save_dict = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }
        
        torch.save(save_dict, save_path)
        self.logger.info(f"Saved pruned model to {save_path}")
    
    def load_pruned_model(
        self,
        model: nn.Module,
        load_path: str
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load pruned model with metadata.
        
        Args:
            model: Model architecture
            load_path: Path to load from
        
        Returns:
            Loaded model and metadata
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        metadata = checkpoint.get('metadata', {})
        
        self.logger.info(f"Loaded pruned model from {load_path}")
        return model, metadata
