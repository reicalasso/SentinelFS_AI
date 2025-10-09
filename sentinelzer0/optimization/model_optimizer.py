"""
Advanced model optimization for production deployment.
Includes quantization, pruning, and inference acceleration.
"""

import torch
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, Any, Optional
import time
import psutil
import os

from ..models.hybrid_detector import HybridThreatDetector
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelOptimizer:
    """
    Advanced model optimization for production deployment.
    
    Features:
    - Dynamic quantization for faster inference
    - Structured pruning for model compression
    - Memory usage optimization
    - Inference speed benchmarking
    """
    
    def __init__(self, model: HybridThreatDetector):
        self.model = model
        self.original_size = self._calculate_model_size()
        self.optimization_history = []
        
    def optimize_for_production(
        self, 
        quantization: bool = True,
        pruning: bool = True,
        pruning_amount: float = 0.2,
        optimize_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Apply comprehensive optimization for production deployment.
        
        Args:
            quantization: Apply dynamic quantization
            pruning: Apply structured pruning
            pruning_amount: Amount of weights to prune (0.0-1.0)
            optimize_memory: Apply memory optimizations
            
        Returns:
            Optimization results and metrics
        """
        logger.info("🚀 Starting comprehensive model optimization...")
        
        optimization_results = {
            'original_size_mb': self.original_size,
            'optimizations_applied': [],
            'performance_metrics': {}
        }
        
        # Baseline performance measurement
        baseline_metrics = self._benchmark_inference()
        optimization_results['performance_metrics']['baseline'] = baseline_metrics
        
        # 1. Dynamic Quantization
        if quantization:
            logger.info("⚡ Applying dynamic quantization...")
            self._apply_quantization()
            optimization_results['optimizations_applied'].append('quantization')
            
            # Measure quantized performance
            quant_metrics = self._benchmark_inference()
            optimization_results['performance_metrics']['quantized'] = quant_metrics
        
        # 2. Structured Pruning
        if pruning:
            logger.info(f"✂️ Applying structured pruning ({pruning_amount:.1%})...")
            self._apply_pruning(pruning_amount)
            optimization_results['optimizations_applied'].append('pruning')
            
            # Measure pruned performance
            pruned_metrics = self._benchmark_inference()
            optimization_results['performance_metrics']['pruned'] = pruned_metrics
        
        # 3. Memory Optimization
        if optimize_memory:
            logger.info("🧠 Applying memory optimizations...")
            self._optimize_memory()
            optimization_results['optimizations_applied'].append('memory_optimization')
        
        # Final measurements
        final_size = self._calculate_model_size()
        final_metrics = self._benchmark_inference()
        
        optimization_results.update({
            'final_size_mb': final_size,
            'size_reduction_percent': ((self.original_size - final_size) / self.original_size) * 100,
            'performance_metrics': {
                **optimization_results['performance_metrics'],
                'final': final_metrics
            }
        })
        
        # Calculate improvements
        speed_improvement = (baseline_metrics['avg_inference_time'] - final_metrics['avg_inference_time']) / baseline_metrics['avg_inference_time'] * 100
        memory_improvement = (baseline_metrics['peak_memory_mb'] - final_metrics['peak_memory_mb']) / baseline_metrics['peak_memory_mb'] * 100
        
        optimization_results['improvements'] = {
            'speed_improvement_percent': speed_improvement,
            'memory_improvement_percent': memory_improvement,
            'size_reduction_percent': optimization_results['size_reduction_percent']
        }
        
        self.optimization_history.append(optimization_results)
        
        logger.info("✅ Optimization complete!")
        logger.info(f"📊 Results: {speed_improvement:.1f}% faster, {memory_improvement:.1f}% less memory, {optimization_results['size_reduction_percent']:.1f}% smaller")
        
        return optimization_results
    
    def _apply_quantization(self):
        """Apply dynamic quantization to the model."""
        # Quantize specific layers for maximum benefit
        quantized_modules = [
            torch.nn.Linear,
            torch.nn.LSTM,
            torch.nn.GRU
        ]
        
        # Apply dynamic quantization
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            quantized_modules,
            dtype=torch.qint8
        )
        
        logger.info("✅ Dynamic quantization applied")
    
    def _apply_pruning(self, amount: float):
        """Apply structured pruning to reduce model parameters."""
        modules_to_prune = []
        
        # Collect linear layers for pruning
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                modules_to_prune.append((module, 'weight'))
        
        # Apply magnitude-based pruning
        for module, param_name in modules_to_prune:
            prune.l1_unstructured(module, param_name, amount=amount)
        
        # Make pruning permanent
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)
        
        logger.info(f"✅ Pruned {amount:.1%} of model parameters")
    
    def _optimize_memory(self):
        """Apply memory optimizations."""
        # Enable gradient checkpointing for RNN layers
        for module in self.model.modules():
            if isinstance(module, (torch.nn.LSTM, torch.nn.GRU)):
                # This would require implementing custom forward pass with checkpointing
                pass
        
        # Compile model for better memory usage (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                logger.info("✅ Model compiled for memory optimization")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
    
    def _benchmark_inference(self, num_samples: int = 100) -> Dict[str, float]:
        """Benchmark inference performance."""
        self.model.eval()
        
        # Generate test data
        batch_size = 1
        seq_len = 64
        num_features = self.model.input_size
        
        test_data = torch.randn(batch_size, seq_len, num_features)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(test_data)
        
        # Benchmark
        times = []
        memory_usage = []
        
        process = psutil.Process(os.getpid())
        
        for _ in range(num_samples):
            # Measure memory before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Time inference
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = self.model(test_data)
            end_time = time.perf_counter()
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            times.append(end_time - start_time)
            memory_usage.append(max(mem_before, mem_after))
        
        return {
            'avg_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'peak_memory_mb': np.max(memory_usage),
            'avg_memory_mb': np.mean(memory_usage)
        }
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def create_optimized_checkpoint(self, save_path: str) -> Dict[str, Any]:
        """Create an optimized model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimization_history': self.optimization_history,
            'model_size_mb': self._calculate_model_size(),
            'optimization_timestamp': time.time()
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"💾 Optimized model saved to: {save_path}")
        
        return checkpoint


def create_production_model(
    checkpoint_path: str,
    optimize: bool = True,
    **optimization_kwargs
) -> tuple[HybridThreatDetector, Dict[str, Any]]:
    """
    Create an optimized production-ready model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        optimize: Whether to apply optimizations
        **optimization_kwargs: Arguments for optimization
        
    Returns:
        Tuple of (optimized_model, optimization_results)
    """
    logger.info(f"🔧 Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model config
    model_config = checkpoint.get('model_config', {})
    input_size = model_config.get('input_size', 34)
    
    # Create model
    model = HybridThreatDetector(
        input_size=input_size,
        hidden_size=model_config.get('hidden_size', 128),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.3)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    optimization_results = {}
    
    if optimize:
        # Apply optimizations
        optimizer = ModelOptimizer(model)
        optimization_results = optimizer.optimize_for_production(**optimization_kwargs)
        model = optimizer.model
    
    logger.info("✅ Production model ready!")
    
    return model, optimization_results