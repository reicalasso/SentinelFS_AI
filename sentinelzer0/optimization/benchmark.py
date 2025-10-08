"""
Performance Benchmarking

Comprehensive performance profiling and benchmarking tools.
Measures latency, throughput, memory usage, and more.
"""

import torch
import torch.nn as nn
import time
import psutil
import numpy as np
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import logging
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    mean_latency_ms: float
    median_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    cpu_percent: float
    gpu_utilization: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mean_latency_ms': self.mean_latency_ms,
            'median_latency_ms': self.median_latency_ms,
            'min_latency_ms': self.min_latency_ms,
            'max_latency_ms': self.max_latency_ms,
            'std_latency_ms': self.std_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'throughput_qps': self.throughput_qps,
            'memory_allocated_mb': self.memory_allocated_mb,
            'memory_reserved_mb': self.memory_reserved_mb,
            'cpu_percent': self.cpu_percent,
            'gpu_utilization': self.gpu_utilization
        }


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite.
    
    Features:
    - Latency measurement (mean, median, percentiles)
    - Throughput testing
    - Memory profiling (CPU & GPU)
    - GPU utilization monitoring
    - Batch size optimization
    - Multi-run averaging
    """
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def benchmark_model(
        self,
        model: nn.Module,
        test_input: torch.Tensor,
        num_iterations: int = 1000,
        warmup_iterations: int = 100,
        measure_memory: bool = True,
        measure_gpu: bool = True
    ) -> BenchmarkResult:
        """
        Comprehensive model benchmarking.
        
        Args:
            model: Model to benchmark
            test_input: Test input tensor
            num_iterations: Number of inference iterations
            warmup_iterations: Number of warmup iterations
            measure_memory: Whether to measure memory usage
            measure_gpu: Whether to measure GPU utilization
        
        Returns:
            Benchmark results
        """
        self.logger.info(f"Starting benchmark: {num_iterations} iterations")
        
        # Move to device
        model = model.to(self.device)
        test_input = test_input.to(self.device)
        model.eval()
        
        # Clear cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(test_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        memory_snapshots = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                # Measure latency
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.time()
                _ = model(test_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                latency = (time.time() - start) * 1000  # ms
                latencies.append(latency)
                
                # Measure memory
                if measure_memory:
                    if self.device.type == 'cuda':
                        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
                    else:
                        allocated = 0.0
                        reserved = 0.0
                    
                    memory_snapshots.append({
                        'allocated': allocated,
                        'reserved': reserved
                    })
        
        # Calculate statistics
        latencies = np.array(latencies)
        
        mean_latency = latencies.mean()
        median_latency = np.median(latencies)
        min_latency = latencies.min()
        max_latency = latencies.max()
        std_latency = latencies.std()
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        throughput = 1000 / mean_latency  # QPS
        
        # Memory statistics
        if measure_memory and memory_snapshots:
            avg_allocated = np.mean([s['allocated'] for s in memory_snapshots])
            avg_reserved = np.mean([s['reserved'] for s in memory_snapshots])
        else:
            avg_allocated = 0.0
            avg_reserved = 0.0
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # GPU utilization
        gpu_util = None
        if measure_gpu and self.device.type == 'cuda':
            gpu_util = self._get_gpu_utilization()
        
        result = BenchmarkResult(
            mean_latency_ms=float(mean_latency),
            median_latency_ms=float(median_latency),
            min_latency_ms=float(min_latency),
            max_latency_ms=float(max_latency),
            std_latency_ms=float(std_latency),
            p95_latency_ms=float(p95_latency),
            p99_latency_ms=float(p99_latency),
            throughput_qps=float(throughput),
            memory_allocated_mb=float(avg_allocated),
            memory_reserved_mb=float(avg_reserved),
            cpu_percent=float(cpu_percent),
            gpu_utilization=float(gpu_util) if gpu_util is not None else None
        )
        
        self.logger.info(f"Benchmark completed: {mean_latency:.3f}ms mean latency")
        return result
    
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage."""
        try:
            import pynvml
            
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            pynvml.nvmlShutdown()
            
            return util.gpu
        
        except Exception as e:
            self.logger.warning(f"Failed to get GPU utilization: {e}")
            return None
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        test_input: torch.Tensor,
        num_iterations: int = 1000
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of model_name -> model
            test_input: Test input tensor
            num_iterations: Number of iterations per model
        
        Returns:
            Dictionary of model_name -> benchmark results
        """
        self.logger.info(f"Comparing {len(models)} models")
        
        results = {}
        for name, model in models.items():
            self.logger.info(f"Benchmarking {name}...")
            results[name] = self.benchmark_model(
                model,
                test_input,
                num_iterations=num_iterations
            )
        
        # Print comparison
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict[str, BenchmarkResult]):
        """Print comparison table."""
        self.logger.info("\n" + "="*80)
        self.logger.info("Model Comparison Results")
        self.logger.info("="*80)
        
        for name, result in results.items():
            self.logger.info(f"\n{name}:")
            self.logger.info(f"  Latency: {result.mean_latency_ms:.3f}ms (mean)")
            self.logger.info(f"  P95: {result.p95_latency_ms:.3f}ms, P99: {result.p99_latency_ms:.3f}ms")
            self.logger.info(f"  Throughput: {result.throughput_qps:.1f} QPS")
            self.logger.info(f"  Memory: {result.memory_allocated_mb:.1f}MB allocated")
        
        self.logger.info("="*80 + "\n")
    
    def profile_memory(
        self,
        model: nn.Module,
        test_input: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Profile memory usage in detail.
        
        Args:
            model: Model to profile
            test_input: Test input tensor
        
        Returns:
            Memory profile
        """
        self.logger.info("Profiling memory usage")
        
        model = model.to(self.device)
        test_input = test_input.to(self.device)
        model.eval()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Before inference
        mem_before = self._get_memory_stats()
        
        # Inference
        with torch.no_grad():
            _ = model(test_input)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        
        # After inference
        mem_after = self._get_memory_stats()
        
        # Peak memory
        mem_peak = self._get_peak_memory_stats()
        
        profile = {
            'before_inference': mem_before,
            'after_inference': mem_after,
            'peak_usage': mem_peak,
            'inference_delta': {
                'allocated_mb': mem_after['allocated_mb'] - mem_before['allocated_mb'],
                'reserved_mb': mem_after['reserved_mb'] - mem_before['reserved_mb']
            }
        }
        
        self.logger.info(f"Memory profile: {profile['inference_delta']}")
        return profile
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        if self.device.type == 'cuda':
            return {
                'allocated_mb': torch.cuda.memory_allocated() / (1024 ** 2),
                'reserved_mb': torch.cuda.memory_reserved() / (1024 ** 2)
            }
        else:
            process = psutil.Process()
            return {
                'allocated_mb': process.memory_info().rss / (1024 ** 2),
                'reserved_mb': 0.0
            }
    
    def _get_peak_memory_stats(self) -> Dict[str, float]:
        """Get peak memory statistics."""
        if self.device.type == 'cuda':
            return {
                'allocated_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
                'reserved_mb': torch.cuda.max_memory_reserved() / (1024 ** 2)
            }
        else:
            return {'allocated_mb': 0.0, 'reserved_mb': 0.0}
    
    def find_optimal_batch_size(
        self,
        model: nn.Module,
        input_shape: tuple,
        max_batch_size: int = 128,
        target_latency_ms: Optional[float] = None
    ) -> int:
        """
        Find optimal batch size for throughput or latency target.
        
        Args:
            model: Model to test
            input_shape: Shape of single input (without batch dimension)
            max_batch_size: Maximum batch size to test
            target_latency_ms: Target latency per sample (optional)
        
        Returns:
            Optimal batch size
        """
        self.logger.info("Finding optimal batch size")
        
        model = model.to(self.device)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        batch_sizes = [bs for bs in batch_sizes if bs <= max_batch_size]
        
        results = {}
        
        for batch_size in batch_sizes:
            # Create test input
            test_input = torch.randn(batch_size, *input_shape).to(self.device)
            
            try:
                # Benchmark
                result = self.benchmark_model(
                    model,
                    test_input,
                    num_iterations=100,
                    warmup_iterations=10
                )
                
                # Calculate per-sample latency
                per_sample_latency = result.mean_latency_ms / batch_size
                
                results[batch_size] = {
                    'total_latency_ms': result.mean_latency_ms,
                    'per_sample_latency_ms': per_sample_latency,
                    'throughput_qps': result.throughput_qps * batch_size,
                    'memory_mb': result.memory_allocated_mb
                }
                
                self.logger.info(
                    f"Batch size {batch_size}: "
                    f"{per_sample_latency:.3f}ms/sample, "
                    f"{results[batch_size]['throughput_qps']:.1f} QPS"
                )
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.warning(f"OOM at batch size {batch_size}")
                    break
                else:
                    raise
        
        # Find optimal batch size
        if target_latency_ms is not None:
            # Find largest batch size meeting latency target
            optimal = 1
            for bs, metrics in results.items():
                if metrics['per_sample_latency_ms'] <= target_latency_ms:
                    optimal = bs
        else:
            # Find batch size with best throughput
            optimal = max(results.keys(), key=lambda bs: results[bs]['throughput_qps'])
        
        self.logger.info(f"Optimal batch size: {optimal}")
        return optimal
    
    def save_benchmark_results(
        self,
        results: Dict[str, BenchmarkResult],
        output_path: str
    ):
        """
        Save benchmark results to file.
        
        Args:
            results: Benchmark results
            output_path: Path to save results
        """
        # Convert to serializable format
        serializable = {
            name: result.to_dict()
            for name, result in results.items()
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        self.logger.info(f"Saved benchmark results to {output_path}")
    
    def load_benchmark_results(
        self,
        input_path: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load benchmark results from file.
        
        Args:
            input_path: Path to load results from
        
        Returns:
            Benchmark results
        """
        with open(input_path, 'r') as f:
            results = json.load(f)
        
        self.logger.info(f"Loaded benchmark results from {input_path}")
        return results
    
    def generate_report(
        self,
        results: Dict[str, BenchmarkResult],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate markdown benchmark report.
        
        Args:
            results: Benchmark results
            output_path: Optional path to save report
        
        Returns:
            Report markdown
        """
        report = "# Performance Benchmark Report\n\n"
        report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary table
        report += "## Summary\n\n"
        report += "| Model | Mean Latency (ms) | P95 (ms) | P99 (ms) | Throughput (QPS) | Memory (MB) |\n"
        report += "|-------|-------------------|----------|----------|------------------|-------------|\n"
        
        for name, result in results.items():
            report += f"| {name} | {result.mean_latency_ms:.3f} | {result.p95_latency_ms:.3f} | "
            report += f"{result.p99_latency_ms:.3f} | {result.throughput_qps:.1f} | "
            report += f"{result.memory_allocated_mb:.1f} |\n"
        
        # Detailed results
        report += "\n## Detailed Results\n\n"
        for name, result in results.items():
            report += f"### {name}\n\n"
            report += f"- **Mean Latency**: {result.mean_latency_ms:.3f} ms\n"
            report += f"- **Median Latency**: {result.median_latency_ms:.3f} ms\n"
            report += f"- **Min Latency**: {result.min_latency_ms:.3f} ms\n"
            report += f"- **Max Latency**: {result.max_latency_ms:.3f} ms\n"
            report += f"- **Std Dev**: {result.std_latency_ms:.3f} ms\n"
            report += f"- **P95 Latency**: {result.p95_latency_ms:.3f} ms\n"
            report += f"- **P99 Latency**: {result.p99_latency_ms:.3f} ms\n"
            report += f"- **Throughput**: {result.throughput_qps:.1f} QPS\n"
            report += f"- **Memory (Allocated)**: {result.memory_allocated_mb:.1f} MB\n"
            report += f"- **Memory (Reserved)**: {result.memory_reserved_mb:.1f} MB\n"
            report += f"- **CPU Usage**: {result.cpu_percent:.1f}%\n"
            if result.gpu_utilization is not None:
                report += f"- **GPU Utilization**: {result.gpu_utilization:.1f}%\n"
            report += "\n"
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Saved report to {output_path}")
        
        return report
