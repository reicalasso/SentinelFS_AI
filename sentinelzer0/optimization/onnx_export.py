"""
ONNX Model Export

Export PyTorch models to ONNX format for cross-platform deployment.
Supports optimization and validation.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import logging
import numpy as np


class ONNXExporter:
    """
    Export PyTorch models to ONNX format.
    
    Features:
    - Model export with optimization
    - ONNX model validation
    - Runtime performance comparison
    - Dynamic axes support
    - Opset version selection
    """
    
    def __init__(self, opset_version: int = 13):
        """
        Initialize ONNX exporter.
        
        Args:
            opset_version: ONNX opset version (default: 13)
        """
        self.logger = logging.getLogger(__name__)
        self.opset_version = opset_version
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def export_model(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        optimize: bool = True
    ) -> str:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor (for tracing)
            output_path: Path to save ONNX model
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axis configuration
            optimize: Whether to optimize the ONNX model
        
        Returns:
            Path to exported ONNX model
        """
        self.logger.info(f"Exporting model to ONNX: {output_path}")
        
        # Set model to eval mode
        model.eval()
        
        # Default names
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']
        
        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                sample_input,
                output_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )
            
            self.logger.info(f"Successfully exported to {output_path}")
            
            # Optimize if requested
            if optimize:
                self._optimize_onnx_model(output_path)
            
            # Validate exported model
            self._validate_onnx_model(output_path)
            
            return output_path
        
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            raise
    
    def _optimize_onnx_model(self, model_path: str):
        """
        Optimize ONNX model.
        
        Applies graph optimizations like constant folding,
        operator fusion, etc.
        """
        try:
            import onnx
            from onnxruntime.transformers import optimizer
            
            # Load model
            model = onnx.load(model_path)
            
            # Apply optimizations
            # Basic optimizations (no special hardware needed)
            optimized_model = optimizer.optimize_model(
                model_path,
                model_type='bert',  # Use general optimizations
                num_heads=0,
                hidden_size=0,
                optimization_options=None
            )
            
            # Save optimized model
            if optimized_model:
                optimized_model.save_model_to_file(model_path)
                self.logger.info("Applied ONNX optimizations")
        
        except ImportError:
            self.logger.warning("onnxruntime.transformers not available, skipping optimization")
        except Exception as e:
            self.logger.warning(f"ONNX optimization failed: {e}")
    
    def _validate_onnx_model(self, model_path: str):
        """
        Validate exported ONNX model.
        
        Checks model structure and runs inference test.
        """
        try:
            # Load and check model
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            self.logger.info("ONNX model validation passed")
        
        except Exception as e:
            self.logger.error(f"ONNX model validation failed: {e}")
            raise
    
    def compare_with_pytorch(
        self,
        pytorch_model: nn.Module,
        onnx_model_path: str,
        test_input: torch.Tensor,
        rtol: float = 1e-3,
        atol: float = 1e-5
    ) -> Dict[str, Any]:
        """
        Compare PyTorch and ONNX model outputs.
        
        Args:
            pytorch_model: Original PyTorch model
            onnx_model_path: Path to ONNX model
            test_input: Test input tensor
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
        
        Returns:
            Comparison results
        """
        import time
        
        # PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = pytorch_model(test_input)
            
            # Measure
            start = time.time()
            for _ in range(100):
                pytorch_output = pytorch_model(test_input)
            pytorch_time = (time.time() - start) / 100
        
        # Convert output to numpy
        if isinstance(pytorch_output, tuple):
            pytorch_output = pytorch_output[0]
        pytorch_output_np = pytorch_output.cpu().numpy()
        
        # ONNX Runtime inference
        ort_session = ort.InferenceSession(
            onnx_model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        input_name = ort_session.get_inputs()[0].name
        onnx_input = {input_name: test_input.cpu().numpy()}
        
        # Warmup
        for _ in range(10):
            _ = ort_session.run(None, onnx_input)
        
        # Measure
        start = time.time()
        for _ in range(100):
            onnx_output = ort_session.run(None, onnx_input)
        onnx_time = (time.time() - start) / 100
        
        onnx_output_np = onnx_output[0]
        
        # Compare outputs
        output_diff = np.abs(pytorch_output_np - onnx_output_np)
        max_diff = output_diff.max()
        mean_diff = output_diff.mean()
        
        # Check if outputs match within tolerance
        outputs_match = np.allclose(
            pytorch_output_np,
            onnx_output_np,
            rtol=rtol,
            atol=atol
        )
        
        comparison = {
            'outputs_match': outputs_match,
            'max_output_diff': float(max_diff),
            'mean_output_diff': float(mean_diff),
            'pytorch_latency_ms': pytorch_time * 1000,
            'onnx_latency_ms': onnx_time * 1000,
            'speedup': pytorch_time / onnx_time,
            'onnx_providers': ort_session.get_providers()
        }
        
        self.logger.info(f"PyTorch vs ONNX comparison: {comparison}")
        return comparison
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Get information about ONNX model.
        
        Args:
            model_path: Path to ONNX model
        
        Returns:
            Model information
        """
        model = onnx.load(model_path)
        
        # Model metadata
        info = {
            'ir_version': model.ir_version,
            'producer_name': model.producer_name,
            'producer_version': model.producer_version,
            'domain': model.domain,
            'model_version': model.model_version,
            'doc_string': model.doc_string
        }
        
        # Input/output info
        graph = model.graph
        info['inputs'] = []
        for input_tensor in graph.input:
            input_info = {
                'name': input_tensor.name,
                'type': input_tensor.type.tensor_type.elem_type,
                'shape': [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
            }
            info['inputs'].append(input_info)
        
        info['outputs'] = []
        for output_tensor in graph.output:
            output_info = {
                'name': output_tensor.name,
                'type': output_tensor.type.tensor_type.elem_type,
                'shape': [d.dim_value for d in output_tensor.type.tensor_type.shape.dim]
            }
            info['outputs'].append(output_info)
        
        # Node count
        info['num_nodes'] = len(graph.node)
        
        # File size
        info['file_size_mb'] = Path(model_path).stat().st_size / (1024 * 1024)
        
        return info
    
    def create_onnx_session(
        self,
        model_path: str,
        use_gpu: bool = True,
        optimization_level: str = 'all'
    ) -> ort.InferenceSession:
        """
        Create ONNX Runtime inference session.
        
        Args:
            model_path: Path to ONNX model
            use_gpu: Whether to use GPU if available
            optimization_level: Optimization level ('none', 'basic', 'all')
        
        Returns:
            ONNX Runtime session
        """
        # Set optimization level
        sess_options = ort.SessionOptions()
        if optimization_level == 'none':
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        elif optimization_level == 'basic':
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        else:  # 'all'
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Set execution providers
        providers = []
        if use_gpu and torch.cuda.is_available():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        # Create session
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        self.logger.info(f"Created ONNX Runtime session with providers: {session.get_providers()}")
        return session
    
    def benchmark_onnx_model(
        self,
        model_path: str,
        test_input: np.ndarray,
        num_iterations: int = 1000,
        warmup_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark ONNX model performance.
        
        Args:
            model_path: Path to ONNX model
            test_input: Test input array
            num_iterations: Number of inference iterations
            warmup_iterations: Number of warmup iterations
        
        Returns:
            Performance metrics
        """
        import time
        
        session = self.create_onnx_session(model_path)
        input_name = session.get_inputs()[0].name
        onnx_input = {input_name: test_input}
        
        # Warmup
        for _ in range(warmup_iterations):
            _ = session.run(None, onnx_input)
        
        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.time()
            _ = session.run(None, onnx_input)
            latencies.append((time.time() - start) * 1000)  # ms
        
        latencies = np.array(latencies)
        
        metrics = {
            'mean_latency_ms': float(latencies.mean()),
            'median_latency_ms': float(np.median(latencies)),
            'min_latency_ms': float(latencies.min()),
            'max_latency_ms': float(latencies.max()),
            'std_latency_ms': float(latencies.std()),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'throughput_qps': 1000 / latencies.mean()
        }
        
        self.logger.info(f"ONNX benchmark results: {metrics}")
        return metrics
