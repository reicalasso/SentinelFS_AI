"""
TensorRT Optimization

Optimize models with NVIDIA TensorRT for maximum GPU performance.
Supports FP16 and INT8 precision modes.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import logging
import numpy as np


class TensorRTOptimizer:
    """
    Optimize models with TensorRT.
    
    Features:
    - FP32/FP16/INT8 precision modes
    - Dynamic shape support
    - Engine building and caching
    - Performance profiling
    - RTX 50-series optimizations
    """
    
    def __init__(self):
        """Initialize TensorRT optimizer."""
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check TensorRT availability
        try:
            import tensorrt as trt
            self.trt = trt
            self.trt_available = True
            self.logger.info(f"TensorRT version: {trt.__version__}")
        except ImportError:
            self.trt = None
            self.trt_available = False
            self.logger.warning("TensorRT not available")
    
    def optimize_onnx_model(
        self,
        onnx_model_path: str,
        engine_path: str,
        precision: str = 'fp16',
        max_batch_size: int = 1,
        workspace_size: int = 1 << 30,  # 1GB
        calibration_cache: Optional[str] = None,
        min_shapes: Optional[Dict[str, List[int]]] = None,
        opt_shapes: Optional[Dict[str, List[int]]] = None,
        max_shapes: Optional[Dict[str, List[int]]] = None
    ) -> str:
        """
        Build TensorRT engine from ONNX model.
        
        Args:
            onnx_model_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            precision: Precision mode ('fp32', 'fp16', 'int8')
            max_batch_size: Maximum batch size
            workspace_size: GPU workspace size in bytes
            calibration_cache: Path to INT8 calibration cache
            min_shapes: Minimum input shapes for dynamic shapes
            opt_shapes: Optimal input shapes for dynamic shapes
            max_shapes: Maximum input shapes for dynamic shapes
        
        Returns:
            Path to TensorRT engine
        """
        if not self.trt_available:
            raise RuntimeError("TensorRT not available")
        
        self.logger.info(f"Building TensorRT engine: {engine_path}")
        
        # Create builder and network
        TRT_LOGGER = self.trt.Logger(self.trt.Logger.WARNING)
        builder = self.trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(self.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = self.trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX model
        with open(onnx_model_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                self.logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    self.logger.error(parser.get_error(error))
                raise RuntimeError("ONNX parsing failed")
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            self.trt.MemoryPoolType.WORKSPACE,
            workspace_size
        )
        
        # Set precision
        if precision == 'fp16':
            if builder.platform_has_fast_fp16:
                config.set_flag(self.trt.BuilderFlag.FP16)
                self.logger.info("Enabled FP16 precision")
            else:
                self.logger.warning("FP16 not supported, using FP32")
        elif precision == 'int8':
            if builder.platform_has_fast_int8:
                config.set_flag(self.trt.BuilderFlag.INT8)
                if calibration_cache:
                    # Load calibration cache
                    config.int8_calibrator = self._create_calibrator(calibration_cache)
                self.logger.info("Enabled INT8 precision")
            else:
                self.logger.warning("INT8 not supported, using FP32")
        
        # Configure dynamic shapes if provided
        if min_shapes and opt_shapes and max_shapes:
            profile = builder.create_optimization_profile()
            for input_name in min_shapes:
                profile.set_shape(
                    input_name,
                    min_shapes[input_name],
                    opt_shapes[input_name],
                    max_shapes[input_name]
                )
            config.add_optimization_profile(profile)
            self.logger.info("Configured dynamic shapes")
        
        # Build engine
        self.logger.info("Building TensorRT engine (this may take a while)...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        self.logger.info(f"TensorRT engine saved to {engine_path}")
        return engine_path
    
    def _create_calibrator(self, calibration_cache: str):
        """Create INT8 calibrator (placeholder)."""
        # This is a simplified version - real implementation would need
        # a proper calibration dataset
        class DummyCalibrator(self.trt.IInt8EntropyCalibrator2):
            def __init__(self, cache_file):
                super().__init__()
                self.cache_file = cache_file
            
            def get_batch_size(self):
                return 1
            
            def get_batch(self, names):
                return None
            
            def read_calibration_cache(self):
                if Path(self.cache_file).exists():
                    with open(self.cache_file, 'rb') as f:
                        return f.read()
                return None
            
            def write_calibration_cache(self, cache):
                with open(self.cache_file, 'wb') as f:
                    f.write(cache)
        
        return DummyCalibrator(calibration_cache)
    
    def load_engine(self, engine_path: str):
        """
        Load TensorRT engine from file.
        
        Args:
            engine_path: Path to engine file
        
        Returns:
            TensorRT engine and context
        """
        if not self.trt_available:
            raise RuntimeError("TensorRT not available")
        
        TRT_LOGGER = self.trt.Logger(self.trt.Logger.WARNING)
        runtime = self.trt.Runtime(TRT_LOGGER)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        self.logger.info(f"Loaded TensorRT engine from {engine_path}")
        return engine, context
    
    def infer(
        self,
        engine,
        context,
        input_data: np.ndarray,
        input_name: str = 'input',
        output_name: str = 'output'
    ) -> np.ndarray:
        """
        Run inference with TensorRT engine.
        
        Args:
            engine: TensorRT engine
            context: TensorRT execution context
            input_data: Input numpy array
            input_name: Name of input tensor
            output_name: Name of output tensor
        
        Returns:
            Output numpy array
        """
        if not self.trt_available:
            raise RuntimeError("TensorRT not available")
        
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Allocate buffers
        bindings = []
        stream = cuda.Stream()
        
        # Input
        input_idx = engine.get_binding_index(input_name)
        input_shape = context.get_binding_shape(input_idx)
        input_size = self.trt.volume(input_shape)
        input_dtype = self.trt.nptype(engine.get_binding_dtype(input_idx))
        
        d_input = cuda.mem_alloc(input_data.nbytes)
        bindings.append(int(d_input))
        
        # Output
        output_idx = engine.get_binding_index(output_name)
        output_shape = context.get_binding_shape(output_idx)
        output_size = self.trt.volume(output_shape)
        output_dtype = self.trt.nptype(engine.get_binding_dtype(output_idx))
        
        h_output = np.empty(output_size, dtype=output_dtype)
        d_output = cuda.mem_alloc(h_output.nbytes)
        bindings.append(int(d_output))
        
        # Transfer input data to device
        cuda.memcpy_htod_async(d_input, input_data, stream)
        
        # Execute inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        # Transfer output back to host
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        
        # Synchronize stream
        stream.synchronize()
        
        return h_output.reshape(output_shape)
    
    def benchmark_engine(
        self,
        engine,
        context,
        input_data: np.ndarray,
        input_name: str = 'input',
        num_iterations: int = 1000,
        warmup_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark TensorRT engine performance.
        
        Args:
            engine: TensorRT engine
            context: TensorRT execution context
            input_data: Input numpy array
            input_name: Name of input tensor
            num_iterations: Number of inference iterations
            warmup_iterations: Number of warmup iterations
        
        Returns:
            Performance metrics
        """
        import time
        
        # Warmup
        for _ in range(warmup_iterations):
            _ = self.infer(engine, context, input_data, input_name)
        
        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.infer(engine, context, input_data, input_name)
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
        
        self.logger.info(f"TensorRT benchmark results: {metrics}")
        return metrics
    
    def get_engine_info(self, engine) -> Dict[str, Any]:
        """
        Get information about TensorRT engine.
        
        Args:
            engine: TensorRT engine
        
        Returns:
            Engine information
        """
        if not self.trt_available:
            raise RuntimeError("TensorRT not available")
        
        info = {
            'num_bindings': engine.num_bindings,
            'max_batch_size': engine.max_batch_size,
            'device_memory_size': engine.device_memory_size,
            'bindings': []
        }
        
        for i in range(engine.num_bindings):
            binding_info = {
                'name': engine.get_binding_name(i),
                'shape': engine.get_binding_shape(i),
                'dtype': str(engine.get_binding_dtype(i)),
                'is_input': engine.binding_is_input(i)
            }
            info['bindings'].append(binding_info)
        
        return info
    
    def optimize_for_rtx50(
        self,
        onnx_model_path: str,
        engine_path: str,
        **kwargs
    ) -> str:
        """
        Optimize model specifically for RTX 50-series GPUs.
        
        Uses optimal settings for Ada Lovelace architecture.
        
        Args:
            onnx_model_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            **kwargs: Additional optimization parameters
        
        Returns:
            Path to optimized engine
        """
        # RTX 50-series specific optimizations
        rtx50_config = {
            'precision': 'fp16',  # RTX 50-series has excellent FP16 performance
            'workspace_size': 4 << 30,  # 4GB workspace for large models
            'max_batch_size': 1,  # Optimize for low latency
        }
        
        # Merge with user config
        rtx50_config.update(kwargs)
        
        self.logger.info("Optimizing for RTX 50-series GPU")
        return self.optimize_onnx_model(
            onnx_model_path,
            engine_path,
            **rtx50_config
        )


class TensorRTInferenceEngine:
    """
    High-level inference engine using TensorRT.
    
    Provides a simple interface for TensorRT inference.
    """
    
    def __init__(self, engine_path: str):
        """
        Initialize inference engine.
        
        Args:
            engine_path: Path to TensorRT engine
        """
        self.logger = logging.getLogger(__name__)
        self.optimizer = TensorRTOptimizer()
        
        if not self.optimizer.trt_available:
            raise RuntimeError("TensorRT not available")
        
        self.engine, self.context = self.optimizer.load_engine(engine_path)
        self.logger.info("TensorRT inference engine ready")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run prediction.
        
        Args:
            input_data: Input numpy array
        
        Returns:
            Prediction output
        """
        return self.optimizer.infer(
            self.engine,
            self.context,
            input_data
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Get engine information."""
        return self.optimizer.get_engine_info(self.engine)
