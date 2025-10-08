"""
Performance Optimization Module

Provides model optimization techniques for faster inference:
- Quantization (INT8/FP16)
- ONNX export
- TensorRT optimization
- Model pruning
- Performance benchmarking
"""

# Import with try-except to handle optional dependencies
try:
    from .quantization import ModelQuantizer, QuantizationType
except ImportError as e:
    import warnings
    warnings.warn(f"quantization module import failed: {e}")
    ModelQuantizer = None
    QuantizationType = None

try:
    from .onnx_export import ONNXExporter
except ImportError as e:
    import warnings
    warnings.warn(f"onnx_export module import failed: {e}")
    ONNXExporter = None

try:
    from .tensorrt_optimizer import TensorRTOptimizer, TensorRTInferenceEngine
except ImportError as e:
    import warnings
    warnings.warn(f"tensorrt_optimizer module import failed: {e}")
    TensorRTOptimizer = None
    TensorRTInferenceEngine = None

try:
    from .pruning import ModelPruner, PruningStrategy
except ImportError as e:
    import warnings
    warnings.warn(f"pruning module import failed: {e}")
    ModelPruner = None
    PruningStrategy = None

try:
    from .benchmark import PerformanceBenchmark, BenchmarkResult
except ImportError as e:
    import warnings
    warnings.warn(f"benchmark module import failed: {e}")
    PerformanceBenchmark = None
    BenchmarkResult = None

__all__ = [
    'ModelQuantizer',
    'QuantizationType',
    'ONNXExporter',
    'TensorRTOptimizer',
    'TensorRTInferenceEngine',
    'ModelPruner',
    'PruningStrategy',
    'PerformanceBenchmark',
    'BenchmarkResult'
]

from .quantization import ModelQuantizer, QuantizationType
from .onnx_export import ONNXExporter
from .tensorrt_optimizer import TensorRTOptimizer
from .pruning import ModelPruner, PruningStrategy
from .benchmark import PerformanceBenchmark, BenchmarkResult

__all__ = [
    'ModelQuantizer',
    'QuantizationType',
    'ONNXExporter',
    'TensorRTOptimizer',
    'ModelPruner',
    'PruningStrategy',
    'PerformanceBenchmark',
    'BenchmarkResult',
]

__version__ = '1.0.0'
