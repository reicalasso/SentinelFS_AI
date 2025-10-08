"""
Model Quantization

Provides INT8 and FP16 quantization for faster inference with minimal accuracy loss.
Supports both static and dynamic quantization strategies.
"""

import torch
import torch.nn as nn
from enum import Enum
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import logging


class QuantizationType(Enum):
    """Quantization type options."""
    INT8_DYNAMIC = "int8_dynamic"
    INT8_STATIC = "int8_static"
    FP16 = "fp16"
    NONE = "none"


class ModelQuantizer:
    """
    Model quantization for performance optimization.
    
    Features:
    - INT8 dynamic quantization (CPU/GPU)
    - INT8 static quantization with calibration
    - FP16 mixed precision
    - Automatic fallback for unsupported ops
    - Accuracy validation
    """
    
    def __init__(self):
        """Initialize quantizer."""
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def quantize_model(
        self,
        model: nn.Module,
        quantization_type: QuantizationType = QuantizationType.INT8_DYNAMIC,
        calibration_data: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """
        Quantize a PyTorch model.
        
        Args:
            model: Model to quantize
            quantization_type: Type of quantization
            calibration_data: Calibration data for static quantization
        
        Returns:
            Quantized model
        """
        if quantization_type == QuantizationType.NONE:
            return model
        
        self.logger.info(f"Quantizing model with {quantization_type.value}")
        
        if quantization_type == QuantizationType.FP16:
            return self._quantize_fp16(model)
        elif quantization_type == QuantizationType.INT8_DYNAMIC:
            return self._quantize_int8_dynamic(model)
        elif quantization_type == QuantizationType.INT8_STATIC:
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")
            return self._quantize_int8_static(model, calibration_data)
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    def _quantize_fp16(self, model: nn.Module) -> nn.Module:
        """Quantize model to FP16 (half precision)."""
        try:
            if self.device.type == 'cuda':
                # Use half() for GPU
                quantized_model = model.half()
                self.logger.info("Applied FP16 quantization (GPU)")
            else:
                # FP16 not well supported on CPU, use float32
                self.logger.warning("FP16 not well supported on CPU, keeping FP32")
                quantized_model = model
            
            return quantized_model
        
        except Exception as e:
            self.logger.error(f"FP16 quantization failed: {e}")
            return model
    
    def _quantize_int8_dynamic(self, model: nn.Module) -> nn.Module:
        """
        Apply INT8 dynamic quantization.
        
        Good for models with dynamic input shapes.
        Quantizes weights, activations computed at runtime.
        """
        try:
            # Set model to eval mode
            model.eval()
            
            # Apply dynamic quantization to linear and LSTM layers
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},
                dtype=torch.qint8
            )
            
            self.logger.info("Applied INT8 dynamic quantization")
            return quantized_model
        
        except Exception as e:
            self.logger.error(f"INT8 dynamic quantization failed: {e}")
            return model
    
    def _quantize_int8_static(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor
    ) -> nn.Module:
        """
        Apply INT8 static quantization with calibration.
        
        Requires calibration data to determine optimal quantization parameters.
        Best accuracy but requires representative data.
        """
        try:
            # Set model to eval mode
            model.eval()
            
            # Set quantization config
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Prepare model for quantization
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate with sample data
            with torch.no_grad():
                model(calibration_data)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(model, inplace=False)
            
            self.logger.info("Applied INT8 static quantization")
            return quantized_model
        
        except Exception as e:
            self.logger.error(f"INT8 static quantization failed: {e}")
            return model
    
    def measure_model_size(self, model: nn.Module) -> float:
        """
        Measure model size in MB.
        
        Args:
            model: Model to measure
        
        Returns:
            Model size in MB
        """
        # Save model to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(model.state_dict(), tmp.name)
            size_mb = Path(tmp.name).stat().st_size / (1024 * 1024)
        
        return size_mb
    
    def compare_models(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_input: torch.Tensor,
        test_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Compare original and quantized models.
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            test_input: Test input data
            test_labels: Test labels (optional)
        
        Returns:
            Comparison metrics
        """
        import time
        
        # Model sizes
        original_size = self.measure_model_size(original_model)
        quantized_size = self.measure_model_size(quantized_model)
        
        # Inference time
        original_model.eval()
        quantized_model.eval()
        
        with torch.no_grad():
            # Original model
            start = time.time()
            for _ in range(100):
                original_output = original_model(test_input)
            original_time = (time.time() - start) / 100
            
            # Quantized model
            start = time.time()
            for _ in range(100):
                quantized_output = quantized_model(test_input)
            quantized_time = (time.time() - start) / 100
        
        # Output comparison
        if isinstance(original_output, tuple):
            original_output = original_output[0]
        if isinstance(quantized_output, tuple):
            quantized_output = quantized_output[0]
        
        # Convert to same dtype for comparison
        if quantized_output.dtype != original_output.dtype:
            quantized_output = quantized_output.float()
        
        output_diff = torch.abs(original_output - quantized_output).mean().item()
        
        comparison = {
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'size_reduction': (original_size - quantized_size) / original_size,
            'original_latency_ms': original_time * 1000,
            'quantized_latency_ms': quantized_time * 1000,
            'speedup': original_time / quantized_time,
            'output_diff': output_diff
        }
        
        # Accuracy comparison if labels provided
        if test_labels is not None:
            original_acc = self._calculate_accuracy(original_output, test_labels)
            quantized_acc = self._calculate_accuracy(quantized_output, test_labels)
            comparison['original_accuracy'] = original_acc
            comparison['quantized_accuracy'] = quantized_acc
            comparison['accuracy_drop'] = original_acc - quantized_acc
        
        return comparison
    
    def _calculate_accuracy(
        self,
        output: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """Calculate classification accuracy."""
        if output.dim() == 2:
            predictions = output.argmax(dim=1)
        else:
            predictions = (output > 0.5).long().squeeze()
        
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        return correct / total if total > 0 else 0.0
    
    def save_quantized_model(
        self,
        model: nn.Module,
        save_path: str,
        quantization_info: Optional[Dict[str, Any]] = None
    ):
        """
        Save quantized model with metadata.
        
        Args:
            model: Quantized model
            save_path: Path to save model
            quantization_info: Quantization metadata
        """
        save_dict = {
            'model_state_dict': model.state_dict(),
            'quantization_info': quantization_info or {}
        }
        
        torch.save(save_dict, save_path)
        self.logger.info(f"Saved quantized model to {save_path}")
    
    def load_quantized_model(
        self,
        model: nn.Module,
        load_path: str
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load quantized model with metadata.
        
        Args:
            model: Model architecture (for loading state dict)
            load_path: Path to load model from
        
        Returns:
            Loaded model and quantization info
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        quantization_info = checkpoint.get('quantization_info', {})
        
        self.logger.info(f"Loaded quantized model from {load_path}")
        return model, quantization_info
