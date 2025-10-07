"""
Model lifecycle management: saving, loading, versioning, and optimization.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import pickle
from datetime import datetime
import os
import shutil
from packaging import version as pkg_version

from ..models.behavioral_analyzer import BehavioralAnalyzer
from ..models.advanced_models import (
    TransformerBehavioralAnalyzer, 
    CNNLSTMAnalyzer, 
    EnsembleAnalyzer
)
from ..data.feature_extractor import FeatureExtractor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """
    Manage model lifecycle: training, saving, loading, versioning, and optimization.
    
    Features:
    - Model versioning with semantic versioning
    - Checkpoint management
    - Model export (ONNX, TorchScript, TensorFlow SavedModel)
    - Model quantization and optimization
    - Performance profiling and monitoring
    - Model validation and integrity checking
    - Multi-architecture support
    """
    
    def __init__(self, model_dir: Path = Path('./models')):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.model_dir / 'metadata.json'
        self.scaler_file = self.model_dir / 'scaler.pkl'
        self.export_dir = self.model_dir / 'exports'
        self.export_dir.mkdir(exist_ok=True)
        self.checkpoint_dir = self.model_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Model architecture mapping
        self.architecture_map = {
            'BehavioralAnalyzer': BehavioralAnalyzer,
            'TransformerBehavioralAnalyzer': TransformerBehavioralAnalyzer,
            'CNNLSTMAnalyzer': CNNLSTMAnalyzer,
            'EnsembleAnalyzer': EnsembleAnalyzer
        }
    
    def save_model(
        self, 
        model: nn.Module, 
        version: str, 
        metrics: Dict[str, float], 
        feature_extractor: FeatureExtractor,
        export_formats: Optional[List[str]] = None,
        model_name: Optional[str] = None
    ):
        """
        Save model with versioning and metadata.
        
        Args:
            model: Trained model
            version: Version string (e.g., '1.0.0')
            metrics: Performance metrics
            feature_extractor: Fitted feature extractor with scaler
            export_formats: Optional list of export formats ['onnx', 'torchscript', 'quantized', 'tensorflow']
            model_name: Optional custom name for the model
        """
        # Validate version format (semantic versioning)
        try:
            pkg_version.parse(version)
        except:
            raise ValueError(f"Invalid version format: {version}. Use semantic versioning (e.g., '1.0.0').")
        
        model_name = model_name or type(model).__name__
        model_path = self.model_dir / f'{model_name}_v{version}.pt'
        
        # Get model config
        model_config = {
            'model_type': model_name,
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'architecture_params': {}
        }
        
        # Add optional config based on model type
        if hasattr(model, 'use_attention'):
            model_config['use_attention'] = model.use_attention
        if hasattr(model, 'bidirectional'):
            model_config['bidirectional'] = model.bidirectional
        if hasattr(model, 'd_model'):
            model_config['d_model'] = model.d_model
        if hasattr(model, 'nhead'):
            model_config['nhead'] = model.nhead
        
        # Add architecture-specific parameters
        for attr in ['use_attention', 'bidirectional', 'd_model', 'nhead', 
                     'num_layers', 'seq_len']:
            if hasattr(model, attr):
                model_config['architecture_params'][attr] = getattr(model, attr)
        
        # Special handling for dropout to ensure it's a float/number
        if hasattr(model, 'dropout'):
            dropout_val = getattr(model, 'dropout')
            # Handle if dropout is a module vs a float/number
            if hasattr(dropout_val, 'p'):  # It's a Dropout module
                model_config['architecture_params']['dropout'] = dropout_val.p
            elif isinstance(dropout_val, (int, float)):
                model_config['architecture_params']['dropout'] = dropout_val
            else:
                model_config['architecture_params']['dropout'] = 0.3  # default
        
        # Save model
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'version': version,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'training_config': getattr(model, 'training_config', {}),
            'pytorch_version': torch.__version__,
            'model_size_mb': self._get_model_size(model)
        }
        
        torch.save(checkpoint, model_path)
        
        # Save scaler with version
        scaler_path = self.model_dir / f'scaler_v{version}.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(feature_extractor.scaler, f)
        
        # Update metadata file
        metadata = self._get_metadata()
        metadata['versions'].setdefault(version, {})
        metadata['versions'][version] = {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'model_config': model_config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'model_size_mb': checkpoint['model_size_mb']
        }
        metadata['current_version'] = version
        
        # Export to different formats if requested
        export_paths = {}
        if export_formats:
            for fmt in export_formats:
                if fmt == 'onnx':
                    export_paths['onnx'] = self.export_onnx(model, version, model_name)
                elif fmt == 'torchscript':
                    export_paths['torchscript'] = self.export_torchscript(model, version, model_name)
                elif fmt == 'quantized':
                    export_paths['quantized'] = self.export_quantized(model, version, model_name)
                elif fmt == 'tensorflow':
                    export_paths['tensorflow'] = self.export_tensorflow(model, version, model_name)
        
        metadata['versions'][version]['export_paths'] = export_paths
        
        # Save updated metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model {model_name} v{version} saved successfully with {len(metrics)} metrics")
    
    def _get_metadata(self) -> Dict:
        """Get or initialize metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'versions': {},
            'current_version': None,
            'created_at': datetime.now().isoformat()
        }
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return round(size_mb, 3)
    
    def load_model(
        self, 
        version: Optional[str] = None,
        model_name: Optional[str] = None,
        use_quantized: bool = False
    ) -> Tuple[nn.Module, FeatureExtractor]:
        """
        Load model and feature extractor.
        
        Args:
            version: Specific version to load (None for latest)
            model_name: Specific model name to load (None for current)
            use_quantized: Whether to load quantized version for faster inference
            
        Returns:
            Tuple of (model, feature_extractor)
        """
        metadata = self._get_metadata()
        
        if not metadata['versions']:
            raise FileNotFoundError("No saved models found")
        
        # Find model to load
        if version is None:
            version = metadata['current_version'] or max(metadata['versions'].keys(), 
                                                        key=lambda x: pkg_version.parse(x))
        
        if version not in metadata['versions']:
            raise ValueError(f"Version {version} not found. Available: {list(metadata['versions'].keys())}")
        
        version_info = metadata['versions'][version]
        model_path = Path(version_info['model_path'])
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        config = checkpoint['model_config']
        
        # Determine model class
        model_type = config.get('model_type', 'BehavioralAnalyzer')
        if model_type not in self.architecture_map:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = self.architecture_map[model_type]
        
        # Create model with proper config
        # Check if num_layers is in architecture_params to avoid duplicate arguments
        arch_params = config.get('architecture_params', {})
        kwargs = {
            'input_size': config['input_size'],
            'hidden_size': config.get('hidden_size', 64),
            'num_layers': config.get('num_layers', 3)
        }
        # Update with architecture-specific params, overwriting defaults if present
        kwargs.update(arch_params)
        
        model = model_class(**kwargs)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load scaler
        scaler_path = Path(version_info['scaler_path'])
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        feature_extractor = FeatureExtractor()
        with open(scaler_path, 'rb') as f:
            feature_extractor.scaler = pickle.load(f)
            feature_extractor.fitted = True
        
        logger.info(f"Model {model_type} v{version} loaded successfully ({checkpoint.get('model_size_mb', 'unknown')} MB)")
        return model, feature_extractor
    
    def export_onnx(self, model: nn.Module, version: str, model_name: str = 'model') -> str:
        """
        Export model to ONNX format for cross-platform deployment.
        
        Args:
            model: Model to export
            version: Version string
            model_name: Name of the model
            
        Returns:
            Path to exported ONNX model
        """
        try:
            onnx_path = self.export_dir / f'{model_name}_v{version}.onnx'
            
            # Create dummy input with proper shape
            seq_len = getattr(model, 'seq_len', 20)
            dummy_input = torch.randn(1, seq_len, model.input_size)
            
            # Export with dynamic axes for flexibility
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=13,  # Use newer opset for better compatibility
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size', 1: 'seq_len'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"Model exported to ONNX: {onnx_path}")
            return str(onnx_path)
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
            return ""
    
    def export_torchscript(self, model: nn.Module, version: str, model_name: str = 'model') -> str:
        """
        Export model to TorchScript for optimized inference.
        
        Args:
            model: Model to export
            version: Version string
            model_name: Name of the model
            
        Returns:
            Path to exported TorchScript model
        """
        try:
            script_path = self.export_dir / f'{model_name}_v{version}_script.pt'
            
            # Trace the model with example input
            seq_len = getattr(model, 'seq_len', 20)
            example_input = torch.randn(1, seq_len, model.input_size)
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save(str(script_path))
            
            logger.info(f"Model exported to TorchScript: {script_path}")
            return str(script_path)
        except Exception as e:
            logger.warning(f"TorchScript export failed: {e}")
            return ""
    
    def export_quantized(self, model: nn.Module, version: str, model_name: str = 'model') -> str:
        """
        Export quantized model for faster CPU inference.
        
        Args:
            model: Model to export
            version: Version string
            model_name: Name of the model
            
        Returns:
            Path to quantized model
        """
        try:
            quantized_path = self.export_dir / f'{model_name}_v{version}_quantized.pt'
            
            # Prepare model for quantization
            model_quant = model.cpu()
            model_quant.eval()
            
            # Use static quantization for better performance
            model_quant.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            model_quant_prepared = torch.quantization.prepare(model_quant, inplace=False)
            
            # Calibrate with sample data (in real usage, you'd use actual calibration data)
            # For now, use a dummy forward pass to simulate calibration
            with torch.no_grad():
                dummy_input = torch.randn(10, 20, model.input_size)
                _ = model_quant_prepared(dummy_input)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(model_quant_prepared, inplace=False)
            
            # Save quantized model
            torch.save(quantized_model, quantized_path)
            
            logger.info(f"Model quantized and saved: {quantized_path}")
            return str(quantized_path)
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            try:
                # Fallback: use dynamic quantization
                quantized_path = self.export_dir / f'{model_name}_v{version}_quantized_fallback.pt'
                quantized_model = torch.quantization.quantize_dynamic(
                    model.cpu(),
                    {nn.LSTM, nn.Linear},
                    dtype=torch.qint8
                )
                torch.save(quantized_model, quantized_path)
                logger.info(f"Fallback quantization saved: {quantized_path}")
                return str(quantized_path)
            except Exception as fallback_e:
                logger.error(f"Fallback quantization also failed: {fallback_e}")
                return ""
    
    def export_tensorflow(self, model: nn.Module, version: str, model_name: str = 'model') -> str:
        """
        Export model to TensorFlow SavedModel format.
        
        Args:
            model: Model to export
            version: Version string
            model_name: Name of the model
            
        Returns:
            Path to exported TensorFlow model (empty string if not available)
        """
        # TensorFlow export requires tensorflow and torch2trt which may not be available
        try:
            import tensorflow as tf
            import torch.onnx as onnx
            
            # First export to ONNX, then to TensorFlow
            onnx_path = self.export_onnx(model, version, model_name)
            if not onnx_path:
                return ""
            
            # Convert ONNX to TensorFlow (if onnx_tf is available)
            try:
                import onnx
                import tf2onnx
                
                onnx_model = onnx.load(onnx_path)
                tf_path = self.export_dir / f'{model_name}_v{version}_tf'
                
                # Convert and save
                tf_rep = tf2onnx.convert.from_onnx(onnx_model)
                tf_rep.export_graph(str(tf_path))
                
                logger.info(f"Model exported to TensorFlow: {tf_path}")
                return str(tf_path)
            except ImportError:
                logger.warning("TensorFlow export requires 'tf2onnx'. Install with: pip install tf2onnx")
        except ImportError:
            logger.warning("TensorFlow export requires 'tensorflow'. Install with: pip install tensorflow")
        except Exception as e:
            logger.warning(f"TensorFlow export failed: {e}")
        
        return ""
    
    def validate_model(self, model: nn.Module, test_data: torch.Tensor, expected_output_shape: Tuple) -> Dict[str, Any]:
        """
        Validate model integrity and functionality.
        
        Args:
            model: Model to validate
            test_data: Test input tensor
            expected_output_shape: Expected output shape
            
        Returns:
            Dictionary with validation results
        """
        model.eval()
        
        validation_results = {
            'model_valid': True,
            'output_shape_correct': False,
            'no_nan_values': True,
            'output_range_reasonable': True,
            'inference_successful': True,
            'error_message': None
        }
        
        try:
            with torch.no_grad():
                output = model(test_data)
                
                # Check output shape
                validation_results['output_shape_correct'] = tuple(output.shape) == expected_output_shape
                
                # Check for NaN or Inf values
                validation_results['no_nan_values'] = not (torch.isnan(output).any() or torch.isinf(output).any())
                
                # Check output range (for binary classification, should be 0-1)
                if len(expected_output_shape) > 1 and expected_output_shape[-1] == 1:
                    out_min, out_max = output.min().item(), output.max().item()
                    validation_results['output_range_reasonable'] = (out_min >= -1e-6 and out_max <= 1.001)
                
        except Exception as e:
            validation_results['model_valid'] = False
            validation_results['inference_successful'] = False
            validation_results['error_message'] = str(e)
        
        # Overall validation status
        validation_results['model_valid'] = (
            validation_results['output_shape_correct'] and
            validation_results['no_nan_values'] and
            validation_results['output_range_reasonable'] and
            validation_results['inference_successful']
        )
        
        return validation_results
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models with their metadata.
        
        Returns:
            List of model metadata dictionaries
        """
        metadata = self._get_metadata()
        models = []
        
        for version, info in metadata['versions'].items():
            models.append({
                'version': version,
                'model_name': info.get('model_name', 'unknown'),
                'model_size_mb': info.get('model_size_mb', 0),
                'timestamp': info.get('timestamp', ''),
                'metrics': info.get('metrics', {}),
                'has_onnx': 'onnx' in info.get('export_paths', {}),
                'has_torchscript': 'torchscript' in info.get('export_paths', {}),
                'has_quantized': 'quantized' in info.get('export_paths', {})
            })
        
        return sorted(models, key=lambda x: pkg_version.parse(x['version']), reverse=True)
    
    def delete_model(self, version: str) -> bool:
        """
        Delete a model version and its associated files.
        
        Args:
            version: Version to delete
            
        Returns:
            True if deletion was successful
        """
        metadata = self._get_metadata()
        
        if version not in metadata['versions']:
            logger.warning(f"Version {version} not found for deletion")
            return False
        
        version_info = metadata['versions'][version]
        
        # Remove model files
        files_to_remove = [
            version_info.get('model_path'),
            version_info.get('scaler_path')
        ]
        
        # Remove export files
        for export_path in version_info.get('export_paths', {}).values():
            if export_path:
                files_to_remove.append(export_path)
        
        for file_path in files_to_remove:
            if file_path and Path(file_path).exists():
                try:
                    os.remove(file_path)
                    logger.info(f"Removed file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove {file_path}: {e}")
        
        # Remove from metadata
        del metadata['versions'][version]
        
        # Update current version if necessary
        if metadata['current_version'] == version:
            if metadata['versions']:
                # Set to latest available version
                latest_version = max(metadata['versions'].keys(), key=pkg_version.parse)
                metadata['current_version'] = latest_version
            else:
                metadata['current_version'] = None
        
        # Save updated metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model version {version} deleted successfully")
        return True
    
    def benchmark_model(
        self, 
        model: nn.Module, 
        input_shape: Tuple[int, int, int],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark model performance with warmup.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape (batch, seq_len, features)
            num_iterations: Number of iterations for benchmarking
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with performance metrics
        """
        import time
        
        model.eval()
        device = next(model.parameters()).device
        
        # Warmup
        dummy_input = torch.randn(*input_shape).to(device)
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
        
        # Ensure CUDA operations are completed before timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark loop
        times = []
        for _ in range(num_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        times = np.array(times)
        
        return {
            'avg_latency_ms': float(np.mean(times)),
            'median_latency_ms': float(np.median(times)),
            'min_latency_ms': float(np.min(times)),
            'max_latency_ms': float(np.max(times)),
            'std_latency_ms': float(np.std(times)),
            'throughput_per_sec': float(1000 / np.mean(times)),  # Inferences per second
            'total_time_sec': float(np.sum(times) / 1000),  # Total time in seconds
            'iterations': num_iterations
        }
    
    def get_model_info(self, version: str) -> Dict[str, Any]:
        """
        Get detailed information about a model version.
        
        Args:
            version: Model version to get info for
            
        Returns:
            Dictionary with model information
        """
        metadata = self._get_metadata()
        
        if version not in metadata['versions']:
            raise ValueError(f"Version {version} not found")
        
        version_info = metadata['versions'][version]
        
        # Get model file size
        model_path = Path(version_info['model_path'])
        size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0
        
        return {
            'version': version,
            'model_name': version_info.get('model_name', ''),
            'file_path': str(model_path),
            'file_size_mb': round(size_mb, 3),
            'timestamp': version_info.get('timestamp', ''),
            'metrics': version_info.get('metrics', {}),
            'config': version_info.get('model_config', {}),
            'exports': version_info.get('export_paths', {}),
            'is_current': metadata['current_version'] == version
        }
