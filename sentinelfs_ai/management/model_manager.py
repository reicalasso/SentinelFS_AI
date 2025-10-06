"""
Model lifecycle management: saving, loading, versioning, and optimization.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pickle
from datetime import datetime

from ..models.behavioral_analyzer import BehavioralAnalyzer
from ..data.feature_extractor import FeatureExtractor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """
    Manage model lifecycle: training, saving, loading, versioning, and optimization.
    
    Features:
    - Model versioning
    - Checkpoint management
    - Model export (ONNX, TorchScript)
    - Model quantization
    - Performance profiling
    """
    
    def __init__(self, model_dir: Path = Path('./models')):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.model_dir / 'metadata.json'
        self.scaler_file = self.model_dir / 'scaler.pkl'
        self.export_dir = self.model_dir / 'exports'
        self.export_dir.mkdir(exist_ok=True)
    
    def save_model(
        self, 
        model: nn.Module, 
        version: str, 
        metrics: Dict[str, float], 
        feature_extractor: FeatureExtractor,
        export_formats: Optional[List[str]] = None
    ):
        """
        Save model with versioning and metadata.
        
        Args:
            model: Trained model
            version: Version string (e.g., '1.0.0')
            metrics: Performance metrics
            feature_extractor: Fitted feature extractor with scaler
            export_formats: Optional list of export formats ['onnx', 'torchscript', 'quantized']
        """
        model_path = self.model_dir / f'model_v{version}.pt'
        
        # Get model config
        model_config = {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers
        }
        
        # Add optional config
        if hasattr(model, 'use_attention'):
            model_config['use_attention'] = model.use_attention
        if hasattr(model, 'bidirectional'):
            model_config['bidirectional'] = model.bidirectional
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }, model_path)
        
        # Save scaler
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(feature_extractor.scaler, f)
        
        # Export to different formats if requested
        export_paths = {}
        if export_formats:
            for fmt in export_formats:
                if fmt == 'onnx':
                    export_paths['onnx'] = self.export_onnx(model, version)
                elif fmt == 'torchscript':
                    export_paths['torchscript'] = self.export_torchscript(model, version)
                elif fmt == 'quantized':
                    export_paths['quantized'] = self.export_quantized(model, version)
        
        # Update metadata
        metadata = {
            'current_version': version,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'model_path': str(model_path),
            'scaler_path': str(self.scaler_file),
            'model_config': model_config,
            'export_paths': export_paths
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model v{version} saved successfully")
    
    def load_model(
        self, 
        version: Optional[str] = None, 
        use_quantized: bool = False
    ) -> Tuple[nn.Module, FeatureExtractor]:
        """
        Load model and feature extractor.
        
        Args:
            version: Specific version to load (None for latest)
            use_quantized: Whether to load quantized version for faster inference
            
        Returns:
            Tuple of (model, feature_extractor)
        """
        # Load metadata
        if not self.metadata_file.exists():
            raise FileNotFoundError("No saved models found")
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if version is None:
            version = metadata['current_version']
        
        # Load quantized model if requested
        if use_quantized and 'export_paths' in metadata and 'quantized' in metadata['export_paths']:
            quantized_path = Path(metadata['export_paths']['quantized'])
            if quantized_path.exists():
                model = torch.load(quantized_path, map_location='cpu')
                logger.info(f"Loaded quantized model v{version}")
            else:
                logger.warning("Quantized model not found, loading normal model")
                use_quantized = False
        
        if not use_quantized:
            model_path = self.model_dir / f'model_v{version}.pt'
            
            # Load model
            checkpoint = torch.load(model_path, map_location='cpu')
            config = checkpoint['model_config']
            
            # Create model with proper config
            model = BehavioralAnalyzer(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                use_attention=config.get('use_attention', True),
                bidirectional=config.get('bidirectional', True)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        
        # Load scaler
        feature_extractor = FeatureExtractor()
        with open(self.scaler_file, 'rb') as f:
            feature_extractor.scaler = pickle.load(f)
            feature_extractor.fitted = True
        
        logger.info(f"Model v{version} loaded successfully")
        return model, feature_extractor
    
    def export_onnx(self, model: nn.Module, version: str) -> str:
        """
        Export model to ONNX format for cross-platform deployment.
        
        Args:
            model: Model to export
            version: Version string
            
        Returns:
            Path to exported ONNX model
        """
        try:
            onnx_path = self.export_dir / f'model_v{version}.onnx'
            
            # Create dummy input
            dummy_input = torch.randn(1, 10, model.input_size)
            
            # Export
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
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
    
    def export_torchscript(self, model: nn.Module, version: str) -> str:
        """
        Export model to TorchScript for optimized inference.
        
        Args:
            model: Model to export
            version: Version string
            
        Returns:
            Path to exported TorchScript model
        """
        try:
            script_path = self.export_dir / f'model_v{version}_script.pt'
            
            # Create scripted model
            scripted_model = torch.jit.script(model)
            scripted_model.save(str(script_path))
            
            logger.info(f"Model exported to TorchScript: {script_path}")
            return str(script_path)
        except Exception as e:
            logger.warning(f"TorchScript export failed: {e}")
            return ""
    
    def export_quantized(self, model: nn.Module, version: str) -> str:
        """
        Export quantized model for faster CPU inference.
        
        Args:
            model: Model to export
            version: Version string
            
        Returns:
            Path to quantized model
        """
        try:
            quantized_path = self.export_dir / f'model_v{version}_quantized.pt'
            
            # Prepare model for quantization
            model.eval()
            model.cpu()
            
            # Dynamic quantization (works on LSTM layers)
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.LSTM, nn.Linear},
                dtype=torch.qint8
            )
            
            # Save quantized model
            torch.save(quantized_model, quantized_path)
            
            logger.info(f"Model quantized and saved: {quantized_path}")
            return str(quantized_path)
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return ""
    
    def benchmark_model(
        self, 
        model: nn.Module, 
        input_shape: Tuple[int, int, int],
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape (batch, seq_len, features)
            num_iterations: Number of iterations for benchmarking
            
        Returns:
            Dictionary with performance metrics
        """
        import time
        
        model.eval()
        device = next(model.parameters()).device
        
        # Warmup
        dummy_input = torch.randn(*input_shape).to(device)
        for _ in range(10):
            _ = model(dummy_input)
        
        # Benchmark
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = 1.0 / avg_time
        
        return {
            'avg_latency_ms': avg_time * 1000,
            'throughput_per_sec': throughput,
            'total_time_sec': total_time,
            'iterations': num_iterations
        }
