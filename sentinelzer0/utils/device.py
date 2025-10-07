"""
Device utilities for PyTorch operations.
"""

import torch
from typing import Optional
import os


def get_device(prefer_cuda: bool = True, gpu_index: int = 0) -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        gpu_index: Index of GPU to use (if multiple GPUs available)
        
    Returns:
        PyTorch device (cuda or cpu)
    """
    if prefer_cuda and torch.cuda.is_available():
        if gpu_index < torch.cuda.device_count():
            return torch.device(f'cuda:{gpu_index}')
        else:
            # Use first available GPU if requested index is not available
            return torch.device('cuda:0')
    return torch.device('cpu')


def get_device_info(gpu_index: int = 0) -> dict:
    """
    Get information about available devices.
    
    Args:
        gpu_index: Index of GPU to get info for
        
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': str(get_device()),
        'available_devices': []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_info = {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'major': torch.cuda.get_device_capability(i)[0],
                'minor': torch.cuda.get_device_capability(i)[1],
                'total_memory_GB': torch.cuda.get_device_properties(i).total_memory / 1024**3,
                'is_available': i == gpu_index
            }
            info['available_devices'].append(device_info)
        
        # Specific device info for requested index
        if gpu_index < torch.cuda.device_count():
            info['cuda_device_name'] = torch.cuda.get_device_name(gpu_index)
            info['cuda_memory_allocated'] = torch.cuda.memory_allocated(gpu_index)
            info['cuda_memory_reserved'] = torch.cuda.memory_reserved(gpu_index)
            info['gpu_index'] = gpu_index
        else:
            # Fallback to first GPU
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
            info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0)
            info['cuda_memory_reserved'] = torch.cuda.memory_reserved(0)
            info['gpu_index'] = 0
    
    return info


def set_memory_efficient(device: torch.device, memory_fraction: float = 0.8):
    """
    Set memory fraction to prevent out-of-memory errors.
    
    Args:
        device: Device to configure
        memory_fraction: Fraction of memory to use (0.0 to 1.0)
    """
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device.index)
        # Enable memory caching optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


def clear_device_cache(device: torch.device):
    """
    Clear device cache to free up memory.
    
    Args:
        device: Device to clear cache for
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_optimal_batch_size(
    model: torch.nn.Module,
    input_shape: tuple,
    device: torch.device,
    max_memory_gb: float = 4.0
) -> int:
    """
    Estimate optimal batch size based on model and device memory.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (without batch dimension)
        device: Device to run on
        max_memory_gb: Maximum memory to use in GB
        
    Returns:
        Optimal batch size
    """
    if device.type != 'cuda':
        return 32  # Default for CPU
    
    # Get available memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    available_memory = total_memory * 0.8  # Use 80% of total memory
    available_memory_gb = available_memory / 1024**3
    
    # Calculate memory for a single sample (rough estimate)
    try:
        # Use a small batch to estimate memory usage
        sample_input = torch.randn(1, *input_shape).to(device)
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats()
            
            # Forward pass
            with torch.no_grad():
                _ = model(sample_input)
            
            # Estimate memory usage
            memory_per_sample = torch.cuda.max_memory_allocated() / (1024**3)  # in GB
            torch.cuda.reset_peak_memory_stats()
        
        # Calculate optimal batch size
        optimal_batch_size = int((max_memory_gb / memory_per_sample) * 0.8)  # Use 80% of estimated capacity
        optimal_batch_size = max(1, optimal_batch_size)  # Ensure at least batch size 1
        
        return min(optimal_batch_size, 128)  # Cap at 128 for practicality
    except:
        # Fallback if estimation fails
        return 32
