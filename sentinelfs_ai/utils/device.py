"""
Device utilities for PyTorch operations.
"""

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        PyTorch device (cuda or cpu)
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': str(get_device())
    }
    
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0)
        info['cuda_memory_reserved'] = torch.cuda.memory_reserved(0)
    
    return info
