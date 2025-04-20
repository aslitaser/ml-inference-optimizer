"""
GPU utility functions for profiling and optimization.
"""

import torch
from typing import Dict, List, Optional


def get_gpu_memory_usage(device_ids: Optional[List[int]] = None) -> Dict[int, Dict[str, int]]:
    """
    Get the memory usage of specified GPU devices.
    
    Args:
        device_ids: List of device IDs to check. If None, check all available devices.
        
    Returns:
        Dictionary mapping device ID to memory usage statistics in bytes
    """
    memory_stats = {}
    
    # If no devices specified, check all available ones
    if device_ids is None:
        if torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
        else:
            return {}  # No CUDA devices available
    
    # Collect memory stats for each device
    for device_id in device_ids:
        if not torch.cuda.is_available():
            memory_stats[device_id] = {
                'allocated': 0,
                'reserved': 0,
                'free': 0,
                'total': 0
            }
            continue
            
        try:
            # Get memory usage from PyTorch
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            
            # Get total memory from CUDA
            torch.cuda.set_device(device_id)
            device_props = torch.cuda.get_device_properties(device_id)
            total = device_props.total_memory
            free = total - reserved
            
            memory_stats[device_id] = {
                'allocated': allocated,
                'reserved': reserved,
                'free': free,
                'total': total
            }
        except RuntimeError:
            # Device might not exist or be available
            memory_stats[device_id] = {
                'allocated': 0,
                'reserved': 0,
                'free': 0,
                'total': 0,
                'error': 'Device not available'
            }
    
    return memory_stats


def clear_gpu_memory() -> None:
    """
    Clear unused memory cached by PyTorch on all available CUDA devices.
    """
    if torch.cuda.is_available():
        # Loop through all available devices
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
            
        # Force a garbage collection to release any dangling references
        import gc
        gc.collect()


def calculate_memory_needed(
    batch_size: int, 
    seq_length: int, 
    hidden_size: int, 
    dtype_size: int = 4,
    layers: int = 1,
    activation_factor: float = 2.0
) -> int:
    """
    Approximate the memory needed for a transformer model forward pass.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        hidden_size: Hidden dimension size
        dtype_size: Size of data type in bytes (4 for float32, 2 for float16/bfloat16)
        layers: Number of transformer layers
        activation_factor: Multiplier for memory used by activations
        
    Returns:
        Approximate memory usage in bytes
    """
    # Base memory for input activations
    base_memory = batch_size * seq_length * hidden_size * dtype_size
    
    # Memory for one transformer layer (attention + MLP)
    # Attention: 4 * base (Q, K, V, O) + attention matrix
    attention_memory = 4 * base_memory + batch_size * seq_length * seq_length * dtype_size
    
    # MLP: typically 4 * hidden_size for intermediate size
    mlp_memory = 2 * base_memory * 4  # Up and down projections with 4x intermediate size
    
    # Total for all layers, including activation memory
    total_memory = (attention_memory + mlp_memory) * layers * activation_factor
    
    return int(total_memory)


def gpu_info_string() -> str:
    """
    Get a formatted string with GPU information.
    
    Returns:
        Formatted string with GPU details
    """
    if not torch.cuda.is_available():
        return "No CUDA devices available"
    
    info_lines = ["GPU Information:"]
    
    # Get number of devices
    device_count = torch.cuda.device_count()
    info_lines.append(f"  Devices available: {device_count}")
    
    # Get memory info for each device
    memory_stats = get_gpu_memory_usage()
    
    for device_id in range(device_count):
        # Get device properties
        props = torch.cuda.get_device_properties(device_id)
        device_name = props.name
        total_memory_gb = props.total_memory / (1024**3)
        
        # Get memory usage
        if device_id in memory_stats:
            stats = memory_stats[device_id]
            allocated_gb = stats['allocated'] / (1024**3)
            reserved_gb = stats['reserved'] / (1024**3)
            free_gb = stats['free'] / (1024**3)
            
            info_lines.append(f"  Device {device_id}: {device_name}")
            info_lines.append(f"    Total memory: {total_memory_gb:.2f} GB")
            info_lines.append(f"    Allocated: {allocated_gb:.2f} GB")
            info_lines.append(f"    Reserved: {reserved_gb:.2f} GB")
            info_lines.append(f"    Free: {free_gb:.2f} GB")
        else:
            info_lines.append(f"  Device {device_id}: {device_name} (Memory information unavailable)")
    
    return "\n".join(info_lines)


def is_enough_gpu_memory(
    batch_size: int, 
    seq_length: int, 
    hidden_size: int,
    dtype_size: int = 2,  # Default to fp16
    required_free_memory_gb: float = 2.0,
    device_id: int = 0
) -> bool:
    """
    Check if there's enough GPU memory for a given model configuration.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        hidden_size: Hidden dimension size
        dtype_size: Size of data type in bytes (4 for float32, 2 for float16/bfloat16)
        required_free_memory_gb: Required free memory in GB for operation safety margin
        device_id: Device ID to check
        
    Returns:
        True if there's likely enough memory, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    # Calculate approximate memory needed
    memory_needed = calculate_memory_needed(batch_size, seq_length, hidden_size, dtype_size)
    memory_needed_gb = memory_needed / (1024**3)
    
    # Get available free memory
    memory_stats = get_gpu_memory_usage([device_id])
    if device_id not in memory_stats:
        return False
        
    free_memory_gb = memory_stats[device_id]['free'] / (1024**3)
    
    # Check if there's enough memory with safety margin
    return free_memory_gb >= (memory_needed_gb + required_free_memory_gb)