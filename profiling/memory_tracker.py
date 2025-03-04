"""
Memory tracking utilities for ML models.

This module provides tools for tracking and analyzing GPU memory usage
during model inference and training.
"""

import gc
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class GPUMemoryTracker:
    """Tracks GPU memory usage during model execution."""

    def __init__(self, device: int = 0):
        """
        Initialize the GPU memory tracker.

        Args:
            device: GPU device ID to track
        """
        self.device = device
        self.tracking = False
        self.start_memory = 0
        self.peak_memory = 0
        self.current_memory = 0
        self.memory_trace = []
        
        # Verify CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot track GPU memory.")
        
        # Verify device exists
        if device >= torch.cuda.device_count():
            raise ValueError(f"Device {device} does not exist. Available devices: 0-{torch.cuda.device_count()-1}")
    
    def start_tracking(self) -> None:
        """
        Start tracking GPU memory usage.
        
        Records the current memory usage as a baseline.
        """
        # Force garbage collection before tracking
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        
        # Record starting memory
        self.start_memory = torch.cuda.memory_allocated(self.device)
        self.current_memory = self.start_memory
        self.peak_memory = self.start_memory
        self.memory_trace = [(time.time(), self.start_memory)]
        self.tracking = True
    
    def _update_memory_stats(self) -> None:
        """Update the internal memory statistics."""
        if self.tracking:
            current = torch.cuda.memory_allocated(self.device)
            peak = torch.cuda.max_memory_allocated(self.device)
            
            self.current_memory = current
            self.peak_memory = max(peak, self.peak_memory)
            self.memory_trace.append((time.time(), current))
    
    def stop_tracking(self) -> Dict[str, Any]:
        """
        Stop tracking and return memory usage statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        if not self.tracking:
            return {"error": "Tracking was not started"}
        
        self._update_memory_stats()
        self.tracking = False
        
        # Compute statistics
        memory_stats = {
            "start_memory_bytes": self.start_memory,
            "final_memory_bytes": self.current_memory,
            "peak_memory_bytes": self.peak_memory,
            "memory_increase_bytes": self.current_memory - self.start_memory,
            "peak_increase_bytes": self.peak_memory - self.start_memory,
            "memory_trace": self.memory_trace,
        }
        
        # Add human-readable versions (in MB)
        memory_stats["start_memory_mb"] = self.start_memory / (1024 * 1024)
        memory_stats["final_memory_mb"] = self.current_memory / (1024 * 1024)
        memory_stats["peak_memory_mb"] = self.peak_memory / (1024 * 1024)
        memory_stats["memory_increase_mb"] = memory_stats["memory_increase_bytes"] / (1024 * 1024)
        memory_stats["peak_increase_mb"] = memory_stats["peak_increase_bytes"] / (1024 * 1024)
        
        return memory_stats
    
    def track_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Track memory usage while executing a function.
        
        Args:
            func: Function to track
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Tuple of (function result, memory statistics)
        """
        self.start_tracking()
        
        try:
            result = func(*args, **kwargs)
        finally:
            memory_stats = self.stop_tracking()
        
        return result, memory_stats
    
    def track_peak_memory(self, func: Callable, *args, **kwargs) -> Tuple[Any, int]:
        """
        Track peak memory usage while executing a function.
        
        Args:
            func: Function to track
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Tuple of (function result, peak memory in bytes)
        """
        result, stats = self.track_function(func, *args, **kwargs)
        return result, stats["peak_memory_bytes"]


def analyze_memory_by_layer(model: nn.Module, inputs: Any) -> Dict[str, int]:
    """
    Analyze memory usage for each layer in a model.
    
    Args:
        model: PyTorch model to analyze
        inputs: Input data for the model
        
    Returns:
        Dictionary mapping layer names to memory usage in bytes
    """
    result = {}
    hooks = []
    
    # Verify CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot track GPU memory.")
    
    # Ensure model is on CUDA
    device = next(model.parameters()).device
    if not device.type == 'cuda':
        raise ValueError("Model must be on a CUDA device for memory analysis")
    
    # Prepare inputs if needed
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    
    # Create hooks for all modules
    def get_memory_hook(name):
        def hook(module, inp, output):
            # Reset peak stats to isolate this layer's contribution
            torch.cuda.reset_peak_memory_stats(device.index)
            # Get initial memory
            memory_before = torch.cuda.memory_allocated(device.index)
            
            # Dummy forward pass to measure this layer's peak memory
            with torch.no_grad():
                if isinstance(inp, tuple):
                    module(*inp)
                else:
                    module(inp)
            
            # Record peak memory
            peak_memory = torch.cuda.max_memory_allocated(device.index)
            memory_usage = peak_memory - memory_before
            result[name] = memory_usage
            
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if list(module.children()):  # Skip container modules
            continue
        if name == '':  # Skip the root module
            continue
        
        hook = module.register_forward_hook(get_memory_hook(name))
        hooks.append(hook)
    
    # Forward pass to activate hooks
    with torch.no_grad():
        model(*inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return result


def detect_memory_leaks(model: nn.Module, inputs: Any, iterations: int = 100) -> bool:
    """
    Detect potential memory leaks by running multiple iterations.
    
    Args:
        model: PyTorch model to test
        inputs: Input data for the model
        iterations: Number of iterations to run
        
    Returns:
        True if a memory leak is detected, False otherwise
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot track GPU memory.")
    
    # Ensure model is on CUDA
    device = next(model.parameters()).device
    if not device.type == 'cuda':
        raise ValueError("Model must be on a CUDA device for memory leak detection")
    
    # Convert inputs to a tuple if needed
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    
    # Clean up memory before starting
    gc.collect()
    torch.cuda.empty_cache()
    
    # First pass to initialize everything
    with torch.no_grad():
        model(*inputs)
    
    # Baseline memory
    gc.collect()
    torch.cuda.empty_cache()
    memory_start = torch.cuda.memory_allocated(device.index)
    
    # Run model repeatedly
    for _ in range(iterations):
        with torch.no_grad():
            model(*inputs)
    
    # Final memory check
    gc.collect()
    torch.cuda.empty_cache()
    memory_end = torch.cuda.memory_allocated(device.index)
    
    # Determine if there's a leak
    memory_diff = memory_end - memory_start
    memory_percent_increase = (memory_diff / memory_start) * 100 if memory_start > 0 else 0
    
    # Consider it a leak if memory increased significantly over iterations
    # This threshold can be adjusted based on your needs
    return memory_percent_increase > 5  # 5% increase suggests a leak


def estimate_max_batch_size(model: nn.Module, sample_input: Any, max_memory: int) -> int:
    """
    Estimate the maximum batch size that can fit in the specified memory.
    
    Args:
        model: PyTorch model to analyze
        sample_input: Sample input with batch size 1
        max_memory: Maximum memory to use (in bytes)
        
    Returns:
        Estimated maximum batch size
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot estimate batch size.")
    
    # Ensure model is on CUDA
    device = next(model.parameters()).device
    if not device.type == 'cuda':
        raise ValueError("Model must be on a CUDA device for batch size estimation")
    
    # Function to create a batch of the specified size
    def create_batch(batch_size):
        if isinstance(sample_input, torch.Tensor):
            # For a single tensor input
            batch_shape = list(sample_input.shape)
            batch_shape[0] = batch_size
            return torch.rand(batch_shape, device=device, dtype=sample_input.dtype)
        elif isinstance(sample_input, tuple):
            # For tuple of inputs
            return tuple(
                torch.rand(
                    [batch_size] + list(x.shape)[1:],
                    device=device,
                    dtype=x.dtype
                ) if isinstance(x, torch.Tensor) else x
                for x in sample_input
            )
        else:
            raise ValueError("Unsupported input type. Provide a tensor or tuple of tensors.")
    
    # Create a function to measure memory for a given batch size
    def measure_memory(batch_size):
        batch = create_batch(batch_size)
        
        # Clean memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device.index)
        
        # Run the model
        with torch.no_grad():
            if isinstance(batch, tuple):
                model(*batch)
            else:
                model(batch)
                
        # Return peak memory
        return torch.cuda.max_memory_allocated(device.index)
    
    # Binary search for the maximum batch size
    left, right = 1, 1024  # Start with a reasonable range
    
    # First, find an upper bound that exceeds memory limit
    while measure_memory(right) < max_memory:
        right *= 2
        if right > 65536:  # Avoid going to extreme values
            break
    
    # Binary search for the largest batch size that fits
    best_batch_size = 1
    while left <= right:
        mid = (left + right) // 2
        memory_used = measure_memory(mid)
        
        if memory_used <= max_memory:
            best_batch_size = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return best_batch_size