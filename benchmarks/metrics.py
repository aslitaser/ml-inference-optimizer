"""
Performance metrics for ML inference optimization.

This module provides functions for calculating various performance metrics
for ML models, including throughput, latency, memory efficiency, and more.
"""

import math
import statistics
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Union


def calculate_throughput(batch_size: int, seq_len: int, time_seconds: float) -> float:
    """
    Calculate model throughput in samples per second.
    
    Args:
        batch_size: Batch size used for inference.
        seq_len: Sequence length used for inference.
        time_seconds: Time taken for inference in seconds.
        
    Returns:
        Throughput in samples per second.
    """
    return batch_size / time_seconds if time_seconds > 0 else 0


def calculate_latency_statistics(latencies: List[float]) -> Dict[str, float]:
    """
    Calculate latency statistics including percentiles.
    
    Args:
        latencies: List of latency measurements in seconds.
        
    Returns:
        Dictionary containing latency statistics.
    """
    if not latencies:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "stddev": 0.0
        }
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    # Calculate basic statistics
    mean = statistics.mean(sorted_latencies)
    median = statistics.median(sorted_latencies)
    min_val = min(sorted_latencies)
    max_val = max(sorted_latencies)
    stddev = statistics.stdev(sorted_latencies) if n > 1 else 0.0
    
    # Calculate percentiles
    p50 = median
    p90 = sorted_latencies[int(n * 0.9)]
    p95 = sorted_latencies[int(n * 0.95)]
    p99 = sorted_latencies[int(n * 0.99)]
    
    return {
        "mean": mean,
        "median": median,
        "min": min_val,
        "max": max_val,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "stddev": stddev
    }


def calculate_memory_efficiency(theoretical_memory: int, actual_memory: int) -> float:
    """
    Calculate memory efficiency as the ratio of theoretical to actual memory usage.
    
    Args:
        theoretical_memory: Theoretical memory usage in MB.
        actual_memory: Actual memory usage in MB.
        
    Returns:
        Memory efficiency as a ratio (higher is better).
    """
    if actual_memory <= 0:
        return 0.0
    
    return theoretical_memory / actual_memory


def calculate_gpu_utilization(profile_data: Dict[str, Any]) -> float:
    """
    Calculate GPU utilization from profiling data.
    
    Args:
        profile_data: Dictionary containing profiling data.
        
    Returns:
        GPU utilization as a percentage.
    """
    if not profile_data or "gpu_utilization" not in profile_data:
        return 0.0
    
    return profile_data["gpu_utilization"]


def calculate_flops_utilization(theoretical_flops: int, runtime: float, batch_size: int) -> float:
    """
    Calculate FLOPS utilization as a percentage of peak theoretical FLOPS.
    
    Args:
        theoretical_flops: Theoretical peak FLOPS of the hardware.
        runtime: Runtime in seconds.
        batch_size: Batch size used for inference.
        
    Returns:
        FLOPS utilization as a percentage.
    """
    if runtime <= 0 or theoretical_flops <= 0:
        return 0.0
    
    # This is a simplified calculation - in reality, you would need the actual
    # FLOPs performed by the model during inference
    actual_flops = batch_size * 1e9  # Placeholder value
    
    return (actual_flops / runtime) / theoretical_flops * 100


def calculate_speedup(baseline_time: float, optimized_time: float) -> float:
    """
    Calculate speedup as the ratio of baseline time to optimized time.
    
    Args:
        baseline_time: Time taken by the baseline model in seconds.
        optimized_time: Time taken by the optimized model in seconds.
        
    Returns:
        Speedup ratio (higher is better).
    """
    if optimized_time <= 0:
        return float('inf')
    
    return baseline_time / optimized_time


def calculate_memory_reduction(baseline_memory: int, optimized_memory: int) -> float:
    """
    Calculate memory reduction as a percentage.
    
    Args:
        baseline_memory: Memory usage of the baseline model in MB.
        optimized_memory: Memory usage of the optimized model in MB.
        
    Returns:
        Memory reduction as a percentage.
    """
    if baseline_memory <= 0:
        return 0.0
    
    reduction = (baseline_memory - optimized_memory) / baseline_memory
    return max(0.0, reduction * 100)


def calculate_scaling_efficiency(single_gpu_time: float, multi_gpu_time: float, num_gpus: int) -> float:
    """
    Calculate scaling efficiency for multi-GPU training/inference.
    
    Args:
        single_gpu_time: Time taken on a single GPU in seconds.
        multi_gpu_time: Time taken on multiple GPUs in seconds.
        num_gpus: Number of GPUs used.
        
    Returns:
        Scaling efficiency as a percentage.
    """
    if multi_gpu_time <= 0 or single_gpu_time <= 0 or num_gpus <= 1:
        return 0.0
    
    ideal_speedup = num_gpus
    actual_speedup = single_gpu_time / multi_gpu_time
    
    return (actual_speedup / ideal_speedup) * 100


def calculate_communication_overhead(total_time: float, computation_time: float) -> float:
    """
    Calculate communication overhead as a percentage of total time.
    
    Args:
        total_time: Total runtime in seconds.
        computation_time: Time spent on computation in seconds.
        
    Returns:
        Communication overhead as a percentage.
    """
    if total_time <= 0:
        return 0.0
    
    communication_time = total_time - computation_time
    return (communication_time / total_time) * 100


def calculate_relative_error(baseline: torch.Tensor, optimized: torch.Tensor) -> float:
    """
    Calculate the relative error between baseline and optimized model outputs.
    
    Args:
        baseline: Tensor containing baseline model outputs.
        optimized: Tensor containing optimized model outputs.
        
    Returns:
        Relative error as a percentage.
    """
    if baseline.numel() == 0 or optimized.numel() == 0:
        return float('inf')
    
    # Ensure tensors have the same shape
    if baseline.shape != optimized.shape:
        raise ValueError(f"Tensor shapes do not match: {baseline.shape} vs {optimized.shape}")
    
    # Calculate element-wise relative error
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    abs_diff = torch.abs(baseline - optimized)
    abs_baseline = torch.abs(baseline) + epsilon
    
    # Calculate mean relative error
    rel_err = (abs_diff / abs_baseline).mean().item() * 100
    
    return rel_err


def calculate_max_absolute_error(baseline: torch.Tensor, optimized: torch.Tensor) -> float:
    """
    Calculate the maximum absolute error between baseline and optimized model outputs.
    
    Args:
        baseline: Tensor containing baseline model outputs.
        optimized: Tensor containing optimized model outputs.
        
    Returns:
        Maximum absolute error.
    """
    if baseline.numel() == 0 or optimized.numel() == 0:
        return float('inf')
    
    # Ensure tensors have the same shape
    if baseline.shape != optimized.shape:
        raise ValueError(f"Tensor shapes do not match: {baseline.shape} vs {optimized.shape}")
    
    # Calculate maximum absolute error
    max_abs_err = torch.max(torch.abs(baseline - optimized)).item()
    
    return max_abs_err


def validate_numerical_stability(results: List[torch.Tensor]) -> bool:
    """
    Validate numerical stability by checking for NaNs and Infs in results.
    
    Args:
        results: List of tensor results from multiple runs.
        
    Returns:
        True if results are numerically stable, False otherwise.
    """
    for tensor in results:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return False
    
    return True


def verify_determinism(model: torch.nn.Module, inputs: Dict[str, torch.Tensor], num_runs: int = 5) -> bool:
    """
    Verify that a model produces deterministic outputs given the same inputs.
    
    Args:
        model: PyTorch model to test.
        inputs: Model inputs.
        num_runs: Number of runs to perform.
        
    Returns:
        True if model produces deterministic outputs, False otherwise.
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(**inputs)
            
            if isinstance(output, torch.Tensor):
                results.append(output.detach().clone())
            elif isinstance(output, tuple) and all(isinstance(o, torch.Tensor) for o in output):
                results.append(output[0].detach().clone())  # Use first tensor for comparison
            elif isinstance(output, dict) and any(isinstance(o, torch.Tensor) for o in output.values()):
                # Use the first tensor found in the dict
                for o in output.values():
                    if isinstance(o, torch.Tensor):
                        results.append(o.detach().clone())
                        break
            else:
                return False  # Can't verify non-tensor outputs
    
    # Check if all results are equal
    reference = results[0]
    for res in results[1:]:
        if not torch.allclose(reference, res, rtol=1e-5, atol=1e-5):
            return False
    
    return True