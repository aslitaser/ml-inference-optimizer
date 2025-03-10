"""
Layer Normalization Triton Kernels.

This module implements optimized Triton kernels for Layer Normalization operations.
These kernels provide highly efficient normalization by fusing multiple operations
and utilizing shared memory for better performance.

Key features:
- Fused mean and variance calculation
- Efficient parallel reduction using shared memory
- Combined normalization, scaling, and bias addition
- Support for residual connections
"""

import torch
import math
import time
from typing import Dict, Optional, Tuple, Union

# Try to import Triton, but gracefully handle if not available
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Triton not available, using PyTorch fallback for LayerNorm kernels")


#-----------------------------------------------------------------------------
# Core LayerNorm Triton Kernels
#-----------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    def _layernorm_fwd_kernel(
        x_ptr, scale_ptr, bias_ptr, out_ptr,
        n_rows, n_cols,
        x_row_stride, x_col_stride,
        scale_stride, bias_stride,
        out_row_stride, out_col_stride,
        eps,
        BLOCK_SIZE: tl.constexpr
    ):
        """
        Compute Layer Normalization in a single fused kernel.
        
        This kernel efficiently computes the Layer Normalization operation by:
        1. Computing mean and variance using parallel reduction
        2. Normalizing the input using the computed statistics
        3. Applying scale and bias in the same kernel
        
        Args:
            x_ptr: Pointer to input tensor [n_rows, n_cols]
            scale_ptr: Pointer to scale tensor [n_cols]
            bias_ptr: Pointer to bias tensor [n_cols]
            out_ptr: Pointer to output tensor [n_rows, n_cols]
            n_rows, n_cols: Dimensions of the input tensor
            Various strides for efficient memory access
            eps: Small constant for numerical stability
            BLOCK_SIZE: Block size for tiling
        """
        # Program ID - the row index
        row_idx = tl.program_id(0)
        
        # Offsets for the current row
        x_row_off = row_idx * x_row_stride
        out_row_off = row_idx * out_row_stride
        
        # Column indices for the current block
        col_indices = tl.arange(0, BLOCK_SIZE)
        # Create a mask for valid columns
        mask = col_indices < n_cols
        
        # Load data for the current row
        x_row_offs = x_row_off + col_indices * x_col_stride
        x_row = tl.load(x_ptr + x_row_offs, mask=mask, other=0.0)
        
        # Step 1: Compute mean using parallel reduction
        # Sum the elements and divide by n_cols
        row_sum = tl.sum(x_row, axis=0)
        row_mean = row_sum / n_cols
        
        # Step 2: Compute variance
        # Calculate squared differences from mean
        x_diff = x_row - row_mean
        x_diff_sq = x_diff * x_diff
        # Sum squared differences and divide by n_cols
        row_var_sum = tl.sum(x_diff_sq, axis=0)
        row_var = row_var_sum / n_cols
        
        # Step 3: Normalize using mean and variance
        # Add epsilon for numerical stability
        row_inv_std = 1.0 / tl.sqrt(row_var + eps)
        
        # Apply normalization
        x_norm = x_diff * row_inv_std
        
        # Step 4: Apply scale and bias
        # Load scale and bias
        scale = tl.load(scale_ptr + col_indices * scale_stride, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + col_indices * bias_stride, mask=mask, other=0.0)
        
        # Apply scale and bias
        out_row = x_norm * scale + bias
        
        # Store the result
        out_row_offs = out_row_off + col_indices * out_col_stride
        tl.store(out_ptr + out_row_offs, out_row, mask=mask)
        
    @triton.jit
    def _layernorm_residual_fwd_kernel(
        x_ptr, residual_ptr, scale_ptr, bias_ptr, out_ptr,
        n_rows, n_cols,
        x_row_stride, x_col_stride,
        residual_row_stride, residual_col_stride,
        scale_stride, bias_stride,
        out_row_stride, out_col_stride,
        eps, alpha,
        BLOCK_SIZE: tl.constexpr
    ):
        """
        Compute Layer Normalization with residual connection in a single fused kernel.
        
        This kernel efficiently computes: 
        out = LayerNorm(x + alpha * residual)
        
        Args:
            x_ptr: Pointer to input tensor [n_rows, n_cols]
            residual_ptr: Pointer to residual tensor [n_rows, n_cols]
            scale_ptr: Pointer to scale tensor [n_cols]
            bias_ptr: Pointer to bias tensor [n_cols]
            out_ptr: Pointer to output tensor [n_rows, n_cols]
            n_rows, n_cols: Dimensions of the input tensor
            Various strides for efficient memory access
            eps: Small constant for numerical stability
            alpha: Scaling factor for residual connection
            BLOCK_SIZE: Block size for tiling
        """
        # Program ID - the row index
        row_idx = tl.program_id(0)
        
        # Offsets for the current row
        x_row_off = row_idx * x_row_stride
        residual_row_off = row_idx * residual_row_stride
        out_row_off = row_idx * out_row_stride
        
        # Column indices for the current block
        col_indices = tl.arange(0, BLOCK_SIZE)
        # Create a mask for valid columns
        mask = col_indices < n_cols
        
        # Load input and residual data for the current row
        x_row_offs = x_row_off + col_indices * x_col_stride
        x_row = tl.load(x_ptr + x_row_offs, mask=mask, other=0.0)
        
        residual_row_offs = residual_row_off + col_indices * residual_col_stride
        residual_row = tl.load(residual_ptr + residual_row_offs, mask=mask, other=0.0)
        
        # Add residual connection with scaling factor
        combined_row = x_row + alpha * residual_row
        
        # Step 1: Compute mean using parallel reduction
        row_sum = tl.sum(combined_row, axis=0)
        row_mean = row_sum / n_cols
        
        # Step 2: Compute variance
        x_diff = combined_row - row_mean
        x_diff_sq = x_diff * x_diff
        row_var_sum = tl.sum(x_diff_sq, axis=0)
        row_var = row_var_sum / n_cols
        
        # Step 3: Normalize using mean and variance
        row_inv_std = 1.0 / tl.sqrt(row_var + eps)
        x_norm = x_diff * row_inv_std
        
        # Step 4: Apply scale and bias
        scale = tl.load(scale_ptr + col_indices * scale_stride, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + col_indices * bias_stride, mask=mask, other=0.0)
        out_row = x_norm * scale + bias
        
        # Store the result
        out_row_offs = out_row_off + col_indices * out_col_stride
        tl.store(out_ptr + out_row_offs, out_row, mask=mask)


#-----------------------------------------------------------------------------
# Python Wrapper Functions
#-----------------------------------------------------------------------------

def triton_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    residual: Optional[torch.Tensor] = None,
    residual_alpha: float = 1.0
) -> torch.Tensor:
    """
    Compute Layer Normalization using optimized Triton kernels.
    
    Args:
        x: Input tensor of shape [batch_size, sequence_length, hidden_size]
                            or [batch_size, hidden_size]
        weight: Scale tensor of shape [hidden_size]
        bias: Bias tensor of shape [hidden_size]
        eps: Small constant for numerical stability
        residual: Optional residual tensor to add before normalization
        residual_alpha: Scaling factor for residual connection
        
    Returns:
        Normalized tensor of the same shape as the input
    """
    # Fall back to PyTorch if Triton is not available
    if not HAS_TRITON:
        return pytorch_layernorm(
            x, weight, bias, eps, residual, residual_alpha
        )
    
    # Get original shape
    orig_shape = x.shape
    
    # Reshape to 2D if necessary - LayerNorm usually works on the last dimension
    if len(orig_shape) > 2:
        x = x.reshape(-1, orig_shape[-1])
    
    # Extract dimensions
    n_rows, n_cols = x.shape
    
    # Create empty output tensor
    output = torch.empty_like(x)
    
    # Use zero bias if not provided
    if bias is None:
        bias = torch.zeros_like(weight)
    
    # Determine block size based on the hidden dimension
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4096)
    
    # Compute grid dimensions
    grid = (n_rows,)
    
    # If a residual connection is provided, use the residual kernel
    if residual is not None:
        # Reshape residual if necessary
        if len(residual.shape) > 2:
            residual = residual.reshape(-1, orig_shape[-1])
        
        # Launch the kernel with residual connection
        _layernorm_residual_fwd_kernel[grid](
            x, residual, weight, bias, output,
            n_rows, n_cols,
            x.stride(0), x.stride(1),
            residual.stride(0), residual.stride(1),
            weight.stride(0), bias.stride(0),
            output.stride(0), output.stride(1),
            eps, residual_alpha,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # Launch the standard kernel
        _layernorm_fwd_kernel[grid](
            x, weight, bias, output,
            n_rows, n_cols,
            x.stride(0), x.stride(1),
            weight.stride(0), bias.stride(0),
            output.stride(0), output.stride(1),
            eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    # Reshape the output back to the original shape if necessary
    if len(orig_shape) > 2:
        output = output.reshape(orig_shape)
    
    return output


def pytorch_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    residual: Optional[torch.Tensor] = None,
    residual_alpha: float = 1.0
) -> torch.Tensor:
    """
    PyTorch implementation of Layer Normalization (used when Triton is not available).
    
    Args:
        x: Input tensor
        weight: Scale tensor
        bias: Bias tensor
        eps: Small constant for numerical stability
        residual: Optional residual tensor to add before normalization
        residual_alpha: Scaling factor for residual connection
        
    Returns:
        Normalized tensor
    """
    # Add residual connection if provided
    if residual is not None:
        x = x + residual_alpha * residual
    
    # Compute mean and variance along the last dimension
    u = x.mean(dim=-1, keepdim=True)
    s = (x - u).pow(2).mean(dim=-1, keepdim=True)
    x = (x - u) / torch.sqrt(s + eps)
    
    # Apply scale and bias
    return weight * x + (bias if bias is not None else 0.0)


#-----------------------------------------------------------------------------
# Benchmarking Functions
#-----------------------------------------------------------------------------

def benchmark_layernorm(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    device: str = "cuda",
    iterations: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Benchmark Layer Normalization performance.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden size
        device: Device to run benchmark on
        iterations: Number of iterations for benchmarking
        warmup: Number of warmup iterations
        
    Returns:
        Dictionary with benchmark results
    """
    # Skip benchmark if CUDA is not available and device is cuda
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "triton_layernorm_ms": 0.0,
            "pytorch_layernorm_ms": 0.0,
            "speedup": 0.0
        }
    
    # Create test tensors
    x = torch.randn((batch_size, seq_len, hidden_size), device=device)
    weight = torch.randn((hidden_size,), device=device)
    bias = torch.randn((hidden_size,), device=device)
    residual = torch.randn((batch_size, seq_len, hidden_size), device=device)
    
    # Warm up
    for _ in range(warmup):
        if HAS_TRITON:
            _ = triton_layernorm(x, weight, bias)
            _ = triton_layernorm(x, weight, bias, residual=residual)
        _ = pytorch_layernorm(x, weight, bias)
        _ = pytorch_layernorm(x, weight, bias, residual=residual)
    
    # Benchmark PyTorch implementation
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    for _ in range(iterations):
        _ = pytorch_layernorm(x, weight, bias)
    
    torch.cuda.synchronize() if device == "cuda" else None
    pytorch_time = (time.time() - start_time) * 1000 / iterations  # ms
    
    # Benchmark PyTorch implementation with residual
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    for _ in range(iterations):
        _ = pytorch_layernorm(x, weight, bias, residual=residual)
    
    torch.cuda.synchronize() if device == "cuda" else None
    pytorch_residual_time = (time.time() - start_time) * 1000 / iterations  # ms
    
    # Benchmark Triton implementation
    if HAS_TRITON:
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()
        
        for _ in range(iterations):
            _ = triton_layernorm(x, weight, bias)
        
        torch.cuda.synchronize() if device == "cuda" else None
        triton_time = (time.time() - start_time) * 1000 / iterations  # ms
        
        # Benchmark Triton implementation with residual
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()
        
        for _ in range(iterations):
            _ = triton_layernorm(x, weight, bias, residual=residual)
        
        torch.cuda.synchronize() if device == "cuda" else None
        triton_residual_time = (time.time() - start_time) * 1000 / iterations  # ms
        
        speedup = pytorch_time / triton_time
        speedup_residual = pytorch_residual_time / triton_residual_time
    else:
        triton_time = pytorch_time
        triton_residual_time = pytorch_residual_time
        speedup = 1.0
        speedup_residual = 1.0
    
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "triton_layernorm_ms": triton_time,
        "pytorch_layernorm_ms": pytorch_time,
        "triton_layernorm_residual_ms": triton_residual_time,
        "pytorch_layernorm_residual_ms": pytorch_residual_time,
        "speedup": speedup,
        "speedup_residual": speedup_residual
    }


def compare_with_torch_layernorm(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compare the results of Triton LayerNorm with PyTorch nn.LayerNorm.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden size
        device: Device to run comparison on
        
    Returns:
        Dictionary with comparison results
    """
    # Skip if CUDA is not available and device is cuda
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping comparison")
        return {
            "max_difference": 0.0,
            "is_correct": False
        }
    
    # Create test tensors
    x = torch.randn((batch_size, seq_len, hidden_size), device=device)
    weight = torch.randn((hidden_size,), device=device)
    bias = torch.randn((hidden_size,), device=device)
    residual = torch.randn((batch_size, seq_len, hidden_size), device=device)
    
    # Create PyTorch LayerNorm module
    torch_ln = torch.nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)
    torch_ln.weight.data.copy_(weight)
    torch_ln.bias.data.copy_(bias)
    torch_ln = torch_ln.to(device)
    
    # Compute with PyTorch LayerNorm
    torch_output = torch_ln(x)
    
    # Compute with residual connection manually
    x_residual = x + residual
    torch_residual_output = torch_ln(x_residual)
    
    # Compute with Triton LayerNorm
    if HAS_TRITON:
        triton_output = triton_layernorm(x, weight, bias)
        triton_residual_output = triton_layernorm(x, weight, bias, residual=residual)
    else:
        # Fall back to our PyTorch implementation
        triton_output = pytorch_layernorm(x, weight, bias)
        triton_residual_output = pytorch_layernorm(x, weight, bias, residual=residual)
    
    # Compare outputs
    max_diff = torch.max(torch.abs(torch_output - triton_output)).item()
    max_residual_diff = torch.max(torch.abs(torch_residual_output - triton_residual_output)).item()
    
    # Determine if results are correct (within tolerance)
    is_correct = max_diff < 1e-3
    is_residual_correct = max_residual_diff < 1e-3
    
    return {
        "max_difference": max_diff,
        "max_residual_difference": max_residual_diff,
        "is_correct": is_correct,
        "is_residual_correct": is_residual_correct,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size
    }


def profile_memory_usage(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Profile memory usage of LayerNorm implementations.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden size
        device: Device to run profiling on
        
    Returns:
        Dictionary with memory usage statistics
    """
    # Skip if CUDA is not available and device is cuda
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping profiling")
        return {
            "torch_memory_mb": 0.0,
            "triton_memory_mb": 0.0,
            "memory_saving_percent": 0.0
        }
    
    # Create test tensors
    x = torch.randn((batch_size, seq_len, hidden_size), device=device)
    weight = torch.randn((hidden_size,), device=device)
    bias = torch.randn((hidden_size,), device=device)
    
    # Create PyTorch LayerNorm module
    torch_ln = torch.nn.LayerNorm(hidden_size).to(device)
    torch_ln.weight.data.copy_(weight)
    torch_ln.bias.data.copy_(bias)
    
    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Run PyTorch LayerNorm
    _ = torch_ln(x)
    
    # Measure memory usage
    if device == "cuda":
        torch_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    else:
        torch_memory = 0.0
    
    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Run Triton LayerNorm
    if HAS_TRITON:
        _ = triton_layernorm(x, weight, bias)
    else:
        _ = pytorch_layernorm(x, weight, bias)
    
    # Measure memory usage
    if device == "cuda":
        triton_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    else:
        triton_memory = 0.0
    
    # Calculate memory savings
    memory_saving = torch_memory - triton_memory
    memory_saving_percent = (memory_saving / torch_memory) * 100 if torch_memory > 0 else 0.0
    
    return {
        "torch_memory_mb": torch_memory,
        "triton_memory_mb": triton_memory,
        "memory_saving_mb": memory_saving,
        "memory_saving_percent": memory_saving_percent,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size
    }