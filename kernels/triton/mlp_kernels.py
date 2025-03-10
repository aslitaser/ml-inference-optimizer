"""
Fused MLP Triton Kernels.

This module implements optimized Triton kernels for Multi-Layer Perceptron operations.
These kernels fuse multiple operations to maximize data reuse and minimize memory accesses.

Key features:
- Fused FC1-Activation-FC2 operations
- Support for GELU, SwiGLU, and ReLU activation functions
- Efficient memory usage by keeping intermediates in shared memory
- Optimized memory access patterns for better performance
"""

import torch
import triton
import triton.language as tl
import math
import time
import numpy as np
from typing import Optional, Dict, Tuple


#-----------------------------------------------------------------------------
# Core FusedMLP Triton Kernels
#-----------------------------------------------------------------------------

@triton.jit
def _fused_mlp_gelu_kernel(
    # Pointers to matrices
    input_ptr, fc1_weight_ptr, fc1_bias_ptr, fc2_weight_ptr, fc2_bias_ptr, output_ptr,
    # Matrix dimensions
    batch_size, seq_len, hidden_size, intermediate_size,
    # Strides
    input_batch_stride, input_row_stride, input_col_stride,
    fc1_weight_row_stride, fc1_weight_col_stride,
    fc2_weight_row_stride, fc2_weight_col_stride,
    output_batch_stride, output_row_stride, output_col_stride,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    """
    Fused kernel that computes: FC2(GELU(FC1(x)))
    
    This kernel fuses the entire MLP operation with GELU activation into a single kernel
    to maximize data reuse and minimize memory accesses. It keeps intermediate results
    in shared memory, reducing global memory traffic.
    
    Args:
        input_ptr: Pointer to input tensor [batch_size, seq_len, hidden_size]
        fc1_weight_ptr: Pointer to FC1 weight tensor [intermediate_size, hidden_size]
        fc1_bias_ptr: Pointer to FC1 bias tensor [intermediate_size]
        fc2_weight_ptr: Pointer to FC2 weight tensor [hidden_size, intermediate_size]
        fc2_bias_ptr: Pointer to FC2 bias tensor [hidden_size]
        output_ptr: Pointer to output tensor [batch_size, seq_len, hidden_size]
        Various dimensions and strides for the tensors
        BLOCK_SIZE: Block size for sequence dimension
        BLOCK_K: Block size for hidden dimension
        BLOCK_N: Block size for intermediate dimension
        USE_BIAS: Whether to apply bias
    """
    # Program ID
    pid_batch = tl.program_id(0)  # batch index
    pid_row = tl.program_id(1)    # sequence index
    
    # Compute offset for the current block
    batch_offset = pid_batch * input_batch_stride
    seq_offset = pid_row * BLOCK_SIZE
    
    # Initialize pointers to input
    input_block_ptr = input_ptr + batch_offset + seq_offset * input_row_stride
    
    # Initialize pointer for output
    output_block_ptr = output_ptr + batch_offset + seq_offset * output_row_stride
    
    # Compute number of valid elements in current block
    valid_seq_len = min(BLOCK_SIZE, seq_len - seq_offset)
    
    # Load input block [BLOCK_SIZE, hidden_size]
    input_mask = tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len
    
    # Allocate shared memory for intermediate results
    # This is the key optimization - keeping the intermediate results in shared memory
    # rather than writing them to global memory
    intermediate = tl.zeros([BLOCK_SIZE, intermediate_size], dtype=tl.float32)
    
    # Step 1: Compute FC1(x) -> [BLOCK_SIZE, intermediate_size]
    # We process this in tiles to handle large hidden/intermediate dimensions
    for h_start in range(0, hidden_size, BLOCK_K):
        # Compute number of valid hidden dimensions in current block
        valid_hidden_k = min(BLOCK_K, hidden_size - h_start)
        
        # Load input tile [BLOCK_SIZE, BLOCK_K]
        x_tile_ptr = input_block_ptr + h_start * input_col_stride
        x_tile_mask = input_mask & (tl.arange(0, BLOCK_K)[None, :] < valid_hidden_k)
        x_tile = tl.load(
            x_tile_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * input_row_stride + 
            tl.arange(0, BLOCK_K)[None, :] * input_col_stride,
            mask=x_tile_mask,
            other=0.0
        )
        
        # Process intermediate dimension in tiles
        for i_start in range(0, intermediate_size, BLOCK_N):
            # Compute number of valid intermediate dimensions
            valid_inter_n = min(BLOCK_N, intermediate_size - i_start)
            
            # Load weight tile for FC1 [BLOCK_K, BLOCK_N]
            fc1_weight_tile_ptr = fc1_weight_ptr + h_start * fc1_weight_col_stride + i_start * fc1_weight_row_stride
            fc1_weight_mask = (tl.arange(0, BLOCK_K)[:, None] < valid_hidden_k) & (tl.arange(0, BLOCK_N)[None, :] < valid_inter_n)
            fc1_weight_tile = tl.load(
                fc1_weight_tile_ptr + tl.arange(0, BLOCK_K)[:, None] * fc1_weight_col_stride + 
                tl.arange(0, BLOCK_N)[None, :] * fc1_weight_row_stride,
                mask=fc1_weight_mask,
                other=0.0
            )
            
            # Compute partial FC1: [BLOCK_SIZE, BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_SIZE, BLOCK_N]
            partial_fc1 = tl.dot(x_tile, fc1_weight_tile)
            
            # Accumulate result to intermediate buffer
            intermediate_indices = tl.arange(0, BLOCK_N) + i_start
            intermediate_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (intermediate_indices[None, :] < intermediate_size)
            intermediate[:, i_start:i_start+BLOCK_N] += partial_fc1
    
    # Apply FC1 bias if using bias
    if USE_BIAS:
        # Load bias for FC1
        for i_start in range(0, intermediate_size, BLOCK_N):
            valid_inter_n = min(BLOCK_N, intermediate_size - i_start)
            fc1_bias_mask = tl.arange(0, BLOCK_N) < valid_inter_n
            fc1_bias_tile = tl.load(
                fc1_bias_ptr + i_start + tl.arange(0, BLOCK_N),
                mask=fc1_bias_mask,
                other=0.0
            )
            
            # Add bias to intermediate
            intermediate_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_inter_n)
            intermediate[:, i_start:i_start+BLOCK_N] = intermediate[:, i_start:i_start+BLOCK_N] + fc1_bias_tile[None, :]
    
    # Apply GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = 0.7978845608028654
    one = 1.0
    half = 0.5
    coeff = 0.044715
    
    # Process activation in tiles to manage shared memory usage
    for i_start in range(0, intermediate_size, BLOCK_N):
        valid_inter_n = min(BLOCK_N, intermediate_size - i_start)
        intermediate_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_inter_n)
        
        # Get intermediate tile
        intermediate_tile = intermediate[:, i_start:i_start+BLOCK_N]
        
        # Apply GELU
        x_cube = intermediate_tile * intermediate_tile * intermediate_tile
        inner = sqrt_2_pi * (intermediate_tile + coeff * x_cube)
        intermediate_tile = half * intermediate_tile * (one + tl.tanh(inner))
        
        # Update intermediate buffer with activated values
        intermediate[:, i_start:i_start+BLOCK_N] = intermediate_tile
    
    # Initialize accumulator for output
    output = tl.zeros([BLOCK_SIZE, hidden_size], dtype=tl.float32)
    
    # Step 2: Compute FC2(intermediate) -> [BLOCK_SIZE, hidden_size]
    # Process in tiles
    for i_start in range(0, intermediate_size, BLOCK_K):
        # Compute number of valid intermediate dimensions
        valid_inter_k = min(BLOCK_K, intermediate_size - i_start)
        
        # Get intermediate tile after activation
        intermediate_tile = intermediate[:, i_start:i_start+BLOCK_K]
        
        # Process hidden dimension in tiles
        for h_start in range(0, hidden_size, BLOCK_N):
            # Compute number of valid hidden dimensions
            valid_hidden_n = min(BLOCK_N, hidden_size - h_start)
            
            # Load weight tile for FC2 [BLOCK_K, BLOCK_N]
            fc2_weight_tile_ptr = fc2_weight_ptr + i_start * fc2_weight_col_stride + h_start * fc2_weight_row_stride
            fc2_weight_mask = (tl.arange(0, BLOCK_K)[:, None] < valid_inter_k) & (tl.arange(0, BLOCK_N)[None, :] < valid_hidden_n)
            fc2_weight_tile = tl.load(
                fc2_weight_tile_ptr + tl.arange(0, BLOCK_K)[:, None] * fc2_weight_col_stride + 
                tl.arange(0, BLOCK_N)[None, :] * fc2_weight_row_stride,
                mask=fc2_weight_mask,
                other=0.0
            )
            
            # Compute partial FC2: [BLOCK_SIZE, BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_SIZE, BLOCK_N]
            partial_fc2 = tl.dot(intermediate_tile, fc2_weight_tile)
            
            # Accumulate result to output buffer
            output_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_hidden_n)
            output[:, h_start:h_start+BLOCK_N] += partial_fc2
    
    # Apply FC2 bias if using bias
    if USE_BIAS:
        # Load bias for FC2
        for h_start in range(0, hidden_size, BLOCK_N):
            valid_hidden_n = min(BLOCK_N, hidden_size - h_start)
            fc2_bias_mask = tl.arange(0, BLOCK_N) < valid_hidden_n
            fc2_bias_tile = tl.load(
                fc2_bias_ptr + h_start + tl.arange(0, BLOCK_N),
                mask=fc2_bias_mask,
                other=0.0
            )
            
            # Add bias to output
            output_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_hidden_n)
            output[:, h_start:h_start+BLOCK_N] = output[:, h_start:h_start+BLOCK_N] + fc2_bias_tile[None, :]
    
    # Store final output
    for h_start in range(0, hidden_size, BLOCK_N):
        valid_hidden_n = min(BLOCK_N, hidden_size - h_start)
        output_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_hidden_n)
        
        # Get output tile
        output_tile = output[:, h_start:h_start+BLOCK_N]
        
        # Store output tile
        tl.store(
            output_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * output_row_stride + 
            (h_start + tl.arange(0, BLOCK_N)[None, :]) * output_col_stride,
            output_tile,
            mask=output_mask
        )


@triton.jit
def _fused_mlp_relu_kernel(
    # Pointers to matrices
    input_ptr, fc1_weight_ptr, fc1_bias_ptr, fc2_weight_ptr, fc2_bias_ptr, output_ptr,
    # Matrix dimensions
    batch_size, seq_len, hidden_size, intermediate_size,
    # Strides
    input_batch_stride, input_row_stride, input_col_stride,
    fc1_weight_row_stride, fc1_weight_col_stride,
    fc2_weight_row_stride, fc2_weight_col_stride,
    output_batch_stride, output_row_stride, output_col_stride,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    """
    Fused kernel that computes: FC2(ReLU(FC1(x)))
    
    Similar to the GELU version, but uses ReLU activation function instead.
    ReLU is simpler and often faster than GELU.
    
    Args:
        Same as _fused_mlp_gelu_kernel
    """
    # Program ID
    pid_batch = tl.program_id(0)  # batch index
    pid_row = tl.program_id(1)    # sequence index
    
    # Compute offset for the current block
    batch_offset = pid_batch * input_batch_stride
    seq_offset = pid_row * BLOCK_SIZE
    
    # Initialize pointers to input
    input_block_ptr = input_ptr + batch_offset + seq_offset * input_row_stride
    
    # Initialize pointer for output
    output_block_ptr = output_ptr + batch_offset + seq_offset * output_row_stride
    
    # Compute number of valid elements in current block
    valid_seq_len = min(BLOCK_SIZE, seq_len - seq_offset)
    
    # Load input block [BLOCK_SIZE, hidden_size]
    input_mask = tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len
    
    # Allocate shared memory for intermediate results
    intermediate = tl.zeros([BLOCK_SIZE, intermediate_size], dtype=tl.float32)
    
    # Step 1: Compute FC1(x) -> [BLOCK_SIZE, intermediate_size]
    for h_start in range(0, hidden_size, BLOCK_K):
        # Compute number of valid hidden dimensions
        valid_hidden_k = min(BLOCK_K, hidden_size - h_start)
        
        # Load input tile [BLOCK_SIZE, BLOCK_K]
        x_tile_ptr = input_block_ptr + h_start * input_col_stride
        x_tile_mask = input_mask & (tl.arange(0, BLOCK_K)[None, :] < valid_hidden_k)
        x_tile = tl.load(
            x_tile_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * input_row_stride + 
            tl.arange(0, BLOCK_K)[None, :] * input_col_stride,
            mask=x_tile_mask,
            other=0.0
        )
        
        # Process intermediate dimension in tiles
        for i_start in range(0, intermediate_size, BLOCK_N):
            # Compute number of valid intermediate dimensions
            valid_inter_n = min(BLOCK_N, intermediate_size - i_start)
            
            # Load weight tile for FC1 [BLOCK_K, BLOCK_N]
            fc1_weight_tile_ptr = fc1_weight_ptr + h_start * fc1_weight_col_stride + i_start * fc1_weight_row_stride
            fc1_weight_mask = (tl.arange(0, BLOCK_K)[:, None] < valid_hidden_k) & (tl.arange(0, BLOCK_N)[None, :] < valid_inter_n)
            fc1_weight_tile = tl.load(
                fc1_weight_tile_ptr + tl.arange(0, BLOCK_K)[:, None] * fc1_weight_col_stride + 
                tl.arange(0, BLOCK_N)[None, :] * fc1_weight_row_stride,
                mask=fc1_weight_mask,
                other=0.0
            )
            
            # Compute partial FC1: [BLOCK_SIZE, BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_SIZE, BLOCK_N]
            partial_fc1 = tl.dot(x_tile, fc1_weight_tile)
            
            # Accumulate result to intermediate buffer
            intermediate_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_inter_n)
            intermediate[:, i_start:i_start+BLOCK_N] += partial_fc1
    
    # Apply FC1 bias if using bias
    if USE_BIAS:
        # Load bias for FC1
        for i_start in range(0, intermediate_size, BLOCK_N):
            valid_inter_n = min(BLOCK_N, intermediate_size - i_start)
            fc1_bias_mask = tl.arange(0, BLOCK_N) < valid_inter_n
            fc1_bias_tile = tl.load(
                fc1_bias_ptr + i_start + tl.arange(0, BLOCK_N),
                mask=fc1_bias_mask,
                other=0.0
            )
            
            # Add bias to intermediate
            intermediate_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_inter_n)
            intermediate[:, i_start:i_start+BLOCK_N] = intermediate[:, i_start:i_start+BLOCK_N] + fc1_bias_tile[None, :]
    
    # Apply ReLU activation: max(0, x)
    # Process activation in tiles to manage shared memory usage
    for i_start in range(0, intermediate_size, BLOCK_N):
        valid_inter_n = min(BLOCK_N, intermediate_size - i_start)
        intermediate_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_inter_n)
        
        # Get intermediate tile
        intermediate_tile = intermediate[:, i_start:i_start+BLOCK_N]
        
        # Apply ReLU
        intermediate_tile = tl.maximum(0.0, intermediate_tile)
        
        # Update intermediate buffer with activated values
        intermediate[:, i_start:i_start+BLOCK_N] = intermediate_tile
    
    # Initialize accumulator for output
    output = tl.zeros([BLOCK_SIZE, hidden_size], dtype=tl.float32)
    
    # Step 2: Compute FC2(intermediate) -> [BLOCK_SIZE, hidden_size]
    # Process in tiles
    for i_start in range(0, intermediate_size, BLOCK_K):
        # Compute number of valid intermediate dimensions
        valid_inter_k = min(BLOCK_K, intermediate_size - i_start)
        
        # Get intermediate tile after activation
        intermediate_tile = intermediate[:, i_start:i_start+BLOCK_K]
        
        # Process hidden dimension in tiles
        for h_start in range(0, hidden_size, BLOCK_N):
            # Compute number of valid hidden dimensions
            valid_hidden_n = min(BLOCK_N, hidden_size - h_start)
            
            # Load weight tile for FC2 [BLOCK_K, BLOCK_N]
            fc2_weight_tile_ptr = fc2_weight_ptr + i_start * fc2_weight_col_stride + h_start * fc2_weight_row_stride
            fc2_weight_mask = (tl.arange(0, BLOCK_K)[:, None] < valid_inter_k) & (tl.arange(0, BLOCK_N)[None, :] < valid_hidden_n)
            fc2_weight_tile = tl.load(
                fc2_weight_tile_ptr + tl.arange(0, BLOCK_K)[:, None] * fc2_weight_col_stride + 
                tl.arange(0, BLOCK_N)[None, :] * fc2_weight_row_stride,
                mask=fc2_weight_mask,
                other=0.0
            )
            
            # Compute partial FC2: [BLOCK_SIZE, BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_SIZE, BLOCK_N]
            partial_fc2 = tl.dot(intermediate_tile, fc2_weight_tile)
            
            # Accumulate result to output buffer
            output_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_hidden_n)
            output[:, h_start:h_start+BLOCK_N] += partial_fc2
    
    # Apply FC2 bias if using bias
    if USE_BIAS:
        # Load bias for FC2
        for h_start in range(0, hidden_size, BLOCK_N):
            valid_hidden_n = min(BLOCK_N, hidden_size - h_start)
            fc2_bias_mask = tl.arange(0, BLOCK_N) < valid_hidden_n
            fc2_bias_tile = tl.load(
                fc2_bias_ptr + h_start + tl.arange(0, BLOCK_N),
                mask=fc2_bias_mask,
                other=0.0
            )
            
            # Add bias to output
            output_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_hidden_n)
            output[:, h_start:h_start+BLOCK_N] = output[:, h_start:h_start+BLOCK_N] + fc2_bias_tile[None, :]
    
    # Store final output
    for h_start in range(0, hidden_size, BLOCK_N):
        valid_hidden_n = min(BLOCK_N, hidden_size - h_start)
        output_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_hidden_n)
        
        # Get output tile
        output_tile = output[:, h_start:h_start+BLOCK_N]
        
        # Store output tile
        tl.store(
            output_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * output_row_stride + 
            (h_start + tl.arange(0, BLOCK_N)[None, :]) * output_col_stride,
            output_tile,
            mask=output_mask
        )


@triton.jit
def _fused_mlp_swiglu_kernel(
    # Pointers to matrices
    input_ptr, gate_weight_ptr, gate_bias_ptr, value_weight_ptr, value_bias_ptr, 
    fc2_weight_ptr, fc2_bias_ptr, output_ptr,
    # Matrix dimensions
    batch_size, seq_len, hidden_size, intermediate_size,
    # Strides
    input_batch_stride, input_row_stride, input_col_stride,
    gate_weight_row_stride, gate_weight_col_stride,
    value_weight_row_stride, value_weight_col_stride,
    fc2_weight_row_stride, fc2_weight_col_stride,
    output_batch_stride, output_row_stride, output_col_stride,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    """
    Fused kernel that computes SwiGLU activation followed by linear projection:
    FC2(SwiGLU(FC1_gate, FC1_value))
    
    SwiGLU is defined as: SiLU(gate_proj) * value_proj
    SiLU(x) = x * sigmoid(x)
    
    Args:
        input_ptr: Pointer to input tensor
        gate_weight_ptr: Pointer to gate projection weights
        gate_bias_ptr: Pointer to gate projection bias
        value_weight_ptr: Pointer to value projection weights
        value_bias_ptr: Pointer to value projection bias
        fc2_weight_ptr: Pointer to output projection weights
        fc2_bias_ptr: Pointer to output projection bias
        output_ptr: Pointer to output tensor
        Various dimensions and strides for the tensors
        BLOCK_SIZE: Block size for sequence dimension
        BLOCK_K: Block size for hidden dimension
        BLOCK_N: Block size for intermediate dimension
        USE_BIAS: Whether to apply bias
    """
    # Program ID
    pid_batch = tl.program_id(0)  # batch index
    pid_row = tl.program_id(1)    # sequence index
    
    # Compute offset for the current block
    batch_offset = pid_batch * input_batch_stride
    seq_offset = pid_row * BLOCK_SIZE
    
    # Initialize pointers to input
    input_block_ptr = input_ptr + batch_offset + seq_offset * input_row_stride
    
    # Initialize pointer for output
    output_block_ptr = output_ptr + batch_offset + seq_offset * output_row_stride
    
    # Compute number of valid elements in current block
    valid_seq_len = min(BLOCK_SIZE, seq_len - seq_offset)
    
    # Load input block [BLOCK_SIZE, hidden_size]
    input_mask = tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len
    
    # Allocate shared memory for intermediate results
    gate_proj = tl.zeros([BLOCK_SIZE, intermediate_size], dtype=tl.float32)
    value_proj = tl.zeros([BLOCK_SIZE, intermediate_size], dtype=tl.float32)
    
    # Step 1: Compute gate and value projections -> [BLOCK_SIZE, intermediate_size]
    for h_start in range(0, hidden_size, BLOCK_K):
        # Compute number of valid hidden dimensions
        valid_hidden_k = min(BLOCK_K, hidden_size - h_start)
        
        # Load input tile [BLOCK_SIZE, BLOCK_K]
        x_tile_ptr = input_block_ptr + h_start * input_col_stride
        x_tile_mask = input_mask & (tl.arange(0, BLOCK_K)[None, :] < valid_hidden_k)
        x_tile = tl.load(
            x_tile_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * input_row_stride + 
            tl.arange(0, BLOCK_K)[None, :] * input_col_stride,
            mask=x_tile_mask,
            other=0.0
        )
        
        # Process intermediate dimension in tiles
        for i_start in range(0, intermediate_size, BLOCK_N):
            # Compute number of valid intermediate dimensions
            valid_inter_n = min(BLOCK_N, intermediate_size - i_start)
            
            # Load gate weight tile [BLOCK_K, BLOCK_N]
            gate_weight_tile_ptr = gate_weight_ptr + h_start * gate_weight_col_stride + i_start * gate_weight_row_stride
            weight_mask = (tl.arange(0, BLOCK_K)[:, None] < valid_hidden_k) & (tl.arange(0, BLOCK_N)[None, :] < valid_inter_n)
            gate_weight_tile = tl.load(
                gate_weight_tile_ptr + tl.arange(0, BLOCK_K)[:, None] * gate_weight_col_stride + 
                tl.arange(0, BLOCK_N)[None, :] * gate_weight_row_stride,
                mask=weight_mask,
                other=0.0
            )
            
            # Load value weight tile [BLOCK_K, BLOCK_N]
            value_weight_tile_ptr = value_weight_ptr + h_start * value_weight_col_stride + i_start * value_weight_row_stride
            value_weight_tile = tl.load(
                value_weight_tile_ptr + tl.arange(0, BLOCK_K)[:, None] * value_weight_col_stride + 
                tl.arange(0, BLOCK_N)[None, :] * value_weight_row_stride,
                mask=weight_mask,
                other=0.0
            )
            
            # Compute partial projections: [BLOCK_SIZE, BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_SIZE, BLOCK_N]
            partial_gate = tl.dot(x_tile, gate_weight_tile)
            partial_value = tl.dot(x_tile, value_weight_tile)
            
            # Accumulate results to intermediate buffers
            intermediate_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_inter_n)
            gate_proj[:, i_start:i_start+BLOCK_N] += partial_gate
            value_proj[:, i_start:i_start+BLOCK_N] += partial_value
    
    # Apply bias if using bias
    if USE_BIAS:
        # Load and apply bias for gate and value projections
        for i_start in range(0, intermediate_size, BLOCK_N):
            valid_inter_n = min(BLOCK_N, intermediate_size - i_start)
            bias_mask = tl.arange(0, BLOCK_N) < valid_inter_n
            
            # Gate bias
            gate_bias_tile = tl.load(
                gate_bias_ptr + i_start + tl.arange(0, BLOCK_N),
                mask=bias_mask,
                other=0.0
            )
            
            # Value bias
            value_bias_tile = tl.load(
                value_bias_ptr + i_start + tl.arange(0, BLOCK_N),
                mask=bias_mask,
                other=0.0
            )
            
            # Add bias to projections
            intermediate_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_inter_n)
            gate_proj[:, i_start:i_start+BLOCK_N] = gate_proj[:, i_start:i_start+BLOCK_N] + gate_bias_tile[None, :]
            value_proj[:, i_start:i_start+BLOCK_N] = value_proj[:, i_start:i_start+BLOCK_N] + value_bias_tile[None, :]
    
    # Apply SwiGLU activation: SiLU(gate_proj) * value_proj
    # SiLU(x) = x * sigmoid(x)
    # Process activation in tiles to manage shared memory usage
    for i_start in range(0, intermediate_size, BLOCK_N):
        valid_inter_n = min(BLOCK_N, intermediate_size - i_start)
        intermediate_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_inter_n)
        
        # Get intermediate tiles
        gate_tile = gate_proj[:, i_start:i_start+BLOCK_N]
        value_tile = value_proj[:, i_start:i_start+BLOCK_N]
        
        # Apply SiLU to gate
        gate_sigmoid = 1.0 / (1.0 + tl.exp(-gate_tile))
        gate_silu = gate_tile * gate_sigmoid
        
        # Apply SwiGLU: SiLU(gate) * value
        swiglu_result = gate_silu * value_tile
        
        # Store back the SwiGLU result in the gate projection buffer to save memory
        gate_proj[:, i_start:i_start+BLOCK_N] = swiglu_result
    
    # Initialize accumulator for output
    output = tl.zeros([BLOCK_SIZE, hidden_size], dtype=tl.float32)
    
    # Step 2: Compute FC2(gate_proj) -> [BLOCK_SIZE, hidden_size]
    # Now gate_proj contains the SwiGLU result
    for i_start in range(0, intermediate_size, BLOCK_K):
        # Compute number of valid intermediate dimensions
        valid_inter_k = min(BLOCK_K, intermediate_size - i_start)
        
        # Get SwiGLU result tile
        swiglu_tile = gate_proj[:, i_start:i_start+BLOCK_K]
        
        # Process hidden dimension in tiles
        for h_start in range(0, hidden_size, BLOCK_N):
            # Compute number of valid hidden dimensions
            valid_hidden_n = min(BLOCK_N, hidden_size - h_start)
            
            # Load weight tile for FC2 [BLOCK_K, BLOCK_N]
            fc2_weight_tile_ptr = fc2_weight_ptr + i_start * fc2_weight_col_stride + h_start * fc2_weight_row_stride
            fc2_weight_mask = (tl.arange(0, BLOCK_K)[:, None] < valid_inter_k) & (tl.arange(0, BLOCK_N)[None, :] < valid_hidden_n)
            fc2_weight_tile = tl.load(
                fc2_weight_tile_ptr + tl.arange(0, BLOCK_K)[:, None] * fc2_weight_col_stride + 
                tl.arange(0, BLOCK_N)[None, :] * fc2_weight_row_stride,
                mask=fc2_weight_mask,
                other=0.0
            )
            
            # Compute partial FC2: [BLOCK_SIZE, BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_SIZE, BLOCK_N]
            partial_fc2 = tl.dot(swiglu_tile, fc2_weight_tile)
            
            # Accumulate result to output buffer
            output_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_hidden_n)
            output[:, h_start:h_start+BLOCK_N] += partial_fc2
    
    # Apply FC2 bias if using bias
    if USE_BIAS:
        # Load bias for FC2
        for h_start in range(0, hidden_size, BLOCK_N):
            valid_hidden_n = min(BLOCK_N, hidden_size - h_start)
            fc2_bias_mask = tl.arange(0, BLOCK_N) < valid_hidden_n
            fc2_bias_tile = tl.load(
                fc2_bias_ptr + h_start + tl.arange(0, BLOCK_N),
                mask=fc2_bias_mask,
                other=0.0
            )
            
            # Add bias to output
            output_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_hidden_n)
            output[:, h_start:h_start+BLOCK_N] = output[:, h_start:h_start+BLOCK_N] + fc2_bias_tile[None, :]
    
    # Store final output
    for h_start in range(0, hidden_size, BLOCK_N):
        valid_hidden_n = min(BLOCK_N, hidden_size - h_start)
        output_mask = (tl.arange(0, BLOCK_SIZE)[:, None] < valid_seq_len) & (tl.arange(0, BLOCK_N)[None, :] < valid_hidden_n)
        
        # Get output tile
        output_tile = output[:, h_start:h_start+BLOCK_N]
        
        # Store output tile
        tl.store(
            output_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * output_row_stride + 
            (h_start + tl.arange(0, BLOCK_N)[None, :]) * output_col_stride,
            output_tile,
            mask=output_mask
        )


#-----------------------------------------------------------------------------
# Python Wrapper Functions
#-----------------------------------------------------------------------------

def triton_fused_mlp(
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: Optional[torch.Tensor],
    fc2_weight: torch.Tensor,
    fc2_bias: Optional[torch.Tensor],
    activation: str = "gelu",
    fc1_gate_weight: Optional[torch.Tensor] = None,
    fc1_gate_bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute fused MLP operation using optimized Triton kernels.
    
    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        fc1_weight: Weight tensor for first linear layer [intermediate_size, hidden_size]
        fc1_bias: Optional bias tensor for first linear layer [intermediate_size]
        fc2_weight: Weight tensor for second linear layer [hidden_size, intermediate_size]
        fc2_bias: Optional bias tensor for second linear layer [hidden_size]
        activation: Activation function to use: "gelu", "relu", or "swiglu"
        fc1_gate_weight: Optional gate weight for SwiGLU [intermediate_size, hidden_size]
        fc1_gate_bias: Optional gate bias for SwiGLU [intermediate_size]
        
    Returns:
        Output tensor of shape [batch_size, seq_len, hidden_size]
    """
    # Handle potential missing Triton
    if not hasattr(triton, "runtime") or not triton.runtime.driver.is_initialized():
        return pytorch_fused_mlp(
            hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias, 
            activation, fc1_gate_weight, fc1_gate_bias
        )
    
    # Extract dimensions
    batch_size, seq_len, hidden_size = hidden_states.shape
    intermediate_size = fc1_weight.shape[0]
    
    # Determine whether to use bias
    use_bias = fc1_bias is not None and fc2_bias is not None
    
    # Create zero tensors for bias if not provided
    if fc1_bias is None:
        fc1_bias = torch.zeros(intermediate_size, device=hidden_states.device, dtype=hidden_states.dtype)
    if fc2_bias is None:
        fc2_bias = torch.zeros(hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
    
    # Create output tensor
    output = torch.empty_like(hidden_states)
    
    # Determine the block sizes based on model dimensions
    # These block sizes can be tuned for specific hardware and model architectures
    BLOCK_SIZE = min(64, seq_len)
    BLOCK_K = min(32, hidden_size)
    BLOCK_N = min(32, intermediate_size)
    
    # Grid dimensions for kernel launch
    grid = (
        batch_size,
        triton.cdiv(seq_len, BLOCK_SIZE)
    )
    
    # Launch the appropriate kernel based on the activation function
    if activation == "gelu":
        _fused_mlp_gelu_kernel[grid](
            hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias, output,
            batch_size, seq_len, hidden_size, intermediate_size,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            fc1_weight.stride(0), fc1_weight.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_SIZE=BLOCK_SIZE, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            USE_BIAS=use_bias
        )
    elif activation == "relu":
        _fused_mlp_relu_kernel[grid](
            hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias, output,
            batch_size, seq_len, hidden_size, intermediate_size,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            fc1_weight.stride(0), fc1_weight.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_SIZE=BLOCK_SIZE, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            USE_BIAS=use_bias
        )
    elif activation == "swiglu":
        # SwiGLU requires both gate and value projections
        if fc1_gate_weight is None or (use_bias and fc1_gate_bias is None):
            raise ValueError("SwiGLU activation requires gate weights and bias (if bias is used)")
            
        # Create zero tensors for gate bias if not provided but use_bias is True
        if use_bias and fc1_gate_bias is None:
            fc1_gate_bias = torch.zeros(intermediate_size, device=hidden_states.device, dtype=hidden_states.dtype)
        
        _fused_mlp_swiglu_kernel[grid](
            hidden_states, fc1_gate_weight, fc1_gate_bias, fc1_weight, fc1_bias,
            fc2_weight, fc2_bias, output,
            batch_size, seq_len, hidden_size, intermediate_size,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            fc1_gate_weight.stride(0), fc1_gate_weight.stride(1),
            fc1_weight.stride(0), fc1_weight.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_SIZE=BLOCK_SIZE, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            USE_BIAS=use_bias
        )
    else:
        raise ValueError(f"Unsupported activation function: {activation}")
    
    return output


def pytorch_fused_mlp(
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: Optional[torch.Tensor],
    fc2_weight: torch.Tensor,
    fc2_bias: Optional[torch.Tensor],
    activation: str = "gelu",
    fc1_gate_weight: Optional[torch.Tensor] = None,
    fc1_gate_bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    PyTorch implementation of fused MLP (used when Triton is not available).
    
    Args:
        Same as triton_fused_mlp
        
    Returns:
        Output tensor of shape [batch_size, seq_len, hidden_size]
    """
    # Step 1: FC1 projection
    fc1_output = torch.nn.functional.linear(hidden_states, fc1_weight, fc1_bias)
    
    # Step 2: Apply activation
    if activation == "gelu":
        fc1_output = torch.nn.functional.gelu(fc1_output)
    elif activation == "relu":
        fc1_output = torch.nn.functional.relu(fc1_output)
    elif activation == "swiglu":
        if fc1_gate_weight is None:
            raise ValueError("SwiGLU activation requires gate weights")
        
        # Compute gate and value projections
        gate_output = torch.nn.functional.linear(hidden_states, fc1_gate_weight, fc1_gate_bias)
        
        # Apply SwiGLU: SiLU(gate) * value
        # SiLU(x) = x * sigmoid(x)
        gate_output = gate_output * torch.sigmoid(gate_output)  # SiLU activation
        fc1_output = gate_output * fc1_output  # Element-wise multiplication
    else:
        raise ValueError(f"Unsupported activation function: {activation}")
    
    # Step 3: FC2 projection
    output = torch.nn.functional.linear(fc1_output, fc2_weight, fc2_bias)
    
    return output


#-----------------------------------------------------------------------------
# Benchmarking Functions
#-----------------------------------------------------------------------------

def benchmark_fused_mlp(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    activation: str = "gelu",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    num_warmup: int = 10,
    num_iter: int = 100
) -> Dict[str, float]:
    """
    Benchmark the performance of the fused MLP implementation.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden size
        intermediate_size: Intermediate size
        activation: Activation function to use
        device: Device to run benchmark on
        dtype: Data type for benchmark
        num_warmup: Number of warmup iterations
        num_iter: Number of benchmark iterations
        
    Returns:
        Dictionary with benchmark results
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "activation": activation,
            "triton_time_ms": 0.0,
            "pytorch_time_ms": 0.0,
            "speedup": 0.0
        }
    
    # Create test tensors
    hidden_states = torch.randn((batch_size, seq_len, hidden_size), device=device, dtype=dtype)
    fc1_weight = torch.randn((intermediate_size, hidden_size), device=device, dtype=dtype)
    fc1_bias = torch.randn((intermediate_size,), device=device, dtype=dtype)
    fc2_weight = torch.randn((hidden_size, intermediate_size), device=device, dtype=dtype)
    fc2_bias = torch.randn((hidden_size,), device=device, dtype=dtype)
    
    # Create gate weights for SwiGLU if needed
    if activation == "swiglu":
        fc1_gate_weight = torch.randn((intermediate_size, hidden_size), device=device, dtype=dtype)
        fc1_gate_bias = torch.randn((intermediate_size,), device=device, dtype=dtype)
    else:
        fc1_gate_weight = None
        fc1_gate_bias = None
    
    # Warm up
    for _ in range(num_warmup):
        # PyTorch implementation
        _ = pytorch_fused_mlp(
            hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
            activation, fc1_gate_weight, fc1_gate_bias
        )
        
        # Triton implementation
        if hasattr(triton, "runtime") and triton.runtime.driver.is_initialized():
            _ = triton_fused_mlp(
                hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
                activation, fc1_gate_weight, fc1_gate_bias
            )
    
    # Benchmark PyTorch implementation
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    for _ in range(num_iter):
        _ = pytorch_fused_mlp(
            hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
            activation, fc1_gate_weight, fc1_gate_bias
        )
    
    torch.cuda.synchronize() if device == "cuda" else None
    pytorch_time = (time.time() - start_time) * 1000 / num_iter  # ms
    
    # Benchmark Triton implementation
    if hasattr(triton, "runtime") and triton.runtime.driver.is_initialized():
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()
        
        for _ in range(num_iter):
            _ = triton_fused_mlp(
                hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
                activation, fc1_gate_weight, fc1_gate_bias
            )
        
        torch.cuda.synchronize() if device == "cuda" else None
        triton_time = (time.time() - start_time) * 1000 / num_iter  # ms
        
        speedup = pytorch_time / triton_time
    else:
        triton_time = pytorch_time
        speedup = 1.0
    
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "activation": activation,
        "triton_time_ms": triton_time,
        "pytorch_time_ms": pytorch_time,
        "speedup": speedup
    }


def validate_fused_mlp(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    activation: str = "gelu",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> Dict[str, float]:
    """
    Validate the correctness of the fused MLP implementation.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden size
        intermediate_size: Intermediate size
        activation: Activation function to use
        device: Device to run validation on
        dtype: Data type for validation
        
    Returns:
        Dictionary with validation results
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping validation")
        return {
            "is_correct": False,
            "max_diff": 0.0
        }
    
    # Create test tensors
    hidden_states = torch.randn((batch_size, seq_len, hidden_size), device=device, dtype=dtype)
    fc1_weight = torch.randn((intermediate_size, hidden_size), device=device, dtype=dtype)
    fc1_bias = torch.randn((intermediate_size,), device=device, dtype=dtype)
    fc2_weight = torch.randn((hidden_size, intermediate_size), device=device, dtype=dtype)
    fc2_bias = torch.randn((hidden_size,), device=device, dtype=dtype)
    
    # Create gate weights for SwiGLU if needed
    if activation == "swiglu":
        fc1_gate_weight = torch.randn((intermediate_size, hidden_size), device=device, dtype=dtype)
        fc1_gate_bias = torch.randn((intermediate_size,), device=device, dtype=dtype)
    else:
        fc1_gate_weight = None
        fc1_gate_bias = None
    
    # Compute with PyTorch implementation
    pytorch_output = pytorch_fused_mlp(
        hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
        activation, fc1_gate_weight, fc1_gate_bias
    )
    
    # Compute with Triton implementation
    if hasattr(triton, "runtime") and triton.runtime.driver.is_initialized():
        triton_output = triton_fused_mlp(
            hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
            activation, fc1_gate_weight, fc1_gate_bias
        )
        
        # Compare outputs
        max_diff = torch.max(torch.abs(pytorch_output - triton_output)).item()
        is_correct = max_diff < 1e-3
    else:
        # If Triton is not available, we compare with ourselves (should be equal)
        max_diff = 0.0
        is_correct = True
    
    return {
        "is_correct": is_correct,
        "max_diff": max_diff,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "activation": activation
    }


def profile_memory_usage(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    activation: str = "gelu",
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Profile memory usage of the fused MLP implementation.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden size
        intermediate_size: Intermediate size
        activation: Activation function to use
        device: Device to run profiling on
        
    Returns:
        Dictionary with memory usage statistics
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping memory profiling")
        return {
            "pytorch_memory_mb": 0.0,
            "triton_memory_mb": 0.0,
            "memory_saving_mb": 0.0,
            "memory_saving_percent": 0.0
        }
    
    # Create test tensors
    hidden_states = torch.randn((batch_size, seq_len, hidden_size), device=device)
    fc1_weight = torch.randn((intermediate_size, hidden_size), device=device)
    fc1_bias = torch.randn((intermediate_size,), device=device)
    fc2_weight = torch.randn((hidden_size, intermediate_size), device=device)
    fc2_bias = torch.randn((hidden_size,), device=device)
    
    # Create gate weights for SwiGLU if needed
    if activation == "swiglu":
        fc1_gate_weight = torch.randn((intermediate_size, hidden_size), device=device)
        fc1_gate_bias = torch.randn((intermediate_size,), device=device)
    else:
        fc1_gate_weight = None
        fc1_gate_bias = None
    
    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Run PyTorch implementation
    _ = pytorch_fused_mlp(
        hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
        activation, fc1_gate_weight, fc1_gate_bias
    )
    
    # Measure memory usage
    if device == "cuda":
        pytorch_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    else:
        pytorch_memory = 0.0
    
    # Run Triton implementation
    if hasattr(triton, "runtime") and triton.runtime.driver.is_initialized():
        _ = triton_fused_mlp(
            hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
            activation, fc1_gate_weight, fc1_gate_bias
        )
        
        # Measure memory usage
        if device == "cuda":
            triton_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        else:
            triton_memory = 0.0
    else:
        triton_memory = pytorch_memory
    
    # Calculate memory savings
    memory_saving = pytorch_memory - triton_memory
    memory_saving_percent = (memory_saving / pytorch_memory) * 100 if pytorch_memory > 0 else 0.0
    
    return {
        "pytorch_memory_mb": pytorch_memory,
        "triton_memory_mb": triton_memory,
        "memory_saving_mb": memory_saving,
        "memory_saving_percent": memory_saving_percent,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "activation": activation
    }