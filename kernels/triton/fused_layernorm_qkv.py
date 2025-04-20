"""
Fused LayerNorm + QKV Projection Triton Kernels.

This module implements optimized Triton kernels that fuse LayerNorm and QKV projection
operations into a single kernel. This reduces memory traffic and improves performance
by keeping normalized hidden states in registers/shared memory.

Key features:
- Fused layernorm computation and QKV projection
- Efficient parallel reduction for mean/variance calculation
- Support for multiple precision modes (FP16, BF16, FP32)
- Variable sequence length optimization
- High-performance tiling strategies
- Compatible with both Flash and Ring Attention modules
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, Any

# Try to import Triton, but gracefully handle if not available
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Triton not available, using PyTorch fallback for fused LayerNorm+QKV kernels")


#-----------------------------------------------------------------------------
# Core Fused LayerNorm + QKV Triton Kernels
#-----------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    @triton.autotune(
        configs=[
            # Configurations for different sequence lengths and model sizes
            triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'NUM_WARPS': 4}),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'NUM_WARPS': 4}),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'NUM_WARPS': 8}),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'NUM_WARPS': 8}),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'NUM_WARPS': 8}),
            # For larger hidden sizes, use more warps
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'NUM_WARPS': 16}),
        ],
        key=['batch_size', 'seq_len', 'hidden_size'],
    )
    def _fused_layernorm_qkv_kernel(
        # Pointers to input/output matrices
        input_ptr, gamma_ptr, beta_ptr, 
        q_weight_ptr, k_weight_ptr, v_weight_ptr,
        q_bias_ptr, k_bias_ptr, v_bias_ptr,
        q_out_ptr, k_out_ptr, v_out_ptr,
        # Matrix dimensions
        batch_size, seq_len, hidden_size, head_size, num_heads, num_kv_heads,
        # Strides for efficient memory access
        stride_input_batch, stride_input_seq, stride_input_hidden,
        stride_gamma, stride_beta,
        stride_qw_out, stride_qw_in, 
        stride_kw_out, stride_kw_in,
        stride_vw_out, stride_vw_in,
        stride_qb, stride_kb, stride_vb,
        stride_qout_batch, stride_qout_seq, stride_qout_head, stride_qout_hidden,
        stride_kout_batch, stride_kout_seq, stride_kout_head, stride_kout_hidden,
        stride_vout_batch, stride_vout_seq, stride_vout_head, stride_vout_hidden,
        # Additional parameters
        eps, 
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        NUM_WARPS: tl.constexpr,
        # Flags
        HAS_BETA: tl.constexpr, 
        HAS_QBIAS: tl.constexpr, HAS_KBIAS: tl.constexpr, HAS_VBIAS: tl.constexpr,
        USE_FP16: tl.constexpr, USE_BF16: tl.constexpr
    ):
        """
        Fused kernel for LayerNorm + QKV projection operation.
        
        This kernel efficiently combines LayerNorm and QKV projection by:
        1. Computing mean and variance using parallel reduction
        2. Normalizing the input using the computed statistics
        3. Keeping normalized vectors in registers/shared memory
        4. Directly applying Q, K, V projections on normalized vectors
        
        Args:
            input_ptr: Pointer to input tensor [batch_size, seq_len, hidden_size]
            gamma_ptr: Pointer to LayerNorm scale tensor [hidden_size]
            beta_ptr: Pointer to LayerNorm bias tensor [hidden_size]
            q_weight_ptr: Pointer to Q projection weights [head_size * num_heads, hidden_size]
            k_weight_ptr: Pointer to K projection weights [head_size * num_kv_heads, hidden_size]
            v_weight_ptr: Pointer to V projection weights [head_size * num_kv_heads, hidden_size]
            q_bias_ptr, k_bias_ptr, v_bias_ptr: Pointers to Q, K, V projection biases
            q_out_ptr, k_out_ptr, v_out_ptr: Pointers to Q, K, V output tensors
                [batch_size, seq_len, num_heads, head_size] for Q
                [batch_size, seq_len, num_kv_heads, head_size] for K, V
            batch_size, seq_len, hidden_size: Input tensor dimensions
            head_size: Size of each attention head
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (for GQA support)
            Various strides for navigating tensors efficiently
            eps: Epsilon for numerical stability in LayerNorm
            BLOCK_SIZE_M: Block size for sequence dimension
            BLOCK_SIZE_N: Block size for hidden dimension
            NUM_WARPS: Number of warps for parallelism
            HAS_BETA, HAS_QBIAS, HAS_KBIAS, HAS_VBIAS: Flags for optional components
            USE_FP16, USE_BF16: Flags for different precision modes
        """
        # Get program ID and compute offsets
        pid = tl.program_id(0)
        
        # Compute batch and sequence index from program ID
        batch_id = pid // (seq_len // BLOCK_SIZE_M + 1)
        seq_id = pid % (seq_len // BLOCK_SIZE_M + 1)
        
        # Calculate starting indices in the sequence
        start_seq_idx = seq_id * BLOCK_SIZE_M
        
        # Block dimensions may exceed the actual size, apply bounds checking
        seq_block_size = min(BLOCK_SIZE_M, seq_len - start_seq_idx)
        if seq_block_size <= 0:
            return
        
        # Initialize offsets for the current batch and sequence block
        batch_seq_offset = batch_id * stride_input_batch + start_seq_idx * stride_input_seq
        
        # --------------------- STEP 1: Compute LayerNorm -----------------------
        
        # Create shared memory for accumulating mean and variance
        reduced_mean = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
        reduced_var = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
        
        # Process input in tiles to compute mean
        for n_offset in range(0, hidden_size, BLOCK_SIZE_N):
            # Compute valid size for this tile
            n_size = min(BLOCK_SIZE_N, hidden_size - n_offset)
            
            # Create offsets and masks for loading
            block_offset = batch_seq_offset + n_offset * stride_input_hidden
            m_indices = tl.arange(0, BLOCK_SIZE_M)
            n_indices = tl.arange(0, BLOCK_SIZE_N)
            
            # Create mask for valid sequence and hidden dimensions
            mask = (m_indices[:, None] < seq_block_size) & (n_indices[None, :] < n_size)
            
            # Load input block
            x = tl.load(
                input_ptr + block_offset + 
                m_indices[:, None] * stride_input_seq + 
                n_indices[None, :] * stride_input_hidden,
                mask=mask, other=0.0
            )
            
            # Cast to float32 for stable reduction
            x = x.to(tl.float32)
            
            # Compute partial sum for mean and sum of squares for variance
            reduced_mean += tl.sum(x, axis=1)
            reduced_var += tl.sum(x * x, axis=1)
        
        # Compute mean and variance
        reduced_mean = reduced_mean / hidden_size
        reduced_var = (reduced_var / hidden_size) - (reduced_mean * reduced_mean)
        
        # Add epsilon for numerical stability
        inv_std = 1.0 / tl.sqrt(reduced_var + eps)
        
        # --------------------- STEP 2: Normalize and Project -----------------------
        
        # Initialize output accumulators for Q, K, V projections
        q_out = tl.zeros([BLOCK_SIZE_M, num_heads, head_size], dtype=tl.float32)
        k_out = tl.zeros([BLOCK_SIZE_M, num_kv_heads, head_size], dtype=tl.float32)
        v_out = tl.zeros([BLOCK_SIZE_M, num_kv_heads, head_size], dtype=tl.float32)
        
        # Process input in tiles for normalization and projection
        for n_offset in range(0, hidden_size, BLOCK_SIZE_N):
            # Compute valid size for this tile
            n_size = min(BLOCK_SIZE_N, hidden_size - n_offset)
            
            # Create offsets and masks
            block_offset = batch_seq_offset + n_offset * stride_input_hidden
            m_indices = tl.arange(0, BLOCK_SIZE_M)
            n_indices = tl.arange(0, BLOCK_SIZE_N)
            
            # Create mask for valid sequence and hidden dimensions
            mask = (m_indices[:, None] < seq_block_size) & (n_indices[None, :] < n_size)
            
            # Load input block, LN parameters
            x = tl.load(
                input_ptr + block_offset + 
                m_indices[:, None] * stride_input_seq + 
                n_indices[None, :] * stride_input_hidden,
                mask=mask, other=0.0
            )
            
            # Cast to float32 for stable computation
            x = x.to(tl.float32)
            
            # Load gamma (scale) for LayerNorm
            gamma = tl.load(
                gamma_ptr + n_offset + n_indices * stride_gamma,
                mask=n_indices < n_size, other=0.0
            )
            
            # Normalize input (x_norm = (x - mean) * inv_std)
            x_centered = x - reduced_mean[:, None]
            x_norm = x_centered * inv_std[:, None]
            
            # Apply gamma (scale)
            x_norm = x_norm * gamma[None, :]
            
            # Apply beta (shift) if provided
            if HAS_BETA:
                beta = tl.load(
                    beta_ptr + n_offset + n_indices * stride_beta,
                    mask=n_indices < n_size, other=0.0
                )
                x_norm = x_norm + beta[None, :]
            
            # Apply Q, K, V projections for each head
            # We compute one block of normalized input against multiple blocks of weights
            
            # Loop over heads for Q projection
            for h_idx in range(num_heads):
                head_offset = h_idx * head_size
                
                for d_offset in range(0, head_size, BLOCK_SIZE_N):
                    d_size = min(BLOCK_SIZE_N, head_size - d_offset)
                    d_indices = tl.arange(0, BLOCK_SIZE_N)
                    
                    # Load Q weights for this head
                    q_weight = tl.load(
                        q_weight_ptr + 
                        (head_offset + d_offset) * stride_qw_out + 
                        n_offset * stride_qw_in + 
                        d_indices[:, None] * stride_qw_out + 
                        n_indices[None, :] * stride_qw_in,
                        mask=(d_indices[:, None] < d_size) & (n_indices[None, :] < n_size),
                        other=0.0
                    )
                    
                    # Compute matrix multiplication: x_norm @ q_weight.T
                    q_part = tl.dot(x_norm, tl.trans(q_weight))
                    
                    # Add to Q output accumulator
                    q_out[:, h_idx, d_offset:d_offset+d_size] += q_part[:, :d_size]
            
            # Loop over heads for K, V projections
            for h_idx in range(num_kv_heads):
                head_offset = h_idx * head_size
                
                for d_offset in range(0, head_size, BLOCK_SIZE_N):
                    d_size = min(BLOCK_SIZE_N, head_size - d_offset)
                    d_indices = tl.arange(0, BLOCK_SIZE_N)
                    
                    # Load K weights for this head
                    k_weight = tl.load(
                        k_weight_ptr + 
                        (head_offset + d_offset) * stride_kw_out + 
                        n_offset * stride_kw_in + 
                        d_indices[:, None] * stride_kw_out + 
                        n_indices[None, :] * stride_kw_in,
                        mask=(d_indices[:, None] < d_size) & (n_indices[None, :] < n_size),
                        other=0.0
                    )
                    
                    # Load V weights for this head
                    v_weight = tl.load(
                        v_weight_ptr + 
                        (head_offset + d_offset) * stride_vw_out + 
                        n_offset * stride_vw_in + 
                        d_indices[:, None] * stride_vw_out + 
                        n_indices[None, :] * stride_vw_in,
                        mask=(d_indices[:, None] < d_size) & (n_indices[None, :] < n_size),
                        other=0.0
                    )
                    
                    # Compute matrix multiplications
                    k_part = tl.dot(x_norm, tl.trans(k_weight))
                    v_part = tl.dot(x_norm, tl.trans(v_weight))
                    
                    # Add to K, V output accumulators
                    k_out[:, h_idx, d_offset:d_offset+d_size] += k_part[:, :d_size]
                    v_out[:, h_idx, d_offset:d_offset+d_size] += v_part[:, :d_size]
        
        # --------------------- STEP 3: Add Biases and Store Results -----------------------
        
        # Add Q bias if provided
        if HAS_QBIAS:
            for h_idx in range(num_heads):
                bias_offset = h_idx * head_size
                
                for d_offset in range(0, head_size, BLOCK_SIZE_N):
                    d_size = min(BLOCK_SIZE_N, head_size - d_offset)
                    d_indices = tl.arange(0, BLOCK_SIZE_N)
                    
                    # Load Q bias
                    q_bias = tl.load(
                        q_bias_ptr + bias_offset + d_offset + d_indices * stride_qb,
                        mask=d_indices < d_size, other=0.0
                    )
                    
                    # Add bias to output
                    q_out[:, h_idx, d_offset:d_offset+d_size] += q_bias[None, :d_size]
        
        # Add K, V biases if provided
        if HAS_KBIAS:
            for h_idx in range(num_kv_heads):
                bias_offset = h_idx * head_size
                
                for d_offset in range(0, head_size, BLOCK_SIZE_N):
                    d_size = min(BLOCK_SIZE_N, head_size - d_offset)
                    d_indices = tl.arange(0, BLOCK_SIZE_N)
                    
                    # Load K bias
                    k_bias = tl.load(
                        k_bias_ptr + bias_offset + d_offset + d_indices * stride_kb,
                        mask=d_indices < d_size, other=0.0
                    )
                    
                    # Add bias to output
                    k_out[:, h_idx, d_offset:d_offset+d_size] += k_bias[None, :d_size]
        
        if HAS_VBIAS:
            for h_idx in range(num_kv_heads):
                bias_offset = h_idx * head_size
                
                for d_offset in range(0, head_size, BLOCK_SIZE_N):
                    d_size = min(BLOCK_SIZE_N, head_size - d_offset)
                    d_indices = tl.arange(0, BLOCK_SIZE_N)
                    
                    # Load V bias
                    v_bias = tl.load(
                        v_bias_ptr + bias_offset + d_offset + d_indices * stride_vb,
                        mask=d_indices < d_size, other=0.0
                    )
                    
                    # Add bias to output
                    v_out[:, h_idx, d_offset:d_offset+d_size] += v_bias[None, :d_size]
        
        # Store Q, K, V outputs
        m_indices = tl.arange(0, BLOCK_SIZE_M)
        out_m_mask = m_indices < seq_block_size
        
        # Store Q output
        for h_idx in range(num_heads):
            # Compute output offset for this head
            q_out_head_offset = batch_id * stride_qout_batch + start_seq_idx * stride_qout_seq + h_idx * stride_qout_head
            
            # Store output for each position in the head dimension
            for d_offset in range(0, head_size, BLOCK_SIZE_N):
                d_size = min(BLOCK_SIZE_N, head_size - d_offset)
                d_indices = tl.arange(0, BLOCK_SIZE_N)
                
                out_mask = (m_indices[:, None] < seq_block_size) & (d_indices[None, :] < d_size)
                
                # Cast to appropriate output type
                if USE_FP16:
                    out_data = q_out[:, h_idx, d_offset:d_offset+d_size].to(tl.float16)
                elif USE_BF16:
                    out_data = q_out[:, h_idx, d_offset:d_offset+d_size].to(tl.bfloat16)
                else:
                    out_data = q_out[:, h_idx, d_offset:d_offset+d_size]
                
                # Store Q output
                tl.store(
                    q_out_ptr + q_out_head_offset + d_offset * stride_qout_hidden + 
                    m_indices[:, None] * stride_qout_seq + 
                    d_indices[None, :] * stride_qout_hidden,
                    out_data,
                    mask=out_mask
                )
        
        # Store K, V outputs
        for h_idx in range(num_kv_heads):
            # Compute output offsets for this head
            k_out_head_offset = batch_id * stride_kout_batch + start_seq_idx * stride_kout_seq + h_idx * stride_kout_head
            v_out_head_offset = batch_id * stride_vout_batch + start_seq_idx * stride_vout_seq + h_idx * stride_vout_head
            
            # Store output for each position in the head dimension
            for d_offset in range(0, head_size, BLOCK_SIZE_N):
                d_size = min(BLOCK_SIZE_N, head_size - d_offset)
                d_indices = tl.arange(0, BLOCK_SIZE_N)
                
                out_mask = (m_indices[:, None] < seq_block_size) & (d_indices[None, :] < d_size)
                
                # Cast to appropriate output type
                if USE_FP16:
                    k_out_data = k_out[:, h_idx, d_offset:d_offset+d_size].to(tl.float16)
                    v_out_data = v_out[:, h_idx, d_offset:d_offset+d_size].to(tl.float16)
                elif USE_BF16:
                    k_out_data = k_out[:, h_idx, d_offset:d_offset+d_size].to(tl.bfloat16)
                    v_out_data = v_out[:, h_idx, d_offset:d_offset+d_size].to(tl.bfloat16)
                else:
                    k_out_data = k_out[:, h_idx, d_offset:d_offset+d_size]
                    v_out_data = v_out[:, h_idx, d_offset:d_offset+d_size]
                
                # Store K, V outputs
                tl.store(
                    k_out_ptr + k_out_head_offset + d_offset * stride_kout_hidden + 
                    m_indices[:, None] * stride_kout_seq + 
                    d_indices[None, :] * stride_kout_hidden,
                    k_out_data,
                    mask=out_mask
                )
                
                tl.store(
                    v_out_ptr + v_out_head_offset + d_offset * stride_vout_hidden + 
                    m_indices[:, None] * stride_vout_seq + 
                    d_indices[None, :] * stride_vout_hidden,
                    v_out_data,
                    mask=out_mask
                )


#-----------------------------------------------------------------------------
# Python Wrapper Functions
#-----------------------------------------------------------------------------

def triton_fused_layernorm_qkv(
    hidden_states: torch.Tensor,
    layernorm_weight: torch.Tensor,
    layernorm_bias: Optional[torch.Tensor],
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    query_bias: Optional[torch.Tensor] = None,
    key_bias: Optional[torch.Tensor] = None,
    value_bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    num_heads: int = 0,
    num_kv_heads: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused LayerNorm + QKV projection using optimized Triton kernels.
    
    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        layernorm_weight: LayerNorm scale (gamma) tensor of shape [hidden_size]
        layernorm_bias: LayerNorm shift (beta) tensor of shape [hidden_size], can be None
        query_weight: Query projection weight of shape [hidden_size, hidden_size]
        key_weight: Key projection weight of shape [hidden_size, hidden_size]
        value_weight: Value projection weight of shape [hidden_size, hidden_size]
        query_bias: Query projection bias of shape [hidden_size], can be None
        key_bias: Key projection bias of shape [hidden_size], can be None
        value_bias: Value projection bias of shape [hidden_size], can be None
        eps: Small constant for numerical stability
        num_heads: Number of attention heads; if 0, inferred from hidden_size
        num_kv_heads: Number of key/value heads; if None, same as num_heads (for GQA/MQA)
        
    Returns:
        Tuple of (query, key, value) tensors after LayerNorm + projection
    """
    # Fall back to PyTorch if Triton is not available
    if not HAS_TRITON:
        return pytorch_fused_layernorm_qkv(
            hidden_states, layernorm_weight, layernorm_bias,
            query_weight, key_weight, value_weight,
            query_bias, key_bias, value_bias,
            eps, num_heads, num_kv_heads
        )
    
    # Get tensor dimensions
    batch_size, seq_len, hidden_size = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype
    
    # Auto-detect number of heads if not provided
    if num_heads == 0:
        # Try common head sizes (64, 80, 128)
        for head_size in [64, 80, 128]:
            if hidden_size % head_size == 0:
                num_heads = hidden_size // head_size
                break
        
        # Fallback if no common head size fits
        if num_heads == 0:
            # Just divide by 64 and adjust head size accordingly
            num_heads = max(1, hidden_size // 64)
    
    # Set num_kv_heads to num_heads if not provided (standard MHA)
    if num_kv_heads is None:
        num_kv_heads = num_heads
    
    # Calculate head dimension
    head_size = hidden_size // num_heads
    
    # Ensure kernel compatibility by checking if hidden_size is divisible by head_size
    if hidden_size % head_size != 0:
        raise ValueError(f"Hidden size ({hidden_size}) must be divisible by head size ({head_size})")
    
    # Ensure KV heads are compatible
    kv_head_size = hidden_size // num_kv_heads
    if hidden_size % kv_head_size != 0:
        raise ValueError(f"Hidden size ({hidden_size}) must be divisible by KV head count ({num_kv_heads})")
    
    # Reshape weights for proper kernel layout
    # For Q: [hidden_size, hidden_size] -> [num_heads * head_size, hidden_size]
    # For K,V: [hidden_size, hidden_size] -> [num_kv_heads * kv_head_size, hidden_size]
    q_weight_reshaped = query_weight.view(num_heads, head_size, hidden_size)
    k_weight_reshaped = key_weight.view(num_kv_heads, kv_head_size, hidden_size)
    v_weight_reshaped = value_weight.view(num_kv_heads, kv_head_size, hidden_size)
    
    # Reshape biases if provided
    q_bias_reshaped = None
    k_bias_reshaped = None
    v_bias_reshaped = None
    
    if query_bias is not None:
        q_bias_reshaped = query_bias.view(num_heads, head_size)
    
    if key_bias is not None:
        k_bias_reshaped = key_bias.view(num_kv_heads, kv_head_size)
    
    if value_bias is not None:
        v_bias_reshaped = value_bias.view(num_kv_heads, kv_head_size)
    
    # Create output tensors
    query = torch.empty((batch_size, seq_len, num_heads, head_size), device=device, dtype=dtype)
    key = torch.empty((batch_size, seq_len, num_kv_heads, kv_head_size), device=device, dtype=dtype)
    value = torch.empty((batch_size, seq_len, num_kv_heads, kv_head_size), device=device, dtype=dtype)
    
    # Determine if we need to use FP16 or BF16
    use_fp16 = dtype == torch.float16
    use_bf16 = dtype == torch.bfloat16
    
    # Calculate grid dimensions
    grid = (batch_size * ((seq_len + 128 - 1) // 128),)
    
    # Compute strides for efficient memory access
    stride_input_batch = hidden_states.stride(0)
    stride_input_seq = hidden_states.stride(1)
    stride_input_hidden = 1  # Assuming contiguous in innermost dimension
    
    stride_gamma = 1  # Assuming contiguous
    stride_beta = 1 if layernorm_bias is not None else 0
    
    # For query weight
    stride_qw_out = query_weight.stride(0)  # Stride for output dimension
    stride_qw_in = 1  # Stride for input dimension
    
    # For key weight
    stride_kw_out = key_weight.stride(0)
    stride_kw_in = 1
    
    # For value weight
    stride_vw_out = value_weight.stride(0)
    stride_vw_in = 1
    
    # For biases
    stride_qb = 1 if query_bias is not None else 0
    stride_kb = 1 if key_bias is not None else 0
    stride_vb = 1 if value_bias is not None else 0
    
    # For output tensors
    stride_qout_batch = query.stride(0)
    stride_qout_seq = query.stride(1)
    stride_qout_head = query.stride(2)
    stride_qout_hidden = 1  # Assuming contiguous in innermost dimension
    
    stride_kout_batch = key.stride(0)
    stride_kout_seq = key.stride(1)
    stride_kout_head = key.stride(2)
    stride_kout_hidden = 1
    
    stride_vout_batch = value.stride(0)
    stride_vout_seq = value.stride(1)
    stride_vout_head = value.stride(2)
    stride_vout_hidden = 1
    
    # Launch the fused kernel
    _fused_layernorm_qkv_kernel[grid](
        # Pointers to input/output matrices
        hidden_states, layernorm_weight, 
        layernorm_bias if layernorm_bias is not None else 0,
        query_weight, key_weight, value_weight,
        query_bias if query_bias is not None else 0,
        key_bias if key_bias is not None else 0,
        value_bias if value_bias is not None else 0,
        query, key, value,
        # Matrix dimensions
        batch_size, seq_len, hidden_size, head_size, num_heads, num_kv_heads,
        # Strides for efficient memory access
        stride_input_batch, stride_input_seq, stride_input_hidden,
        stride_gamma, stride_beta,
        stride_qw_out, stride_qw_in,
        stride_kw_out, stride_kw_in,
        stride_vw_out, stride_vw_in,
        stride_qb, stride_kb, stride_vb,
        stride_qout_batch, stride_qout_seq, stride_qout_head, stride_qout_hidden,
        stride_kout_batch, stride_kout_seq, stride_kout_head, stride_kout_hidden,
        stride_vout_batch, stride_vout_seq, stride_vout_head, stride_vout_hidden,
        # Additional parameters
        eps,
        # Meta-parameters
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        NUM_WARPS=8,
        # Flags
        HAS_BETA=layernorm_bias is not None,
        HAS_QBIAS=query_bias is not None,
        HAS_KBIAS=key_bias is not None,
        HAS_VBIAS=value_bias is not None,
        USE_FP16=use_fp16,
        USE_BF16=use_bf16
    )
    
    return query, key, value


def pytorch_fused_layernorm_qkv(
    hidden_states: torch.Tensor,
    layernorm_weight: torch.Tensor,
    layernorm_bias: Optional[torch.Tensor],
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    query_bias: Optional[torch.Tensor] = None,
    key_bias: Optional[torch.Tensor] = None,
    value_bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    num_heads: int = 0,
    num_kv_heads: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of fused LayerNorm + QKV projection (used when Triton is not available).
    
    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        layernorm_weight: LayerNorm scale (gamma) tensor of shape [hidden_size]
        layernorm_bias: LayerNorm shift (beta) tensor of shape [hidden_size], can be None
        query_weight: Query projection weight of shape [hidden_size, hidden_size]
        key_weight: Key projection weight of shape [hidden_size, hidden_size]
        value_weight: Value projection weight of shape [hidden_size, hidden_size]
        query_bias: Query projection bias of shape [hidden_size], can be None
        key_bias: Key projection bias of shape [hidden_size], can be None
        value_bias: Value projection bias of shape [hidden_size], can be None
        eps: Small constant for numerical stability
        num_heads: Number of attention heads; if 0, inferred from hidden_size
        num_kv_heads: Number of key/value heads; if None, same as num_heads (for GQA/MQA)
        
    Returns:
        Tuple of (query, key, value) tensors after LayerNorm + projection
    """
    # Get tensor dimensions
    batch_size, seq_len, hidden_size = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype
    
    # Auto-detect number of heads if not provided
    if num_heads == 0:
        # Try common head sizes (64, 80, 128)
        for head_size in [64, 80, 128]:
            if hidden_size % head_size == 0:
                num_heads = hidden_size // head_size
                break
        
        # Fallback if no common head size fits
        if num_heads == 0:
            # Just divide by 64 and adjust head size accordingly
            num_heads = max(1, hidden_size // 64)
    
    # Set num_kv_heads to num_heads if not provided (standard MHA)
    if num_kv_heads is None:
        num_kv_heads = num_heads
    
    # Calculate head dimensions
    head_size = hidden_size // num_heads
    kv_head_size = hidden_size // num_kv_heads
    
    # Step 1: LayerNorm
    # Calculate mean and variance along last dimension
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = ((hidden_states - mean) ** 2).mean(dim=-1, keepdim=True)
    
    # Normalize
    hidden_norm = (hidden_states - mean) / torch.sqrt(var + eps)
    
    # Apply scale and shift
    if layernorm_bias is not None:
        hidden_norm = layernorm_weight * hidden_norm + layernorm_bias
    else:
        hidden_norm = layernorm_weight * hidden_norm
    
    # Step 2: Apply QKV projections
    # Query projection
    query = F.linear(hidden_norm, query_weight, query_bias)
    # Key projection
    key = F.linear(hidden_norm, key_weight, key_bias)
    # Value projection
    value = F.linear(hidden_norm, value_weight, value_bias)
    
    # Reshape outputs to multi-head format
    query = query.view(batch_size, seq_len, num_heads, head_size)
    key = key.view(batch_size, seq_len, num_kv_heads, kv_head_size)
    value = value.view(batch_size, seq_len, num_kv_heads, kv_head_size)
    
    return query, key, value


#-----------------------------------------------------------------------------
# Benchmarking Functions
#-----------------------------------------------------------------------------

def benchmark_fused_layernorm_qkv(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int = 0,
    num_kv_heads: Optional[int] = None,
    device: str = "cuda",
    iterations: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Benchmark the performance of fused LayerNorm + QKV projection.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden size
        num_heads: Number of attention heads (if 0, will be inferred)
        num_kv_heads: Number of key/value heads (if None, same as num_heads)
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
            "triton_fused_ms": 0.0,
            "separate_pytorch_ms": 0.0,
            "speedup": 0.0
        }
    
    if num_heads == 0:
        # Try to infer a reasonable number of heads
        for head_size in [64, 80, 128]:
            if hidden_size % head_size == 0:
                num_heads = hidden_size // head_size
                break
        
        # Fallback
        if num_heads == 0:
            num_heads = max(1, hidden_size // 64)
    
    if num_kv_heads is None:
        num_kv_heads = num_heads
    
    import time
    
    # Create test tensors
    hidden_states = torch.randn((batch_size, seq_len, hidden_size), device=device)
    
    # LayerNorm parameters
    ln_weight = torch.randn((hidden_size,), device=device)
    ln_bias = torch.randn((hidden_size,), device=device)
    
    # QKV projection parameters
    q_weight = torch.randn((hidden_size, hidden_size), device=device)
    k_weight = torch.randn((hidden_size, hidden_size), device=device)
    v_weight = torch.randn((hidden_size, hidden_size), device=device)
    
    q_bias = torch.randn((hidden_size,), device=device)
    k_bias = torch.randn((hidden_size,), device=device)
    v_bias = torch.randn((hidden_size,), device=device)
    
    # Warm up
    for _ in range(warmup):
        if HAS_TRITON:
            _ = triton_fused_layernorm_qkv(
                hidden_states, ln_weight, ln_bias,
                q_weight, k_weight, v_weight,
                q_bias, k_bias, v_bias,
                num_heads=num_heads, num_kv_heads=num_kv_heads
            )
        
        # Separate PyTorch operations
        norm = F.layer_norm(hidden_states, (hidden_size,), ln_weight, ln_bias)
        _ = F.linear(norm, q_weight, q_bias)
        _ = F.linear(norm, k_weight, k_bias)
        _ = F.linear(norm, v_weight, v_bias)
    
    # Benchmark separate PyTorch operations
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    for _ in range(iterations):
        norm = F.layer_norm(hidden_states, (hidden_size,), ln_weight, ln_bias)
        q = F.linear(norm, q_weight, q_bias)
        k = F.linear(norm, k_weight, k_bias)
        v = F.linear(norm, v_weight, v_bias)
    
    torch.cuda.synchronize() if device == "cuda" else None
    pytorch_time = (time.time() - start_time) * 1000 / iterations  # ms
    
    # Benchmark triton fused implementation
    triton_time = 0.0
    if HAS_TRITON:
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()
        
        for _ in range(iterations):
            _ = triton_fused_layernorm_qkv(
                hidden_states, ln_weight, ln_bias,
                q_weight, k_weight, v_weight,
                q_bias, k_bias, v_bias,
                num_heads=num_heads, num_kv_heads=num_kv_heads
            )
        
        torch.cuda.synchronize() if device == "cuda" else None
        triton_time = (time.time() - start_time) * 1000 / iterations  # ms
        
        speedup = pytorch_time / triton_time
    else:
        triton_time = pytorch_time
        speedup = 1.0
    
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "triton_fused_ms": triton_time,
        "separate_pytorch_ms": pytorch_time,
        "speedup": speedup
    }


def compare_with_unfused_implementation(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int = 0,
    num_kv_heads: Optional[int] = None,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compare the results of fused LayerNorm + QKV projection with unfused operations.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden size
        num_heads: Number of attention heads (if 0, will be inferred)
        num_kv_heads: Number of key/value heads (if None, same as num_heads)
        device: Device to run comparison on
        
    Returns:
        Dictionary with comparison results
    """
    # Skip if CUDA is not available and device is cuda
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping comparison")
        return {
            "max_difference_q": 0.0,
            "max_difference_k": 0.0,
            "max_difference_v": 0.0,
            "is_correct": False
        }
    
    # Auto-detect number of heads if not provided
    if num_heads == 0:
        for head_size in [64, 80, 128]:
            if hidden_size % head_size == 0:
                num_heads = hidden_size // head_size
                break
        
        # Fallback
        if num_heads == 0:
            num_heads = max(1, hidden_size // 64)
    
    if num_kv_heads is None:
        num_kv_heads = num_heads
    
    # Create test tensors
    hidden_states = torch.randn((batch_size, seq_len, hidden_size), device=device)
    
    # LayerNorm parameters
    ln_weight = torch.randn((hidden_size,), device=device)
    ln_bias = torch.randn((hidden_size,), device=device)
    
    # QKV projection parameters
    q_weight = torch.randn((hidden_size, hidden_size), device=device)
    k_weight = torch.randn((hidden_size, hidden_size), device=device)
    v_weight = torch.randn((hidden_size, hidden_size), device=device)
    
    q_bias = torch.randn((hidden_size,), device=device)
    k_bias = torch.randn((hidden_size,), device=device)
    v_bias = torch.randn((hidden_size,), device=device)
    
    # Calculate head sizes
    head_size = hidden_size // num_heads
    kv_head_size = hidden_size // num_kv_heads
    
    # Compute with separate PyTorch operations
    norm = F.layer_norm(hidden_states, (hidden_size,), ln_weight, ln_bias)
    pytorch_q = F.linear(norm, q_weight, q_bias).view(batch_size, seq_len, num_heads, head_size)
    pytorch_k = F.linear(norm, k_weight, k_bias).view(batch_size, seq_len, num_kv_heads, kv_head_size)
    pytorch_v = F.linear(norm, v_weight, v_bias).view(batch_size, seq_len, num_kv_heads, kv_head_size)
    
    # Compute with fused implementation
    if HAS_TRITON:
        triton_q, triton_k, triton_v = triton_fused_layernorm_qkv(
            hidden_states, ln_weight, ln_bias,
            q_weight, k_weight, v_weight,
            q_bias, k_bias, v_bias,
            num_heads=num_heads, num_kv_heads=num_kv_heads
        )
    else:
        # Fall back to our PyTorch implementation
        triton_q, triton_k, triton_v = pytorch_fused_layernorm_qkv(
            hidden_states, ln_weight, ln_bias,
            q_weight, k_weight, v_weight,
            q_bias, k_bias, v_bias,
            num_heads=num_heads, num_kv_heads=num_kv_heads
        )
    
    # Compare outputs
    q_diff = torch.max(torch.abs(pytorch_q - triton_q)).item()
    k_diff = torch.max(torch.abs(pytorch_k - triton_k)).item()
    v_diff = torch.max(torch.abs(pytorch_v - triton_v)).item()
    
    # Determine if results are correct (within tolerance)
    tolerance = 1e-3
    is_correct = (q_diff < tolerance) and (k_diff < tolerance) and (v_diff < tolerance)
    
    return {
        "max_difference_q": q_diff,
        "max_difference_k": k_diff,
        "max_difference_v": v_diff,
        "is_correct": is_correct,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads
    }


def profile_memory_usage(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int = 0,
    num_kv_heads: Optional[int] = None,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Profile memory usage of fused LayerNorm + QKV projection compared to unfused operations.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden size
        num_heads: Number of attention heads (if 0, will be inferred)
        num_kv_heads: Number of key/value heads (if None, same as num_heads)
        device: Device to run profiling on
        
    Returns:
        Dictionary with memory usage statistics
    """
    # Skip if CUDA is not available and device is cuda
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping profiling")
        return {
            "unfused_memory_mb": 0.0,
            "fused_memory_mb": 0.0,
            "memory_saving_percent": 0.0
        }
    
    # Auto-detect number of heads if not provided
    if num_heads == 0:
        for head_size in [64, 80, 128]:
            if hidden_size % head_size == 0:
                num_heads = hidden_size // head_size
                break
        
        # Fallback
        if num_heads == 0:
            num_heads = max(1, hidden_size // 64)
    
    if num_kv_heads is None:
        num_kv_heads = num_heads
    
    # Create test tensors
    hidden_states = torch.randn((batch_size, seq_len, hidden_size), device=device)
    
    # LayerNorm parameters
    ln_weight = torch.randn((hidden_size,), device=device)
    ln_bias = torch.randn((hidden_size,), device=device)
    
    # QKV projection parameters
    q_weight = torch.randn((hidden_size, hidden_size), device=device)
    k_weight = torch.randn((hidden_size, hidden_size), device=device)
    v_weight = torch.randn((hidden_size, hidden_size), device=device)
    
    q_bias = torch.randn((hidden_size,), device=device)
    k_bias = torch.randn((hidden_size,), device=device)
    v_bias = torch.randn((hidden_size,), device=device)
    
    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Run unfused implementation
    norm = F.layer_norm(hidden_states, (hidden_size,), ln_weight, ln_bias)
    q = F.linear(norm, q_weight, q_bias)
    k = F.linear(norm, k_weight, k_bias)
    v = F.linear(norm, v_weight, v_bias)
    
    # Measure memory usage
    if device == "cuda":
        unfused_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    else:
        unfused_memory = 0.0
    
    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Run fused implementation
    if HAS_TRITON:
        _ = triton_fused_layernorm_qkv(
            hidden_states, ln_weight, ln_bias,
            q_weight, k_weight, v_weight,
            q_bias, k_bias, v_bias,
            num_heads=num_heads, num_kv_heads=num_kv_heads
        )
    else:
        _ = pytorch_fused_layernorm_qkv(
            hidden_states, ln_weight, ln_bias,
            q_weight, k_weight, v_weight,
            q_bias, k_bias, v_bias,
            num_heads=num_heads, num_kv_heads=num_kv_heads
        )
    
    # Measure memory usage
    if device == "cuda":
        fused_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    else:
        fused_memory = 0.0
    
    # Calculate memory savings
    memory_saving = unfused_memory - fused_memory
    memory_saving_percent = (memory_saving / unfused_memory) * 100 if unfused_memory > 0 else 0.0
    
    return {
        "unfused_memory_mb": unfused_memory,
        "fused_memory_mb": fused_memory,
        "memory_saving_mb": memory_saving,
        "memory_saving_percent": memory_saving_percent,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads
    }


def flash_compatible_wrapper(
    hidden_states: torch.Tensor,
    layernorm_weight: torch.Tensor,
    layernorm_bias: Optional[torch.Tensor],
    qkv_weight: torch.Tensor,  # Combined weight [3*hidden_size, hidden_size]
    qkv_bias: Optional[torch.Tensor] = None,  # Combined bias [3*hidden_size]
    eps: float = 1e-5,
    num_heads: int = 0,
    num_kv_heads: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adapter function for compatibility with Flash Attention models using combined QKV weights.
    
    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        layernorm_weight: LayerNorm scale tensor of shape [hidden_size]
        layernorm_bias: LayerNorm shift tensor of shape [hidden_size], can be None
        qkv_weight: Combined QKV projection weight of shape [3*hidden_size, hidden_size]
        qkv_bias: Combined QKV projection bias of shape [3*hidden_size], can be None
        eps: Small constant for numerical stability
        num_heads: Number of attention heads; if 0, inferred from hidden_size
        num_kv_heads: Number of key/value heads; if None, same as num_heads (for GQA/MQA)
        
    Returns:
        Tuple of (query, key, value) tensors after LayerNorm + projection
    """
    hidden_size = hidden_states.shape[-1]
    
    # Split combined QKV weights
    q_weight, k_weight, v_weight = torch.split(qkv_weight, hidden_size, dim=0)
    
    # Split combined QKV bias if provided
    q_bias, k_bias, v_bias = None, None, None
    if qkv_bias is not None:
        q_bias, k_bias, v_bias = torch.split(qkv_bias, hidden_size, dim=0)
    
    # Call the main function
    return triton_fused_layernorm_qkv(
        hidden_states, layernorm_weight, layernorm_bias,
        q_weight, k_weight, v_weight,
        q_bias, k_bias, v_bias,
        eps, num_heads, num_kv_heads
    )


def ring_compatible_wrapper(
    hidden_states: torch.Tensor,
    layernorm_weight: torch.Tensor,
    layernorm_bias: Optional[torch.Tensor],
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_bias: Optional[torch.Tensor] = None,
    k_bias: Optional[torch.Tensor] = None,
    v_bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    num_heads: int = 0,
    num_kv_heads: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adapter function for compatibility with Ring Attention models.
    
    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        layernorm_weight: LayerNorm scale tensor of shape [hidden_size]
        layernorm_bias: LayerNorm shift tensor of shape [hidden_size], can be None
        q_weight: Query projection weight [hidden_size, hidden_size]
        k_weight: Key projection weight [hidden_size, hidden_size]
        v_weight: Value projection weight [hidden_size, hidden_size]
        q_bias, k_bias, v_bias: Q, K, V projection biases
        eps: Small constant for numerical stability
        num_heads: Number of attention heads; if 0, inferred from hidden_size
        num_kv_heads: Number of key/value heads; if None, same as num_heads (for GQA/MQA)
        
    Returns:
        Tuple of (query, key, value) tensors with shape compatible for Ring Attention
    """
    q, k, v = triton_fused_layernorm_qkv(
        hidden_states, layernorm_weight, layernorm_bias,
        q_weight, k_weight, v_weight,
        q_bias, k_bias, v_bias,
        eps, num_heads, num_kv_heads
    )
    
    # Reshape from [batch, seq, heads, head_dim] to [batch, heads, seq, head_dim] for Ring Attention
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    
    return q, k, v