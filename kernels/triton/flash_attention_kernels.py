"""
Flash Attention 3 Triton Kernels.

This module implements optimized Triton kernels for Flash Attention 3 algorithm.
These kernels provide highly efficient attention computation with better memory
scaling and performance characteristics compared to standard implementations.

Key features:
- Block-based tiling for memory efficiency
- Fused operations to minimize memory transfers
- Support for causal masking
- Backward pass optimizations

For systems without Triton, PyTorch fallback implementations are provided.
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# Try to import Triton, but gracefully handle if not available
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Triton not available, using PyTorch fallback for Flash Attention")


#-----------------------------------------------------------------------------
# Core Flash Attention Triton Kernels
#-----------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    @triton.autotune(
        configs=[
            # For short sequences - smaller block sizes but more parallelism
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_DMODEL': 32}, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_DMODEL': 32}, num_warps=4),
            # For medium sequences - balanced blocks
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DMODEL': 32}, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DMODEL': 64}, num_warps=8),
            # For long sequences - larger blocks for better reuse
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_DMODEL': 64}, num_warps=8),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_DMODEL': 64}, num_warps=16),
        ],
        key=['batch_size', 'seq_len', 'num_heads', 'head_dim', 'CAUSAL', 'USE_MASK'],
    )
    def _flash_attention_forward_kernel(
        # Pointers to matrices
        q_ptr, k_ptr, v_ptr, o_ptr, 
        # Attention mask (optional)
        mask_ptr,
        # Matrix dimensions
        batch_size, seq_len, num_heads, head_dim,
        # Strides for different dimensions
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_km, stride_kd,
        stride_vb, stride_vh, stride_vm, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_maskb, stride_maskh, stride_maskm,
        # Other parameters
        scale, 
        # Meta-parameters
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr,
        USE_MASK: tl.constexpr,
        # FP8 related meta-parameters
        IS_FP8: tl.constexpr,
        USE_ORTHO_MAT: tl.constexpr,
        STORE_L_M: tl.constexpr, # Flag to control storing L/m
        ortho_matrix_ptr, # Optional pointer to orthogonal matrix for FP8
        # Pointers for L and M statistics (optional, only needed if STORE_L_M)
        l_ptr, m_ptr, 
        stride_lb, stride_lh, stride_lm, # L strides
        stride_mb, stride_mh, stride_mm, # m strides
    ):
        """
        Compute flash attention for a block of the output (Flash Attention 3).
        
        This kernel implements the Flash Attention algorithm with tiled block-based computation
        to avoid materializing the full attention matrix. It processes blocks of the sequence
        length to keep memory usage proportional to sqrt(N) rather than N^2.
        
        The implementation includes:
        - Tiled matrix multiplication for efficient QK^T computation
        - Progressive softmax algorithm for numerical stability
        - Optimized memory access patterns for coalesced reads/writes
        - Support for causal masking with early termination
        - Flexible attention mask handling
        
        Args:
            q_ptr: Pointer to query tensor [batch_size, seq_len, num_heads, head_dim]
            k_ptr: Pointer to key tensor [batch_size, seq_len, num_heads, head_dim]
            v_ptr: Pointer to value tensor [batch_size, seq_len, num_heads, head_dim]
            o_ptr: Pointer to output tensor [batch_size, seq_len, num_heads, head_dim]
            mask_ptr: Pointer to mask tensor (optional)
            batch_size, seq_len, num_heads, head_dim: Tensor dimensions
            Various strides for navigating tensors efficiently
            scale: Scaling factor for attention scores (typically 1/sqrt(head_dim))
            BLOCK_M: Number of parallel query vectors to process
            BLOCK_DMODEL: Block size for the head dimension
            BLOCK_N: Block size for processing keys and values
            CAUSAL: Whether to apply causal masking
            USE_MASK: Whether to apply an attention mask
            IS_FP8: Whether FP8 precision is used
            USE_ORTHO_MAT: Whether to apply orthogonal matrix transformation for FP8
            ortho_matrix_ptr: Pointer to the orthogonal matrix (FP32) if USE_ORTHO_MAT is True
            STORE_L_M: Whether to store L and m statistics for backward pass
            l_ptr, m_ptr: Pointers to L and M tensors [batch_size, num_heads, seq_len]
            stride_lb, stride_lh, stride_lm: Strides for L tensor
            stride_mb, stride_mh, stride_mm: Strides for M tensor
        """
        # Compute which part of the output this program should compute
        pid_batch = tl.program_id(0)  # Batch index
        pid_head = tl.program_id(1)   # Head index
        pid_m = tl.program_id(2)      # Query vector index
        
        # Initialize offsets for q, k, v, and o matrices
        q_offset = (pid_batch * stride_qb + 
                   pid_head * stride_qh + 
                   pid_m * BLOCK_M * stride_qm)
        k_offset = (pid_batch * stride_kb + 
                   pid_head * stride_kh)
        v_offset = (pid_batch * stride_vb + 
                   pid_head * stride_vh)
        o_offset = (pid_batch * stride_ob + 
                   pid_head * stride_oh +
                   pid_m * BLOCK_M * stride_om)
        
        # Load q block [BLOCK_M, BLOCK_DMODEL]
        q_block_ptr = q_ptr + q_offset
        # Compute range of valid queries for this block
        q_valid_range = min(BLOCK_M, seq_len - pid_m * BLOCK_M)
        # Determine data type for loading (FP8 or standard)
        # TODO: Ensure tl.float8e4m3nv is the correct type if using E4M3
        tl_dtype = tl.float8e4m3fn if IS_FP8 else q_ptr.dtype.element_ty 
        
        # Create mask for valid queries and dimensions
        q_mask = (tl.arange(0, BLOCK_M)[:, None] < q_valid_range) & (tl.arange(0, BLOCK_DMODEL)[None, :] < head_dim)
        # Load query block with masking
        q_block = tl.load(
            q_block_ptr + (tl.arange(0, BLOCK_M)[:, None] * stride_qm +
                           tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qd),
            mask=q_mask,
            other=0.0,
            eviction_policy="evict_first" # Hint for Triton scheduler
        ).to(tl.float32) # Always cast to float32 for internal computation
        
        # Apply orthogonal transformation if FP8 and enabled
        if IS_FP8 and USE_ORTHO_MAT:
            # Load the orthogonal matrix block [BLOCK_DMODEL, BLOCK_DMODEL]
            # Assuming ortho_matrix is [head_dim, head_dim] and stored contiguously
            ortho_offs_d1 = tl.arange(0, BLOCK_DMODEL)
            ortho_offs_d2 = tl.arange(0, BLOCK_DMODEL)
            ortho_mask = (ortho_offs_d1[:, None] < head_dim) & (ortho_offs_d2[None, :] < head_dim)
            ortho_block = tl.load(
                ortho_matrix_ptr + ortho_offs_d1[:, None] * head_dim + ortho_offs_d2[None, :],
                mask=ortho_mask,
                other=0.0
            ) # Matrix is expected to be FP32
            
            # Apply transformation: q_block = q_block @ ortho_block
            # Ensure shapes are compatible: [BLOCK_M, BLOCK_DMODEL] @ [BLOCK_DMODEL, BLOCK_DMODEL]
            # Note: Triton matmul expects specific layouts/dtypes, may need casting/transposing
            q_block = tl.dot(q_block, ortho_block, out_dtype=tl.float32)
        
        # Initialize accumulators for stable softmax algorithm
        # o: Output accumulator
        # m_i: Running maximum for numerical stability
        # l_i: Running sum of exponentials for normalization
        o = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        
        # Define offsets for the L and M tensors for the current block if storing
        l_block_offset = 0
        m_block_offset = 0
        if STORE_L_M:
            l_block_offset = (pid_batch * stride_lb + pid_head * stride_lh + pid_m * BLOCK_M * stride_lm)
            m_block_offset = (pid_batch * stride_mb + pid_head * stride_mh + pid_m * BLOCK_M * stride_mm)
        
        # Loop over k, v blocks - this is the key to Flash Attention's memory efficiency
        # We process the sequence in chunks to avoid materializing the full attention matrix
        for start_n in range(0, seq_len, BLOCK_N):
            # Early termination optimization for causal masking
            # If we're processing a block that's entirely masked out, we can skip it
            if CAUSAL and start_n > (pid_m + 1) * BLOCK_M - 1:
                break
                
            # Calculate valid range for this key/value block
            kv_valid_range = min(BLOCK_N, seq_len - start_n)
            # Create mask for valid keys/values and dimensions
            kv_mask = (tl.arange(0, BLOCK_N)[:, None] < kv_valid_range) & (tl.arange(0, BLOCK_DMODEL)[None, :] < head_dim)
            
            # Load k, v blocks for current chunk
            k_block_ptr = k_ptr + k_offset + start_n * stride_km
            v_block_ptr = v_ptr + v_offset + start_n * stride_vm
            
            # Load key block with masking for boundary conditions
            k_block = tl.load(
                k_block_ptr + (tl.arange(0, BLOCK_N)[:, None] * stride_km +
                               tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kd),
                mask=kv_mask,
                other=0.0,
                eviction_policy="evict_first"
            ).to(tl.float32) # Cast to float32
            
            # Apply orthogonal transformation if FP8 and enabled
            if IS_FP8 and USE_ORTHO_MAT:
                # Reload ortho_block if not already loaded or kept in registers
                # Note: Re-loading might be inefficient; ideally, it's loaded once per kernel
                # Or, if it's small enough, kept entirely in registers/shared memory.
                # For simplicity here, we reload. Optimization needed.
                ortho_offs_d1 = tl.arange(0, BLOCK_DMODEL)
                ortho_offs_d2 = tl.arange(0, BLOCK_DMODEL)
                ortho_mask = (ortho_offs_d1[:, None] < head_dim) & (ortho_offs_d2[None, :] < head_dim)
                ortho_block = tl.load(
                    ortho_matrix_ptr + ortho_offs_d1[:, None] * head_dim + ortho_offs_d2[None, :],
                    mask=ortho_mask,
                    other=0.0
                ) # Matrix is expected to be FP32
                k_block = tl.dot(k_block, ortho_block, out_dtype=tl.float32)
            
            # Load value block with masking for boundary conditions
            v_block = tl.load(
                v_block_ptr + (tl.arange(0, BLOCK_N)[:, None] * stride_vm +
                               tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vd),
                mask=kv_mask,
                other=0.0,
                eviction_policy="evict_first"
            ).to(tl.float32) # Cast V to FP32 as well, necessary for accumulation
            
            # Compute attention scores for this block: QK^T
            # Optimized matrix multiplication using Triton's tl.dot operation
            scores = tl.dot(q_block, tl.trans(k_block))
            # Apply scaling factor (1/sqrt(d_k)) for stable training
            scores = scores * scale
            
            # Apply causal masking if needed (for autoregressive models)
            if CAUSAL:
                # Compute absolute position indices for queries and keys
                row_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                col_ids = start_n + tl.arange(0, BLOCK_N)
                # Create causal mask where row_ids >= col_ids (upper triangular including diagonal)
                # This ensures each token can only attend to itself and previous tokens
                causal_mask = row_ids[:, None] >= col_ids[None, :]
                # Apply mask by zeroing out future positions and setting them to a large negative value
                scores = scores * causal_mask + (-1e9) * ~causal_mask
                
            # Apply attention mask if provided (e.g., for padding tokens)
            if USE_MASK:
                # Calculate mask offset
                mask_block_ptr = mask_ptr + (pid_batch * stride_maskb +
                                            pid_head * stride_maskh +
                                            pid_m * BLOCK_M * stride_maskm)
                
                # Create mask for valid attention mask values
                mask_load_mask = (tl.arange(0, BLOCK_M)[:, None] < q_valid_range) & (tl.arange(0, BLOCK_N)[None, :] < kv_valid_range)
                
                # Load attention mask block
                mask_block = tl.load(
                    mask_block_ptr + (tl.arange(0, BLOCK_M)[:, None] * stride_maskm +
                                      tl.arange(0, BLOCK_N)[None, :]),
                    mask=mask_load_mask,
                    other=0.0
                )
                
                # Apply attention mask (0 → -inf, 1 → keep value)
                scores = scores * mask_block + (-1e9) * (1.0 - mask_block)
                
            # Implement the stable softmax algorithm from the Flash Attention paper
            
            # 1. Find current block's max score per query for numerical stability
            m_block = tl.max(scores, axis=1)
            
            # 2. Compute new running max by comparing with previous blocks
            m_new = tl.maximum(m_i, m_block)
            
            # 3. Calculate scaling factors for the softmax stability algorithm
            # Alpha scales the existing accumulated values
            # Beta scales the new block's values
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_block - m_new)
            
            # 4. Update output and normalization accumulators
            # New normalization factor: L_new = α·L_old + β·∑exp(S - m_block)
            l_i_new = alpha * l_i + beta * tl.sum(tl.exp(scores - m_block[:, None]), axis=1)
            
            # New output accumulator: O_new = α·O_old + ∑exp(S - m_new)·V
            # This implements the efficient O(N) version of softmax described in Flash Attention
            o_new = (alpha[:, None] * o + 
                    tl.dot(tl.exp(scores - m_new[:, None]), v_block))
            
            # Update accumulators for next iteration
            l_i = l_i_new
            m_i = m_new
            o = o_new
            
        # Final normalization of output by dividing by the sum of attention weights
        o = o / l_i[:, None]
        
        # Store L and M statistics if requested (for backward pass)
        if STORE_L_M:
            # Create mask for storing L and M (only for valid query positions)
            l_m_store_mask = (tl.arange(0, BLOCK_M) < q_valid_range)
            # Store l_i for the current block
            tl.store(l_ptr + l_block_offset + tl.arange(0, BLOCK_M), l_i, mask=l_m_store_mask)
            # Store m_i for the current block
            tl.store(m_ptr + m_block_offset + tl.arange(0, BLOCK_M), m_i, mask=l_m_store_mask)
        
        # Calculate output mask for valid positions
        out_mask = (tl.arange(0, BLOCK_M)[:, None] < q_valid_range) & (tl.arange(0, BLOCK_DMODEL)[None, :] < head_dim)
        
        # Write normalized output back to global memory
        tl.store(
            o_ptr + o_offset + (tl.arange(0, BLOCK_M)[:, None] * stride_om +
                               tl.arange(0, BLOCK_DMODEL)[None, :] * stride_od),
            o,
            mask=out_mask
        ).to(tl_dtype) # Cast output back to original dtype (e.g., FP8)

    @triton.jit
    @triton.autotune(
        configs=[
            # For short sequences with smaller models
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_DMODEL': 32}, num_warps=4),
            # For medium sequences
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_DMODEL': 32}, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DMODEL': 32}, num_warps=8),
            # For large models with medium sequences
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DMODEL': 64}, num_warps=8),
            # For long sequences with large hidden dimensions
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DMODEL': 128}, num_warps=8),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_DMODEL': 64}, num_warps=8),
        ],
        key=['batch_size', 'seq_len', 'hidden_size', 'num_heads', 'head_dim', 'CAUSAL', 'USE_MASK', 'HAS_BIAS', 'IS_FP8', 'USE_ORTHO_MAT'], # Add FP8 flags to key
        prune_configs_by={
            'batch_size': lambda n: n <= 32,  # Only attempt large blocks with small batches
            'seq_len': lambda n: n <= 8192,   # Constrain for very long sequences
        },
    )
    def _fused_attention_kernel(
        # Pointers to matrices
        hidden_states_ptr, qkv_weight_ptr, qkv_bias_ptr, 
        out_weight_ptr, out_bias_ptr, output_ptr,
        # Attention mask (optional)
        mask_ptr,
        # Matrix dimensions
        batch_size, seq_len, hidden_size, num_heads, head_dim,
        # Strides for different dimensions
        stride_hsb, stride_hsm, stride_hsd,
        stride_qkvw_o, stride_qkvw_i,
        stride_qkvb,
        stride_outw_o, stride_outw_i,
        stride_outb,
        stride_out_b, stride_out_m, stride_out_d,
        stride_mask_b, stride_mask_h, stride_mask_m,
        # Other parameters
        scale, dropout_p, 
        # Meta-parameters
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr,
        USE_MASK: tl.constexpr, USE_DROPOUT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        # FP8 meta-parameters (potentially needed if fusing)
        IS_FP8: tl.constexpr,
        USE_ORTHO_MAT: tl.constexpr,
        ortho_matrix_ptr # Optional pointer
    ):
        """
        Fully fused attention kernel that combines multiple operations:
        - QKV projection
        - Attention computation
        - Output projection
        
        This kernel minimizes memory transfers by keeping intermediate values
        in high-bandwidth on-chip memory.
        
        Args:
            hidden_states_ptr: Pointer to input hidden states [batch_size, seq_len, hidden_size]
            qkv_weight_ptr: Pointer to QKV projection weights [3 * hidden_size, hidden_size]
            qkv_bias_ptr: Pointer to QKV projection bias [3 * hidden_size]
            out_weight_ptr: Pointer to output projection weights [hidden_size, hidden_size]
            out_bias_ptr: Pointer to output projection bias [hidden_size]
            output_ptr: Pointer to output tensor [batch_size, seq_len, hidden_size]
            mask_ptr: Pointer to attention mask (optional)
            Various dimensions, strides, and parameters
            Meta-parameters controlling kernel behavior
        """
        # Get program IDs for parallelism
        pid_batch = tl.program_id(0)  # Batch index
        pid_m = tl.program_id(1)      # Block of queries (BLOCK_M x num_heads)

        # Compute header index and offsets in the query sequence
        head_idx = pid_m // tl.cdiv(seq_len, BLOCK_M)
        block_start_m = (pid_m % tl.cdiv(seq_len, BLOCK_M)) * BLOCK_M
        
        # Compute ranges and offsets
        # Number of query vectors to process in this block
        q_range = min(BLOCK_M, seq_len - block_start_m)
        
        # --------------------- STEP 1: Load QKV Weights -----------------------
        # Calculate offsets for the QKV weights and bias
        q_weight_offset = head_idx * head_dim
        k_weight_offset = hidden_size + head_idx * head_dim
        v_weight_offset = 2 * hidden_size + head_idx * head_dim
        
        # Create offsets for the QKV weight matrix
        offs_qkvw_n = tl.arange(0, BLOCK_DMODEL)
        offs_qkvw_k = tl.arange(0, hidden_size)
        
        # Create mask for QKV weight matrix
        qkvw_mask = (offs_qkvw_n[:, None] < head_dim) & (offs_qkvw_k[None, :] < hidden_size)
        
        # Load Q weights
        q_weight_ptr = qkv_weight_ptr + q_weight_offset * stride_qkvw_o
        q_weight = tl.load(
            q_weight_ptr + 
            offs_qkvw_n[:, None] * stride_qkvw_o + 
            offs_qkvw_k[None, :] * stride_qkvw_i,
            mask=qkvw_mask,
            other=0.0
        )
        
        # Load K weights
        k_weight_ptr = qkv_weight_ptr + k_weight_offset * stride_qkvw_o
        k_weight = tl.load(
            k_weight_ptr + 
            offs_qkvw_n[:, None] * stride_qkvw_o + 
            offs_qkvw_k[None, :] * stride_qkvw_i,
            mask=qkvw_mask,
            other=0.0
        )
        
        # Load V weights
        v_weight_ptr = qkv_weight_ptr + v_weight_offset * stride_qkvw_o
        v_weight = tl.load(
            v_weight_ptr + 
            offs_qkvw_n[:, None] * stride_qkvw_o + 
            offs_qkvw_k[None, :] * stride_qkvw_i,
            mask=qkvw_mask,
            other=0.0
        )
        
        # Load QKV bias if needed
        if HAS_BIAS:
            # Calculate offsets for the QKV bias
            q_bias_offset = head_idx * head_dim
            k_bias_offset = hidden_size + head_idx * head_dim
            v_bias_offset = 2 * hidden_size + head_idx * head_dim
            
            # Create offset for bias
            offs_bias = tl.arange(0, BLOCK_DMODEL)
            
            # Create mask for bias
            bias_mask = offs_bias < head_dim
            
            # Load Q bias
            q_bias = tl.load(
                qkv_bias_ptr + q_bias_offset + offs_bias * stride_qkvb,
                mask=bias_mask,
                other=0.0
            )
            
            # Load K bias
            k_bias = tl.load(
                qkv_bias_ptr + k_bias_offset + offs_bias * stride_qkvb,
                mask=bias_mask,
                other=0.0
            )
            
            # Load V bias
            v_bias = tl.load(
                qkv_bias_ptr + v_bias_offset + offs_bias * stride_qkvb,
                mask=bias_mask,
                other=0.0
            )
        
        # --------------------- STEP 2: Process each query block -----------------------
        # Initialize shared memory for storing intermediate Q, K, V values
        q_block = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        
        # Initialize accumulators for flash attention algorithm
        m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)  # Max scores for numerical stability
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                     # Sum of exponentials
        o = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)         # Output accumulator
        
        # Create offset arrays for Q inputs
        offs_q_m = tl.arange(0, BLOCK_M)
        offs_q_n = tl.arange(0, hidden_size)
        
        # Compute base offset for hidden states
        hs_offset = pid_batch * stride_hsb + block_start_m * stride_hsm
        
        # Create mask for input loading
        input_mask = (offs_q_m[:, None] < q_range) & (offs_q_n[None, :] < hidden_size)
        
        # Load hidden states for this block
        hs_block = tl.load(
            hidden_states_ptr + hs_offset + 
            offs_q_m[:, None] * stride_hsm + 
            offs_q_n[None, :] * stride_hsd,
            mask=input_mask,
            other=0.0
        )
        
        # Compute Q, K, V projections
        # Compute: q_block = hs_block @ q_weight.T + q_bias
        q_block = tl.dot(hs_block, tl.trans(q_weight))
        if HAS_BIAS:
            q_block += q_bias[None, :]
        
        # Process K, V blocks for attention computation
        for block_start_n in range(0, seq_len, BLOCK_N):
            # If causal masking is enabled, we can early exit when processing
            # blocks that will be fully masked 
            if CAUSAL and block_start_n > block_start_m + BLOCK_M - 1:
                break
            
            # Compute the actual size of this key/value block
            k_range = min(BLOCK_N, seq_len - block_start_n)
            
            # Create offset arrays for K, V inputs
            offs_kv_m = tl.arange(0, BLOCK_N)
            
            # Create mask for K, V loading
            kv_mask = (offs_kv_m[:, None] < k_range) & (offs_q_n[None, :] < hidden_size)
            
            # Calculate hidden states offset for K, V
            hs_kv_offset = pid_batch * stride_hsb + block_start_n * stride_hsm
            
            # Load hidden states for K, V computation
            hs_kv_block = tl.load(
                hidden_states_ptr + hs_kv_offset + 
                offs_kv_m[:, None] * stride_hsm + 
                offs_q_n[None, :] * stride_hsd,
                mask=kv_mask,
                other=0.0
            )
            
            # Compute K, V projections
            # K = hs_kv_block @ k_weight.T + k_bias
            k_block = tl.dot(hs_kv_block, tl.trans(k_weight))
            if HAS_BIAS:
                k_block += k_bias[None, :]
                
            # V = hs_kv_block @ v_weight.T + v_bias
            v_block = tl.dot(hs_kv_block, tl.trans(v_weight))
            if HAS_BIAS:
                v_block += v_bias[None, :]
            
            # --------------------- STEP 3: Compute Attention ---------------------
            # Compute attention scores: QK^T / sqrt(d_k)
            scores = tl.dot(q_block, tl.trans(k_block))
            scores = scores * scale
            
            # Apply causal masking if enabled
            if CAUSAL:
                # Compute absolute position indices for queries and keys
                row_ids = block_start_m + tl.arange(0, BLOCK_M)
                col_ids = block_start_n + tl.arange(0, BLOCK_N)
                # Create causal mask where row_ids >= col_ids (upper triangular including diagonal)
                causal_mask = row_ids[:, None] >= col_ids[None, :]
                # Apply mask by zeroing out future positions
                scores = scores * causal_mask + (-1e9) * ~causal_mask
            
            # Apply attention mask if provided
            if USE_MASK:
                # Calculate mask offset
                mask_offset = (pid_batch * stride_mask_b +
                              head_idx * stride_mask_h +
                              block_start_m * stride_mask_m)
                
                # Create offset arrays for mask
                offs_mask_m = tl.arange(0, BLOCK_M)
                offs_mask_n = tl.arange(0, BLOCK_N)
                
                # Create mask for loading attention mask
                mask_load_mask = (offs_mask_m[:, None] < q_range) & (offs_mask_n[None, :] < k_range)
                
                # Load attention mask
                attn_mask = tl.load(
                    mask_ptr + mask_offset +
                    offs_mask_m[:, None] * stride_mask_m +
                    offs_mask_n[None, :],
                    mask=mask_load_mask,
                    other=0.0
                )
                
                # Apply mask: 0 → -inf, 1 → keep value
                scores = scores * attn_mask + (-1e9) * (1.0 - attn_mask)
            
            # --------------------- STEP 4: Flash Attention Algorithm ---------------------
            # Implement stable softmax algorithm from Flash Attention
            
            # 1. Find current block's max score per query for numerical stability
            m_block = tl.max(scores, axis=1)
            
            # 2. Compute new running max by comparing with previous blocks
            m_new = tl.maximum(m_i, m_block)
            
            # 3. Calculate scaling factors for the softmax stability algorithm
            alpha = tl.exp(m_i - m_new)   # Scale existing accumulated values
            beta = tl.exp(m_block - m_new) # Scale the new block's values
            
            # 4. Update normalization factor: L_new = α·L_old + β·∑exp(S - m_block)
            l_i_new = alpha * l_i + beta * tl.sum(tl.exp(scores - m_block[:, None]), axis=1)
            
            # 5. Update output accumulator: O_new = α·O_old + ∑exp(S - m_new)·V
            o_new = alpha[:, None] * o + tl.dot(tl.exp(scores - m_new[:, None]), v_block)
            
            # Update accumulators for next iteration
            l_i = l_i_new
            m_i = m_new
            o = o_new
        
        # --------------------- STEP 5: Final normalization ---------------------
        # Normalize output by dividing by the sum of attention weights
        o = o / l_i[:, None]
        
        # Apply dropout if needed
        if USE_DROPOUT:
            dropout_scale = 1.0 / (1.0 - dropout_p)
            # Generate random mask according to dropout probability
            dropout_mask = tl.rand(BLOCK_M, BLOCK_DMODEL) >= dropout_p
            o = o * dropout_scale * dropout_mask
        
        # --------------------- STEP 6: Output projection ---------------------
        # Load output projection weights
        offs_outw_n = tl.arange(0, BLOCK_DMODEL)
        offs_outw_k = tl.arange(0, hidden_size)
        
        # Create mask for output weights
        outw_mask = (offs_outw_n[:, None] < head_dim) & (offs_outw_k[None, :] < hidden_size)
        
        # Load output weights
        out_weight = tl.load(
            out_weight_ptr + 
            offs_outw_n[:, None] * stride_outw_o + 
            offs_outw_k[None, :] * stride_outw_i,
            mask=outw_mask,
            other=0.0
        )
        
        # Compute output projection
        output = tl.dot(o, out_weight)
        
        # Add output bias if enabled
        if HAS_BIAS:
            offs_outb = tl.arange(0, hidden_size)
            outb_mask = offs_outb < hidden_size
            
            out_bias = tl.load(
                out_bias_ptr + offs_outb * stride_outb,
                mask=outb_mask,
                other=0.0
            )
            
            output += out_bias[None, :]
        
        # --------------------- STEP 7: Write output to memory ---------------------
        # Calculate output offset
        out_offset = pid_batch * stride_out_b + block_start_m * stride_out_m
        
        # Create offset arrays for output
        offs_out_m = tl.arange(0, BLOCK_M)
        offs_out_d = tl.arange(0, hidden_size)
        
        # Create mask for output writing
        out_mask = (offs_out_m[:, None] < q_range) & (offs_out_d[None, :] < hidden_size)
        
        # Write output to global memory
        tl.store(
            output_ptr + out_offset + 
            offs_out_m[:, None] * stride_out_m + 
            offs_out_d[None, :] * stride_out_d,
            output,
            mask=out_mask
        ).to(tl_dtype) # Cast output back to original dtype (e.g., FP8)

    @triton.jit
    @triton.autotune(
        configs=[
            # Configurations for smaller models and sequences
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_DMODEL': 32}, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_DMODEL': 32}, num_warps=4),
            # Balanced configurations for medium cases
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DMODEL': 32}, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DMODEL': 64}, num_warps=8),
            # Configurations for larger models or longer sequences
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_DMODEL': 64}, num_warps=8),
        ],
        key=['batch_size', 'seq_len', 'num_heads', 'head_dim', 'CAUSAL', 'USE_MASK', 'IS_FP8', 'USE_ORTHO_MAT'], # Add FP8 flags to key
    )
    def _flash_attention_backward_kernel(
        # Pointers to matrices
        grad_output_ptr, q_ptr, k_ptr, v_ptr, o_ptr, # Need o_ptr for delta recomputation
        grad_q_ptr, grad_k_ptr, grad_v_ptr,
        # L and m statistics from forward pass (or recomputed)
        l_ptr, m_ptr, 
        # Optional attention mask
        mask_ptr,
        # Matrix dimensions
        batch_size, seq_len, num_heads, head_dim,
        # Strides for different dimensions
        stride_dob, stride_dom, stride_doh, stride_dod,  # Grad output strides
        stride_qb, stride_qm, stride_qh, stride_qd,      # Q strides
        stride_kb, stride_km, stride_kh, stride_kd,      # K strides
        stride_vb, stride_vm, stride_vh, stride_vd,      # V strides
        stride_ob, stride_om, stride_oh, stride_od,      # O strides
        stride_gqb, stride_gqm, stride_gqh, stride_gqd,  # Grad Q strides
        stride_gkb, stride_gkm, stride_gkh, stride_gkd,  # Grad K strides
        stride_gvb, stride_gvm, stride_gvh, stride_gvd,  # Grad V strides
        stride_lb, stride_lh, stride_lm,                # L strides
        stride_mb, stride_mh, stride_mm,                # m strides
        stride_maskb, stride_maskh, stride_maskm,        # Mask strides (if applicable)
        # Other parameters
        scale, 
        # Meta-parameters
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr,
        USE_MASK: tl.constexpr,
       # FP8 related meta-parameters (Needed if recomputing FP8 forward pass)
       IS_FP8: tl.constexpr,
       USE_ORTHO_MAT: tl.constexpr,
       ortho_matrix_ptr # Optional pointer
    ):
        """
        Compute the backward pass of flash attention.
        
        Follows the algorithm described in the FlashAttention-2 paper (Section 3.2).
        Requires the output O and the softmax normalization statistics (L, m) 
        from the forward pass, which might need to be recomputed if not passed in.
        Handles FP8 with optional incoherent processing.
        """
        # Get program IDs for parallel processing
        pid_batch = tl.program_id(0)  # Batch index
        pid_head = tl.program_id(1)   # Head index
        pid_m = tl.program_id(2)      # Block index for query sequence
        
        # Compute offsets for the current block
        start_m = pid_m * BLOCK_M
        q_offset = (pid_batch * stride_qb + pid_head * stride_qh + start_m * stride_qm)
        do_offset = (pid_batch * stride_dob + pid_head * stride_doh + start_m * stride_dom)
        o_offset = (pid_batch * stride_ob + pid_head * stride_oh + start_m * stride_om)
        grad_q_offset = (pid_batch * stride_gqb + pid_head * stride_gqh + start_m * stride_gqm)
        l_offset = (pid_batch * stride_lb + pid_head * stride_lh + start_m * stride_lm) # Offset for l stat
        m_offset = (pid_batch * stride_mb + pid_head * stride_mh + start_m * stride_mm) # Offset for m stat
        
        # Compute range of valid queries for this block (handle boundary)
        m_range = min(BLOCK_M, seq_len - start_m)
        
        # Define offsets for the head dimension
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_m = tl.arange(0, BLOCK_M)
        
        # --- Load Inputs for the Block --- 
        # Load dO (gradient of output)
        do_mask = (offs_m[:, None] < m_range) & (offs_d[None, :] < head_dim)
        do_block = tl.load(
            grad_output_ptr + do_offset + (offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod),
            mask=do_mask,
            other=0.0
        ).to(tl.float32)

        # Load O (output from forward pass)
        o_mask = (offs_m[:, None] < m_range) & (offs_d[None, :] < head_dim)
        o_block = tl.load(
            o_ptr + o_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_od),
            mask=o_mask,
            other=0.0
        ).to(tl.float32)
        
        # Load l and m statistics for stable softmax recomputation
        l_mask = (offs_m < m_range)
        m_mask = (offs_m < m_range)
        l_block = tl.load(l_ptr + l_offset + offs_m, mask=l_mask, other=0.0)
        m_block = tl.load(m_ptr + m_offset + offs_m, mask=m_mask, other=-float('inf'))

        # Load Q block
        tl_q_dtype = tl.float8e4m3fn if IS_FP8 else q_ptr.dtype.element_ty
        q_mask = (offs_m[:, None] < m_range) & (offs_d[None, :] < head_dim)
        q_block = tl.load(
            q_ptr + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd),
            mask=q_mask,
            other=0.0
        ).to(tl.float32)
        
        # --- Handle FP8 Orthogonal Transformation --- 
        ortho_block = None
        if IS_FP8 and USE_ORTHO_MAT:
            # Load the orthogonal matrix once
            ortho_offs_d1 = tl.arange(0, BLOCK_DMODEL)
            ortho_offs_d2 = tl.arange(0, BLOCK_DMODEL)
            ortho_mask = (ortho_offs_d1[:, None] < head_dim) & (ortho_offs_d2[None, :] < head_dim)
            ortho_block = tl.load(
                ortho_matrix_ptr + ortho_offs_d1[:, None] * head_dim + ortho_offs_d2[None, :],
                mask=ortho_mask,
                other=0.0
            ) # Expects FP32 matrix
            
            # Apply transformation Q_transformed = Q @ Ortho
            q_block = tl.dot(q_block, ortho_block, out_dtype=tl.float32)
            # Note: o_block and do_block remain untransformed as they relate to the final output
            
        # --- Compute delta term --- 
        # delta = rowsum(dO * O) [BLOCK_M]
        delta = tl.sum(do_block * o_block, axis=1)

        # --- Initialize Gradient Accumulators --- 
        grad_q = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        # --- Loop over Key/Value blocks --- 
        for start_n in range(0, seq_len, BLOCK_N):
            # Early termination check for causal masking
            if CAUSAL and start_n > (start_m + BLOCK_M - 1):
                break
                
            # Compute range of valid keys/values for this block
            n_range = min(BLOCK_N, seq_len - start_n)
            offs_n = tl.arange(0, BLOCK_N)

            # --- Load K and V blocks --- 
            tl_kv_dtype = tl.float8e4m3fn if IS_FP8 else k_ptr.dtype.element_ty
            
            k_offset = (pid_batch * stride_kb + pid_head * stride_kh + start_n * stride_km)
            k_block_ptr = k_ptr + k_offset
            k_mask = (offs_n[:, None] < n_range) & (offs_d[None, :] < head_dim)
            k_block = tl.load(
                k_block_ptr + (offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd),
                mask=k_mask,
                other=0.0
            ).to(tl.float32)

            v_offset = (pid_batch * stride_vb + pid_head * stride_vh + start_n * stride_vm)
            v_block_ptr = v_ptr + v_offset
            v_mask = (offs_n[:, None] < n_range) & (offs_d[None, :] < head_dim)
            v_block = tl.load(
                v_block_ptr + (offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd),
                mask=v_mask,
                other=0.0
            ).to(tl.float32)
            
            # Apply orthogonal transformation to K if needed
            if IS_FP8 and USE_ORTHO_MAT:
                k_block = tl.dot(k_block, ortho_block, out_dtype=tl.float32)
            
            # --- Recompute Scores and Softmax (P) --- 
            # S = Q K^T (using potentially transformed Q and K)
            scores = tl.dot(q_block, tl.trans(k_block), out_dtype=tl.float32) * scale

            # Apply causal/attention masking
            if CAUSAL:
                row_ids = start_m + tl.arange(0, BLOCK_M)
                col_ids = start_n + tl.arange(0, BLOCK_N)
                causal_mask = row_ids[:, None] >= col_ids[None, :]
                scores = scores * causal_mask + (-1e9) * ~causal_mask
            if USE_MASK:
                # Load mask block - adjust strides/logic based on actual mask format
                mask_offset = (pid_batch * stride_maskb + pid_head * stride_maskh + start_m * stride_maskm) 
                mask_load_mask = (offs_m[:, None] < m_range) & (offs_n[None, :] < n_range)
                mask_block = tl.load(
                    mask_ptr + (offs_m[:, None] * stride_maskm + offs_n[None, :]), # Check mask indexing
                    mask=mask_load_mask,
                    other=0.0
                )
                scores = scores * mask_block + (-1e9) * (1.0 - mask_block)
            
            # P = softmax(S) using loaded l and m stats
            # p = exp(S - m) / l
            p = tl.exp(scores - m_block[:, None]) / l_block[:, None] # [BLOCK_M, BLOCK_N]
            # Clamp P for stability if needed, e.g., p = tl.where(mask_load_mask, p, 0.0)
            
            # --- Compute Gradients for this block --- 
            # Compute dV = P^T @ dO 
            # Shapes: P^T [N, M], dO [M, D] -> dV [N, D]
            grad_v_block = tl.dot(tl.trans(p), do_block, out_dtype=tl.float32)
            
            # Compute dP = dO @ V^T
            # Shapes: dO [M, D], V^T [D, N] -> dP [M, N]
            dp = tl.dot(do_block, tl.trans(v_block), out_dtype=tl.float32)
            
            # Compute dS = P * (dP - delta)
            ds = p * (dp - delta[:, None]) # delta[M] broadcast to [M, N]

            # Compute dK = dS^T @ Q
            # Shapes: dS^T [N, M], Q [M, D] -> dK [N, D]
            # Q is potentially transformed Q_transformed
            grad_k_block = tl.dot(tl.trans(ds), q_block, out_dtype=tl.float32)
            
            # Compute dQ contribution from this block: dQ += dS @ K
            # Shapes: dS [M, N], K [N, D] -> dQ_contrib [M, D]
            # K is potentially transformed K_transformed
            grad_q += tl.dot(ds, k_block, out_dtype=tl.float32)
            
            # --- Atomically Add dV and dK to Global Gradients --- 
            grad_v_offset = (pid_batch * stride_gvb + pid_head * stride_gvh + start_n * stride_gvm)
            grad_k_offset = (pid_batch * stride_gkb + pid_head * stride_gkh + start_n * stride_gkm)
           
            grad_v_block_ptr = grad_v_ptr + grad_v_offset
            grad_k_block_ptr = grad_k_ptr + grad_k_offset
           
            store_mask = (offs_n[:, None] < n_range) & (offs_d[None, :] < head_dim)
            
            # Store dV (no transformation needed)
            # Accumulation happens across block_m, potentially requiring atomics
            tl.atomic_add(grad_v_block_ptr + (offs_n[:, None] * stride_gvm + offs_d[None, :] * stride_gvd), grad_v_block, mask=store_mask)
            
            # Store dK (gradient w.r.t potentially transformed K)
            # Accumulation happens across block_m, potentially requiring atomics
            tl.atomic_add(grad_k_block_ptr + (offs_n[:, None] * stride_gkm + offs_d[None, :] * stride_gkd), grad_k_block, mask=store_mask)
            # TODO: Inverse transform for dK needs to happen *after* all atomic adds complete.
            # This is complex and might require a separate kernel or careful synchronization.
            # It cannot be done correctly within this loop if USE_ORTHO_MAT is true.

        # --- End of Loop --- 

        # --- Finalize and Store dQ --- 
        # Apply inverse orthogonal transformation to dQ if needed
        if IS_FP8 and USE_ORTHO_MAT:
            # grad_q currently holds dQ w.r.t Q_transformed
            # We need dQ w.r.t Q_original = dQ @ Ortho^T
            grad_q = tl.dot(grad_q, tl.trans(ortho_block), out_dtype=tl.float32)

        # Write final dQ gradient
        grad_q_block_ptr = grad_q_ptr + grad_q_offset
        grad_q_mask = (offs_m[:, None] < m_range) & (offs_d[None, :] < head_dim)

        # Store final dQ (converting back to original Q dtype if necessary)
        tl.store(
            grad_q_block_ptr + (offs_m[:, None] * stride_gqm + offs_d[None, :] * stride_gqd), \
            grad_q.to(tl_q_dtype), # Cast to original type (e.g., FP8)
            mask=grad_q_mask
        )


#-----------------------------------------------------------------------------
# Python Wrappers for Triton Kernels
#-----------------------------------------------------------------------------

if HAS_TRITON:
    class _FlashAttentionFunction(torch.autograd.Function):
        """Flash Attention implemented via Triton kernels, supporting FP8."""
        
        @staticmethod
        def forward(ctx, q, k, v, mask, causal, softmax_scale, dropout_p, 
                    return_softmax, block_size, precision, ortho_matrix):
            """Forward pass calling the Triton kernel."""
            # Determine if L/m need to be stored (if any input requires grad)
            store_l_m = q.requires_grad or k.requires_grad or v.requires_grad

            # Input checks are mostly handled by the calling function (triton_flash_attention)
            batch_size, seq_len, num_heads, head_dim = q.shape
            scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(head_dim))
            
            # Prepare output tensor and L, m tensors for backward pass
            output = torch.empty_like(q, dtype=q.dtype)
            # Allocate L, m only if needed for backward pass
            l = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32) if store_l_m else None
            m = torch.full((batch_size, num_heads, seq_len), -float('inf'), device=q.device, dtype=torch.float32) if store_l_m else None
            
            # Prepare mask pointer and strides
            mask_ptr = 0
            stride_maskb, stride_maskh, stride_maskm = 0, 0, 0
            USE_MASK = (mask is not None)
            if USE_MASK:
                mask = mask.contiguous() # Ensure contiguous
                mask_ptr = mask.data_ptr()
                # Simplified stride calculation - needs refinement based on mask format
                stride_maskb = mask.stride(0) if mask.dim() > 0 else 0
                stride_maskh = mask.stride(1) if mask.dim() > 1 else 0 # Assuming B, H, ... format potential
                stride_maskm = mask.stride(2) if mask.dim() > 2 else 0 # Assuming B, H, M, ... format potential

            # --- FP8 Handling ---
            IS_FP8 = (precision == "fp8")
            USE_ORTHO_MAT = (IS_FP8 and ortho_matrix is not None)
            ortho_matrix_ptr = 0
            if USE_ORTHO_MAT:
                # Validation done in launcher, assume ortho_matrix is valid FP32 contiguous tensor here
                ortho_matrix_ptr = ortho_matrix.data_ptr()
            # --- End FP8 Handling ---

            # Grid dimensions
            # TODO: Need separate BLOCK_M for forward kernel grid calculation
            # Using fixed 128 for now, but should ideally use autotuner result or config
            grid_m_block = 128 
            grid = (batch_size, num_heads, triton.cdiv(seq_len, grid_m_block))
            
            # Launch forward kernel (Now includes L and m pointers)
            _flash_attention_forward_kernel[grid](
                q_ptr=q.data_ptr(), k_ptr=k.data_ptr(), v_ptr=v.data_ptr(), o_ptr=output.data_ptr(),
                l_ptr=l.data_ptr() if store_l_m else 0, m_ptr=m.data_ptr() if store_l_m else 0, # Pass L and m pointers
                mask_ptr=mask_ptr,
                batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim,
                stride_qb=q.stride(0), stride_qh=q.stride(2), stride_qm=q.stride(1), stride_qd=q.stride(3),
                stride_kb=k.stride(0), stride_kh=k.stride(2), stride_km=k.stride(1), stride_kd=k.stride(3),
                stride_vb=v.stride(0), stride_vh=v.stride(2), stride_vm=v.stride(1), stride_vd=v.stride(3),
                stride_ob=output.stride(0), stride_oh=output.stride(2), stride_om=output.stride(1), stride_od=output.stride(3),
                stride_lb=l.stride(0) if store_l_m else 0, stride_lh=l.stride(1) if store_l_m else 0, stride_lm=l.stride(2) if store_l_m else 0, # L strides
                stride_mb=m.stride(0) if store_l_m else 0, stride_mh=m.stride(1) if store_l_m else 0, stride_mm=m.stride(2) if store_l_m else 0, # m strides
                stride_maskb=stride_maskb, stride_maskh=stride_maskh, stride_maskm=stride_maskm,
                scale=scale,
                # BLOCK_M, BLOCK_N, BLOCK_DMODEL from autotuner config
                CAUSAL=causal,
                USE_MASK=USE_MASK,
                IS_FP8=IS_FP8,
                USE_ORTHO_MAT=USE_ORTHO_MAT,
                STORE_L_M=store_l_m, # Pass flag to kernel
                ortho_matrix_ptr=ortho_matrix_ptr,
                # Pass L/m pointers and strides only if they were allocated
                l_ptr=l.data_ptr() if store_l_m else 0, 
                m_ptr=m.data_ptr() if store_l_m else 0,
                stride_lb=l.stride(0) if store_l_m else 0,
                stride_lh=l.stride(1) if store_l_m else 0,
                stride_lm=l.stride(2) if store_l_m else 0,
                stride_mb=m.stride(0) if store_l_m else 0,
                stride_mh=m.stride(1) if store_l_m else 0,
                stride_mm=m.stride(2) if store_l_m else 0
            )
            
            # Save tensors for backward pass
            # Conditionally save tensors based on requires_grad and FP8 usage
            tensors_to_save = [q, k, v, output]
            if store_l_m:
                tensors_to_save.extend([l, m])
            # Only save ortho_matrix if it was actually used (FP8 and provided)
            if USE_ORTHO_MAT and ortho_matrix is not None:
                tensors_to_save.append(ortho_matrix)
            ctx.save_for_backward(*tensors_to_save)
            
            ctx.store_l_m = store_l_m # Save the flag for backward logic
            ctx.use_ortho_mat = USE_ORTHO_MAT # Save the flag for backward logic
            ctx.causal = causal
            ctx.softmax_scale = scale
            ctx.precision = precision
            ctx.use_mask = USE_MASK
            ctx.mask = mask # Save mask itself if needed for backward masking logic
            
            # TODO: Handle return_softmax correctly if needed
            if return_softmax:
                raise NotImplementedError("Returning softmax weights not yet implemented in Triton kernel")
            
            return output

        @staticmethod
        def backward(ctx, grad_output):
            """Backward pass calling the Triton backward kernel."""
            # Unpack saved tensors based on flags saved in forward context
            saved_tensors = ctx.saved_tensors
            q, k, v, output = saved_tensors[:4] # Q, K, V, O are always saved
            saved_idx = 4
            
            # Load L, m if they were saved
            if ctx.store_l_m:
                l, m = saved_tensors[saved_idx:saved_idx+2]
                saved_idx += 2
            else:
                # L and m are required for backward, raise error or recompute? Recompute is safer but slower.
                # For now, assume they MUST be available if requires_grad is true.
                raise RuntimeError("Internal Error: L and m required for backward but not found in saved tensors.")

            # Load ortho_matrix if it was saved
            ortho_matrix = None
            if ctx.use_ortho_mat:
                if saved_idx < len(saved_tensors):
                    ortho_matrix = saved_tensors[saved_idx]
                    saved_idx += 1
                else:
                    # This indicates an issue if use_ortho_mat was true but matrix wasn't saved
                    raise RuntimeError("Internal Error: Orthogonal matrix required but not found in saved tensors.")

            causal = ctx.causal
            scale = ctx.softmax_scale
            precision = ctx.precision # Get precision from context
            use_mask = ctx.use_mask
            mask = ctx.mask
            
            # Input checks
            if not grad_output.is_contiguous():
                grad_output = grad_output.contiguous()
            
            batch_size, seq_len, num_heads, head_dim = q.shape
            
            # Allocate gradient tensors (matching input dtypes, including FP8)
            grad_q = torch.zeros_like(q, dtype=q.dtype)
            grad_k = torch.zeros_like(k, dtype=k.dtype)
            grad_v = torch.zeros_like(v, dtype=v.dtype)
            
            # Prepare mask pointer and strides for backward
            mask_ptr = 0
            stride_maskb, stride_maskh, stride_maskm = 0, 0, 0
            if use_mask and mask is not None:
                 mask_ptr = mask.data_ptr()
                 stride_maskb = mask.stride(0) if mask.dim() > 0 else 0
                 stride_maskh = mask.stride(1) if mask.dim() > 1 else 0 
                 stride_maskm = mask.stride(2) if mask.dim() > 2 else 0
                 
            # --- FP8 Handling --- 
            IS_FP8 = (precision == "fp8")
            USE_ORTHO_MAT = (IS_FP8 and ortho_matrix is not None)
            ortho_matrix_ptr = 0
            if USE_ORTHO_MAT:
                # Ortho matrix should already be FP32 contiguous from forward
                ortho_matrix_ptr = ortho_matrix.data_ptr()
            # --- End FP8 Handling ---
            
            # Grid dimensions for backward kernel
            # TODO: Needs separate BLOCK_M for backward kernel grid calculation
            grid_m_block_bwd = 128 # Use fixed 128 for now
            grid = (batch_size, num_heads, triton.cdiv(seq_len, grid_m_block_bwd))
            
            # Launch backward kernel
            _flash_attention_backward_kernel[grid](
                grad_output_ptr=grad_output.data_ptr(), 
                q_ptr=q.data_ptr(), k_ptr=k.data_ptr(), v_ptr=v.data_ptr(), o_ptr=output.data_ptr(),
                grad_q_ptr=grad_q.data_ptr(), grad_k_ptr=grad_k.data_ptr(), grad_v_ptr=grad_v.data_ptr(),
                l_ptr=l.data_ptr(), m_ptr=m.data_ptr(), 
                mask_ptr=mask_ptr,
                batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim,
                stride_dob=grad_output.stride(0), stride_dom=grad_output.stride(1), stride_doh=grad_output.stride(2), stride_dod=grad_output.stride(3),
                stride_qb=q.stride(0), stride_qm=q.stride(1), stride_qh=q.stride(2), stride_qd=q.stride(3),
                stride_kb=k.stride(0), stride_km=k.stride(1), stride_kh=k.stride(2), stride_kd=k.stride(3),
                stride_vb=v.stride(0), stride_vm=v.stride(1), stride_vh=v.stride(2), stride_vd=v.stride(3),
                stride_ob=output.stride(0), stride_om=output.stride(1), stride_oh=output.stride(2), stride_od=output.stride(3),
                stride_gqb=grad_q.stride(0), stride_gqm=grad_q.stride(1), stride_gqh=grad_q.stride(2), stride_gqd=grad_q.stride(3),
                stride_gkb=grad_k.stride(0), stride_gkm=grad_k.stride(1), stride_gkh=grad_k.stride(2), stride_gkd=grad_k.stride(3),
                stride_gvb=grad_v.stride(0), stride_gvm=grad_v.stride(1), stride_gvh=grad_v.stride(2), stride_gvd=grad_v.stride(3),
                stride_lb=l.stride(0), stride_lh=l.stride(1), stride_lm=l.stride(2),
                stride_mb=m.stride(0), stride_mh=m.stride(1), stride_mm=m.stride(2),
                stride_maskb=stride_maskb, stride_maskh=stride_maskh, stride_maskm=stride_maskm,
                scale=scale,
                # BLOCK_M, BLOCK_N, BLOCK_DMODEL from autotuner config
                CAUSAL=causal,
                USE_MASK=use_mask,
                IS_FP8=IS_FP8,
                USE_ORTHO_MAT=USE_ORTHO_MAT,
                ortho_matrix_ptr=ortho_matrix_ptr
            )

            # Return gradients for q, k, v, and None for other inputs 
            # (mask, causal, softmax_scale, dropout_p, return_softmax, block_size, precision, ortho_matrix)
            return grad_q, grad_k, grad_v, None, None, None, None, None, None, None, None


def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
    return_softmax: bool = False,
    block_size: int = 128
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Python wrapper for the Triton flash attention kernel.
    
    Args:
        q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
        v: Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
        mask: Optional attention mask
        causal: Whether to apply causal masking
        softmax_scale: Scale factor for softmax (default: 1/sqrt(head_dim))
        dropout_p: Attention dropout probability
        return_softmax: Whether to return softmax attention weights
        block_size: Block size for tiling
        
    Returns:
        output: Attention output of shape [batch_size, seq_len, num_heads, head_dim]
        (Optional) attention_weights: If return_softmax=True, returns attention weights
    """
    # Check if Triton is available, fallback to PyTorch implementation if not
    if not HAS_TRITON:
        print("Triton not available, falling back to PyTorch implementation")
        return pytorch_flash_attention(q, k, v, mask, causal, softmax_scale, 
                                      dropout_p, return_softmax)
    
    # Use autograd-enabled implementation for training
    if q.requires_grad or k.requires_grad or v.requires_grad:
        return _FlashAttentionFunction.apply(
            q, k, v, mask, causal, softmax_scale, dropout_p, return_softmax, block_size, "fp8", None
        )
    
    # Validate input tensors
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"Expected 4D tensors for q, k, v but got shapes: "
                         f"q={q.shape}, k={k.shape}, v={v.shape}")
    
    # Extract dimensions
    batch_size, seq_len, num_heads, head_dim = q.shape
    k_batch_size, k_seq_len, k_num_heads, k_head_dim = k.shape
    v_batch_size, v_seq_len, v_num_heads, v_head_dim = v.shape
    
    # Validate shapes match
    if (batch_size != k_batch_size or batch_size != v_batch_size or
        num_heads != k_num_heads or num_heads != v_num_heads or
        head_dim != k_head_dim or head_dim != v_head_dim):
        raise ValueError(
            f"Inconsistent tensor dimensions: "
            f"q={q.shape}, k={k.shape}, v={v.shape}"
        )
    
    # Validate all tensors are on the same device
    if q.device != k.device or q.device != v.device:
        raise ValueError(
            f"All tensors must be on the same device, but got: "
            f"q.device={q.device}, k.device={k.device}, v.device={v.device}"
        )
    
    # Validate tensor datatypes
    if q.dtype != k.dtype or q.dtype != v.dtype:
        # Automatically convert to match the query dtype
        k = k.to(q.dtype)
        v = v.to(q.dtype)
        print(f"Warning: Converted k and v to {q.dtype} to match q")
    
    # Set softmax scale if not provided
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    
    # Create output tensor
    output = torch.empty_like(q)
    
    # Prepare attention mask if provided
    use_mask = mask is not None
    if use_mask:
        # Ensure mask has the right shape
        if mask.dim() == 2:  # [batch_size, seq_len]
            # Expand to [batch_size, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.dim() == 3:
            if mask.shape[1] == 1:  # [batch_size, 1, seq_len]
                # Expand to [batch_size, 1, 1, seq_len]
                mask = mask.unsqueeze(2)
            else:  # [batch_size, seq_len, seq_len]
                # Reshape to [batch_size, 1, seq_len, seq_len]
                mask = mask.unsqueeze(1)
        elif mask.dim() == 4:  # [batch_size, num_heads, seq_len, seq_len]
            # Already in the right format
            pass
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")
        
        # Ensure mask is on the same device and has the right dtype
        if mask.device != q.device:
            mask = mask.to(q.device)
        
        # Convert mask to float for compatibility with the kernel
        if not mask.is_floating_point():
            mask = mask.to(torch.float32)
    
    # Select optimal block size based on sequence length and GPU capabilities
    # Larger block sizes are more efficient but use more shared memory
    if block_size > 256:
        # For very long sequences, use smaller blocks to avoid shared memory limits
        if seq_len > 4096:
            block_size = 64
        elif seq_len > 2048:
            block_size = 128
        else:
            block_size = 256
    
    # Ensure block size doesn't exceed sequence length
    block_size = min(block_size, seq_len)
    
    # Determine blocks for parallelism
    grid = (
        batch_size,                          # Batch dimension
        num_heads,                           # Head dimension
        triton.cdiv(seq_len, block_size)     # Sequence dimension
    )
    
    # Handle attention weights if requested
    attention_weights = None
    if return_softmax:
        attention_weights = torch.zeros(
            (batch_size, num_heads, seq_len, k_seq_len),
            device=q.device,
            dtype=q.dtype
        )
    
    # Launch kernel with appropriate grid dimensions and parameters
    try:
        _flash_attention_forward_kernel[grid](
            q, k, v, output, mask if use_mask else torch.empty(0, device=q.device),
            batch_size, seq_len, num_heads, head_dim,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            v.stride(0), v.stride(2), v.stride(1), v.stride(3),
            output.stride(0), output.stride(2), output.stride(1), output.stride(3),
            mask.stride(0) if use_mask else 0,
            mask.stride(1) if use_mask else 0,
            mask.stride(3) if use_mask else 0,
            softmax_scale,
            BLOCK_M=block_size,
            BLOCK_DMODEL=head_dim,
            BLOCK_N=block_size,
            CAUSAL=causal,
            USE_MASK=use_mask,
            IS_FP8=False,
            USE_ORTHO_MAT=False,
            ortho_matrix_ptr=0
        )
    except Exception as e:
        print(f"Triton kernel launch failed with error: {e}")
        print("Falling back to PyTorch implementation")
        return pytorch_flash_attention(q, k, v, mask, causal, softmax_scale, 
                                      dropout_p, return_softmax)
    
    # Apply dropout if needed
    if dropout_p > 0.0 and q.device.type == "cuda":
        if not q.is_cuda:
            # Move to CUDA for dropout if not already
            output = output.cuda()
        
        dropout_mask = torch.empty_like(output).bernoulli_(1 - dropout_p)
        dropout_mask = dropout_mask / (1 - dropout_p)
        output = output * dropout_mask
    
    # Handle attention weights if requested
    if return_softmax:
        # In a full implementation, this would extract attention weights
        # from the kernel or compute them from q, k
        
        # Here we're using a simple approximation since the kernel doesn't return weights
        # This is a simplified version that will not be perfectly accurate
        # but provides a reasonable approximation for visualization/debugging
        with torch.no_grad():
            scores = torch.einsum("bshd,bkhd->bhsk", q, k) * softmax_scale
            
            # Apply causal mask if needed
            if causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, k_seq_len, device=scores.device, dtype=torch.bool),
                    diagonal=1
                )
                scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
            
            # Apply attention mask if provided
            if use_mask:
                scores.masked_fill_(~mask.bool(), -1e9)
            
            attention_weights = torch.softmax(scores, dim=-1)
            
            # Apply dropout to weights if needed
            if dropout_p > 0.0:
                attention_weights = F.dropout(attention_weights, p=dropout_p)
        
        return output, attention_weights
    else:
        return output


def triton_fused_attention(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: Optional[torch.Tensor],
    out_weight: torch.Tensor,
    out_bias: Optional[torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    num_heads: int = 8,
    head_dim: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    block_size: int = 128
) -> torch.Tensor:
    """
    Fused attention implementation that combines QKV projection, attention, and output projection.
    
    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        qkv_weight: QKV projection weight of shape [3*hidden_size, hidden_size]
        qkv_bias: Optional QKV projection bias of shape [3*hidden_size]
        out_weight: Output projection weight of shape [hidden_size, hidden_size]
        out_bias: Optional output projection bias of shape [hidden_size]
        mask: Optional attention mask
        causal: Whether to apply causal masking
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head (default: hidden_size // num_heads)
        dropout_p: Attention dropout probability
        softmax_scale: Scale factor for softmax (default: 1/sqrt(head_dim))
        block_size: Block size for tiling
        
    Returns:
        output: Output tensor of shape [batch_size, seq_len, hidden_size]
    """
    # Check if Triton is available, fallback to PyTorch implementation if not
    if not HAS_TRITON:
        print("Triton not available, falling back to PyTorch implementation")
        return pytorch_fused_attention(hidden_states, qkv_weight, qkv_bias, 
                                      out_weight, out_bias, mask, causal,
                                      num_heads, head_dim, dropout_p, softmax_scale)
    
    # Validate input tensors
    if hidden_states.dim() != 3:
        raise ValueError(f"Expected 3D tensor for hidden_states but got shape: {hidden_states.shape}")
    
    if qkv_weight.dim() != 2:
        raise ValueError(f"Expected 2D tensor for qkv_weight but got shape: {qkv_weight.shape}")
    
    if qkv_bias is not None and qkv_bias.dim() != 1:
        raise ValueError(f"Expected 1D tensor for qkv_bias but got shape: {qkv_bias.shape}")
    
    if out_weight.dim() != 2:
        raise ValueError(f"Expected 2D tensor for out_weight but got shape: {out_weight.shape}")
    
    if out_bias is not None and out_bias.dim() != 1:
        raise ValueError(f"Expected 1D tensor for out_bias but got shape: {out_bias.shape}")
    
    # Extract dimensions
    batch_size, seq_len, hidden_size = hidden_states.shape
    
    # Validate QKV weights shape
    if qkv_weight.shape[0] != 3 * hidden_size:
        raise ValueError(f"Expected qkv_weight shape [{3 * hidden_size}, {hidden_size}] but got {qkv_weight.shape}")
    
    # Calculate head dimension if not provided
    if head_dim is None:
        # Ensure hidden_size is divisible by num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by num_heads {num_heads}")
        head_dim = hidden_size // num_heads
    
    # Check output weight shape
    if out_weight.shape != (hidden_size, hidden_size):
        raise ValueError(f"Expected out_weight shape [{hidden_size}, {hidden_size}] but got {out_weight.shape}")
    
    # Set default softmax scale if not provided
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    
    # Create output tensor
    output = torch.zeros_like(hidden_states)
    
    # Prepare attention mask if provided
    use_mask = mask is not None
    has_bias = qkv_bias is not None and out_bias is not None
    use_dropout = dropout_p > 0.0
    
    # Process mask if provided
    if use_mask:
        # Ensure mask has the right shape
        if mask.dim() == 2:  # [batch_size, seq_len]
            # Expand to [batch_size, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.dim() == 3:
            if mask.shape[1] == 1:  # [batch_size, 1, seq_len]
                # Expand to [batch_size, 1, 1, seq_len]
                mask = mask.unsqueeze(2)
            else:  # [batch_size, seq_len, seq_len]
                # Reshape to [batch_size, 1, seq_len, seq_len]
                mask = mask.unsqueeze(1)
        elif mask.dim() == 4:  # [batch_size, num_heads, seq_len, seq_len]
            # Already in the right format
            pass
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")
        
        # Ensure mask is on the same device and has the right dtype
        if mask.device != hidden_states.device:
            mask = mask.to(hidden_states.device)
        
        # Convert mask to float for compatibility with the kernel
        if not mask.is_floating_point():
            mask = mask.to(torch.float32)
    
    # Select optimal block sizes based on sequence length and hardware
    # Larger block sizes are more efficient but use more shared memory
    dmodel_block = min(head_dim, 64)  # Typical head dimension is 64-128
    
    if block_size > 256:
        # For very long sequences, use smaller blocks to avoid shared memory limits
        if seq_len > 4096:
            block_size = 64
        elif seq_len > 2048:
            block_size = 128
        else:
            block_size = 256
    
    # Ensure block size doesn't exceed sequence length
    block_size = min(block_size, seq_len)
    
    # Calculate number of heads per block to avoid exceeding shared memory
    heads_per_block = max(1, min(16, num_heads))
    
    # Determine grid dimensions for parallelism (batch, heads*seq_blocks)
    grid = (
        batch_size,                           # Batch dimension
        num_heads * triton.cdiv(seq_len, block_size)  # Head and sequence dimensions
    )
    
    # Prepare empty tensors for bias if not provided
    qkv_bias_ptr = qkv_bias if qkv_bias is not None else torch.zeros(3 * hidden_size, device=hidden_states.device)
    out_bias_ptr = out_bias if out_bias is not None else torch.zeros(hidden_size, device=hidden_states.device)
    mask_ptr = mask if use_mask else torch.zeros((1, 1, 1, 1), device=hidden_states.device)
    
    # Try to launch fused kernel with error handling
    try:
        _fused_attention_kernel[grid](
            # Pointers to matrices
            hidden_states, qkv_weight, qkv_bias_ptr, out_weight, out_bias_ptr, output, mask_ptr,
            # Matrix dimensions
            batch_size, seq_len, hidden_size, num_heads, head_dim,
            # Strides for different dimensions
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            qkv_weight.stride(0), qkv_weight.stride(1),
            qkv_bias_ptr.stride(0) if has_bias else 0,
            out_weight.stride(0), out_weight.stride(1),
            out_bias_ptr.stride(0) if has_bias else 0,
            output.stride(0), output.stride(1), output.stride(2),
            mask_ptr.stride(0) if use_mask else 0,
            mask_ptr.stride(1) if use_mask else 0,
            mask_ptr.stride(3) if use_mask else 0,
            # Other parameters
            softmax_scale, dropout_p,
            # Meta-parameters
            BLOCK_M=block_size,
            BLOCK_DMODEL=dmodel_block,
            BLOCK_N=block_size,
            CAUSAL=causal,
            USE_MASK=use_mask,
            USE_DROPOUT=use_dropout,
            HAS_BIAS=has_bias,
            IS_FP8=False,
            USE_ORTHO_MAT=False,
            ortho_matrix_ptr=0
        )
        return output
    except Exception as e:
        print(f"Triton kernel launch failed with error: {e}")
        print("Falling back to separate operations implementation")
        
        # Fallback implementation using separate operations
        # 1. QKV projection
        qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
        qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # 2. Self-attention
        attention_output = triton_flash_attention(
            q=q,
            k=k,
            v=v,
            mask=mask,
            causal=causal,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            return_softmax=False,
            block_size=block_size
        )
        
        # 3. Reshape output
        attention_output = attention_output.view(batch_size, seq_len, hidden_size)
        
        # 4. Output projection
        output = F.linear(attention_output, out_weight, out_bias)
        
        return output


def pytorch_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
    return_softmax: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    PyTorch implementation of flash attention (used when Triton is not available).
    
    This is a block-based implementation that avoids materializing the full attention matrix.
    
    Args:
        q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
        v: Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
        mask: Optional attention mask
        causal: Whether to apply causal masking
        softmax_scale: Scale factor for softmax (default: 1/sqrt(head_dim))
        dropout_p: Attention dropout probability
        return_softmax: Whether to return softmax attention weights
        
    Returns:
        output: Attention output
        (Optional) attention_weights: If return_softmax=True, returns attention weights
    """
    # Extract dimensions
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Set softmax scale if not provided
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    
    # Create output tensor
    output = torch.zeros_like(q)
    
    # Initialize tracking variables for stable softmax
    m_i = torch.full((batch_size, num_heads, seq_len), float('-inf'), device=q.device)
    l_i = torch.zeros((batch_size, num_heads, seq_len), device=q.device)
    
    # Store attention weights if needed
    if return_softmax:
        attention_weights = torch.zeros(
            (batch_size, num_heads, seq_len, seq_len), 
            device=q.device, 
            dtype=q.dtype
        )
    
    # Process in blocks
    block_size = min(128, seq_len)
    
    for block_start in range(0, seq_len, block_size):
        block_end = min(block_start + block_size, seq_len)
        
        # Current block range
        curr_block = slice(block_start, block_end)
        
        # If causal, we only need to compute keys up to current position
        k_end = block_end if causal else seq_len
        k_range = slice(0, k_end)
        
        # Extract current blocks
        q_block = q[:, curr_block, :, :]  # [B, block_size, H, D]
        k_block = k[:, k_range, :, :]     # [B, k_end, H, D]
        v_block = v[:, k_range, :, :]     # [B, k_end, H, D]
        
        # Compute attention scores
        scores = torch.einsum("bshd,bkhd->bhsk", q_block, k_block) * softmax_scale
        
        # Apply causal mask if needed
        if causal:
            causal_mask = torch.ones((block_end - block_start, k_end), device=scores.device, dtype=torch.bool)
            causal_mask = torch.triu(causal_mask, diagonal=block_start+1)
            scores.masked_fill_(causal_mask.view(1, 1, block_end - block_start, k_end), -float('inf'))
        
        # Apply attention mask if provided
        if mask is not None:
            # Handle different mask shapes
            if mask.dim() == 2:  # [B, S]
                mask_block = mask[:, k_range].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, k_end]
            elif mask.dim() == 3 and mask.shape[1] == 1:  # [B, 1, S]
                mask_block = mask[:, :, k_range].unsqueeze(2)  # [B, 1, 1, k_end]
            elif mask.dim() == 3:  # [B, S, S]
                mask_block = mask[:, curr_block, k_range].unsqueeze(1)  # [B, 1, block_size, k_end]
            elif mask.dim() == 4:  # [B, H, S, S]
                mask_block = mask[:, :, curr_block, k_range]  # [B, H, block_size, k_end]
            else:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")
            
            scores.masked_fill_(~mask_block.bool(), -float('inf'))
        
        # Find max for numerical stability
        m_block = torch.max(scores, dim=-1, keepdim=True)[0]  # [B, H, block_size, 1]
        scores_scaled = scores - m_block  # [B, H, block_size, k_end]
        
        # Compute local softmax
        exp_scores = torch.exp(scores_scaled)  # [B, H, block_size, k_end]
        exp_sum = torch.sum(exp_scores, dim=-1, keepdim=True)  # [B, H, block_size, 1]
        
        # Update tracking variables for global softmax
        m_i_prev = m_i[:, :, curr_block].unsqueeze(-1)  # [B, H, block_size, 1]
        l_i_prev = l_i[:, :, curr_block].unsqueeze(-1)  # [B, H, block_size, 1]
        
        # Compute new max values
        m_i_new = torch.maximum(m_block, m_i_prev)  # [B, H, block_size, 1]
        
        # Update normalizing factors
        exp_diff1 = torch.exp(m_i_prev - m_i_new)  # [B, H, block_size, 1]
        exp_diff2 = torch.exp(m_block - m_i_new)  # [B, H, block_size, 1]
        l_i_new = exp_diff1 * l_i_prev + exp_diff2 * exp_sum  # [B, H, block_size, 1]
        
        # Update tracking variables
        m_i[:, :, curr_block] = m_i_new.squeeze(-1)
        l_i[:, :, curr_block] = l_i_new.squeeze(-1)
        
        # Compute weighted values
        weighted_values = torch.einsum("bhsk,bkhd->bshd", exp_scores, v_block)  # [B, block_size, H, D]
        
        # Update output
        output[:, curr_block] = output[:, curr_block] + weighted_values * exp_diff1.permute(0, 2, 1, 3)
        
        # Store attention weights if requested
        if return_softmax:
            attention_weights[:, :, curr_block, :k_end] = exp_scores / exp_sum
    
    # Normalize output
    output = output / l_i.unsqueeze(-1).permute(0, 2, 1, 3)
    
    # Apply dropout if needed
    if dropout_p > 0.0 and return_softmax:
        dropout_mask = torch.empty_like(attention_weights).bernoulli_(1 - dropout_p)
        dropout_mask = dropout_mask / (1 - dropout_p)
        attention_weights = attention_weights * dropout_mask
        
        # Recompute output with dropout applied weights
        output = torch.einsum("bhsk,bkhd->bshd", attention_weights, v)
    elif dropout_p > 0.0:
        dropout_mask = torch.empty_like(output).bernoulli_(1 - dropout_p)
        dropout_mask = dropout_mask / (1 - dropout_p)
        output = output * dropout_mask
    
    if return_softmax:
        return output, attention_weights
    else:
        return output


def pytorch_fused_attention(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: Optional[torch.Tensor],
    out_weight: torch.Tensor,
    out_bias: Optional[torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    num_heads: int = 8,
    head_dim: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None
) -> torch.Tensor:
    """
    PyTorch implementation of fused attention (used when Triton is not available).
    
    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        qkv_weight: QKV projection weight
        qkv_bias: Optional QKV projection bias
        out_weight: Output projection weight
        out_bias: Optional output projection bias
        mask: Optional attention mask
        causal: Whether to apply causal masking
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head (default: hidden_size // num_heads)
        dropout_p: Attention dropout probability
        softmax_scale: Scale factor for softmax
        
    Returns:
        output: Output tensor of shape [batch_size, seq_len, hidden_size]
    """
    # Extract dimensions
    batch_size, seq_len, hidden_size = hidden_states.shape
    if head_dim is None:
        head_dim = hidden_size // num_heads
    
    # 1. QKV projection
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv = qkv.view(batch_size, seq_len, 3, num_heads, head_dim)
    q, k, v = qkv.unbind(dim=2)
    
    # 2. Self-attention
    attention_output = pytorch_flash_attention(
        q=q,
        k=k,
        v=v,
        mask=mask,
        causal=causal,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        return_softmax=False
    )
    
    # 3. Reshape output
    attention_output = attention_output.view(batch_size, seq_len, hidden_size)
    
    # 4. Output projection
    output = F.linear(attention_output, out_weight, out_bias)
    
    return output


#-----------------------------------------------------------------------------
# Benchmarking Functions
#-----------------------------------------------------------------------------

def benchmark_flash_attention(
    seq_len: int, 
    batch_size: int,
    num_heads: int,
    head_dim: int,
    device: str = "cuda",
    causal: bool = False,
    iterations: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Benchmark Flash Attention performance.
    
    Args:
        seq_len: Sequence length
        batch_size: Batch size
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        device: Device to run benchmark on
        causal: Whether to use causal masking
        iterations: Number of iterations for benchmarking
        warmup: Number of warmup iterations
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    # Skip benchmark if CUDA is not available and device is cuda
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return {
            "sequence_length": seq_len,
            "batch_size": batch_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "causal": causal,
            "flash_attention_ms": 0.0,
            "pytorch_attention_ms": 0.0,
            "speedup": 0.0
        }
    
    # Create test tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    
    # Warm up
    for _ in range(warmup):
        if HAS_TRITON:
            _ = triton_flash_attention(q, k, v, causal=causal)
        _ = pytorch_flash_attention(q, k, v, causal=causal)
    
    # Benchmark PyTorch implementation
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    for _ in range(iterations):
        _ = pytorch_flash_attention(q, k, v, causal=causal)
        
    torch.cuda.synchronize() if device == "cuda" else None
    pytorch_time = (time.time() - start_time) * 1000 / iterations  # ms
    
    # Benchmark Flash Attention
    if HAS_TRITON:
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()
        
        for _ in range(iterations):
            _ = triton_flash_attention(q, k, v, causal=causal)
            
        torch.cuda.synchronize() if device == "cuda" else None
        flash_time = (time.time() - start_time) * 1000 / iterations  # ms
        speedup = pytorch_time / flash_time
    else:
        flash_time = pytorch_time
        speedup = 1.0
    
    return {
        "sequence_length": seq_len,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "causal": causal,
        "flash_attention_ms": flash_time,
        "pytorch_attention_ms": pytorch_time,
        "speedup": speedup
    }


def compare_with_standard_attention(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compare Flash Attention with standard attention.
    
    Args:
        seq_len: Sequence length
        batch_size: Batch size
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        device: Device to run benchmark on
        
    Returns:
        Dictionary with comparison results
    """
    # Skip if CUDA is not available and device is cuda
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping comparison")
        return {
            "max_difference": 0.0,
            "is_correct": False,
            "memory_standard_mb": 0.0,
            "memory_flash_mb": 0.0,
            "memory_reduction": 0.0
        }
    
    # Create test tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    
    # Standard attention function
    def standard_attention(q, k, v):
        # [batch_size, num_heads, seq_len, seq_len]
        scores = torch.einsum("bshd,bkhd->bhsk", q, k) / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.einsum("bhsk,bkhd->bshd", attn_weights, v)
        return output
    
    # Compute with standard attention (record memory usage)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
    standard_output = standard_attention(q, k, v)
    
    if device == "cuda":
        memory_standard = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    else:
        memory_standard = 0.0
    
    # Compute with Flash Attention (record memory usage)
    if HAS_TRITON:
        flash_fn = triton_flash_attention
    else:
        flash_fn = pytorch_flash_attention
        
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
    flash_output = flash_fn(q, k, v)
    
    if device == "cuda":
        memory_flash = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    else:
        memory_flash = 0.0
    
    # Compare outputs
    max_diff = torch.max(torch.abs(standard_output - flash_output)).item()
    is_correct = max_diff < 1e-3
    
    return {
        "max_difference": max_diff,
        "is_correct": is_correct,
        "memory_standard_mb": memory_standard,
        "memory_flash_mb": memory_flash,
        "memory_reduction": memory_standard / max(memory_flash, 1e-6),
        "sequence_length": seq_len,
        "can_handle_longer_sequences": seq_len < 4096
    }


def compare_with_xformers(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compare Flash Attention with xFormers memory-efficient attention.
    
    Args:
        seq_len: Sequence length
        batch_size: Batch size
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        device: Device to run benchmark on
        
    Returns:
        Dictionary with comparison results
    """
    # Check if xFormers is available
    try:
        import xformers.ops as xops
        has_xformers = True
    except ImportError:
        has_xformers = False
    
    if not has_xformers:
        print("xFormers not available, skipping comparison")
        return {
            "has_xformers": False,
            "flash_attention_ms": 0.0,
            "xformers_attention_ms": 0.0,
            "speedup_ratio": 0.0
        }
    
    # Skip if CUDA is not available and device is cuda
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping comparison")
        return {
            "has_xformers": True,
            "flash_attention_ms": 0.0,
            "xformers_attention_ms": 0.0,
            "speedup_ratio": 0.0
        }
    
    import time
    
    # Create test tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    
    # Benchmark xFormers
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    for _ in range(10):  # Just a few iterations for comparison
        # Reshape for xFormers [B, S, H, D] -> [B, H, S, D]
        q_xf = q.transpose(1, 2)
        k_xf = k.transpose(1, 2)
        v_xf = v.transpose(1, 2)
        
        # Run xFormers attention
        output_xf = xops.memory_efficient_attention(q_xf, k_xf, v_xf)
        
        # Reshape back [B, H, S, D] -> [B, S, H, D]
        output_xf = output_xf.transpose(1, 2)
        
    torch.cuda.synchronize() if device == "cuda" else None
    xformers_time = (time.time() - start_time) * 1000 / 10  # ms
    
    # Benchmark Flash Attention
    if HAS_TRITON:
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()
        
        for _ in range(10):  # Same iterations for fair comparison
            _ = triton_flash_attention(q, k, v)
            
        torch.cuda.synchronize() if device == "cuda" else None
        flash_time = (time.time() - start_time) * 1000 / 10  # ms
    else:
        flash_time = pytorch_flash_attention(q, k, v)
    
    return {
        "has_xformers": True,
        "flash_attention_ms": flash_time,
        "xformers_attention_ms": xformers_time,
        "speedup_ratio": xformers_time / flash_time,
        "sequence_length": seq_len,
        "batch_size": batch_size
    }