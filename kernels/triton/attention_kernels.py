"""
Optimized Triton Kernels for Attention Mechanisms

This module provides highly-optimized Triton kernels for attention mechanisms,
with a focus on Ring Attention for memory-efficient processing of long sequences.
These kernels are designed to maximize throughput and minimize memory usage on
modern GPU architectures.

Key features:
- Block-based processing for better cache utilization
- Fused operations to reduce memory bandwidth usage
- Support for different precision modes (FP16, BF16, FP32)
- Optimized specifically for large sequence lengths
- Ring attention pattern implementation for O(N) memory scaling

Note: Triton is a language and compiler for writing highly efficient GPU kernels
with a focus on deep learning applications.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Union, Any

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Triton is not available. Using PyTorch fallback for attention kernels.")


if TRITON_AVAILABLE:
    @triton.jit
    def _ring_attention_forward_kernel(
        # Pointers to matrices
        q_ptr, k_ptr, v_ptr, o_ptr,
        # Matrix dimensions
        batch_size, num_heads, seq_len, head_dim,
        # Strides for accessing memory
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        # Attention mask pointer and strides if applicable
        mask_ptr, stride_mb, stride_mh, stride_ms1, stride_ms2,
        # Chunk sizes for ring attention
        q_chunk_size, k_chunk_size,
        # Additional parameters
        scale_factor: tl.constexpr,
        use_mask: tl.constexpr,
        # Block sizes for the grid
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        # Number of warps for parallelism
        num_warps: tl.constexpr,
        num_stages: tl.constexpr
    ):
        """Triton kernel for ring attention forward pass.
        
        This kernel computes attention in a memory-efficient manner by processing data
        in blocks and using a ring communication pattern.
        
        Args:
            q_ptr: Query tensor pointer
            k_ptr: Key tensor pointer
            v_ptr: Value tensor pointer
            o_ptr: Output tensor pointer
            (various dimensions and strides): Matrix dimensions and memory layout information
            mask_ptr: Optional attention mask pointer
            (chunk sizes): Sizes for chunked processing
            scale_factor: Scale factor for attention scores
            BLOCK_SIZE_*: Block sizes for tiled computation
            num_warps: Number of warps for parallelism
            num_stages: Number of pipeline stages
        """
        # Get the program ID
        pid = tl.program_id(0)
        
        # Compute batch and head indices
        batch_id = pid // (num_heads * ((seq_len + q_chunk_size - 1) // q_chunk_size))
        head_id = (pid % (num_heads * ((seq_len + q_chunk_size - 1) // q_chunk_size))) // ((seq_len + q_chunk_size - 1) // q_chunk_size)
        q_chunk_id = pid % ((seq_len + q_chunk_size - 1) // q_chunk_size)
        
        # Compute actual query chunk size (handling edge cases)
        curr_q_chunk_size = min(q_chunk_size, seq_len - q_chunk_id * q_chunk_size)
        
        # Exit if this is an empty chunk
        if curr_q_chunk_size <= 0:
            return
            
        # Calculate the start of this query chunk
        q_chunk_start = q_chunk_id * q_chunk_size
        
        # Compute pointers to the query chunk
        q_block_ptr = q_ptr + batch_id * stride_qb + head_id * stride_qh + q_chunk_start * stride_qs
        o_block_ptr = o_ptr + batch_id * stride_ob + head_id * stride_oh + q_chunk_start * stride_os
        
        # Allocate accumulator for the output
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        
        # We'll accumulate softmax normalization info
        m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        
        # Iterate through key chunks using a ring pattern
        for k_chunk_id in range((seq_len + k_chunk_size - 1) // k_chunk_size):
            # Compute actual key chunk size (handling edge cases)
            curr_k_chunk_size = min(k_chunk_size, seq_len - k_chunk_id * k_chunk_size)
            
            # Skip if this is an empty chunk
            if curr_k_chunk_size <= 0:
                continue
                
            # Calculate the start of this key chunk
            k_chunk_start = k_chunk_id * k_chunk_size
            
            # Compute pointers to the key and value chunks
            k_block_ptr = k_ptr + batch_id * stride_kb + head_id * stride_kh + k_chunk_start * stride_ks
            v_block_ptr = v_ptr + batch_id * stride_vb + head_id * stride_vh + k_chunk_start * stride_vs
            
            # Optional mask pointer for this chunk
            mask_block_ptr = 0
            if use_mask:
                mask_block_ptr = mask_ptr + batch_id * stride_mb + head_id * stride_mh + q_chunk_start * stride_ms1 + k_chunk_start * stride_ms2
            
            # Process blocks within each chunk
            for block_m in range(0, curr_q_chunk_size, BLOCK_SIZE_M):
                for block_n in range(0, curr_k_chunk_size, BLOCK_SIZE_N):
                    # Compute ranges for this block
                    m_size = min(BLOCK_SIZE_M, curr_q_chunk_size - block_m)
                    n_size = min(BLOCK_SIZE_N, curr_k_chunk_size - block_n)
                    
                    # Load the query block
                    q_block = tl.load(
                        q_block_ptr + block_m * stride_qs + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_qs + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_qd,
                        mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < m_size) & (tl.arange(0, BLOCK_SIZE_K)[None, :] < head_dim),
                        other=0.0
                    )
                    
                    # Load the key block and transpose it
                    k_block = tl.load(
                        k_block_ptr + block_n * stride_ks + tl.arange(0, BLOCK_SIZE_N)[:, None] * stride_ks + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_kd,
                        mask=(tl.arange(0, BLOCK_SIZE_N)[:, None] < n_size) & (tl.arange(0, BLOCK_SIZE_K)[None, :] < head_dim),
                        other=0.0
                    )
                    k_block = tl.trans(k_block)
                    
                    # Compute attention scores for this block
                    scores = tl.dot(q_block, k_block, allow_tf32=False)
                    scores = scores * scale_factor
                    
                    # Apply mask if provided
                    if use_mask:
                        mask_chunk = tl.load(
                            mask_block_ptr + block_m * stride_ms1 + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_ms1 +
                            block_n * stride_ms2 + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_ms2,
                            mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < m_size) & (tl.arange(0, BLOCK_SIZE_N)[None, :] < n_size),
                            other=0.0
                        )
                        scores = scores + mask_chunk
                    
                    # Compute local attention weight normalization components
                    m_ij = tl.max(scores, axis=1)
                    p_ij = tl.exp(scores - m_ij[:, None])
                    l_ij = tl.sum(p_ij, axis=1)
                    
                    # Load value block
                    v_block = tl.load(
                        v_block_ptr + block_n * stride_vs + tl.arange(0, BLOCK_SIZE_N)[:, None] * stride_vs + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_vd,
                        mask=(tl.arange(0, BLOCK_SIZE_N)[:, None] < n_size) & (tl.arange(0, BLOCK_SIZE_K)[None, :] < head_dim),
                        other=0.0
                    )
                    
                    # Update running softmax stats for normalization across chunks
                    m_i_new = tl.maximum(m_i, m_ij)
                    alpha = tl.exp(m_i - m_i_new)
                    beta = tl.exp(m_ij - m_i_new)
                    
                    # Correct the running sum
                    acc = acc * alpha[:, None]
                    l_i = l_i * alpha + beta * l_ij
                    m_i = m_i_new
                    
                    # Compute and accumulate attention outputs for this block
                    chunk_out = tl.dot(p_ij, v_block, allow_tf32=False)
                    acc = acc + chunk_out * beta[:, None]
            
            # End of key chunk loop
        
        # Final normalization of accumulated outputs
        acc = acc / l_i[:, None]
        
        # Write output
        for block_m in range(0, curr_q_chunk_size, BLOCK_SIZE_M):
            m_size = min(BLOCK_SIZE_M, curr_q_chunk_size - block_m)
            tl.store(
                o_block_ptr + block_m * stride_os + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_os + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_od,
                acc,
                mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < m_size) & (tl.arange(0, BLOCK_SIZE_K)[None, :] < head_dim)
            )


    @triton.jit
    def _fused_ring_attention_kernel(
        # Pointers to matrices
        hidden_states_ptr, qkv_weight_ptr, qkv_bias_ptr,
        output_ptr, output_weight_ptr, output_bias_ptr,
        # Matrix dimensions
        batch_size, seq_len, hidden_size, num_heads, head_dim,
        # Strides for accessing memory
        stride_hsb, stride_hss, stride_hsh,
        stride_qkvw1, stride_qkvw2,
        stride_qkvb,
        stride_outw1, stride_outw2,
        stride_outb,
        stride_ob, stride_os, stride_oh,
        # Attention mask pointer and strides if applicable
        mask_ptr, stride_mb, stride_mh, stride_ms1, stride_ms2,
        # Chunk sizes for ring attention
        chunk_size,
        # Additional parameters
        scale_factor: tl.constexpr,
        use_mask: tl.constexpr,
        use_bias: tl.constexpr,
        # Block sizes for the grid
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        # Number of warps for parallelism
        num_warps: tl.constexpr,
        num_stages: tl.constexpr
    ):
        """Triton kernel for fused ring attention with QKV projection.
        
        This kernel combines QKV projection, attention computation, and output
        projection into a single kernel to maximize efficiency.
        
        Args:
            hidden_states_ptr: Input hidden states pointer
            qkv_weight_ptr: QKV projection weight pointer
            qkv_bias_ptr: QKV projection bias pointer
            output_ptr: Output tensor pointer
            output_weight_ptr: Output projection weight pointer
            output_bias_ptr: Output projection bias pointer
            (various dimensions and strides): Matrix dimensions and memory layout information
            mask_ptr: Optional attention mask pointer
            chunk_size: Size for chunked processing
            scale_factor: Scale factor for attention scores
            use_mask: Whether to use attention mask
            use_bias: Whether to use bias
            BLOCK_SIZE_*: Block sizes for tiled computation
            num_warps: Number of warps for parallelism
            num_stages: Number of pipeline stages
        """
        # Get the program ID
        pid = tl.program_id(0)
        
        # Compute batch and block indices
        batch_id = pid // ((seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
        block_id = pid % ((seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
        seq_start = block_id * BLOCK_SIZE_M
        
        # Compute actual block size (handling edge cases)
        curr_block_size = min(BLOCK_SIZE_M, seq_len - seq_start)
        
        # Exit if this is an empty block
        if curr_block_size <= 0:
            return
        
        # Input pointers for this block
        hidden_ptr = hidden_states_ptr + batch_id * stride_hsb + seq_start * stride_hss
        
        # Output pointers for this block
        out_ptr = output_ptr + batch_id * stride_ob + seq_start * stride_os
        
        # Allocate shared memory for intermediate QKV values
        qkv_buffer = tl.zeros((BLOCK_SIZE_M, 3, hidden_size), dtype=tl.float32)
        
        # Compute QKV projections
        for m in range(0, curr_block_size, 1):
            # Load one token of hidden states
            h = tl.load(
                hidden_ptr + m * stride_hss + tl.arange(0, hidden_size) * stride_hsh,
                mask=tl.arange(0, hidden_size) < hidden_size,
                other=0.0
            )
            
            # Compute QKV projections for this token
            for qkv_idx in range(3):
                for j in range(0, hidden_size, BLOCK_SIZE_N):
                    # Compute actual column block size
                    n_size = min(BLOCK_SIZE_N, hidden_size - j)
                    
                    # Load weight block
                    w = tl.load(
                        qkv_weight_ptr + qkv_idx * hidden_size * stride_qkvw1 + j * stride_qkvw2 + tl.arange(0, BLOCK_SIZE_N) * stride_qkvw2,
                        mask=tl.arange(0, BLOCK_SIZE_N) < n_size,
                        other=0.0
                    )
                    
                    # Compute partial sum for this weight block
                    partial = h[j:j+n_size] * w
                    
                    # Accumulate
                    for k in range(n_size):
                        qkv_buffer[m, qkv_idx, j+k] += partial[k]
            
            # Add bias if specified
            if use_bias:
                for qkv_idx in range(3):
                    bias = tl.load(
                        qkv_bias_ptr + qkv_idx * hidden_size * stride_qkvb + tl.arange(0, hidden_size) * stride_qkvb,
                        mask=tl.arange(0, hidden_size) < hidden_size,
                        other=0.0
                    )
                    qkv_buffer[m, qkv_idx, :] += bias
        
        # Reshape QKV for attention computation 
        # [block_size, 3, hidden_size] -> [3, block_size, num_heads, head_dim]
        q_buffer = tl.reshape(qkv_buffer[:curr_block_size, 0], (curr_block_size, num_heads, head_dim))
        k_buffer = tl.reshape(qkv_buffer[:curr_block_size, 1], (curr_block_size, num_heads, head_dim))
        v_buffer = tl.reshape(qkv_buffer[:curr_block_size, 2], (curr_block_size, num_heads, head_dim))
        
        # Transposed for attention computation
        q = tl.reshape(q_buffer, (curr_block_size, num_heads, head_dim))
        k = tl.reshape(k_buffer, (curr_block_size, num_heads, head_dim))
        v = tl.reshape(v_buffer, (curr_block_size, num_heads, head_dim))
        
        # Scale query
        q = q * scale_factor
        
        # Initialize output buffer
        out_buffer = tl.zeros((curr_block_size, num_heads, head_dim), dtype=tl.float32)
        
        # Compute chunked attention
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        # Initialize normalization accumulators
        max_scores = tl.zeros((curr_block_size, num_heads), dtype=tl.float32) - float("inf")
        sum_exp = tl.zeros((curr_block_size, num_heads), dtype=tl.float32)
        
        # Process all key/value chunks
        for chunk_idx in range(num_chunks):
            # Determine chunk boundaries
            k_start = chunk_idx * chunk_size
            k_end = min(k_start + chunk_size, seq_len)
            
            # Skip empty chunks
            if k_end <= k_start:
                continue
            
            # Process this chunk for all heads and tokens in the current block
            for head_idx in range(num_heads):
                for q_idx in range(curr_block_size):
                    # Extract the query vector for the current token and head
                    q_vec = q[q_idx, head_idx]
                    
                    # Initialize local accumulators for this token and head
                    local_max = float("-inf")
                    local_sum_exp = 0.0
                    local_out = tl.zeros((head_dim,), dtype=tl.float32)
                    
                    # Process all keys/values in the current chunk
                    for k_idx in range(k_start, k_end):
                        # Extract key and value vectors
                        k_vec = k[k_idx - k_start, head_idx] if k_idx - k_start < curr_block_size else \
                                tl.load(
                                    hidden_states_ptr + batch_id * stride_hsb + k_idx * stride_hss + 
                                    (hidden_size + head_idx * head_dim) * stride_hsh + tl.arange(0, head_dim) * stride_hsh,
                                    mask=tl.arange(0, head_dim) < head_dim,
                                    other=0.0
                                )
                        
                        # Compute attention score
                        score = tl.sum(q_vec * k_vec)
                        
                        # Apply mask if provided
                        if use_mask:
                            mask_val = tl.load(
                                mask_ptr + batch_id * stride_mb + head_idx * stride_mh + 
                                (seq_start + q_idx) * stride_ms1 + k_idx * stride_ms2
                            )
                            score = score + mask_val
                        
                        # Update local max score
                        local_max = tl.maximum(local_max, score)
                        
                        # Compute attention weight
                        exp_score = tl.exp(score - local_max)
                        local_sum_exp += exp_score
                        
                        # Get value vector
                        v_vec = v[k_idx - k_start, head_idx] if k_idx - k_start < curr_block_size else \
                                tl.load(
                                    hidden_states_ptr + batch_id * stride_hsb + k_idx * stride_hss + 
                                    (2 * hidden_size + head_idx * head_dim) * stride_hsh + tl.arange(0, head_dim) * stride_hsh,
                                    mask=tl.arange(0, head_dim) < head_dim,
                                    other=0.0
                                )
                        
                        # Accumulate weighted value
                        local_out += exp_score * v_vec
                    
                    # Update the global accumulators with the local results from this chunk
                    old_max = max_scores[q_idx, head_idx]
                    new_max = tl.maximum(old_max, local_max)
                    
                    # Reweight sums based on max score differences
                    old_scale = tl.exp(old_max - new_max)
                    new_scale = tl.exp(local_max - new_max)
                    
                    # Update output buffer
                    out_buffer[q_idx, head_idx] = out_buffer[q_idx, head_idx] * old_scale + local_out * new_scale
                    
                    # Update normalization terms
                    sum_exp[q_idx, head_idx] = sum_exp[q_idx, head_idx] * old_scale + local_sum_exp * new_scale
                    max_scores[q_idx, head_idx] = new_max
        
        # Apply final normalization
        for q_idx in range(curr_block_size):
            for head_idx in range(num_heads):
                if sum_exp[q_idx, head_idx] > 0:
                    out_buffer[q_idx, head_idx] = out_buffer[q_idx, head_idx] / sum_exp[q_idx, head_idx]
        
        # Reshape output buffer back to [curr_block_size, hidden_size]
        out_proj_buffer = tl.reshape(out_buffer, (curr_block_size, hidden_size))
        
        # Compute output projection
        final_output = tl.zeros((curr_block_size, hidden_size), dtype=tl.float32)
        
        for m in range(curr_block_size):
            for j in range(0, hidden_size, BLOCK_SIZE_N):
                # Compute actual column block size
                n_size = min(BLOCK_SIZE_N, hidden_size - j)
                
                # Compute output projection
                for k in range(0, hidden_size, BLOCK_SIZE_N):
                    # Compute actual inner dimension size
                    k_size = min(BLOCK_SIZE_N, hidden_size - k)
                    
                    # Load output projection weights
                    w = tl.load(
                        output_weight_ptr + j * stride_outw1 + k * stride_outw2 + 
                        tl.arange(0, BLOCK_SIZE_N)[:, None] * stride_outw1 + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_outw2,
                        mask=(tl.arange(0, BLOCK_SIZE_N)[:, None] < n_size) & (tl.arange(0, BLOCK_SIZE_N)[None, :] < k_size),
                        other=0.0
                    )
                    
                    # Compute matrix multiplication
                    for n in range(n_size):
                        for kk in range(k_size):
                            final_output[m, j+n] += out_proj_buffer[m, k+kk] * w[n, kk]
            
            # Add output bias if specified
            if use_bias:
                bias = tl.load(
                    output_bias_ptr + tl.arange(0, hidden_size) * stride_outb,
                    mask=tl.arange(0, hidden_size) < hidden_size,
                    other=0.0
                )
                final_output[m, :] += bias
        
        # Write final output
        for m in range(curr_block_size):
            tl.store(
                out_ptr + m * stride_os + tl.arange(0, hidden_size) * stride_oh,
                final_output[m],
                mask=tl.arange(0, hidden_size) < hidden_size
            )


    @triton.jit
    def _chunk_based_attention_kernel(
        # Pointers to matrices
        q_ptr, k_ptr, v_ptr, o_ptr,
        # Matrix dimensions
        batch_size, num_heads, seq_len, head_dim,
        # Strides for accessing memory
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        # Attention mask pointer and strides if applicable
        mask_ptr, stride_mb, stride_mh, stride_ms1, stride_ms2,
        # Chunk size for processing
        chunk_size: tl.constexpr,
        # Additional parameters
        scale_factor: tl.constexpr,
        use_mask: tl.constexpr,
        # Block sizes for the grid
        BLOCK_SIZE: tl.constexpr,
        # Number of warps for parallelism
        num_warps: tl.constexpr
    ):
        """Triton kernel for chunk-based attention processing.
        
        This kernel computes attention by processing data in fixed chunks to 
        optimize memory usage.
        
        Args:
            q_ptr, k_ptr, v_ptr, o_ptr: Pointers to input/output tensors
            (various dimensions and strides): Matrix dimensions and memory layout
            mask_ptr: Optional attention mask pointer
            chunk_size: Size of chunks for processing
            scale_factor: Scale factor for attention scores
            use_mask: Whether to use attention mask
            BLOCK_SIZE: Block size for computation
            num_warps: Number of warps for parallelism
        """
        # Get the program ID
        pid = tl.program_id(0)
        
        # Compute batch, head, and position indices
        batch_id = pid // (num_heads * seq_len)
        head_id = (pid % (num_heads * seq_len)) // seq_len
        seq_pos = pid % seq_len
        
        # Calculate pointers for this position
        q_pos_ptr = q_ptr + batch_id * stride_qb + head_id * stride_qh + seq_pos * stride_qs
        o_pos_ptr = o_ptr + batch_id * stride_ob + head_id * stride_oh + seq_pos * stride_os
        
        # Load query vector for this position
        query = tl.load(
            q_pos_ptr + tl.arange(0, head_dim) * stride_qd,
            mask=tl.arange(0, head_dim) < head_dim,
            other=0.0
        )
        
        # Initialize accumulators for softmax and output
        m_i = float("-inf")  # Max score for numerical stability
        l_i = 0.0  # Sum of exponentials for normalization
        acc = tl.zeros((head_dim,), dtype=tl.float32)
        
        # Process key/value tokens in chunks
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_length = chunk_end - chunk_start
            
            # Skip empty chunks
            if chunk_length <= 0:
                continue
            
            # Initialize local accumulators for this chunk
            m_ij = float("-inf")
            l_ij = 0.0
            chunk_acc = tl.zeros((head_dim,), dtype=tl.float32)
            
            # Process tokens within this chunk
            for k_pos in range(chunk_start, chunk_end, BLOCK_SIZE):
                # Determine actual block size
                block_size = min(BLOCK_SIZE, chunk_end - k_pos)
                
                # Skip empty blocks
                if block_size <= 0:
                    continue
                
                # Load key vectors for this block
                k_pos_ptr = k_ptr + batch_id * stride_kb + head_id * stride_kh + k_pos * stride_ks
                key_block = tl.load(
                    k_pos_ptr + tl.arange(0, block_size)[:, None] * stride_ks + tl.arange(0, head_dim)[None, :] * stride_kd,
                    mask=(tl.arange(0, block_size)[:, None] < block_size) & (tl.arange(0, head_dim)[None, :] < head_dim),
                    other=0.0
                )
                
                # Compute attention scores for this block
                scores = tl.zeros((block_size,), dtype=tl.float32)
                for i in range(block_size):
                    scores[i] = tl.sum(query * key_block[i]) * scale_factor
                
                # Apply mask if provided
                if use_mask:
                    for i in range(block_size):
                        mask_val = tl.load(
                            mask_ptr + batch_id * stride_mb + head_id * stride_mh + 
                            seq_pos * stride_ms1 + (k_pos + i) * stride_ms2,
                            mask=i < block_size,
                            other=0.0
                        )
                        scores[i] = scores[i] + mask_val
                
                # Update max score for numerical stability
                block_max = tl.maximum(scores)
                m_ij = tl.maximum(m_ij, block_max)
                
                # Load value vectors for this block
                v_pos_ptr = v_ptr + batch_id * stride_vb + head_id * stride_vh + k_pos * stride_vs
                value_block = tl.load(
                    v_pos_ptr + tl.arange(0, block_size)[:, None] * stride_vs + tl.arange(0, head_dim)[None, :] * stride_vd,
                    mask=(tl.arange(0, block_size)[:, None] < block_size) & (tl.arange(0, head_dim)[None, :] < head_dim),
                    other=0.0
                )
                
                # Compute exponentials and update normalization sum
                for i in range(block_size):
                    exp_score = tl.exp(scores[i] - m_ij)
                    l_ij += exp_score
                    chunk_acc += exp_score * value_block[i]
            
            # Merge chunk results with running accumulators using softmax decomposition rules
            if m_i == float("-inf"):
                # First chunk
                m_i = m_ij
                l_i = l_ij
                acc = chunk_acc
            else:
                # Subsequent chunks
                m_new = tl.maximum(m_i, m_ij)
                scale_i = tl.exp(m_i - m_new)
                scale_ij = tl.exp(m_ij - m_new)
                
                # Update accumulators
                acc = acc * scale_i + chunk_acc * scale_ij
                l_i = l_i * scale_i + l_ij * scale_ij
                m_i = m_new
        
        # Final normalization
        if l_i > 0:
            acc = acc / l_i
        
        # Write output
        tl.store(
            o_pos_ptr + tl.arange(0, head_dim) * stride_od,
            acc,
            mask=tl.arange(0, head_dim) < head_dim
        )


    # Python wrapper functions for the Triton kernels
    def triton_ring_attention_forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention using Triton's optimized ring attention kernel.
        
        Args:
            query: Query tensor [batch_size, num_heads, q_seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, kv_seq_len, head_dim]
            value: Value tensor [batch_size, num_heads, kv_seq_len, head_dim]
            attention_mask: Optional attention mask [batch_size, 1, q_seq_len, kv_seq_len]
            
        Returns:
            Output tensor after attention [batch_size, q_seq_len, hidden_size]
        """
        # Extract dimensions
        batch_size, num_heads, q_seq_len, head_dim = query.shape
        _, _, kv_seq_len, _ = key.shape
        
        # Determine compute dtype based on input
        dtype = query.dtype
        
        # Create output tensor
        output = torch.zeros(
            (batch_size, num_heads, q_seq_len, head_dim),
            dtype=dtype,
            device=query.device
        )
        
        # Calculate scale factor
        scale_factor = 1.0 / math.sqrt(head_dim)
        
        # Determine chunk sizes based on sequence lengths
        q_chunk_size = min(128, q_seq_len)
        k_chunk_size = min(128, kv_seq_len)
        
        # Calculate grid size
        grid = (batch_size * num_heads * ((q_seq_len + q_chunk_size - 1) // q_chunk_size),)
        
        # Calculate strides for tensor access
        q_strides = (
            query.stride(0), query.stride(1), query.stride(2), query.stride(3)
        )
        k_strides = (
            key.stride(0), key.stride(1), key.stride(2), key.stride(3)
        )
        v_strides = (
            value.stride(0), value.stride(1), value.stride(2), value.stride(3)
        )
        o_strides = (
            output.stride(0), output.stride(1), output.stride(2), output.stride(3)
        )
        
        # Handle optional mask
        use_mask = attention_mask is not None
        if use_mask:
            mask_strides = (
                attention_mask.stride(0),
                attention_mask.stride(1) if attention_mask.dim() > 2 else 0,
                attention_mask.stride(-2),
                attention_mask.stride(-1)
            )
            mask_ptr = attention_mask.data_ptr()
        else:
            mask_strides = (0, 0, 0, 0)
            mask_ptr = 0
        
        # Launch kernel
        _ring_attention_forward_kernel[grid](
            query.data_ptr(), key.data_ptr(), value.data_ptr(), output.data_ptr(),
            batch_size, num_heads, q_seq_len, head_dim,
            q_strides[0], q_strides[1], q_strides[2], q_strides[3],
            k_strides[0], k_strides[1], k_strides[2], k_strides[3],
            v_strides[0], v_strides[1], v_strides[2], v_strides[3],
            o_strides[0], o_strides[1], o_strides[2], o_strides[3],
            mask_ptr, mask_strides[0], mask_strides[1], mask_strides[2], mask_strides[3],
            q_chunk_size, k_chunk_size,
            scale_factor, use_mask,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=head_dim,
            num_warps=8,
            num_stages=1
        )
        
        # Reshape output to match PyTorch's format
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, q_seq_len, num_heads * head_dim)
        
        return output


    def triton_fused_ring_attention(
        hidden_states: torch.Tensor,
        qkv_weight: torch.Tensor,
        qkv_bias: Optional[torch.Tensor],
        output_weight: torch.Tensor,
        output_bias: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention with fused QKV projection using Triton's optimized kernel.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            qkv_weight: QKV projection weight [3 * hidden_size, hidden_size]
            qkv_bias: Optional QKV projection bias [3 * hidden_size]
            output_weight: Output projection weight [hidden_size, hidden_size]
            output_bias: Optional output projection bias [hidden_size]
            attention_mask: Optional attention mask [batch_size, 1, seq_len, seq_len]
            
        Returns:
            Output tensor after attention [batch_size, seq_len, hidden_size]
        """
        # Extract dimensions
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Determine if model has biases
        use_bias = qkv_bias is not None and output_bias is not None
        
        # Check if we're on GPU and the sizes are compatible
        if not hidden_states.is_cuda:
            raise ValueError("Triton kernels require CUDA tensors")
        
        # Head size calculation - assume it's a power of 2 for optimal performance
        num_heads = 0
        head_dim = 0
        for i in range(1, 13):  # Try head sizes from 2 to 4096
            test_dim = 2 ** i
            if hidden_size % test_dim == 0:
                potential_heads = hidden_size // test_dim
                # Prefer more heads with smaller dimensions
                if potential_heads >= 1 and (num_heads == 0 or potential_heads > num_heads):
                    num_heads = potential_heads
                    head_dim = test_dim
        
        if num_heads == 0:
            # Fallback for unusual hidden sizes
            head_dim = 64
            num_heads = hidden_size // head_dim
        
        # Create output tensor
        output = torch.zeros(
            (batch_size, seq_len, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        # Calculate scale factor
        scale_factor = 1.0 / math.sqrt(head_dim)
        
        # Determine chunk size based on sequence length
        chunk_size = min(128, seq_len)
        
        # Calculate grid size - one block per sequence segment
        grid = (batch_size * ((seq_len + 32 - 1) // 32),)
        
        # Calculate strides for tensor access
        hs_strides = (
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2)
        )
        qkvw_strides = (qkv_weight.stride(0), qkv_weight.stride(1))
        qkvb_stride = 1 if use_bias else 0
        outw_strides = (output_weight.stride(0), output_weight.stride(1))
        outb_stride = 1 if use_bias else 0
        out_strides = (output.stride(0), output.stride(1), output.stride(2))
        
        # Handle optional mask
        use_mask = attention_mask is not None
        if use_mask:
            mask_strides = (
                attention_mask.stride(0),
                attention_mask.stride(1) if attention_mask.dim() > 2 else 0,
                attention_mask.stride(-2),
                attention_mask.stride(-1)
            )
            mask_ptr = attention_mask.data_ptr()
        else:
            mask_strides = (0, 0, 0, 0)
            mask_ptr = 0
        
        # Handle optional biases
        qkv_bias_ptr = qkv_bias.data_ptr() if use_bias else 0
        output_bias_ptr = output_bias.data_ptr() if use_bias else 0
        
        # Launch kernel
        _fused_ring_attention_kernel[grid](
            hidden_states.data_ptr(), qkv_weight.data_ptr(), qkv_bias_ptr,
            output.data_ptr(), output_weight.data_ptr(), output_bias_ptr,
            batch_size, seq_len, hidden_size, num_heads, head_dim,
            hs_strides[0], hs_strides[1], hs_strides[2],
            qkvw_strides[0], qkvw_strides[1],
            qkvb_stride,
            outw_strides[0], outw_strides[1],
            outb_stride,
            out_strides[0], out_strides[1], out_strides[2],
            mask_ptr, mask_strides[0], mask_strides[1], mask_strides[2], mask_strides[3],
            chunk_size,
            scale_factor, use_mask, use_bias,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=32,
            num_warps=4,
            num_stages=1
        )
        
        return output


    def triton_chunk_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 128
    ) -> torch.Tensor:
        """Compute attention using Triton's optimized chunk-based kernel.
        
        This version processes each position independently with chunked computation
        for better memory efficiency.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len, head_dim]
            attention_mask: Optional attention mask [batch_size, 1, seq_len, seq_len]
            chunk_size: Size of chunks for processing
            
        Returns:
            Output tensor after attention [batch_size, seq_len, hidden_size]
        """
        # Extract dimensions
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Determine compute dtype based on input
        dtype = query.dtype
        
        # Create output tensor
        output = torch.zeros(
            (batch_size, num_heads, seq_len, head_dim),
            dtype=dtype,
            device=query.device
        )
        
        # Calculate scale factor
        scale_factor = 1.0 / math.sqrt(head_dim)
        
        # Calculate grid size - one thread per output position
        grid = (batch_size * num_heads * seq_len,)
        
        # Calculate strides for tensor access
        q_strides = (
            query.stride(0), query.stride(1), query.stride(2), query.stride(3)
        )
        k_strides = (
            key.stride(0), key.stride(1), key.stride(2), key.stride(3)
        )
        v_strides = (
            value.stride(0), value.stride(1), value.stride(2), value.stride(3)
        )
        o_strides = (
            output.stride(0), output.stride(1), output.stride(2), output.stride(3)
        )
        
        # Handle optional mask
        use_mask = attention_mask is not None
        if use_mask:
            mask_strides = (
                attention_mask.stride(0),
                attention_mask.stride(1) if attention_mask.dim() > 2 else 0,
                attention_mask.stride(-2),
                attention_mask.stride(-1)
            )
            mask_ptr = attention_mask.data_ptr()
        else:
            mask_strides = (0, 0, 0, 0)
            mask_ptr = 0
        
        # Launch kernel
        _chunk_based_attention_kernel[grid](
            query.data_ptr(), key.data_ptr(), value.data_ptr(), output.data_ptr(),
            batch_size, num_heads, seq_len, head_dim,
            q_strides[0], q_strides[1], q_strides[2], q_strides[3],
            k_strides[0], k_strides[1], k_strides[2], k_strides[3],
            v_strides[0], v_strides[1], v_strides[2], v_strides[3],
            o_strides[0], o_strides[1], o_strides[2], o_strides[3],
            mask_ptr, mask_strides[0], mask_strides[1], mask_strides[2], mask_strides[3],
            chunk_size,
            scale_factor, use_mask,
            BLOCK_SIZE=16,
            num_warps=4
        )
        
        # Reshape output to match PyTorch's format
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, seq_len, num_heads * head_dim)
        
        return output


    # Utility functions for benchmarking and comparing
    def benchmark_attention_kernels(
        seq_len: int,
        batch_size: int,
        hidden_size: int,
        num_heads: int,
        num_trials: int = 10
    ) -> Dict[str, float]:
        """Benchmark various attention kernel implementations.
        
        Args:
            seq_len: Length of the sequence
            batch_size: Batch size
            hidden_size: Size of hidden dimension
            num_heads: Number of attention heads
            num_trials: Number of trials for timing
            
        Returns:
            Dictionary with benchmark results
        """
        import time
        import torch
        
        head_dim = hidden_size // num_heads
        
        # Create test tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda().half()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda().half()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda().half()
        
        # Hidden states for fused attention
        hidden_states = torch.randn(batch_size, seq_len, hidden_size).cuda().half()
        qkv_weight = torch.randn(3 * hidden_size, hidden_size).cuda().half()
        qkv_bias = torch.randn(3 * hidden_size).cuda().half()
        output_weight = torch.randn(hidden_size, hidden_size).cuda().half()
        output_bias = torch.randn(hidden_size).cuda().half()
        
        # Benchmark PyTorch attention
        torch.cuda.synchronize()
        pt_times = []
        for _ in range(num_trials):
            start = time.time()
            q_scaled = q * (1.0 / math.sqrt(head_dim))
            attn_weights = torch.matmul(q_scaled, k.transpose(-1, -2))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, hidden_size)
            torch.cuda.synchronize()
            pt_times.append(time.time() - start)
        pt_time = sum(pt_times) / len(pt_times)
        
        # Benchmark Triton ring attention
        torch.cuda.synchronize()
        ring_times = []
        for _ in range(num_trials):
            start = time.time()
            output = triton_ring_attention_forward(q, k, v)
            torch.cuda.synchronize()
            ring_times.append(time.time() - start)
        ring_time = sum(ring_times) / len(ring_times)
        
        # Benchmark Triton chunk attention
        torch.cuda.synchronize()
        chunk_times = []
        for _ in range(num_trials):
            start = time.time()
            output = triton_chunk_attention(q, k, v)
            torch.cuda.synchronize()
            chunk_times.append(time.time() - start)
        chunk_time = sum(chunk_times) / len(chunk_times)
        
        # Benchmark fused attention
        torch.cuda.synchronize()
        fused_times = []
        for _ in range(num_trials):
            start = time.time()
            output = triton_fused_ring_attention(
                hidden_states, qkv_weight, qkv_bias, output_weight, output_bias
            )
            torch.cuda.synchronize()
            fused_times.append(time.time() - start)
        fused_time = sum(fused_times) / len(fused_times)
        
        # Calculate memory usage
        pytorch_bytes = 2 * batch_size * num_heads * seq_len * seq_len  # attention matrix in FP16
        pytorch_bytes += 3 * batch_size * seq_len * hidden_size  # QKV tensors
        pytorch_bytes += batch_size * seq_len * hidden_size  # output tensor
        
        triton_bytes = 3 * batch_size * seq_len * hidden_size  # QKV tensors
        triton_bytes += batch_size * seq_len * hidden_size  # output tensor
        # Ring attention doesn't store full attention matrix
        
        return {
            "pytorch_time_ms": pt_time * 1000,
            "ring_attention_time_ms": ring_time * 1000,
            "chunk_attention_time_ms": chunk_time * 1000,
            "fused_attention_time_ms": fused_time * 1000,
            "pytorch_speedup": 1.0,
            "ring_attention_speedup": pt_time / ring_time,
            "chunk_attention_speedup": pt_time / chunk_time,
            "fused_attention_speedup": pt_time / fused_time,
            "pytorch_memory_mb": pytorch_bytes / (1024 * 1024),
            "triton_memory_mb": triton_bytes / (1024 * 1024),
            "memory_savings_factor": pytorch_bytes / triton_bytes
        }


# Fallback PyTorch implementations for systems without Triton
else:
    def triton_ring_attention_forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """PyTorch fallback for ring attention when Triton is not available."""
        batch_size, num_heads, seq_len, head_dim = query.shape
        _, _, kv_seq_len, _ = key.shape
        
        # Scale query
        query = query * (1.0 / math.sqrt(head_dim))
        
        # Determine chunk size
        chunk_size = min(128, kv_seq_len)
        
        # Initialize output tensor
        attn_output = torch.zeros(
            (batch_size, num_heads, seq_len, head_dim),
            dtype=query.dtype,
            device=query.device
        )
        
        # Compute chunked attention using PyTorch operations
        m_i = torch.ones((batch_size, num_heads, seq_len, 1), 
                         dtype=torch.float32, 
                         device=query.device) * float("-inf")
        l_i = torch.zeros((batch_size, num_heads, seq_len, 1), 
                         dtype=torch.float32, 
                         device=query.device)
        
        # Process in chunks
        for k_start in range(0, kv_seq_len, chunk_size):
            k_end = min(k_start + chunk_size, kv_seq_len)
            
            # Get key/value chunk
            k_chunk = key[:, :, k_start:k_end]
            v_chunk = value[:, :, k_start:k_end]
            
            # Compute attention scores for this chunk
            attn_weights = torch.matmul(query, k_chunk.transpose(-1, -2))
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, k_start:k_end]
            
            # Softmax stabilization
            m_ij = attn_weights.max(dim=-1, keepdim=True)[0]
            p_ij = torch.exp(attn_weights - m_ij)
            l_ij = p_ij.sum(dim=-1, keepdim=True)
            
            # Update running softmax stats for normalization across chunks
            m_i_new = torch.maximum(m_i, m_ij)
            alpha = torch.exp(m_i - m_i_new)
            beta = torch.exp(m_ij - m_i_new)
            
            # Correct the running sum using the decomposition rule for softmax
            attn_output = attn_output * alpha
            l_i = l_i * alpha + beta * l_ij
            m_i = m_i_new
            
            # Compute attention outputs for this chunk
            attn_output = attn_output + torch.matmul(p_ij, v_chunk) * beta
        
        # Final normalization
        attn_output = attn_output / l_i
        
        # Reshape output to [batch_size, seq_len, hidden_size]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, num_heads * head_dim)
        
        return attn_output


    def triton_fused_ring_attention(
        hidden_states: torch.Tensor,
        qkv_weight: torch.Tensor,
        qkv_bias: Optional[torch.Tensor],
        output_weight: torch.Tensor,
        output_bias: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """PyTorch fallback for fused ring attention when Triton is not available."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute QKV projections
        qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
        qkv = qkv.reshape(batch_size, seq_len, 3, hidden_size)
        
        # Split into q, k, v
        q, k, v = qkv.unbind(dim=2)
        
        # Reshape for attention computation
        head_dim = hidden_size // (qkv_weight.shape[0] // (3 * hidden_size))
        num_heads = hidden_size // head_dim
        
        q = q.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        
        # Compute attention using PyTorch version of ring attention
        attn_output = triton_ring_attention_forward(q, k, v, attention_mask)
        
        # Apply output projection
        output = F.linear(attn_output, output_weight, output_bias)
        
        return output


    def benchmark_attention_kernels(
        seq_len: int,
        batch_size: int,
        hidden_size: int,
        num_heads: int,
        num_trials: int = 10
    ) -> Dict[str, float]:
        """PyTorch fallback for benchmarking when Triton is not available."""
        return {
            "error": "Triton is not available for benchmarking. Using PyTorch fallbacks."
        }


# Common functions regardless of Triton availability
def compare_with_flash_attention(
    seq_len: int,
    batch_size: int,
    hidden_size: int,
    num_heads: int
) -> Dict[str, float]:
    """Compare Ring Attention with FlashAttention if available.
    
    Args:
        seq_len: Length of the sequence
        batch_size: Batch size
        hidden_size: Size of hidden dimension
        num_heads: Number of attention heads
        
    Returns:
        Dictionary with comparison results
    """
    import torch
    import time
    
    # Check for FlashAttention availability
    try:
        from flash_attn import flash_attn_func
        flash_attn_available = True
    except ImportError:
        flash_attn_available = False
        return {"error": "FlashAttention is not available for comparison"}
    
    # Create test tensors
    head_dim = hidden_size // num_heads
    q = torch.randn(batch_size, seq_len, num_heads, head_dim).cuda().half()
    k = torch.randn(batch_size, seq_len, num_heads, head_dim).cuda().half()
    v = torch.randn(batch_size, seq_len, num_heads, head_dim).cuda().half()
    
    # Reshape for ring attention
    q_ring = q.permute(0, 2, 1, 3)
    k_ring = k.permute(0, 2, 1, 3)
    v_ring = v.permute(0, 2, 1, 3)
    
    # Warm up
    for _ in range(5):
        if TRITON_AVAILABLE:
            triton_ring_attention_forward(q_ring, k_ring, v_ring)
        flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=1.0/math.sqrt(head_dim))
    
    torch.cuda.synchronize()
    
    # Benchmark FlashAttention
    flash_times = []
    for _ in range(10):
        start = time.time()
        flash_output = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=1.0/math.sqrt(head_dim))
        torch.cuda.synchronize()
        flash_times.append(time.time() - start)
    flash_time = sum(flash_times) / len(flash_times)
    
    # Benchmark Ring Attention
    if TRITON_AVAILABLE:
        ring_times = []
        for _ in range(10):
            start = time.time()
            ring_output = triton_ring_attention_forward(q_ring, k_ring, v_ring)
            torch.cuda.synchronize()
            ring_times.append(time.time() - start)
        ring_time = sum(ring_times) / len(ring_times)
    else:
        ring_time = float('inf')
        ring_output = None
    
    # Compare outputs if both methods are available
    if TRITON_AVAILABLE:
        # Reshape outputs to same format
        ring_output = ring_output.reshape(batch_size, seq_len, num_heads, head_dim)
        max_diff = (flash_output - ring_output).abs().max().item()
        mean_diff = (flash_output - ring_output).abs().mean().item()
        
        relative_error = mean_diff / flash_output.abs().mean().item()
    else:
        max_diff = float('nan')
        mean_diff = float('nan')
        relative_error = float('nan')
    
    return {
        "flash_attention_time_ms": flash_time * 1000,
        "ring_attention_time_ms": ring_time * 1000,
        "flash_vs_ring_speedup": ring_time / flash_time if ring_time != float('inf') else float('nan'),
        "max_absolute_diff": max_diff,
        "mean_absolute_diff": mean_diff,
        "relative_error": relative_error,
    }


def calculate_attention_theoretical_flops(
    seq_len: int,
    batch_size: int,
    hidden_size: int,
    num_heads: int
) -> Dict[str, float]:
    """Calculate theoretical FLOPs for different attention mechanisms.
    
    Args:
        seq_len: Length of the sequence
        batch_size: Batch size
        hidden_size: Size of hidden dimension
        num_heads: Number of attention heads
        
    Returns:
        Dictionary with FLOP counts
    """
    head_dim = hidden_size // num_heads
    
    # Common operations in all attention variants
    # QKV projections: 3 * batch_size * seq_len * hidden_size * hidden_size
    qkv_flops = 3 * batch_size * seq_len * hidden_size * hidden_size
    
    # Output projection: batch_size * seq_len * hidden_size * hidden_size
    output_flops = batch_size * seq_len * hidden_size * hidden_size
    
    # Standard attention
    # Q*K^T: batch_size * num_heads * seq_len * seq_len * head_dim
    std_qk_flops = batch_size * num_heads * seq_len * seq_len * head_dim
    
    # softmax: batch_size * num_heads * seq_len * seq_len (consider as 5 FLOPs per element)
    std_softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
    
    # softmax*V: batch_size * num_heads * seq_len * seq_len * head_dim
    std_attn_flops = batch_size * num_heads * seq_len * seq_len * head_dim
    
    std_total = qkv_flops + std_qk_flops + std_softmax_flops + std_attn_flops + output_flops
    
    # Ring Attention - same computation but chunked and potentially with better memory access patterns
    # For large sequences, we also avoid materializing the full attention matrix
    chunk_size = min(128, seq_len)
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    # QK^T for chunks: batch_size * num_heads * seq_len * num_chunks * chunk_size * head_dim
    ring_qk_flops = batch_size * num_heads * seq_len * num_chunks * chunk_size * head_dim
    
    # softmax for chunks (with overhead for progressive softmax): 
    # 7 * batch_size * num_heads * seq_len * num_chunks * chunk_size
    ring_softmax_flops = 7 * batch_size * num_heads * seq_len * num_chunks * chunk_size
    
    # softmax*V for chunks: batch_size * num_heads * seq_len * num_chunks * chunk_size * head_dim
    ring_attn_flops = batch_size * num_heads * seq_len * num_chunks * chunk_size * head_dim
    
    ring_total = qkv_flops + ring_qk_flops + ring_softmax_flops + ring_attn_flops + output_flops
    
    # Flash Attention (specialized kernel that avoids materializing the attention matrix)
    # Uses tiling and shared memory, approximate FLOPs
    flash_compute_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    flash_total = qkv_flops + flash_compute_flops + output_flops
    
    return {
        "standard_attention_gflops": std_total / 1e9,
        "ring_attention_gflops": ring_total / 1e9,
        "flash_attention_gflops": flash_total / 1e9,
        "standard_to_ring_flops_ratio": std_total / ring_total,
        "ring_to_flash_flops_ratio": ring_total / flash_total,
    }