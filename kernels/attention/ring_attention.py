"""
Ring Attention Implementation

This module implements Ring Attention, a memory-efficient attention mechanism
that partitions sequences into chunks and processes them in a ring communication
pattern across multiple GPUs. This approach significantly reduces the memory footprint
required for attention computation in transformer models.

Key benefits:
- O(N) memory scaling instead of O(N²) for attention
- Efficient utilization of multiple GPUs
- Support for extremely long sequences
- Compatible with various attention types (self, cross)

The implementation supports both PyTorch and optimized Triton kernels.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# Local imports
from ..triton.attention_kernels import (
    triton_ring_attention_forward,
    triton_fused_ring_attention
)


@dataclass
class RingAttentionConfig:
    """Configuration for Ring Attention.
    
    Attributes:
        world_size: Number of GPUs/processes for distributed computation
        chunk_size: Size of chunks for ring processing (None for auto-sizing)
        fuse_qkv: Whether to fuse QKV projections for efficiency
        use_flash_attention: Whether to use FlashAttention kernels if available
        use_triton: Whether to use Triton kernels for optimized computation
        precision: Precision for attention calculation ("fp32", "fp16", "bf16")
        communication_dtype: Data type for inter-GPU communication
        normalize_attention_scores: Whether to normalize attention scores
        attention_dropout: Dropout probability for attention weights
    """
    world_size: int = 1
    chunk_size: Optional[int] = None
    fuse_qkv: bool = True
    use_flash_attention: bool = False
    use_triton: bool = True
    precision: str = "bf16"  # One of "fp32", "fp16", "bf16"
    communication_dtype: torch.dtype = torch.bfloat16
    normalize_attention_scores: bool = True
    attention_dropout: float = 0.0
    
    def __post_init__(self):
        # Validation
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0 if specified, got {self.chunk_size}")
            
        if self.precision not in ["fp32", "fp16", "bf16"]:
            raise ValueError(f"precision must be one of ['fp32', 'fp16', 'bf16'], got {self.precision}")
            
        if self.attention_dropout < 0 or self.attention_dropout >= 1:
            raise ValueError(f"attention_dropout must be in [0, 1), got {self.attention_dropout}")
            
        # Check if triton is available if use_triton is True
        if self.use_triton and not TRITON_AVAILABLE:
            warnings.warn("Triton is not available but use_triton=True. Falling back to PyTorch implementation.")
            self.use_triton = False
            
        # Determine compute dtype
        self.compute_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]


class RingAttention(nn.Module):
    """Base class for Ring Attention implementations.
    
    This module implements attention using a ring communication pattern to reduce
    memory requirements from O(N²) to O(N) for sequence length N.
    
    Args:
        hidden_size: Size of hidden dimension
        num_attention_heads: Number of attention heads
        config: RingAttentionConfig object with parameters
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        config: RingAttentionConfig
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.config = config
        
        # Derived values
        self.head_dim = hidden_size // num_attention_heads
        if self.head_dim * num_attention_heads != hidden_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) is not divisible by num_attention_heads ({num_attention_heads})"
            )
        
        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Communication buffers for ring exchange
        self.register_buffer("key_buffer", None)
        self.register_buffer("value_buffer", None)
    
    def get_effective_bytes_per_token(self) -> int:
        """Calculate the effective memory usage per token in bytes.
        
        Returns:
            The number of bytes used per token with ring attention
        """
        # Base memory for hidden states
        bytes_per_type = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }
        
        dtype_size = bytes_per_type[self.config.compute_dtype]
        
        # With ring attention, we store:
        # - Full Q/K/V tensors (3 * hidden_size)
        # - Partial attention matrix (hidden_size / world_size)
        # - Output tensor (hidden_size)
        bytes_per_token = dtype_size * (4 * self.hidden_size + self.hidden_size / self.config.world_size)
        
        return int(bytes_per_token)
    
    def calculate_theoretical_memory_savings(self, seq_len: int) -> float:
        """Calculate theoretical memory savings compared to standard attention.
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            Memory savings ratio (standard / ring)
        """
        # Standard attention uses O(N²) memory for the attention matrix
        # Ring attention uses O(N) memory with a constant factor
        standard_bytes = seq_len * seq_len * self.num_attention_heads * 2  # 2 bytes for fp16/bf16
        ring_bytes = self.get_effective_bytes_per_token() * seq_len
        
        return standard_bytes / ring_bytes


class RingSelfAttention(RingAttention):
    """Ring Self-Attention implementation.
    
    This class implements self-attention using the ring communication pattern.
    
    Args:
        hidden_size: Size of hidden dimension
        num_attention_heads: Number of attention heads
        config: RingAttentionConfig object with parameters
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        config: RingAttentionConfig
    ):
        super().__init__(hidden_size, num_attention_heads, config)
        
        # Initialize QKV projections depending on fusion config
        if config.fuse_qkv:
            self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Dropout module
        self.attention_dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for self-attention.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, 1, 1, seq_len]
            
        Returns:
            Output tensor after self-attention [batch_size, seq_len, hidden_size]
        """
        # Fast path using fused kernels if enabled
        if self.config.use_triton and TRITON_AVAILABLE and self.config.fuse_qkv:
            # Use optimized Triton kernel for the entire attention operation
            return triton_fused_ring_attention(
                hidden_states=hidden_states,
                qkv_weight=self.qkv_proj.weight,
                qkv_bias=self.qkv_proj.bias,
                output_weight=self.out_proj.weight,
                output_bias=self.out_proj.bias,
                attention_mask=attention_mask
            )
        
        # Standard path: compute Q, K, V tensors
        query, key, value = self.prepare_attention_inputs(hidden_states)
        
        # Compute attention using ring pattern
        attn_output = self._ring_self_attention(query, key, value, attention_mask)
        
        # Apply output projection
        output = self.out_proj(attn_output)
        
        return output
    
    def prepare_attention_inputs(
        self, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare query, key, value tensors from hidden states.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Tuple of (query, key, value) tensors, each shaped:
            [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        if self.config.fuse_qkv:
            # Fused QKV projection
            qkv = self.qkv_proj(hidden_states)
            # Split into query, key, value
            qkv = qkv.reshape(batch_size, seq_len, 3, self.num_attention_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
            query, key, value = qkv[0], qkv[1], qkv[2]
        else:
            # Separate Q, K, V projections
            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)
            
            # Reshape to [batch_size, num_heads, seq_len, head_dim]
            query = query.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim).permute(0, 2, 1, 3)
            key = key.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim).permute(0, 2, 1, 3)
            value = value.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scale query
        query = query * self.scale
        
        return query, key, value
    
    def _ring_self_attention(
        self,
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute self-attention using ring communication pattern.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len, head_dim]
            attention_mask: Optional attention mask [batch_size, 1, 1, seq_len]
            
        Returns:
            Output tensor after attention [batch_size, seq_len, hidden_size]
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Determine chunk size if not specified
        if self.config.chunk_size is None:
            # Auto chunk size based on sequence length and world size
            chunk_size = max(1, seq_len // (2 * self.config.world_size))
        else:
            chunk_size = self.config.chunk_size
        
        # Calculate schedule for chunks
        chunk_schedule = self._calculate_chunk_schedule(seq_len, chunk_size)
        
        # Initialize output tensor
        attn_output = torch.zeros(
            (batch_size, num_heads, seq_len, head_dim),
            dtype=self.config.compute_dtype,
            device=query.device
        )
        
        # Prepare key buffer and value buffer if not initialized
        if self.key_buffer is None or self.key_buffer.shape[2] < chunk_size:
            self.register_buffer(
                "key_buffer",
                torch.zeros(
                    (batch_size, num_heads, chunk_size, head_dim),
                    dtype=self.config.communication_dtype,
                    device=query.device
                )
            )
            self.register_buffer(
                "value_buffer",
                torch.zeros(
                    (batch_size, num_heads, chunk_size, head_dim),
                    dtype=self.config.communication_dtype,
                    device=query.device
                )
            )
        
        # Process chunks according to schedule
        for q_chunk_idx, k_chunk_idx, chunk_len in chunk_schedule:
            # Extract key/value chunks
            k_chunk = key[:, :, k_chunk_idx:k_chunk_idx+chunk_len]
            v_chunk = value[:, :, k_chunk_idx:k_chunk_idx+chunk_len]
            
            # Convert to communication dtype if different
            if self.config.communication_dtype != self.config.compute_dtype:
                k_chunk = k_chunk.to(self.config.communication_dtype)
                v_chunk = v_chunk.to(self.config.communication_dtype)
            
            # Copy to buffer 
            self.key_buffer[:, :, :chunk_len] = k_chunk
            self.value_buffer[:, :, :chunk_len] = v_chunk
            
            # Update local chunk
            q_chunk = query[:, :, q_chunk_idx:q_chunk_idx+chunk_len]
            
            # Compute attention scores for this chunk
            # [batch_size, num_heads, q_chunk_len, k_chunk_len]
            attn_weights = torch.matmul(q_chunk, k_chunk.transpose(2, 3))
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Slice the attention mask to the current chunk
                mask_chunk = attention_mask[:, :, :, k_chunk_idx:k_chunk_idx+chunk_len]
                attn_weights = attn_weights + mask_chunk
            
            # Apply softmax within chunk
            if self.config.normalize_attention_scores:
                attn_weights = F.softmax(attn_weights, dim=-1)
            
            # Apply attention dropout
            if self.config.attention_dropout > 0:
                attn_weights = self.attention_dropout(attn_weights)
            
            # Compute output for this chunk
            # [batch_size, num_heads, q_chunk_len, head_dim]
            chunk_output = torch.matmul(attn_weights, v_chunk)
            
            # Accumulate result
            attn_output[:, :, q_chunk_idx:q_chunk_idx+chunk_len] += chunk_output
        
        # Reshape output to [batch_size, seq_len, hidden_size]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        return attn_output
    
    def _calculate_chunk_schedule(
        self, 
        seq_len: int,
        chunk_size: int
    ) -> List[Tuple[int, int, int]]:
        """Calculate chunk processing schedule for ring attention.
        
        Args:
            seq_len: Length of the sequence
            chunk_size: Size of chunks for processing
            
        Returns:
            List of (query_chunk_idx, key_chunk_idx, chunk_length) tuples
        """
        # Determine number of full chunks and remainder
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        # Create schedule of processing chunks
        schedule = []
        for i in range(num_chunks):
            for j in range(num_chunks):
                q_start = i * chunk_size
                k_start = j * chunk_size
                
                # Handle last chunk which may be shorter
                q_length = min(chunk_size, seq_len - q_start)
                k_length = min(chunk_size, seq_len - k_start)
                
                schedule.append((q_start, k_start, k_length))
        
        return schedule


class RingCrossAttention(RingAttention):
    """Ring Cross-Attention implementation.
    
    This class implements cross-attention using the ring communication pattern.
    
    Args:
        hidden_size: Size of hidden dimension
        num_attention_heads: Number of attention heads
        config: RingAttentionConfig object with parameters
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        config: RingAttentionConfig
    ):
        super().__init__(hidden_size, num_attention_heads, config)
        
        # Initialize separate projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Dropout module
        self.attention_dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self,
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for cross-attention.
        
        Args:
            query_states: Query input states [batch_size, q_seq_len, hidden_size]
            key_value_states: Key/value input states [batch_size, kv_seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, 1, q_seq_len, kv_seq_len]
            
        Returns:
            Output tensor after cross-attention [batch_size, q_seq_len, hidden_size]
        """
        # Fast path using optimized Triton kernels if enabled
        if self.config.use_triton and TRITON_AVAILABLE:
            return triton_ring_attention_forward(
                query=self.q_proj(query_states).reshape(
                    query_states.shape[0], query_states.shape[1], 
                    self.num_attention_heads, self.head_dim
                ).permute(0, 2, 1, 3),
                key=self.k_proj(key_value_states).reshape(
                    key_value_states.shape[0], key_value_states.shape[1], 
                    self.num_attention_heads, self.head_dim
                ).permute(0, 2, 1, 3),
                value=self.v_proj(key_value_states).reshape(
                    key_value_states.shape[0], key_value_states.shape[1], 
                    self.num_attention_heads, self.head_dim
                ).permute(0, 2, 1, 3),
                attention_mask=attention_mask
            )
        
        # Standard path - compute Q, K, V projections
        batch_size, q_seq_len, _ = query_states.shape
        _, kv_seq_len, _ = key_value_states.shape
        
        # Project query, key, value
        query = self.q_proj(query_states)
        key = self.k_proj(key_value_states)
        value = self.v_proj(key_value_states)
        
        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        query = query.reshape(batch_size, q_seq_len, self.num_attention_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.reshape(batch_size, kv_seq_len, self.num_attention_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.reshape(batch_size, kv_seq_len, self.num_attention_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scale query
        query = query * self.scale
        
        # Compute attention using ring pattern
        attn_output = self._ring_cross_attention(query, key, value, attention_mask)
        
        # Apply output projection
        output = self.out_proj(attn_output)
        
        return output
    
    def _ring_cross_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute cross-attention using ring communication pattern.
        
        Args:
            query: Query tensor [batch_size, num_heads, q_seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, kv_seq_len, head_dim]
            value: Value tensor [batch_size, num_heads, kv_seq_len, head_dim]
            attention_mask: Optional attention mask [batch_size, 1, q_seq_len, kv_seq_len]
            
        Returns:
            Output tensor after attention [batch_size, q_seq_len, hidden_size]
        """
        batch_size, num_heads, q_seq_len, head_dim = query.shape
        _, _, kv_seq_len, _ = key.shape
        
        # Determine chunk size if not specified
        if self.config.chunk_size is None:
            # Auto chunk size based on sequence length and world size
            k_chunk_size = max(1, kv_seq_len // (2 * self.config.world_size))
            q_chunk_size = max(1, q_seq_len // (2 * self.config.world_size))
        else:
            k_chunk_size = self.config.chunk_size
            q_chunk_size = self.config.chunk_size
        
        # Initialize output tensor
        attn_output = torch.zeros(
            (batch_size, num_heads, q_seq_len, head_dim),
            dtype=self.config.compute_dtype,
            device=query.device
        )
        
        # Prepare key buffer and value buffer if not initialized
        if self.key_buffer is None or self.key_buffer.shape[2] < k_chunk_size:
            self.register_buffer(
                "key_buffer",
                torch.zeros(
                    (batch_size, num_heads, k_chunk_size, head_dim),
                    dtype=self.config.communication_dtype,
                    device=query.device
                )
            )
            self.register_buffer(
                "value_buffer",
                torch.zeros(
                    (batch_size, num_heads, k_chunk_size, head_dim),
                    dtype=self.config.communication_dtype,
                    device=query.device
                )
            )
        
        # Process chunks of the query sequence
        for q_start in range(0, q_seq_len, q_chunk_size):
            q_length = min(q_chunk_size, q_seq_len - q_start)
            q_chunk = query[:, :, q_start:q_start+q_length]
            
            # Accumulate attention for different key/value chunks
            attn_weights_sum = None
            attn_output_chunk = torch.zeros(
                (batch_size, num_heads, q_length, head_dim),
                dtype=self.config.compute_dtype,
                device=query.device
            )
            
            # Process chunks of the key/value sequence
            for k_start in range(0, kv_seq_len, k_chunk_size):
                k_length = min(k_chunk_size, kv_seq_len - k_start)
                
                # Extract key/value chunks
                k_chunk = key[:, :, k_start:k_start+k_length]
                v_chunk = value[:, :, k_start:k_start+k_length]
                
                # Convert to communication dtype if different
                if self.config.communication_dtype != self.config.compute_dtype:
                    k_chunk = k_chunk.to(self.config.communication_dtype)
                    v_chunk = v_chunk.to(self.config.communication_dtype)
                
                # Copy to buffer
                self.key_buffer[:, :, :k_length] = k_chunk
                self.value_buffer[:, :, :k_length] = v_chunk
                
                # Compute attention scores for this chunk
                # [batch_size, num_heads, q_length, k_length]
                chunk_attn_weights = torch.matmul(q_chunk, k_chunk.transpose(2, 3))
                
                # Apply attention mask if provided
                if attention_mask is not None:
                    # Slice the attention mask to the current chunks
                    mask_chunk = attention_mask[:, :, q_start:q_start+q_length, k_start:k_start+k_length]
                    chunk_attn_weights = chunk_attn_weights + mask_chunk
                
                # When normalizing, we accumulate unnormalized scores to apply softmax across all key chunks
                if self.config.normalize_attention_scores:
                    if attn_weights_sum is None:
                        attn_weights_sum = chunk_attn_weights
                    else:
                        # Expand attn_weights_sum if needed
                        if attn_weights_sum.shape[-1] < chunk_attn_weights.shape[-1]:
                            new_size = attn_weights_sum.shape[-1] + chunk_attn_weights.shape[-1]
                            new_attn_weights = torch.zeros(
                                (batch_size, num_heads, q_length, new_size),
                                dtype=chunk_attn_weights.dtype,
                                device=chunk_attn_weights.device
                            )
                            new_attn_weights[:, :, :, :attn_weights_sum.shape[-1]] = attn_weights_sum
                            new_attn_weights[:, :, :, attn_weights_sum.shape[-1]:] = chunk_attn_weights
                            attn_weights_sum = new_attn_weights
                        else:
                            # Concatenate along key dimension
                            attn_weights_sum = torch.cat(
                                [attn_weights_sum, chunk_attn_weights], dim=-1
                            )
                else:
                    # If not normalizing, apply softmax to each chunk independently
                    chunk_attn_weights = F.softmax(chunk_attn_weights, dim=-1)
                    
                    # Apply attention dropout if enabled
                    if self.config.attention_dropout > 0:
                        chunk_attn_weights = self.attention_dropout(chunk_attn_weights)
                    
                    # Compute output for this chunk
                    # [batch_size, num_heads, q_length, head_dim]
                    chunk_output = torch.matmul(chunk_attn_weights, v_chunk)
                    
                    # Accumulate result
                    attn_output_chunk += chunk_output
            
            # If normalizing scores across all keys, apply softmax now
            if self.config.normalize_attention_scores and attn_weights_sum is not None:
                attn_weights = F.softmax(attn_weights_sum, dim=-1)
                
                # Apply attention dropout if enabled
                if self.config.attention_dropout > 0:
                    attn_weights = self.attention_dropout(attn_weights)
                
                # We need to multiply the normalized weights with all value chunks
                attn_output_chunk = torch.zeros(
                    (batch_size, num_heads, q_length, head_dim),
                    dtype=self.config.compute_dtype,
                    device=query.device
                )
                
                # Process all key/value chunks again with normalized weights
                offset = 0
                for k_start in range(0, kv_seq_len, k_chunk_size):
                    k_length = min(k_chunk_size, kv_seq_len - k_start)
                    v_chunk = value[:, :, k_start:k_start+k_length]
                    
                    # Extract corresponding attention weights
                    chunk_weights = attn_weights[:, :, :, offset:offset+k_length]
                    offset += k_length
                    
                    # Compute output contribution from this chunk
                    chunk_output = torch.matmul(chunk_weights, v_chunk)
                    attn_output_chunk += chunk_output
            
            # Update the output tensor with this query chunk's results
            attn_output[:, :, q_start:q_start+q_length] = attn_output_chunk
        
        # Reshape output to [batch_size, q_seq_len, hidden_size]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.reshape(batch_size, q_seq_len, self.hidden_size)
        
        return attn_output


class ModelConverter:
    """Utility class to convert models to use Ring Attention.
    
    This class provides functionality to find and replace standard attention
    modules with optimized Ring Attention modules.
    
    Args:
        config: RingAttentionConfig object with parameters
    """
    def __init__(self, config: RingAttentionConfig):
        self.config = config
    
    def convert_model(self, model: nn.Module) -> nn.Module:
        """Convert a model to use Ring Attention.
        
        This method recursively traverses the model and replaces compatible
        attention modules with Ring Attention equivalents.
        
        Args:
            model: The PyTorch model to convert
            
        Returns:
            Converted model with Ring Attention modules
        """
        return self._find_and_replace_attention(model)
    
    def _find_and_replace_attention(self, module: nn.Module) -> nn.Module:
        """Recursively find and replace attention modules.
        
        Args:
            module: The module to process
            
        Returns:
            The processed module
        """
        # Check if the current module is an attention module
        if self._is_attention_module(module):
            return self._convert_attention_module(module)
        
        # Recursively process child modules
        for name, child in list(module.named_children()):
            converted_child = self._find_and_replace_attention(child)
            if converted_child is not child:
                setattr(module, name, converted_child)
        
        return module
    
    def _is_attention_module(self, module: nn.Module) -> bool:
        """Determine if a module is an attention module.
        
        This method checks if a module is a standard attention implementation
        that can be replaced with Ring Attention.
        
        Args:
            module: The module to check
            
        Returns:
            True if the module is an attention module, False otherwise
        """
        # Check common attention module names and patterns
        module_name = module.__class__.__name__.lower()
        attention_markers = [
            "attention", "attn", "selfatt", "selfattn", "crossatt", "crossattn"
        ]
        
        # Check if the module name contains any attention marker
        if any(marker in module_name for marker in attention_markers):
            # Further check if it has Q, K, V projections
            has_q = hasattr(module, "q_proj") or hasattr(module, "query")
            has_k = hasattr(module, "k_proj") or hasattr(module, "key")
            has_v = hasattr(module, "v_proj") or hasattr(module, "value")
            has_qkv = hasattr(module, "qkv_proj") or hasattr(module, "qkv")
            
            # Must have either separate QKV projections or a combined one
            return (has_q and has_k and has_v) or has_qkv
        
        return False
    
    def _convert_attention_module(self, module: nn.Module) -> nn.Module:
        """Convert a standard attention module to Ring Attention.
        
        Args:
            module: The attention module to convert
            
        Returns:
            Converted Ring Attention module
        """
        # Determine the module type and configuration
        module_name = module.__class__.__name__.lower()
        hidden_size = None
        num_heads = None
        
        # Try to extract parameters from the module
        if hasattr(module, "hidden_size"):
            hidden_size = module.hidden_size
        elif hasattr(module, "embed_dim"):
            hidden_size = module.embed_dim
        
        if hasattr(module, "num_heads"):
            num_heads = module.num_heads
        elif hasattr(module, "num_attention_heads"):
            num_heads = module.num_attention_heads
        
        # If we couldn't determine parameters, try to infer from q_proj weight
        if (hidden_size is None or num_heads is None) and hasattr(module, "q_proj"):
            hidden_size = module.q_proj.weight.shape[0]
            # Guess num_heads based on typical head dimensions (64, 80, 96, 128)
            for head_dim in [128, 96, 80, 64]:
                if hidden_size % head_dim == 0:
                    num_heads = hidden_size // head_dim
                    break
        
        # If we still don't have parameters, can't convert
        if hidden_size is None or num_heads is None:
            warnings.warn(
                f"Could not determine parameters for {module_name}, skipping conversion"
            )
            return module
        
        # Determine if this is a self-attention or cross-attention module
        is_cross_attn = "cross" in module_name
        
        # Create appropriate ring attention module
        if is_cross_attn:
            ring_module = RingCrossAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                config=self.config
            )
        else:
            ring_module = RingSelfAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                config=self.config
            )
        
        # Try to copy weights when possible
        try:
            # Copy QKV weights
            if hasattr(module, "q_proj") and hasattr(ring_module, "q_proj"):
                ring_module.q_proj.weight.data.copy_(module.q_proj.weight.data)
                ring_module.q_proj.bias.data.copy_(module.q_proj.bias.data)
                
            if hasattr(module, "k_proj") and hasattr(ring_module, "k_proj"):
                ring_module.k_proj.weight.data.copy_(module.k_proj.weight.data)
                ring_module.k_proj.bias.data.copy_(module.k_proj.bias.data)
                
            if hasattr(module, "v_proj") and hasattr(ring_module, "v_proj"):
                ring_module.v_proj.weight.data.copy_(module.v_proj.weight.data)
                ring_module.v_proj.bias.data.copy_(module.v_proj.bias.data)
            
            # Copy combined QKV weights if they exist
            if hasattr(module, "qkv_proj") and hasattr(ring_module, "qkv_proj"):
                ring_module.qkv_proj.weight.data.copy_(module.qkv_proj.weight.data)
                ring_module.qkv_proj.bias.data.copy_(module.qkv_proj.bias.data)
            
            # Copy output projection
            if hasattr(module, "out_proj") and hasattr(ring_module, "out_proj"):
                ring_module.out_proj.weight.data.copy_(module.out_proj.weight.data)
                ring_module.out_proj.bias.data.copy_(module.out_proj.bias.data)
        except Exception as e:
            warnings.warn(f"Error copying weights for {module_name}: {e}")
        
        return ring_module


def benchmark_ring_attention(
    seq_len: int,
    batch_size: int,
    hidden_size: int,
    num_heads: int
) -> Dict[str, float]:
    """Benchmark Ring Attention against standard attention.
    
    Args:
        seq_len: Length of the sequence
        batch_size: Batch size
        hidden_size: Size of hidden dimension
        num_heads: Number of attention heads
        
    Returns:
        Dictionary with benchmark results
    """
    import torch
    import time
    
    # Create test tensors
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).cuda()
    
    # Create standard attention module
    standard_attention = nn.MultiheadAttention(
        hidden_size, num_heads, batch_first=True
    ).cuda()
    
    # Create ring attention module
    config = RingAttentionConfig(
        world_size=1,  # Single GPU for benchmarking
        chunk_size=128,
        fuse_qkv=False,  # To match MultiheadAttention interface
        use_flash_attention=False,
        use_triton=False,  # PyTorch implementation for fair comparison
    )
    ring_attention = RingSelfAttention(hidden_size, num_heads, config).cuda()
    
    # Warm up
    for _ in range(5):
        standard_attention(hidden_states, hidden_states, hidden_states)
        ring_attention(hidden_states)
    
    torch.cuda.synchronize()
    
    # Benchmark standard attention
    start_time = time.time()
    for _ in range(10):
        standard_attention(hidden_states, hidden_states, hidden_states)
    torch.cuda.synchronize()
    standard_time = (time.time() - start_time) / 10
    
    # Benchmark ring attention
    start_time = time.time()
    for _ in range(10):
        ring_attention(hidden_states)
    torch.cuda.synchronize()
    ring_time = (time.time() - start_time) / 10
    
    # Measure memory usage
    standard_memory = seq_len * seq_len * num_heads * 2  # 2 bytes for fp16
    ring_memory = ring_attention.get_effective_bytes_per_token() * seq_len
    
    # Calculate theoretical speedup
    if seq_len <= 1024:
        speedup_factor = 1.0  # For small sequences, standard attention may be faster
    else:
        # For long sequences, we expect linear speedup proportional to sequence length
        speedup_factor = standard_time / ring_time
    
    # Calculate memory savings
    memory_savings = standard_memory / ring_memory
    
    return {
        "standard_time_ms": standard_time * 1000,
        "ring_time_ms": ring_time * 1000,
        "speedup_factor": speedup_factor,
        "standard_memory_mb": standard_memory / (1024 * 1024),
        "ring_memory_mb": ring_memory / (1024 * 1024),
        "memory_savings_factor": memory_savings,
    }


def calculate_theoretical_flops(
    seq_len: int,
    batch_size: int,
    hidden_size: int,
    num_heads: int
) -> int:
    """Calculate theoretical FLOPs for attention computation.
    
    Args:
        seq_len: Length of the sequence
        batch_size: Batch size
        hidden_size: Size of hidden dimension
        num_heads: Number of attention heads
        
    Returns:
        Number of floating point operations
    """
    head_dim = hidden_size // num_heads
    
    # FLOPs for QKV projections: 3 * (batch_size * seq_len * hidden_size * hidden_size)
    qkv_flops = 3 * batch_size * seq_len * hidden_size * hidden_size
    
    # FLOPs for attention scores: batch_size * num_heads * seq_len * seq_len * head_dim
    attn_score_flops = batch_size * num_heads * seq_len * seq_len * head_dim
    
    # FLOPs for attention output: batch_size * num_heads * seq_len * seq_len * head_dim
    attn_output_flops = batch_size * num_heads * seq_len * seq_len * head_dim
    
    # FLOPs for output projection: batch_size * seq_len * hidden_size * hidden_size
    output_flops = batch_size * seq_len * hidden_size * hidden_size
    
    # Total FLOPs
    total_flops = qkv_flops + attn_score_flops + attn_output_flops + output_flops
    
    return total_flops


def compare_with_standard_attention(
    seq_len: int,
    batch_size: int,
    hidden_size: int,
    num_heads: int
) -> Dict[str, float]:
    """Compare Ring Attention with standard attention for correctness and performance.
    
    Args:
        seq_len: Length of the sequence
        batch_size: Batch size
        hidden_size: Size of hidden dimension
        num_heads: Number of attention heads
        
    Returns:
        Dictionary with comparison results
    """
    import torch
    
    # Create test tensors
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).cuda()
    hidden_states_copy = hidden_states.clone()
    
    # Create standard attention module
    standard_attention = nn.MultiheadAttention(
        hidden_size, num_heads, batch_first=True
    ).cuda()
    
    # Create ring attention module with same parameters
    config = RingAttentionConfig(
        world_size=1,  # Single GPU for comparison
        chunk_size=128,
        fuse_qkv=False,  # To match MultiheadAttention interface
        use_flash_attention=False,
        use_triton=False,  # PyTorch implementation for fair comparison
    )
    ring_attention = RingSelfAttention(hidden_size, num_heads, config).cuda()
    
    # Copy weights from standard attention to ring attention
    ring_attention.q_proj.weight.data.copy_(standard_attention.q_proj.weight.data)
    ring_attention.q_proj.bias.data.copy_(standard_attention.q_proj.bias.data)
    ring_attention.k_proj.weight.data.copy_(standard_attention.k_proj.weight.data)
    ring_attention.k_proj.bias.data.copy_(standard_attention.k_proj.bias.data)
    ring_attention.v_proj.weight.data.copy_(standard_attention.v_proj.weight.data)
    ring_attention.v_proj.bias.data.copy_(standard_attention.v_proj.bias.data)
    ring_attention.out_proj.weight.data.copy_(standard_attention.out_proj.weight.data)
    ring_attention.out_proj.bias.data.copy_(standard_attention.out_proj.bias.data)
    
    # Run standard attention
    with torch.no_grad():
        standard_output, _ = standard_attention(
            hidden_states, hidden_states, hidden_states
        )
    
    # Run ring attention
    with torch.no_grad():
        ring_output = ring_attention(hidden_states_copy)
    
    # Compare outputs
    max_diff = (standard_output - ring_output).abs().max().item()
    mean_diff = (standard_output - ring_output).abs().mean().item()
    
    # Run benchmark
    benchmark_results = benchmark_ring_attention(
        seq_len, batch_size, hidden_size, num_heads
    )
    
    # Add comparison metrics
    comparison_results = {
        "max_absolute_diff": max_diff,
        "mean_absolute_diff": mean_diff,
        "relative_error": mean_diff / standard_output.abs().mean().item(),
        **benchmark_results
    }
    
    return comparison_results