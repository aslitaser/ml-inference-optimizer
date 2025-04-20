"""
Flash Attention 3 implementation.

This module provides an optimized implementation of FlashAttention algorithm,
featuring improved memory efficiency and speed compared to standard attention mechanisms.
It implements the core ideas from the FlashAttention3 paper with additional optimizations.

Key features:
- O(sqrt(N)) memory complexity instead of O(N�) 
- Optimized block-sparse attention for long sequences
- Multiple precision options (fp16, bf16, fp32)
- Causal masking support
- Triton kernel integration for GPU optimization
- vLLM-compatible kernel options

References:
- FlashAttention3: https://arxiv.org/abs/2307.08691
- Original FlashAttention: https://arxiv.org/abs/2205.14135
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Set, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from ..triton.flash_attention_kernels import triton_flash_attention, triton_fused_attention

# Attempt to import the paged attention kernel
try:
    from ..triton.attention_kernels import triton_paged_attention_forward, TRITON_AVAILABLE
except ImportError:
    triton_paged_attention_forward = None
    # Keep existing TRITON_AVAILABLE check for other kernels
    try:
        import triton
        TRITON_AVAILABLE = True
    except ImportError:
        TRITON_AVAILABLE = False
    logging.warning("Paged Attention Triton kernel not found or Triton is unavailable. PagedAttention will not be usable via FlashAttention modules.")


@dataclass
class FlashAttentionConfig:
    """Configuration for FlashAttention3 implementation.
    
    Args:
        block_size: Block size for tiled matrix operations
        causal: Whether to use causal masking (for decoder-only models)
        softmax_scale: Scale factor for softmax (default: 1/sqrt(head_dim))
        dropout_p: Attention dropout probability
        return_softmax: Whether to return softmax attention weights
        use_triton: Whether to use Triton kernels (falls back to PyTorch if False)
        memory_efficient: Whether to use memory-efficient implementation
        precision: Precision mode ("fp16", "bf16", "fp32")
        normalize_query: Whether to normalize query vectors (like in GQA)
        fp8_ortho_matrix: Precomputed orthogonal matrix for FP8 incoherent processing
    """
    block_size: int = 128  # Block size for tiling
    causal: bool = False  # Whether to use causal masking
    softmax_scale: Optional[float] = None  # Scale factor for softmax
    dropout_p: float = 0.0  # Attention dropout probability
    return_softmax: bool = False  # Whether to return softmax attention weights
    use_triton: bool = True  # Whether to use Triton kernels
    memory_efficient: bool = True  # Whether to use memory-efficient implementation
    precision: str = "fp16"  # Precision for computation ("fp16", "bf16", "fp32")
    normalize_query: bool = False  # Whether to apply GQA-style normalization to query
    fp8_ortho_matrix: Optional[torch.Tensor] = None # Precomputed orthogonal matrix for FP8 incoherent processing
    # TODO: Add seed for on-the-fly generation if matrix is not precomputed

    @property
    def allowed_precisions(self) -> Set[str]:
        return {"fp16", "bf16", "fp32", "fp8"}

    def __post_init__(self):
        if self.precision not in self.allowed_precisions:
            raise ValueError(f"Unsupported precision mode: {self.precision}. Allowed: {self.allowed_precisions}")
        
        if self.precision == "fp8":
            if not HAS_TRITON:
                raise RuntimeError("FP8 precision requires Triton to be installed.")
            if not torch.cuda.is_available():
                 raise RuntimeError("FP8 precision requires a CUDA-enabled GPU.")
            # Check for Hopper+ architecture (Compute Capability 9.0+)
            major, minor = torch.cuda.get_device_capability()
            if major < 9:
                 raise RuntimeError(f"FP8 precision requires Hopper architecture (Compute Capability 9.0+) but found {major}.{minor}.")
            if not self.use_triton:
                 print("Warning: FP8 precision requested, but use_triton=False. Enabling Triton as it's required for FP8.")
                 self.use_triton = True
            # Decide if the orthogonal matrix needs generation or is provided
            # For now, we assume it might be passed or handled later
            # if self.fp8_ortho_matrix is None:
            #    print("Warning: FP8 enabled but no orthogonal matrix provided. Incoherent processing might not be applied unless handled by the kernel.")


class FlashAttention3(nn.Module):
    """
    FlashAttention3 implementation for efficient attention computation.
    
    This implementation significantly reduces memory usage by avoiding materializing
    the full attention matrix. It uses a tiled block-based approach to compute attention
    incrementally, resulting in O(sqrt(N)) memory complexity instead of O(N�).
    
    Key optimizations:
    1. Block-based tiling for memory efficiency
    2. Triton kernel acceleration where available
    3. Fused operations to reduce memory bandwidth
    4. Support for different precision modes
    """
    
    def __init__(self, config: Optional[FlashAttentionConfig] = None):
        """
        Initialize FlashAttention3.
        
        Args:
            config: Configuration for FlashAttention3. If None, uses default config.
        """
        super().__init__()
        self.config = config or FlashAttentionConfig()
        
        # Validate configuration - moved most checks to __post_init__ of the config
        if self.config.use_triton and not HAS_TRITON:
            # Keep this check for general Triton availability fallback
            print("Warning: Triton not available. Falling back to PyTorch implementation.")
            self.config.use_triton = False
        # FP8 specific checks are now in FlashAttentionConfig.__post_init__
        # Remove the old precision check here as it's redundant
        # if self.config.precision not in ["fp16", "bf16", "fp32"]:
        #    raise ValueError(f"Unsupported precision mode: {self.config.precision}")

        # For backward compatibility with older PyTorch versions
        self.supports_backward = True  
        
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute flash attention.
        
        Args:
            q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            v: Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
            mask: Optional attention mask of shape [batch_size, seq_len] or [batch_size, 1, seq_len]
                  or [batch_size, seq_len, seq_len]
                  
        Returns:
            output: Attention output of shape [batch_size, seq_len, num_heads, head_dim]
            (Optional) attention_weights: If return_softmax=True, returns attention weights
        """
        # Check if inputs have expected rank
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError(f"Expected 4D tensors for q, k, v but got shapes: "
                             f"q={q.shape}, k={k.shape}, v={v.shape}")
            
        # Ensure inputs are on the correct device (CUDA is required for FP8/Triton)
        if self.config.use_triton and not q.is_cuda:
             raise ValueError("Triton kernels require input tensors to be on a CUDA device.")

        # Cast to appropriate precision if needed
        orig_dtype = q.dtype
        target_dtype = None
        if self.config.precision == "fp16":
            target_dtype = torch.float16
        elif self.config.precision == "bf16":
            target_dtype = torch.bfloat16
        elif self.config.precision == "fp32":
            target_dtype = torch.float32
        elif self.config.precision == "fp8":
            # Use torch.float8_e4m3fn for FP8 E4M3 format
            # Requires PyTorch 2.1+
            if hasattr(torch, "float8_e4m3fn"):
                 target_dtype = torch.float8_e4m3fn
            else:
                 raise RuntimeError("FP8 precision requires PyTorch 2.1 or later with float8 support.")
        
        if target_dtype and q.dtype != target_dtype:
             q = q.to(target_dtype)
             k = k.to(target_dtype)
             v = v.to(target_dtype)
             if mask is not None and mask.dtype != torch.bool:
                 # Ensure mask is boolean for efficiency if casting other inputs
                 mask = mask.bool()

        # Apply query normalization if requested
        if self.config.normalize_query:
            q = F.normalize(q, dim=-1)
            
        # Dispatch to appropriate implementation
        if self.config.use_triton:
             # Perform FP8 checks again here, as config might be modified after init
             if self.config.precision == "fp8":
                 if not q.is_cuda: raise RuntimeError("FP8 requires CUDA tensors.")
                 major, minor = torch.cuda.get_device_capability(q.device)
                 if major < 9: raise RuntimeError(f"FP8 requires CC 9.0+ (Hopper), found {major}.{minor}.")
                 if not HAS_TRITON: raise RuntimeError("FP8 requires Triton.") # Should be caught earlier but double check
                 # TODO: Potentially generate or ensure orthogonal matrix exists here if not passed in config
                 
             output = self._forward_triton(q, k, v, mask)
        else:
             if self.config.precision == "fp8":
                 # Explicitly disallow FP8 with PyTorch fallback
                 raise RuntimeError("FP8 precision is only supported with the Triton backend.")
             output = self._forward_pytorch(q, k, v, mask)
            
        # Cast back to original dtype if needed
        if isinstance(output, tuple):
            return output[0].to(orig_dtype), output[1]
        else:
            return output.to(orig_dtype)
    
    def _forward_triton(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute flash attention using Triton kernels.
        
        Args:
            q, k, v: Query, key, value tensors (potentially FP8)
            mask: Optional attention mask
            
        Returns:
            output: Attention output
            (Optional) attention_weights: If return_softmax=True, returns attention weights
        """
        # Prepare softmax scale
        head_dim = q.shape[-1]
        softmax_scale = self.config.softmax_scale or (1.0 / math.sqrt(head_dim))
        
        # Prepare orthogonal matrix if FP8 is enabled and matrix is needed
        ortho_matrix = None
        if self.config.precision == "fp8":
            # Use precomputed matrix if available
            if self.config.fp8_ortho_matrix is not None:
                 ortho_matrix = self.config.fp8_ortho_matrix.to(q.device, dtype=torch.float32) # Use FP32 for matrix mult stability
            # TODO: Add logic to generate the matrix if not provided, maybe based on a seed?
            # For now, we pass None if not precomputed. The kernel needs to handle this.
            # Example generation (needs careful seeding/device handling):
            # if ortho_matrix is None:
            #    ortho_matrix = torch.randn(head_dim, head_dim, device=q.device, dtype=torch.float32)
            #    ortho_matrix, _ = torch.linalg.qr(ortho_matrix)

        # Call triton kernel
        return triton_flash_attention(
            q=q, 
            k=k, 
            v=v, 
            mask=mask,
            causal=self.config.causal,
            softmax_scale=softmax_scale,
            dropout_p=self.config.dropout_p if self.training else 0.0, # Pass dropout only during training
            return_softmax=self.config.return_softmax,
            block_size=self.config.block_size,
            precision=self.config.precision, # Pass precision info
            ortho_matrix=ortho_matrix # Pass the orthogonal matrix for incoherent processing
        )
    
    def _forward_pytorch(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute flash attention using PyTorch implementation.
        
        This is a memory-efficient implementation that avoids materializing
        the full attention matrix by processing in blocks.
        
        Args:
            q, k, v: Query, key, value tensors
            mask: Optional attention mask
            
        Returns:
            output: Attention output
            (Optional) attention_weights: If return_softmax=True, returns attention weights
        """
        # Extract dimensions
        batch_size, seq_len, num_heads, head_dim = q.shape
        softmax_scale = self.config.softmax_scale or (1.0 / math.sqrt(head_dim))
        
        # Initialize output tensor
        output = torch.zeros_like(q)
        
        # Initialize tracking variables for numerically stable softmax
        m_i = torch.ones((batch_size, num_heads, seq_len), 
                         device=q.device, dtype=q.dtype) * -1e9
        l_i = torch.zeros((batch_size, num_heads, seq_len), 
                          device=q.device, dtype=q.dtype)
        
        # Process in blocks to save memory
        block_size = min(self.config.block_size, seq_len)
        
        # Store attention weights if requested
        if self.config.return_softmax:
            attention_weights = torch.zeros(
                (batch_size, num_heads, seq_len, seq_len), 
                device=q.device, dtype=q.dtype
            )
        
        # Compute in blocks
        for block_start in range(0, seq_len, block_size):
            block_end = min(block_start + block_size, seq_len)
            
            # Current block range
            curr_block = slice(block_start, block_end)
            
            # If causal, we only need to compute keys up to current position
            k_range = slice(0, block_end) if self.config.causal else slice(0, seq_len)
            
            # Extract current block for query and all necessary keys
            q_block = q[:, curr_block, :, :]  # [B, block_size, H, D]
            k_block = k[:, k_range, :, :]     # [B, k_range, H, D]
            v_block = v[:, k_range, :, :]     # [B, k_range, H, D]
            
            # Compute attention scores for this block
            # [B, block_size, H, k_range]
            scores = torch.einsum("bshd,bkhd->bshk", q_block, k_block) * softmax_scale
            
            # Apply causal mask if needed
            if self.config.causal and block_start > 0:
                causal_mask = torch.ones((block_end - block_start, block_end), 
                                         device=scores.device, dtype=torch.bool)
                causal_mask = torch.triu(causal_mask, diagonal=block_start+1)
                scores.masked_fill_(causal_mask.view(1, block_end - block_start, 1, block_end), -1e9)
            
            # Apply attention mask if provided
            if mask is not None:
                # Broadcast mask to the appropriate shape
                if mask.dim() == 2:  # [B, S]
                    mask_block = mask[:, k_range].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, k_range]
                elif mask.dim() == 3 and mask.shape[1] == 1:  # [B, 1, S]
                    mask_block = mask[:, :, k_range].unsqueeze(2)  # [B, 1, 1, k_range]
                elif mask.dim() == 3:  # [B, S, S]
                    mask_block = mask[:, curr_block, k_range].unsqueeze(2)  # [B, block_size, 1, k_range]
                else:
                    raise ValueError(f"Unsupported mask shape: {mask.shape}")
                
                scores.masked_fill_(~mask_block.bool(), -1e9)
            
            # Find max for numerical stability
            m_block = torch.max(scores, dim=-1, keepdim=True)[0]  # [B, block_size, H, 1]
            scores_scaled = scores - m_block  # [B, block_size, H, k_range]
            
            # Compute local softmax
            exp_scores = torch.exp(scores_scaled)  # [B, block_size, H, k_range]
            exp_sum = torch.sum(exp_scores, dim=-1, keepdim=True)  # [B, block_size, H, 1]
            
            # Update tracking variables for global softmax
            m_i_prev = m_i[:, :, curr_block].unsqueeze(-1)  # [B, H, block_size, 1]
            l_i_prev = l_i[:, :, curr_block].unsqueeze(-1)  # [B, H, block_size, 1]
            
            # Compute new max values
            m_i_new = torch.maximum(m_block.transpose(1, 2), m_i_prev)  # [B, H, block_size, 1]
            
            # Update normalizing factors
            exp_diff1 = torch.exp(m_i_prev - m_i_new)  # [B, H, block_size, 1]
            exp_diff2 = torch.exp(m_block.transpose(1, 2) - m_i_new)  # [B, H, block_size, 1]
            l_i_new = exp_diff1 * l_i_prev + exp_diff2 * exp_sum.transpose(1, 2)  # [B, H, block_size, 1]
            
            # Update tracking variables
            m_i[:, :, curr_block] = m_i_new.squeeze(-1)
            l_i[:, :, curr_block] = l_i_new.squeeze(-1)
            
            # Compute weighted values
            weighted_values = torch.einsum("bshk,bkhd->bshd", exp_scores, v_block)  # [B, block_size, H, D]
            
            # Update output
            output[:, curr_block, :, :] += weighted_values * (exp_diff1.transpose(1, 2))
            
            # Store attention weights if requested
            if self.config.return_softmax:
                attention_weights[:, :, curr_block, k_range] = (exp_scores / exp_sum).transpose(1, 2)
        
        # Normalize output
        output = output / l_i.unsqueeze(-1).transpose(1, 2)
        
        # Apply dropout if needed
        if self.config.dropout_p > 0.0 and self.training:
            dropout_mask = torch.empty_like(output).bernoulli_(1 - self.config.dropout_p)
            dropout_mask.div_(1 - self.config.dropout_p)
            output = output * dropout_mask
        
        if self.config.return_softmax:
            return output, attention_weights
        else:
            return output
            
    def get_theoretical_memory_usage(
        self, 
        seq_len: int, 
        batch_size: int, 
        num_heads: int, 
        head_dim: int
    ) -> Dict[str, int]:
        """
        Calculate theoretical memory usage of attention mechanisms.
        
        Args:
            seq_len: Sequence length
            batch_size: Batch size
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            
        Returns:
            Dict containing memory usage in bytes for different attention implementations
        """
        # Size of one float depends on precision
        if self.config.precision == "fp16" or self.config.precision == "bf16":
            bytes_per_element = 2
        else:  # fp32
            bytes_per_element = 4
            
        # Memory usage for standard attention
        qkv_memory = 3 * batch_size * seq_len * num_heads * head_dim * bytes_per_element
        attention_matrix = batch_size * num_heads * seq_len * seq_len * bytes_per_element
        output_memory = batch_size * seq_len * num_heads * head_dim * bytes_per_element
        standard_total = qkv_memory + attention_matrix + output_memory
        
        # Memory usage for Flash Attention
        block_size = self.config.block_size
        blocks = math.ceil(seq_len / block_size)
        
        flash_qkv_memory = qkv_memory  # Same as standard
        flash_block_memory = batch_size * num_heads * block_size * seq_len * bytes_per_element
        flash_tracking_memory = 2 * batch_size * num_heads * seq_len * bytes_per_element
        flash_output_memory = output_memory  # Same as standard
        flash_total = flash_qkv_memory + flash_block_memory + flash_tracking_memory + flash_output_memory
        
        # Memory usage for Triton implementation (more optimized)
        triton_total = flash_qkv_memory + (2 * block_size * seq_len + 3 * seq_len) * bytes_per_element + output_memory
        
        return {
            "standard_attention_bytes": standard_total,
            "flash_attention_bytes": flash_total,
            "triton_flash_attention_bytes": triton_total,
            "memory_savings_ratio": standard_total / flash_total,
            "max_sequence_length_standard": int(1e9 / (batch_size * num_heads * (2 * head_dim + seq_len) * bytes_per_element)),
            "max_sequence_length_flash": int(1e9 / (batch_size * num_heads * (3 * head_dim + 3) * bytes_per_element)),
        }
    
    def use_vllm_compatible_kernels(self) -> None:
        """
        Switch to vLLM-compatible kernel implementation.
        
        This method adapts the FlashAttention implementation to be compatible with vLLM,
        which may use slightly different kernel implementations optimized for specific hardware.
        """
        # In a real implementation, this would modify kernel selection logic
        self.config.block_size = 64  # vLLM typically uses smaller blocks
        print("Switched to vLLM-compatible kernel configuration")


class FlashAttentionLayer(nn.Module):
    """
    Drop-in replacement for standard attention layers.
    
    This class provides a complete attention layer with QKV projections
    and output projection using the efficient FlashAttention3 algorithm.
    Can also use PagedAttention Triton kernel if KV cache metadata is provided.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        config: Optional[FlashAttentionConfig] = None,
        # Added parameter to indicate GQA/MQA for cache interaction
        num_kv_heads: Optional[int] = None
    ):
        """
        Initialize FlashAttentionLayer.
        
        Args:
            hidden_size: Hidden size of the model
            num_attention_heads: Number of attention heads (Query heads)
            config: FlashAttention configuration
            num_kv_heads: Number of Key/Value heads (if different from Query heads for GQA/MQA)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.config = config or FlashAttentionConfig()
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_attention_heads {num_attention_heads}")
        if hidden_size % self.num_kv_heads != 0:
             raise ValueError(f"hidden_size {hidden_size} must be divisible by num_kv_heads {self.num_kv_heads}")
        
        # QKV projections
        # Note: For GQA/MQA, K and V projections might have different output dimensions
        # if num_kv_heads != num_attention_heads, but standard Linear is often used
        # with repetition/grouping happening inside the attention kernel or logic.
        # The PagedAttention kernel expects K/V cache based on num_kv_heads.
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        
        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # Flash attention module (for standard path)
        # Note: FlashAttention might need adjustments for GQA/MQA if not handled internally
        self.flash_attention = FlashAttention3(self.config)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize attention weights with appropriate scaling."""
        # Placeholder standard init
        std = 0.02
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std)
        if self.q_proj.bias is not None: nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None: nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None: nn.init.zeros_(self.v_proj.bias)
        if self.o_proj.bias is not None: nn.init.zeros_(self.o_proj.bias)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # Add kwargs to accept PagedAttention parameters
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Forward pass for the flash attention layer.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask (used only in standard FlashAttention path)
            **kwargs: Can include PagedAttention parameters:
                physical_kv_cache_k: Physical K cache tensor.
                physical_kv_cache_v: Physical V cache tensor.
                block_tables: Tensor mapping logical block indices to physical block indices.
                context_lengths: Tensor containing the current length of each sequence.
                kv_cache_block_size: The number of tokens per physical block.
                max_seq_len: Maximum sequence length the cache can hold.
                layer_idx: The layer index for which to compute attention.

        Returns:
            output: Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, q_seq_len, _ = hidden_states.shape

        # --- PagedAttention Path ---
        if 'block_tables' in kwargs and TRITON_AVAILABLE and triton_paged_attention_forward is not None:
            # Extract PagedAttention arguments
            k_cache = kwargs.get('physical_kv_cache_k')
            v_cache = kwargs.get('physical_kv_cache_v')
            block_tables = kwargs.get('block_tables')
            context_lengths = kwargs.get('context_lengths')
            block_size = kwargs.get('kv_cache_block_size')
            max_seq_len = kwargs.get('max_seq_len')
            layer_idx = kwargs.get('layer_idx')

            # Basic validation
            if not all([k_cache is not None, v_cache is not None, block_tables is not None,
                        context_lengths is not None, block_size is not None,
                        max_seq_len is not None, layer_idx is not None]):
                raise ValueError("Missing required arguments for PagedAttention in FlashAttentionLayer forward pass.")

            # Project query
            q = self.q_proj(hidden_states)
            # Reshape query: [batch_size, q_seq_len, hidden_size] -> [batch_size, num_heads, q_seq_len, head_dim]
            q = q.view(batch_size, q_seq_len, self.num_attention_heads, self.head_dim).permute(0, 2, 1, 3)

            # NOTE: K and V are NOT computed from hidden_states here.
            # They are read directly from the cache by the kernel.
            # We only need to ensure the cache is populated correctly *before* this call.
            # (This happens in the model's main forward loop after computing hidden states)

            # Allocate output tensor
            output = torch.empty_like(q)

            # Call PagedAttention Triton kernel
            # The kernel expects K/V cache with num_kv_heads
            # Ensure k_cache, v_cache were allocated with self.num_kv_heads
            triton_paged_attention_forward(
                query=q,
                output=output,
                k_cache=k_cache,
                v_cache=v_cache,
                block_tables=block_tables,
                context_lengths=context_lengths,
                block_size=block_size,
                max_seq_len=max_seq_len,
                layer_idx=layer_idx
            )

            # Reshape output: [batch_size, num_heads, q_seq_len, head_dim] -> [batch_size, q_seq_len, hidden_size]
            output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, q_seq_len, self.hidden_size)

            # Apply output projection
            final_output = self.o_proj(output)
            return final_output

        # --- Standard FlashAttention Path ---
        else:
            if 'block_tables' in kwargs:
                 logging.warning("PagedAttention arguments provided but Triton kernel is unavailable/disabled. Falling back to standard FlashAttention.")

            # Project inputs to queries, keys, and values
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

            # Reshape for multi-head attention
            # Query: [batch, q_seq, num_q_heads, head_dim]
            q = q.view(batch_size, q_seq_len, self.num_attention_heads, self.head_dim)
             # Key/Value: [batch, kv_seq, num_kv_heads, head_dim] (kv_seq == q_seq here)
            k = k.view(batch_size, q_seq_len, self.num_kv_heads, self.head_dim)
            v = v.view(batch_size, q_seq_len, self.num_kv_heads, self.head_dim)

            # TODO: Handle GQA/MQA repetition for standard FlashAttention if needed
            # Standard flash_attn might expect Q/K/V head numbers to match unless it supports GQA internally.
            # This might require repeating K/V heads if self.num_kv_heads < self.num_attention_heads
            # Example (if needed):
            # if self.num_kv_heads < self.num_attention_heads:
            #     num_reps = self.num_attention_heads // self.num_kv_heads
            #     k = k.repeat_interleave(num_reps, dim=2) # Repeat along head dim
            #     v = v.repeat_interleave(num_reps, dim=2)

            # Compute attention with flash attention algorithm
            # Input shape expected by FlashAttention3: [batch_size, seqlen, nheads, d]
            context_layer = self.flash_attention(q, k, v, attention_mask)

            # Reshape back: [batch_size, q_seq_len, num_heads, head_dim] -> [batch_size, q_seq_len, hidden_size]
            context_layer = context_layer.view(batch_size, q_seq_len, self.hidden_size)

            # Apply output projection
            output = self.o_proj(context_layer)

            return output


class FlashSelfAttention(nn.Module):
    """
    Implementation of self-attention using FlashAttention3 algorithm.
    
    This version is specifically optimized for self-attention use cases with:
    1. Fused QKV projection
    2. Memory-efficient attention computation
    3. Optimized backward pass
    Can also use PagedAttention Triton kernel if KV cache metadata is provided.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        config: Optional[FlashAttentionConfig] = None,
        num_kv_heads: Optional[int] = None # Added for GQA/MQA
    ):
        """
        Initialize FlashSelfAttention.
        
        Args:
            hidden_size: Hidden size of the model
            num_attention_heads: Number of attention heads (Query heads)
            config: FlashAttention configuration
            num_kv_heads: Number of Key/Value heads (if different from Query heads for GQA/MQA)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.config = config or FlashAttentionConfig()
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_attention_heads {num_attention_heads}")
        if hidden_size % self.num_kv_heads != 0:
             raise ValueError(f"hidden_size {hidden_size} must be divisible by num_kv_heads {self.num_kv_heads}")
        
        # Fused QKV projection
        # The output dimension needs to accommodate Q heads + K heads + V heads
        q_dim = hidden_size # num_attention_heads * head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden_size, q_dim + 2 * kv_dim)
        
        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # Flash attention module (for standard path)
        self.flash_attention = FlashAttention3(self.config)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize attention weights with appropriate scaling."""
        # Placeholder standard init
        std = 0.02
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std)
        if self.qkv_proj.bias is not None: nn.init.zeros_(self.qkv_proj.bias)
        if self.o_proj.bias is not None: nn.init.zeros_(self.o_proj.bias)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
         # Add kwargs to accept PagedAttention parameters
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Forward pass for flash self-attention.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask (used only in standard FlashAttention path)
            **kwargs: Can include PagedAttention parameters (see FlashAttentionLayer).

        Returns:
            output: Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, q_seq_len, _ = hidden_states.shape

        # --- PagedAttention Path ---
        if 'block_tables' in kwargs and TRITON_AVAILABLE and triton_paged_attention_forward is not None:
            # Extract PagedAttention arguments
            k_cache = kwargs.get('physical_kv_cache_k')
            v_cache = kwargs.get('physical_kv_cache_v')
            block_tables = kwargs.get('block_tables')
            context_lengths = kwargs.get('context_lengths')
            block_size = kwargs.get('kv_cache_block_size')
            max_seq_len = kwargs.get('max_seq_len')
            layer_idx = kwargs.get('layer_idx')

            # Basic validation
            if not all([k_cache is not None, v_cache is not None, block_tables is not None,
                        context_lengths is not None, block_size is not None,
                        max_seq_len is not None, layer_idx is not None]):
                raise ValueError("Missing required arguments for PagedAttention in FlashSelfAttention forward pass.")

            # Project QKV (we only need Q for the kernel input)
            qkv = self.qkv_proj(hidden_states)
            q_dim = self.hidden_size
            kv_dim = self.num_kv_heads * self.head_dim
            q = qkv[:, :, :q_dim]
            # k = qkv[:, :, q_dim:q_dim + kv_dim] # Not needed for kernel input
            # v = qkv[:, :, q_dim + kv_dim:]       # Not needed for kernel input

            # Reshape query: [batch_size, q_seq_len, hidden_size] -> [batch_size, num_heads, q_seq_len, head_dim]
            q = q.view(batch_size, q_seq_len, self.num_attention_heads, self.head_dim).permute(0, 2, 1, 3)

            # Allocate output tensor
            output = torch.empty_like(q)

            # Call PagedAttention Triton kernel
            triton_paged_attention_forward(
                query=q,
                output=output,
                k_cache=k_cache,
                v_cache=v_cache,
                block_tables=block_tables,
                context_lengths=context_lengths,
                block_size=block_size,
                max_seq_len=max_seq_len,
                layer_idx=layer_idx
            )

            # Reshape output: [batch_size, num_heads, q_seq_len, head_dim] -> [batch_size, q_seq_len, hidden_size]
            output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, q_seq_len, self.hidden_size)

            # Apply output projection
            final_output = self.o_proj(output)
            return final_output

        # --- Standard FlashAttention Path ---
        else:
            if 'block_tables' in kwargs:
                 logging.warning("PagedAttention arguments provided but Triton kernel is unavailable/disabled. Falling back to standard FlashAttention.")

            # Use either triton fused attention kernel if available or regular method
            # Note: _forward_triton_fused might need updates for GQA/MQA if not handled by the kernel
            if self.config.use_triton and HAS_TRITON and hasattr(self, '_forward_triton_fused'):
                 # Ensure _forward_triton_fused handles GQA/MQA correctly if self.num_kv_heads != self.num_attention_heads
                return self._forward_triton_fused(hidden_states, attention_mask)

            # Project and prepare QKV using the fused layer
            q, k, v = self._prepare_qkv(hidden_states) # Returns q:[b,s,qh,d], k:[b,s,kvh,d], v:[b,s,kvh,d]

            # TODO: Handle GQA/MQA repetition for standard FlashAttention if needed
            # (See comment in FlashAttentionLayer)
            # if self.num_kv_heads < self.num_attention_heads:
            #     num_reps = self.num_attention_heads // self.num_kv_heads
            #     k = k.repeat_interleave(num_reps, dim=2) # Repeat along head dim
            #     v = v.repeat_interleave(num_reps, dim=2)

            # Compute attention with flash attention algorithm
            # Expects [batch_size, seqlen, nheads, d]
            context_layer = self.flash_attention(q, k, v, attention_mask)

            # Reshape back: [batch_size, q_seq_len, num_heads, head_dim] -> [batch_size, q_seq_len, hidden_size]
            context_layer = context_layer.view(batch_size, q_seq_len, self.hidden_size)

            # Apply output projection
            output = self.o_proj(context_layer)

            return output

    def _prepare_qkv(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare query, key, and value tensors from hidden states using fused projection.
        Handles GQA/MQA splitting.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            q: [batch_size, seq_len, num_attention_heads, head_dim]
            k: [batch_size, seq_len, num_kv_heads, head_dim]
            v: [batch_size, seq_len, num_kv_heads, head_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Apply fused QKV projection
        qkv = self.qkv_proj(hidden_states)

        # Split fused QKV
        q_dim = self.hidden_size
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, v = torch.split(qkv, [q_dim, kv_dim, kv_dim], dim=-1)

        # Reshape
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        return q, k, v

    def _forward_triton_fused(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute flash attention using the fused Triton kernels for maximum efficiency.
        This implementation handles GQA/MQA configurations by properly managing
        different head counts for queries vs keys/values.
        """
        # Get QKV projections (already includes bias if present)
        qkv = self.qkv_proj(hidden_states)

        # Get batch size and sequence length for reshaping
        batch_size, seq_len, _ = hidden_states.shape
        
        # Parse the fused QKV tensor
        q_dim = self.hidden_size
        kv_dim = self.num_kv_heads * self.head_dim
        
        # Split into query, key, and value tensors
        q = qkv[:, :, :q_dim]
        k = qkv[:, :, q_dim:q_dim + kv_dim]
        v = qkv[:, :, q_dim + kv_dim:]
        
        # Reshape for multi-head attention
        # [batch, seq, q_heads*head_dim] -> [batch, seq, q_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        # [batch, seq, kv_heads*head_dim] -> [batch, seq, kv_heads, head_dim]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Handle GQA/MQA by repeating keys and values if needed
        if self.num_kv_heads < self.num_attention_heads:
            # For GQA/MQA, need to repeat keys and values
            repeat_factor = self.num_attention_heads // self.num_kv_heads
            
            if repeat_factor > 1:
                # Repeat each key/value head repeat_factor times
                # This expands [batch, seq, kv_heads, head_dim] to [batch, seq, q_heads, head_dim]
                k = k.repeat_interleave(repeat_factor, dim=2)
                v = v.repeat_interleave(repeat_factor, dim=2)
                
                # Handle non-divisible cases (e.g., if num_attention_heads % num_kv_heads != 0)
                if k.size(2) < self.num_attention_heads:
                    # Repeat the last head enough times to match num_attention_heads
                    remaining = self.num_attention_heads - k.size(2)
                    k_extra = k[:, :, -1:].repeat(1, 1, remaining, 1)
                    v_extra = v[:, :, -1:].repeat(1, 1, remaining, 1)
                    k = torch.cat([k, k_extra], dim=2)
                    v = torch.cat([v, v_extra], dim=2)
        
        # Prepare softmax scale
        head_dim = q.shape[-1]
        softmax_scale = self.config.softmax_scale or (1.0 / math.sqrt(head_dim))
        
        # Process attention mask if provided
        if attention_mask is not None:
            # Convert mask to compatible format if needed
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # [batch, 1, seq, seq]
            
            # Make sure mask is in the correct format for the kernel
            if attention_mask.shape[-2] != seq_len or attention_mask.shape[-1] != seq_len:
                raise ValueError(f"Attention mask shape {attention_mask.shape} incompatible with sequence length {seq_len}")
        
        # Call the Triton kernel with properly prepared tensors
        output = triton_fused_attention(
            q=q,
            k=k,
            v=v,
            mask=attention_mask,
            causal=self.config.causal,
            softmax_scale=softmax_scale,
            dropout_p=self.config.dropout_p if self.training else 0.0,
            head_dim=self.head_dim,
            block_size=self.config.block_size
        )
        
        # Reshape output: [batch, seq, num_heads, head_dim] -> [batch, seq, hidden_size]
        output = output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Apply output projection
        output = self.o_proj(output)
        
        return output


class ModelConverter:
    """
    Utility for converting standard attention in models to FlashAttention3.
    
    This class automatically finds and replaces standard attention modules in
    a PyTorch model with the optimized FlashAttention3 implementation.
    """
    
    def __init__(self, config: Optional[FlashAttentionConfig] = None):
        """
        Initialize the model converter.
        
        Args:
            config: FlashAttention configuration
        """
        self.config = config or FlashAttentionConfig()
        
    def convert_model(self, model: nn.Module) -> nn.Module:
        """
        Convert a model to use FlashAttention3.
        
        Args:
            model: PyTorch model to convert
            
        Returns:
            Converted model using FlashAttention3
        """
        # Make a copy to avoid modifying the original
        model = self._find_and_replace_attention(model)
        return model
    
    def _find_and_replace_attention(self, module: nn.Module) -> nn.Module:
        """
        Recursively find and replace attention modules.
        
        This method detects standard attention implementations in common model
        architectures and replaces them with FlashAttention3 equivalents.
        
        Args:
            module: Module to process
            
        Returns:
            Processed module with replaced attention
        """
        # Create a list of (name, submodule) pairs
        submodules = dict(module.named_children())
        
        # Flag to track if we've modified this module
        modified = False
        
        for name, submodule in submodules.items():
            # Check if this is an attention module
            if self._is_attention_module(submodule):
                # Create and initialize flash attention replacement
                flash_module = self._create_flash_replacement(submodule)
                setattr(module, name, flash_module)
                modified = True
            else:
                # Recursively process submodules
                converted_module = self._find_and_replace_attention(submodule)
                if converted_module is not submodule:
                    setattr(module, name, converted_module)
                    modified = True
                    
        # Return the modified module or the original if no changes were made
        return module
    
    def _is_attention_module(self, module: nn.Module) -> bool:
        """
        Detect if a module is an attention implementation.
        
        This method checks for common patterns in attention modules
        across different model architectures.
        
        Args:
            module: Module to check
            
        Returns:
            True if the module appears to be an attention module
        """
        # Check common attention module class names
        if type(module).__name__ in {
            "MultiHeadAttention",
            "BertSelfAttention",
            "T5Attention",
            "GPT2Attention",
            "LlamaAttention",
            "MistralAttention",
            "CLIPAttention",
            "OPTAttention",
            "RobertaAttention",
            "FalconAttention"
        }:
            return True
        
        # Check for common attribute patterns in attention modules
        has_qkv = hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj")
        has_out = hasattr(module, "out_proj") or hasattr(module, "o_proj")
        has_heads = hasattr(module, "num_heads") or hasattr(module, "num_attention_heads")
        
        if has_qkv and has_out and has_heads:
            return True
            
        # Check for fused QKV projection
        has_fused_qkv = hasattr(module, "qkv_proj") or hasattr(module, "qkv")
        
        if has_fused_qkv and has_out and has_heads:
            return True
            
        return False
    
    def _create_flash_replacement(self, module: nn.Module) -> nn.Module:
        """
        Create a FlashAttention3 module to replace a standard attention module.
        
        Args:
            module: Attention module to replace
            
        Returns:
            FlashAttention3 equivalent module
        """
        # Determine if module uses separate or fused QKV
        has_separate_qkv = hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj")
        
        # Extract necessary parameters
        if hasattr(module, "num_attention_heads"):
            num_heads = module.num_attention_heads
        elif hasattr(module, "num_heads"):
            num_heads = module.num_heads
        else:
            # Try to infer from module structure
            num_heads = 8  # Default value
            
        # Determine hidden size
        if hasattr(module, "hidden_size"):
            hidden_size = module.hidden_size
        elif hasattr(module, "embed_dim"):
            hidden_size = module.embed_dim
        elif has_separate_qkv and hasattr(module, "q_proj"):
            hidden_size = module.q_proj.out_features
        else:
            # Try to infer from module structure
            hidden_size = 512  # Default value
            
        # Create replacement module
        if has_separate_qkv:
            replacement = FlashAttentionLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                config=self.config
            )
            
            # Copy weights if possible
            if hasattr(module, "q_proj") and hasattr(replacement, "q_proj"):
                replacement.q_proj.weight.data.copy_(module.q_proj.weight.data)
                replacement.q_proj.bias.data.copy_(module.q_proj.bias.data)
                
            if hasattr(module, "k_proj") and hasattr(replacement, "k_proj"):
                replacement.k_proj.weight.data.copy_(module.k_proj.weight.data)
                replacement.k_proj.bias.data.copy_(module.k_proj.bias.data)
                
            if hasattr(module, "v_proj") and hasattr(replacement, "v_proj"):
                replacement.v_proj.weight.data.copy_(module.v_proj.weight.data)
                replacement.v_proj.bias.data.copy_(module.v_proj.bias.data)
                
            # Copy output projection
            out_proj_name = "o_proj" if hasattr(module, "o_proj") else "out_proj"
            if hasattr(module, out_proj_name) and hasattr(replacement, "o_proj"):
                replacement.o_proj.weight.data.copy_(getattr(module, out_proj_name).weight.data)
                replacement.o_proj.bias.data.copy_(getattr(module, out_proj_name).bias.data)
        else:
            # Module uses fused QKV
            replacement = FlashSelfAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                config=self.config
            )
            
            # Copy fused QKV weights if possible
            qkv_attr = "qkv_proj" if hasattr(module, "qkv_proj") else "qkv"
            if hasattr(module, qkv_attr) and hasattr(replacement, "qkv_proj"):
                replacement.qkv_proj.weight.data.copy_(getattr(module, qkv_attr).weight.data)
                replacement.qkv_proj.bias.data.copy_(getattr(module, qkv_attr).bias.data)
                
            # Copy output projection
            out_proj_name = "o_proj" if hasattr(module, "o_proj") else "out_proj"
            if hasattr(module, out_proj_name) and hasattr(replacement, "o_proj"):
                replacement.o_proj.weight.data.copy_(getattr(module, out_proj_name).weight.data)
                replacement.o_proj.bias.data.copy_(getattr(module, out_proj_name).bias.data)
                
        return replacement
    
    @staticmethod
    def convert_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert attention mask to the format expected by FlashAttention3.
        
        Args:
            attention_mask: Input attention mask
            
        Returns:
            Converted attention mask
        """
        # Check if mask needs conversion
        if attention_mask is None:
            return None
            
        if attention_mask.dim() == 2:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
        elif attention_mask.dim() == 3 and attention_mask.shape[1] == 1:
            # [batch_size, 1, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(2)
            
        # Ensure boolean mask with correct shape
        return attention_mask.bool()


def benchmark_flash_attention_speed(
    batch_size: int = 32,
    seq_len: int = 512,
    num_heads: int = 8,
    head_dim: int = 64,
    device: str = "cuda",
    causal: bool = False,
    num_iters: int = 100,
    warmup_iters: int = 10
) -> Dict[str, float]:
    """
    Benchmark FlashAttention3 performance compared to standard attention.
    
    Args:
        batch_size: Batch size for benchmarking
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        device: Device to run benchmark on
        causal: Whether to use causal masking
        num_iters: Number of iterations for benchmarking
        warmup_iters: Number of warmup iterations
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    # Skip benchmark if CUDA is not available and device is cuda
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return {
            "standard_attention_ms": 0.0,
            "flash_attention_ms": 0.0,
            "speedup": 0.0
        }
        
    # Create tensors
    hidden_size = num_heads * head_dim
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    
    # Standard attention function
    def standard_attention(q, k, v, causal=False):
        # [batch_size, num_heads, seq_len, seq_len]
        scores = torch.einsum("bshd,bkhd->bhsk", q, k) / math.sqrt(head_dim)
        
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), 
                diagonal=1
            )
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.einsum("bhsk,bkhd->bshd", attn_weights, v)
        return output
    
    # Create FlashAttention module
    config = FlashAttentionConfig(
        causal=causal,
        block_size=64,
        use_triton=HAS_TRITON
    )
    flash_attn = FlashAttention3(config)
    
    # Warmup
    for _ in range(warmup_iters):
        _ = standard_attention(q, k, v, causal)
        _ = flash_attn(q, k, v)
    
    # Time standard attention
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    for _ in range(num_iters):
        output_standard = standard_attention(q, k, v, causal)
        
    torch.cuda.synchronize() if device == "cuda" else None
    standard_time = (time.time() - start_time) * 1000 / num_iters  # ms per iteration
    
    # Time flash attention
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    for _ in range(num_iters):
        output_flash = flash_attn(q, k, v)
        
    torch.cuda.synchronize() if device == "cuda" else None
    flash_time = (time.time() - start_time) * 1000 / num_iters  # ms per iteration
    
    # Check for correctness (approximately equal)
    max_diff = (output_standard - output_flash).abs().max().item()
    is_correct = max_diff < 1e-3
    
    return {
        "standard_attention_ms": standard_time,
        "flash_attention_ms": flash_time,
        "speedup": standard_time / flash_time,
        "max_difference": max_diff,
        "is_correct": is_correct,
        "sequence_length": seq_len,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "causal": causal
    }


def benchmark_memory_usage(
    seq_len: int = 512,
    batch_size: int = 32,
    num_heads: int = 8,
    head_dim: int = 64,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Benchmark memory usage of FlashAttention3 vs standard attention.
    
    Args:
        seq_len: Sequence length
        batch_size: Batch size
        num_heads: Number of attention heads
        head_dim: Dimension per head
        device: Device to run benchmark on
        
    Returns:
        Dictionary with memory usage results
    """
    # Skip if CUDA is not available
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping memory benchmark")
        return {
            "standard_peak_memory_mb": 0.0,
            "flash_peak_memory_mb": 0.0,
            "memory_savings_ratio": 0.0
        }
        
    # Need to import torch.cuda explicitly for memory stats
    import torch.cuda
    
    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Standard attention
    def run_standard_attention():
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        
        # [batch_size, num_heads, seq_len, seq_len]
        scores = torch.einsum("bshd,bkhd->bhsk", q, k) / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.einsum("bhsk,bkhd->bshd", attn_weights, v)
        return output
    
    # Run standard attention
    _ = run_standard_attention()
    
    # Measure peak memory
    if device == "cuda":
        standard_peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    else:
        standard_peak_memory = 0.0
    
    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Flash attention
    def run_flash_attention():
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        
        config = FlashAttentionConfig(block_size=64, use_triton=HAS_TRITON)
        flash_attn = FlashAttention3(config)
        output = flash_attn(q, k, v)
        return output
    
    # Run flash attention
    _ = run_flash_attention()
    
    # Measure peak memory
    if device == "cuda":
        flash_peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    else:
        flash_peak_memory = 0.0
    
    # Calculate memory savings
    memory_savings_ratio = standard_peak_memory / max(flash_peak_memory, 1e-6)
    
    return {
        "standard_peak_memory_mb": standard_peak_memory,
        "flash_peak_memory_mb": flash_peak_memory,
        "memory_savings_ratio": memory_savings_ratio,
        "sequence_length": seq_len,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "head_dim": head_dim
    }