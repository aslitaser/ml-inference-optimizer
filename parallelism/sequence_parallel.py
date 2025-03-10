import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple, List, Dict, Any, Callable, Union
import math
import functools
import time
import weakref
from dataclasses import dataclass

from parallelism.communication import (
    get_rank, get_world_size, all_reduce, all_gather, barrier
)
from parallelism.parallel_utils import (
    divide, split_tensor_along_dim, gather_tensor_along_dim,
    set_tensor_model_parallel_attributes, ensure_divisibility,
    get_partition_start_end
)

@dataclass
class SequenceParallelConfig:
    """Configuration class for sequence parallelism.
    
    Attributes:
        world_size: Number of GPUs to use
        sp_size: Sequence parallel size
        overlap_communication: Whether to overlap communication with computation
        attention_handling: How to handle attention ("local", "ring", "full")
        chunk_size: Size of sequence chunks (None for automatic)
        buffer_reuse: Whether to reuse communication buffers
        communication_dtype: Data type for communication
    """
    world_size: int = 1
    sp_size: int = 1
    overlap_communication: bool = True
    attention_handling: str = "ring"  # "local", "ring", or "full"
    chunk_size: Optional[int] = None
    buffer_reuse: bool = True
    communication_dtype: torch.dtype = torch.float16
    
    def __post_init__(self):
        """Validate the configuration."""
        # Ensure sequence parallel size divides world size
        if self.world_size % self.sp_size != 0:
            raise ValueError(f"Sequence parallel size ({self.sp_size}) must divide world size ({self.world_size})")
        
        # Validate attention handling strategy
        if self.attention_handling not in ["local", "ring", "full"]:
            raise ValueError(f"Attention handling strategy '{self.attention_handling}' not supported. "
                            f"Use 'local', 'ring', or 'full'.")
        
        # For chunk_size=None, we'll determine it dynamically during forward pass
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError(f"Chunk size must be positive, got {self.chunk_size}")
    
    def get_sp_group(self) -> Optional[dist.ProcessGroup]:
        """Get sequence parallel process group.
        
        Returns:
            Process group for sequence parallelism
        """
        # In a real implementation, this would return the actual process group
        # that was created by setup_sequence_parallel_group
        from parallelism.communication import setup_sequence_parallel_group
        return setup_sequence_parallel_group(self.world_size, self.sp_size)
    
    def get_dp_size(self) -> int:
        """Get data parallel size.
        
        Returns:
            Data parallel size
        """
        return self.world_size // self.sp_size
    
    def get_rank_info(self) -> Tuple[int, int]:
        """Get sequence parallel and data parallel ranks.
        
        Returns:
            Tuple of (sp_rank, dp_rank)
        """
        rank = get_rank()
        sp_rank = rank % self.sp_size
        dp_rank = rank // self.sp_size
        return sp_rank, dp_rank


class SequenceShardedModule(nn.Module):
    """Base class for sequence-parallel modules.
    
    This module wraps a regular PyTorch module and handles sequence parallelism logic.
    """
    
    def __init__(self, module: nn.Module, config: SequenceParallelConfig):
        """Initialize the sequence-sharded module.
        
        Args:
            module: Module to wrap
            config: Sequence parallel configuration
        """
        super().__init__()
        self.module = module
        self.config = config
        
        # Communication buffers
        self.scatter_buffers = []
        self.gather_buffers = []
        
        # Overlap communication
        self.pending_handles = []
        self.use_async = config.overlap_communication
        
        # Process group
        self.sp_group = config.get_sp_group()
        self.sp_rank, self.dp_rank = config.get_rank_info()
        
        # Track profiling info
        self.last_communication_time = 0.0
        self.compute_time = 0.0
        
        # Initialize buffers
        if config.buffer_reuse:
            self._init_buffers()
    
    def _init_buffers(self):
        """
        Initialize communication buffers for reuse.
        
        This method pre-allocates communication buffers for sequence parallelism
        operations to avoid repeated memory allocations during inference.
        Buffers are properly sized based on typical transformer architectures.
        """
        # These will be populated during the first forward pass when tensor shapes are known,
        # but we can initialize some common sizes in advance
        
        # If we don't want buffer reuse, don't pre-allocate
        if not self.config.buffer_reuse:
            return
            
        # Make sure we're on a CUDA device before allocating
        if not torch.cuda.is_available():
            return
            
        # Get device to allocate buffers
        device = torch.cuda.current_device()
        
        # Create buffers for scatter and gather operations
        # Common sizes for sequence dimensions: 256, 512, 1024, 2048
        # Common batch sizes: 1, 2, 4, 8, 16, 32
        # Common embedding dimensions: 768, 1024, 2048, 4096
        common_seq_lengths = [256, 512, 1024, 2048]
        common_batch_sizes = [1, 2, 4, 8, 16]
        common_hidden_sizes = [768, 1024, 2048, 4096]
        
        # For sequence parallelism, the critical dimension is sequence length
        # We allocate buffers for common combinations
        for seq_len in common_seq_lengths:
            for batch_size in common_batch_sizes[:2]:  # Only smaller batch sizes
                for hidden_size in common_hidden_sizes[:3]:  # Only smaller hidden sizes
                    # Estimate size of sequence chunk
                    seq_chunk = seq_len // self.config.sp_size
                    
                    # Skip if too large to avoid OOM
                    buffer_size_gb = batch_size * seq_chunk * hidden_size * 4 / (1024**3)
                    if buffer_size_gb > 0.5:  # Skip buffers larger than 0.5 GB 
                        continue
                    
                    # Scatter buffers
                    scatter_shape = (batch_size, seq_chunk, hidden_size)
                    for dtype in [torch.float16, torch.float32]:
                        # Limit the number of buffers to avoid excessive memory usage
                        if len(self.scatter_buffers) < 8:
                            self.scatter_buffers.append(
                                torch.empty(scatter_shape, dtype=dtype, device=device)
                            )
                    
                    # Gather buffers
                    gather_shape = (batch_size, seq_len, hidden_size)
                    for dtype in [torch.float16, torch.float32]:
                        # Limit the number of buffers to avoid excessive memory usage
                        if len(self.gather_buffers) < 8:
                            self.gather_buffers.append(
                                torch.empty(gather_shape, dtype=dtype, device=device)
                            )
        
        # For sequence sharding, also allocate buffers for attention patterns
        # These are used in ring attention for key-value exchange
        if self.config.attention_handling == "ring":
            # Allocate buffers for ring exchange
            # We need buffers sized for key and value tensors
            # Typical shape: [batch_size, num_heads, seq_chunk, head_dim]
            for seq_len in common_seq_lengths[:2]:  # Only smaller sequence lengths
                for batch_size in common_batch_sizes[:1]:  # Only smallest batch size
                    # Typical transformer parameters
                    num_heads = 16
                    head_dim = 64
                    
                    # Calculate sequence chunk size
                    seq_chunk = seq_len // self.config.sp_size
                    
                    # Allocate buffers for key and value tensors
                    kv_shape = (batch_size, num_heads, seq_chunk, head_dim)
                    
                    # Create buffer for float16 (most common for inference)
                    self.scatter_buffers.append(
                        torch.empty(kv_shape, dtype=torch.float16, device=device)
                    )
                    
                    # Create buffer for attention mask
                    mask_shape = (batch_size, 1, seq_chunk, seq_chunk)
                    self.scatter_buffers.append(
                        torch.empty(mask_shape, dtype=torch.float16, device=device)
                    )
    
    def _ensure_buffer(self, buffers: list, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """Get or create a buffer for communication.
        
        Args:
            buffers: List of existing buffers
            shape: Required buffer shape
            dtype: Required buffer dtype
            
        Returns:
            Buffer tensor
        """
        if not self.config.buffer_reuse:
            # Don't reuse buffers, create a new one
            return torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
        
        # Check if we have a suitable buffer already
        for i, buffer in enumerate(buffers):
            if buffer.shape == shape and buffer.dtype == dtype:
                # Move buffer to the end of the list to implement LRU caching
                buffers.append(buffers.pop(i))
                return buffers[-1]
        
        # Create a new buffer
        new_buffer = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
        buffers.append(new_buffer)
        return new_buffer
    
    def _shard_sequence_dim(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Shard tensor along sequence dimension.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            Tensor sharded along sequence dimension
        """
        from parallelism.communication import scatter_along_sequence_dim
        
        # Use built-in function or direct implementation
        return scatter_along_sequence_dim(hidden_states, self.config.sp_size)
    
    def _gather_sequence_dim(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gather tensor along sequence dimension.
        
        Args:
            hidden_states: Sharded tensor
            
        Returns:
            Tensor gathered along sequence dimension
        """
        from parallelism.communication import gather_along_sequence_dim
        
        # Use built-in function or direct implementation
        return gather_along_sequence_dim(hidden_states, self.config.sp_size)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with sequence parallelism.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask (optional)
            
        Returns:
            Output tensor
        """
        # Shard along sequence dimension
        start_time = time.time()
        
        # Prepare attention mask for sequence parallelism if needed
        if attention_mask is not None:
            sp_attention_mask = self._prepare_attention_mask(attention_mask)
        else:
            sp_attention_mask = None
        
        # Shard input along sequence dimension
        sharded_hidden_states = self._shard_sequence_dim(hidden_states)
        
        # Record communication time
        self.last_communication_time = time.time() - start_time
        
        # Forward through wrapped module
        compute_start = time.time()
        if sp_attention_mask is not None:
            output = self.module(sharded_hidden_states, sp_attention_mask)
        else:
            output = self.module(sharded_hidden_states)
        
        # Record compute time
        self.compute_time = time.time() - compute_start
        
        # Gather output along sequence dimension
        start_time = time.time()
        gathered_output = self._gather_sequence_dim(output)
        
        # Update communication time
        self.last_communication_time += time.time() - start_time
        
        return gathered_output
    
    def _prepare_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Prepare attention mask for sequence parallelism.
        
        Args:
            attention_mask: Full attention mask
            
        Returns:
            Sequence parallel attention mask
        """
        # Use utility function
        return create_sequence_parallel_attention_mask(attention_mask, self.config.sp_size)[self.sp_rank]
    
    def optimize_for_inference(self) -> None:
        """Apply inference optimizations to the module."""
        # Apply FP16 weights if appropriate
        if self.config.communication_dtype == torch.float16:
            # Convert to FP16 for faster communication
            for param in self.parameters():
                if param.dtype == torch.float32:
                    param.data = param.data.to(torch.float16)
        
        # Apply kernel fusion if available (would call into specialized kernels)
        if hasattr(self.module, 'optimize_for_inference'):
            self.module.optimize_for_inference()
        
        # Pre-allocate buffers with expected sizes
        if self.config.buffer_reuse:
            # This would be device-specific optimization in a real implementation
            pass


class SequenceParallelAttention(nn.Module):
    """Attention module using sequence parallelism.
    
    This implements sequence-parallel multi-head attention with different strategies:
    - local: Each worker processes its local chunk independently
    - ring: Ring communication pattern for partial attention
    - full: Full all-to-all communication for complete attention
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_attention_heads: int, 
        config: SequenceParallelConfig,
        attention_dropout: float = 0.1,
        head_dim: Optional[int] = None,
        bias: bool = True
    ):
        """Initialize sequence parallel attention.
        
        Args:
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            config: Sequence parallel configuration
            attention_dropout: Dropout probability for attention weights
            head_dim: Head dimension (if None, calculated as hidden_size / num_attention_heads)
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.config = config
        
        # Calculate dimensions
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_dim
        
        # Create query, key, value projections
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=bias)
        
        # Output projection
        self.output = nn.Linear(self.all_head_size, hidden_size, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(attention_dropout)
        
        # Communication buffers
        self.key_value_buffers = []
        
        # Process group
        self.sp_group = config.get_sp_group()
        self.sp_rank, self.dp_rank = config.get_rank_info()
        
        # Choose attention implementation based on config
        self.attention_impl = self._select_attention_impl()
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize the weights."""
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.output.weight)
        
        if self.query.bias is not None:
            nn.init.zeros_(self.query.bias)
            nn.init.zeros_(self.key.bias)
            nn.init.zeros_(self.value.bias)
            nn.init.zeros_(self.output.bias)
    
    def _select_attention_impl(self) -> Callable:
        """Select the appropriate attention implementation.
        
        Returns:
            Attention implementation function
        """
        if self.config.attention_handling == "local":
            return self._local_attention
        elif self.config.attention_handling == "ring":
            return self._ring_attention
        elif self.config.attention_handling == "full":
            return self._full_attention
        else:
            raise ValueError(f"Unknown attention handling: {self.config.attention_handling}")
    
    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape and transpose tensor for attention computation.
        
        Args:
            x: Input tensor [batch, seq_len, all_head_size]
            
        Returns:
            Reshaped tensor [batch, num_heads, seq_len, head_dim]
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for sequence parallel attention.
        
        Args:
            hidden_states: Sharded input tensor [batch, seq_len/sp_size, hidden_size]
            attention_mask: Attention mask (optional, adapted for sequence parallelism)
            
        Returns:
            Output tensor [batch, seq_len/sp_size, hidden_size]
        """
        # Calculate query, key, value projections
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        # Use selected attention implementation
        context_layer = self.attention_impl(
            q=self._transpose_for_scores(q),
            k=self._transpose_for_scores(k),
            v=self._transpose_for_scores(v),
            attention_mask=attention_mask
        )
        
        # Reshape context layer
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Project to output size
        output = self.output(context_layer)
        
        return output
    
    def _local_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Perform local-only attention on the sequence chunk.
        
        Args:
            q: Query tensor [batch, num_heads, seq_len/sp_size, head_dim]
            k: Key tensor [batch, num_heads, seq_len/sp_size, head_dim]
            v: Value tensor [batch, num_heads, seq_len/sp_size, head_dim]
            attention_mask: Attention mask (optional)
            
        Returns:
            Output tensor [batch, num_heads, seq_len/sp_size, head_dim]
        """
        # Take the dot product between query and key
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        
        # Scale attention scores
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, v)
        
        return context_layer
    
    def _ring_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Perform attention using ring communication pattern.
        
        Args:
            q: Query tensor [batch, num_heads, seq_len/sp_size, head_dim]
            k: Key tensor [batch, num_heads, seq_len/sp_size, head_dim]
            v: Value tensor [batch, num_heads, seq_len/sp_size, head_dim]
            attention_mask: Attention mask (optional)
            
        Returns:
            Output tensor [batch, num_heads, seq_len/sp_size, head_dim]
        """
        from parallelism.communication import ring_exchange
        
        # Get dimensions
        batch_size = q.size(0)
        num_heads = q.size(1)
        seq_chunk_len = q.size(2)
        head_dim = q.size(3)
        sp_size = self.config.sp_size
        
        # Prepare for accumulation of attention outputs
        context_layer = torch.zeros_like(q)
        
        # Keep track of the current k, v, and attention mask
        current_k = k
        current_v = v
        current_mask = attention_mask
        
        # Process each step in the ring
        for i in range(sp_size):
            # Compute attention scores for current chunk
            attention_scores = torch.matmul(q, current_k.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.head_dim)
            
            # Apply attention mask if provided
            if current_mask is not None:
                attention_scores = attention_scores + current_mask
            
            # Apply softmax for each step independently
            # Note: This is not mathematically equivalent to full attention
            # but approximates it while keeping memory consumption low
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            
            # Apply attention to values and accumulate
            chunk_context = torch.matmul(attention_probs, current_v)
            context_layer += chunk_context
            
            # Move to next chunk in the ring if we haven't processed all chunks
            if i < sp_size - 1:
                # Exchange keys and values with next rank in the ring
                current_k, current_v, current_mask = ring_exchange(
                    current_k, current_v, current_mask, 
                    group=self.sp_group
                )
        
        # Average the context layer to account for multiple accumulations
        context_layer = context_layer / sp_size
        
        return context_layer
    
    def _full_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Perform full attention with all-to-all communication.
        
        Args:
            q: Query tensor [batch, num_heads, seq_len/sp_size, head_dim]
            k: Key tensor [batch, num_heads, seq_len/sp_size, head_dim]
            v: Value tensor [batch, num_heads, seq_len/sp_size, head_dim]
            attention_mask: Attention mask (optional)
            
        Returns:
            Output tensor [batch, num_heads, seq_len/sp_size, head_dim]
        """
        # Get sequence parallel info
        sp_size = self.config.sp_size
        sp_group = self.sp_group
        
        # Gather all key and value tensors
        gathered_k = gather_tensor_along_dim(k, dim=2, world_size=sp_size)
        gathered_v = gather_tensor_along_dim(v, dim=2, world_size=sp_size)
        
        # Gather attention mask if provided
        if attention_mask is not None:
            # Need to handle potentially different attention mask shapes
            # This is a simplified implementation
            gathered_mask = gather_tensor_along_dim(attention_mask, dim=-1, world_size=sp_size)
        else:
            gathered_mask = None
        
        # Calculate attention scores
        attention_scores = torch.matmul(q, gathered_k.transpose(-1, -2))
        
        # Scale attention scores
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if gathered_mask is not None:
            attention_scores = attention_scores + gathered_mask
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, gathered_v)
        
        return context_layer


class SequenceParallelMLP(nn.Module):
    """MLP module using sequence parallelism.
    
    This implements a sequence-parallel MLP with:
    - Chunking along the sequence dimension
    - Independent computation on each chunk
    - Gather at the end for the final result
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        config: SequenceParallelConfig,
        activation: Callable = F.gelu,
        dropout_prob: float = 0.1,
        bias: bool = True
    ):
        """Initialize sequence parallel MLP.
        
        Args:
            hidden_size: Hidden dimension size
            intermediate_size: Intermediate dimension size
            config: Sequence parallel configuration
            activation: Activation function
            dropout_prob: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.config = config
        
        # Create dense layers
        self.dense_h_to_4h = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.dense_4h_to_h = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        # Activation and dropout
        self.activation = activation
        self.dropout = nn.Dropout(dropout_prob)
        
        # Process group
        self.sp_group = config.get_sp_group()
        self.sp_rank, self.dp_rank = config.get_rank_info()
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize the weights."""
        # Apply standard initialization
        nn.init.xavier_uniform_(self.dense_h_to_4h.weight)
        nn.init.xavier_uniform_(self.dense_4h_to_h.weight)
        
        if self.dense_h_to_4h.bias is not None:
            nn.init.zeros_(self.dense_h_to_4h.bias)
            nn.init.zeros_(self.dense_4h_to_h.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for sequence parallel MLP.
        
        Args:
            hidden_states: Input tensor [batch, seq_len/sp_size, hidden_size]
            
        Returns:
            Output tensor [batch, seq_len/sp_size, hidden_size]
        """
        # Project to intermediate size
        intermediate = self.dense_h_to_4h(hidden_states)
        
        # Apply activation
        intermediate = self.activation(intermediate)
        
        # Project back to hidden size
        output = self.dense_4h_to_h(intermediate)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output


class SequenceParallelConverter:
    """Converter for transforming models to use sequence parallelism."""
    
    def __init__(self, config: SequenceParallelConfig):
        """Initialize sequence parallel converter.
        
        Args:
            config: Sequence parallel configuration
        """
        self.config = config
    
    def convert_model(self, model: nn.Module) -> nn.Module:
        """Convert a model to use sequence parallelism.
        
        Args:
            model: Input model
            
        Returns:
            Model converted for sequence parallelism
        """
        import copy
        
        # Create a deep copy to avoid modifying the original
        model_sp = copy.deepcopy(model)
        
        # Convert attention and MLP layers
        model_sp = self.convert_attention_layers(model_sp)
        model_sp = self.convert_mlp_layers(model_sp)
        
        # Wrap the entire model to handle input/output sequence sharding
        model_sp = SequenceShardedModule(model_sp, self.config)
        
        return model_sp
    
    def convert_attention_layers(self, model: nn.Module) -> nn.Module:
        """Convert attention layers to sequence parallel versions.
        
        Args:
            model: Input model
            
        Returns:
            Model with converted attention layers
        """
        # Get all modules
        for name, module in list(model.named_children()):
            # If this is already a sequence parallel module, skip it
            if isinstance(module, (SequenceParallelAttention, SequenceShardedModule)):
                continue
            
            # Handle attention modules
            if ("attention" in name.lower() and 
                hasattr(module, "query") and 
                hasattr(module, "key") and 
                hasattr(module, "value")):
                
                # Extract parameters for the new attention layer
                hidden_size = module.query.weight.size(1)
                
                # Determine number of attention heads
                if hasattr(module, "num_attention_heads"):
                    num_heads = module.num_attention_heads
                elif hasattr(module, "num_heads"):
                    num_heads = module.num_heads
                else:
                    # Try to infer from dimensions
                    all_head_size = module.query.weight.size(0)
                    if hasattr(module, "head_dim"):
                        head_dim = module.head_dim
                    else:
                        # Assume head_dim is a common value like 64
                        head_dim = 64
                    num_heads = all_head_size // head_dim
                
                # Create sequence parallel attention
                sp_attention = SequenceParallelAttention(
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    config=self.config,
                    bias=module.query.bias is not None
                )
                
                # Replace module
                setattr(model, name, sp_attention)
            else:
                # Recursively convert children
                setattr(model, name, self.convert_attention_layers(module))
        
        return model
    
    def convert_mlp_layers(self, model: nn.Module) -> nn.Module:
        """Convert MLP layers to sequence parallel versions.
        
        Args:
            model: Input model
            
        Returns:
            Model with converted MLP layers
        """
        # Get all modules
        for name, module in list(model.named_children()):
            # If this is already a sequence parallel module, skip it
            if isinstance(module, (SequenceParallelMLP, SequenceShardedModule)):
                continue
            
            # Handle MLP modules using naming patterns common in transformer models
            if any(pattern in name.lower() for pattern in ["mlp", "ffn", "feed_forward"]):
                # Try to extract parameters for new MLP layer
                try:
                    # Determine hidden and intermediate sizes
                    # Look for typical layer names in various model architectures
                    
                    # BERT, RoBERTa style
                    if hasattr(module, "dense") and hasattr(module, "output"):
                        hidden_size = module.output.dense.weight.size(0)
                        intermediate_size = module.dense.weight.size(0)
                    
                    # GPT, OPT style
                    elif hasattr(module, "c_fc") and hasattr(module, "c_proj"):
                        hidden_size = module.c_proj.weight.size(0)
                        intermediate_size = module.c_fc.weight.size(0)
                    
                    # T5, BART style
                    elif hasattr(module, "wi") and hasattr(module, "wo"):
                        hidden_size = module.wo.weight.size(0)
                        intermediate_size = module.wi.weight.size(0)
                    
                    # Generic case
                    else:
                        # Find all linear layers
                        linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
                        if len(linear_layers) >= 2:
                            # Assume first linear layer goes from hidden->intermediate
                            # and last goes from intermediate->hidden
                            hidden_size = linear_layers[-1].weight.size(0)
                            intermediate_size = linear_layers[0].weight.size(0)
                        else:
                            raise ValueError(f"Could not determine sizes for MLP conversion in {name}")
                    
                    # Create sequence parallel MLP
                    sp_mlp = SequenceParallelMLP(
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        config=self.config,
                        bias=linear_layers[0].bias is not None if 'linear_layers' in locals() else True
                    )
                    
                    # Replace module
                    setattr(model, name, sp_mlp)
                    
                except (AttributeError, ValueError, IndexError):
                    # If we can't convert, continue with recursion
                    setattr(model, name, self.convert_mlp_layers(module))
            else:
                # Recursively convert children
                setattr(model, name, self.convert_mlp_layers(module))
        
        return model
    
    def partition_input_data(self, inputs: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Partition input data for sequence parallelism.
        
        Args:
            inputs: Input data dictionary
            
        Returns:
            List of partitioned input dictionaries
        """
        sp_size = self.config.sp_size
        partitioned_inputs = []
        
        # Create a copy of inputs for each SP rank
        for sp_rank in range(sp_size):
            rank_inputs = {}
            
            for key, tensor in inputs.items():
                # Partition sequence dimension for sequence tensors
                if key in ["input_ids", "attention_mask", "token_type_ids"]:
                    # Sequence tensors
                    partitioned_tensor = partition_sequence(tensor, sp_size)[sp_rank]
                    rank_inputs[key] = partitioned_tensor
                else:
                    # Non-sequence tensors (labels, etc.)
                    rank_inputs[key] = tensor
            
            partitioned_inputs.append(rank_inputs)
        
        return partitioned_inputs
    
    def gather_output_data(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Gather output data from sequence parallel workers.
        
        Args:
            outputs: List of output tensors from each worker
            
        Returns:
            Gathered output tensor
        """
        return gather_sequence(outputs, dim=1)


# Utility functions

def partition_sequence(tensor: torch.Tensor, sp_size: int) -> List[torch.Tensor]:
    """Partition a tensor along the sequence dimension.
    
    Args:
        tensor: Input tensor with shape [..., seq_len, ...]
        sp_size: Sequence parallel size
        
    Returns:
        List of partitioned tensors
    """
    # Determine sequence dimension based on tensor shape
    if tensor.dim() == 2 and tensor.size(1) > tensor.size(0):
        # [batch_size, seq_len]
        seq_dim = 1
    elif tensor.dim() == 3:
        # [batch_size, seq_len, hidden_size]
        seq_dim = 1
    elif tensor.dim() == 4 and tensor.size(2) == tensor.size(3):
        # [batch_size, num_heads, seq_len, seq_len] (attention mask)
        # We partition the last dimension
        seq_dim = 3
    else:
        # Default to dimension 1
        seq_dim = 1
    
    # Get sequence length and ensure it's divisible by sp_size
    seq_len = tensor.size(seq_dim)
    ensure_divisibility(seq_len, sp_size)
    
    # Calculate partition size
    partition_size = seq_len // sp_size
    
    # Split the tensor
    partitions = []
    for i in range(sp_size):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size
        
        # Create partition using narrow to avoid copy
        partition = tensor.narrow(seq_dim, start_idx, partition_size)
        partitions.append(partition)
    
    return partitions


def gather_sequence(tensor_list: List[torch.Tensor], dim: int = 1) -> torch.Tensor:
    """Gather a list of tensors along the sequence dimension.
    
    Args:
        tensor_list: List of tensors to gather
        dim: Dimension along which to gather
        
    Returns:
        Gathered tensor
    """
    # If only one tensor, return it directly
    if len(tensor_list) == 1:
        return tensor_list[0]
    
    # Ensure all tensors have the same shape except along gather dimension
    base_shape = list(tensor_list[0].shape)
    for i, tensor in enumerate(tensor_list[1:], 1):
        curr_shape = list(tensor.shape)
        for j, (base_dim, curr_dim) in enumerate(zip(base_shape, curr_shape)):
            if j != dim and base_dim != curr_dim:
                raise ValueError(
                    f"Tensor at index {i} has incompatible shape: {curr_shape} vs {base_shape}"
                )
    
    # Gather along the specified dimension
    return torch.cat(tensor_list, dim=dim)


def create_sequence_parallel_attention_mask(
    attention_mask: torch.Tensor, 
    sp_size: int
) -> List[torch.Tensor]:
    """Create attention masks for sequence parallelism.
    
    Args:
        attention_mask: Original attention mask
        sp_size: Sequence parallel size
        
    Returns:
        List of attention masks for each sequence parallel worker
    """
    # Determine the structure of the attention mask
    if attention_mask.dim() == 2:
        # [batch_size, seq_len] -> convert to 4D attention mask
        batch_size, seq_len = attention_mask.shape
        # Convert to 4D mask [batch_size, 1, 1, seq_len]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # Convert from 0/1 to -10000.0/0.0
        attention_mask = (1.0 - attention_mask) * -10000.0
    
    # Handle 4D masks [batch_size, 1, seq_len, seq_len] or [batch_size, 1, 1, seq_len]
    if attention_mask.dim() == 4:
        # Get dimensions
        batch_size = attention_mask.size(0)
        tgt_len = attention_mask.size(2)
        src_len = attention_mask.size(3)
        
        # Split sequence dimension for both target and source
        seq_per_partition = src_len // sp_size
        
        # Create masks for each worker
        masks = []
        for i in range(sp_size):
            # For local attention on each partition
            start_idx = i * seq_per_partition
            end_idx = (i + 1) * seq_per_partition
            
            # Create local mask for this partition
            if tgt_len == src_len:
                # Self-attention mask: partition both dimensions the same way
                local_mask = attention_mask[:, :, start_idx:end_idx, start_idx:end_idx]
            else:
                # Cross-attention mask: only partition the source dimension
                local_mask = attention_mask[:, :, :, start_idx:end_idx]
            
            masks.append(local_mask)
        
        return masks
    
    # Default handling for other mask types
    parts = partition_sequence(attention_mask, sp_size)
    return parts