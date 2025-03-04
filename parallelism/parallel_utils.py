import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, List, Tuple, Dict, Any
import os
import gc
import math
from torch.cuda import memory_stats, memory_allocated, memory_reserved


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """
    Ensure that numerator is divisible by denominator.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
    
    Raises:
        ValueError: If numerator is not divisible by denominator
    """
    if numerator % denominator != 0:
        raise ValueError(
            f"{numerator} is not divisible by {denominator}. "
            f"Please adjust the model configuration to ensure divisibility."
        )

def divide(numerator: int, denominator: int) -> int:
    """
    Ensure that numerator is divisible by denominator and return the division result.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        
    Returns:
        Division result
    """
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def initialize_tensor_parallel(world_size: int, backend: str = "nccl") -> bool:
    """
    Initialize tensor parallel environment.
    
    Args:
        world_size: Number of processes
        backend: PyTorch distributed backend
        
    Returns:
        Whether initialization was successful
    """
    # Check if tensor parallelism is already initialized
    if hasattr(initialize_tensor_parallel, "_initialized") and initialize_tensor_parallel._initialized:
        return True
    
    # Check if torch.distributed is already initialized
    if not torch.distributed.is_initialized():
        # Get local rank from environment variable
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        
        try:
            # Set device based on local rank
            torch.cuda.set_device(local_rank)
            
            # Initialize the process group
            torch.distributed.init_process_group(backend=backend, world_size=world_size)
            
            # Set tensor parallel process group with a single global group
            # This would be enhanced in a real implementation to support
            # more complex process group hierarchies
            _set_tensor_model_parallel_group(torch.distributed.group.WORLD)
            
            # Mark as initialized
            initialize_tensor_parallel._initialized = True
            
            return True
        except Exception as e:
            print(f"Tensor parallel initialization failed: {e}")
            return False
    else:
        # Set tensor parallel process group with the existing world group
        _set_tensor_model_parallel_group(torch.distributed.group.WORLD)
        
        # Mark as initialized
        initialize_tensor_parallel._initialized = True
        
        return True

def _set_tensor_model_parallel_group(group):
    """
    Set the tensor model parallel group.
    
    Args:
        group: Process group for tensor model parallelism
    """
    # Store the group as a module-level attribute
    _set_tensor_model_parallel_group.group = group

def get_tensor_model_parallel_group():
    """
    Get the tensor model parallel process group.
    
    Returns:
        Process group for tensor model parallelism
    """
    if hasattr(_set_tensor_model_parallel_group, "group"):
        return _set_tensor_model_parallel_group.group
    return None

def get_tensor_model_parallel_rank() -> int:
    """
    Get the tensor model parallel rank.
    
    Returns:
        Tensor model parallel rank
    """
    group = get_tensor_model_parallel_group()
    if group is None:
        return 0
    
    return torch.distributed.get_rank(group) if torch.distributed.is_initialized() else 0

def get_tensor_model_parallel_world_size() -> int:
    """
    Get the tensor model parallel world size.
    
    Returns:
        Tensor model parallel world size
    """
    group = get_tensor_model_parallel_group()
    if group is None:
        return 1
    
    return torch.distributed.get_world_size(group) if torch.distributed.is_initialized() else 1

def split_tensor_along_dim(
    tensor: torch.Tensor,
    dim: int,
    world_size: Optional[int] = None,
    contiguous: bool = True
) -> List[torch.Tensor]:
    """
    Split a tensor along the specified dimension.
    
    Args:
        tensor: Input tensor to split
        dim: Dimension along which to split
        world_size: Number of partitions (defaults to tensor parallel world size)
        contiguous: Whether to make each split contiguous in memory
        
    Returns:
        List of split tensors
    """
    if world_size is None:
        world_size = get_tensor_model_parallel_world_size()
    
    if tensor.dim() == 0 or world_size == 1:
        return [tensor]
    
    # Get the size along the dimension to split
    dim_size = tensor.size(dim)
    
    # Check divisibility
    ensure_divisibility(dim_size, world_size)
    
    # Calculate split size
    split_size = dim_size // world_size
    splits = torch.split(tensor, split_size, dim=dim)
    
    if contiguous:
        splits = [split.contiguous() for split in splits]
    
    return splits

def gather_tensor_along_dim(
    tensor: torch.Tensor,
    dim: int,
    world_size: Optional[int] = None,
    dest_rank: Optional[int] = None
) -> torch.Tensor:
    """
    Gather split tensors from all processes along a dimension.
    
    Args:
        tensor: Local tensor to gather
        dim: Dimension along which to gather
        world_size: Total number of processes (defaults to tensor parallel world size)
        dest_rank: Destination rank for gathering (defaults to all ranks)
        
    Returns:
        Gathered tensor
    """
    if world_size is None:
        world_size = get_tensor_model_parallel_world_size()
    
    if world_size == 1:
        return tensor
    
    # Get current rank
    current_rank = get_tensor_model_parallel_rank()
    
    # If dest_rank is specified, only that rank needs to gather
    if dest_rank is not None and current_rank != dest_rank:
        return tensor
    
    # Get tensor parallel group
    group = get_tensor_model_parallel_group()
    
    # All gather
    gathered_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_tensors, tensor, group=group)
    
    # Concatenate along specified dimension
    return torch.cat(gathered_tensors, dim=dim)

def create_attention_mask_for_tp(
    attention_mask: torch.Tensor,
    tp_size: int
) -> torch.Tensor:
    """
    Create an attention mask suitable for tensor parallelism.
    
    Args:
        attention_mask: Original attention mask [batch_size, 1, seq_len, seq_len]
        tp_size: Tensor parallel size
        
    Returns:
        Modified attention mask for tensor parallelism
    """
    # Get current rank
    rank = get_tensor_model_parallel_rank()
    
    # If no parallelism or mask is None, return the original mask
    if tp_size == 1 or attention_mask is None:
        return attention_mask
    
    # For causal attention masks in decoder-only models
    if attention_mask.dim() == 4 and attention_mask.size(2) == attention_mask.size(3):
        seq_len = attention_mask.size(3)
        
        # Split sequence dimension for attention
        seq_per_partition = divide(seq_len, tp_size)
        
        # Compute partition start and end
        start = rank * seq_per_partition
        end = (rank + 1) * seq_per_partition
        
        # Slice the attention mask for this partition
        sliced_mask = attention_mask[:, :, :, start:end]
        
        return sliced_mask
    
    # For encoder-decoder attention masks
    elif attention_mask.dim() == 4:
        # In encoder-decoder attention, typically only the key dimension is split
        # The mask shape is usually [batch_size, 1, tgt_len, src_len]
        src_len = attention_mask.size(3)
        
        # Split source sequence dimension for key
        src_per_partition = divide(src_len, tp_size)
        
        # Compute partition start and end
        start = rank * src_per_partition
        end = (rank + 1) * src_per_partition
        
        # Slice the attention mask for this partition
        sliced_mask = attention_mask[:, :, :, start:end]
        
        return sliced_mask
    
    # For 2D or 3D attention masks
    else:
        # Default handling - try to split the last dimension
        last_dim = attention_mask.size(-1)
        
        # Split last dimension
        per_partition = divide(last_dim, tp_size)
        
        # Compute partition start and end
        start = rank * per_partition
        end = (rank + 1) * per_partition
        
        # Slice the attention mask for this partition
        sliced_mask = attention_mask[..., start:end]
        
        return sliced_mask

def shard_model_parameters(
    model: nn.Module,
    parallel_dim: int,
    tp_size: int
) -> nn.Module:
    """
    Shard model parameters according to tensor parallel strategy.
    
    Args:
        model: Input model
        parallel_dim: Dimension to shard parameters
        tp_size: Tensor parallel size
        
    Returns:
        Model with sharded parameters
    """
    # Get current rank
    rank = get_tensor_model_parallel_rank()
    
    # Skip if no tensor parallelism is used
    if tp_size == 1:
        return model
    
    # Handle shard differently by parameter type
    for name, param in model.named_parameters():
        # Skip parameters that don't need sharding (e.g., embeddings)
        if "embedding" in name:
            continue
        
        # Linear layer weights
        if "weight" in name and param.dim() == 2:
            if "query" in name or "key" in name or "value" in name or "out_proj" in name:
                # Attention layers:
                # For Q, K, V - shard output dimension (out_features)
                # For output projection - shard input dimension (in_features)
                
                if "out_proj" in name:
                    # Shard input dimension for output projection (row-parallel)
                    if parallel_dim == 0:  # Shard along row dimension
                        orig_size = param.size(0)
                        shard_size = divide(orig_size, tp_size)
                        # Create a view starting at this rank's partition
                        param.data = param.data[rank * shard_size:(rank + 1) * shard_size, :]
                    else:  # Shard along column dimension
                        orig_size = param.size(1)
                        shard_size = divide(orig_size, tp_size)
                        param.data = param.data[:, rank * shard_size:(rank + 1) * shard_size]
                else:
                    # Shard output dimension for Q, K, V (column-parallel)
                    if parallel_dim == 0:  # Shard along row dimension
                        orig_size = param.size(0)
                        shard_size = divide(orig_size, tp_size)
                        param.data = param.data[rank * shard_size:(rank + 1) * shard_size, :]
                    else:  # Shard along column dimension
                        orig_size = param.size(1)
                        shard_size = divide(orig_size, tp_size)
                        param.data = param.data[:, rank * shard_size:(rank + 1) * shard_size]
            
            # MLP layers
            elif "fc" in name or "dense" in name:
                if "up" in name or "intermediate" in name:
                    # Intermediate expansion layer (column-parallel)
                    if parallel_dim == 0:  # Shard along row dimension
                        orig_size = param.size(0)
                        shard_size = divide(orig_size, tp_size)
                        param.data = param.data[rank * shard_size:(rank + 1) * shard_size, :]
                    else:  # Shard along column dimension
                        orig_size = param.size(1)
                        shard_size = divide(orig_size, tp_size)
                        param.data = param.data[:, rank * shard_size:(rank + 1) * shard_size]
                else:
                    # Output projection from MLP (row-parallel)
                    if parallel_dim == 0:  # Shard along row dimension
                        orig_size = param.size(0)
                        shard_size = divide(orig_size, tp_size)
                        param.data = param.data[rank * shard_size:(rank + 1) * shard_size, :]
                    else:  # Shard along column dimension
                        orig_size = param.size(1)
                        shard_size = divide(orig_size, tp_size)
                        param.data = param.data[:, rank * shard_size:(rank + 1) * shard_size]
        
        # Bias vectors for tensor-parallel layers
        elif "bias" in name and param.dim() == 1:
            if "query" in name or "key" in name or "value" in name:
                # Shard bias for attention heads
                orig_size = param.size(0)
                shard_size = divide(orig_size, tp_size)
                param.data = param.data[rank * shard_size:(rank + 1) * shard_size]
            elif "fc" in name or "dense" in name:
                if "up" in name or "intermediate" in name:
                    # Shard bias for intermediate expansion
                    orig_size = param.size(0)
                    shard_size = divide(orig_size, tp_size)
                    param.data = param.data[rank * shard_size:(rank + 1) * shard_size]
    
    return model

def get_partition_start_end(
    total_size: int,
    partition_idx: int,
    world_size: int
) -> Tuple[int, int]:
    """
    Get start and end indices for a partition.
    
    Args:
        total_size: Total size of the dimension
        partition_idx: Partition index
        world_size: Total number of partitions
        
    Returns:
        Tuple of (start_idx, end_idx)
    """
    # Ensure total_size is divisible by world_size
    ensure_divisibility(total_size, world_size)
    
    # Calculate partition size
    partition_size = total_size // world_size
    
    # Calculate start and end indices
    start_idx = partition_idx * partition_size
    end_idx = (partition_idx + 1) * partition_size
    
    return start_idx, end_idx

def is_power_of_two(n: int) -> bool:
    """
    Check if a number is a power of two.
    
    Args:
        n: Number to check
        
    Returns:
        True if n is a power of two, False otherwise
    """
    return (n > 0) and (n & (n - 1) == 0)

def split_tensor_into_1d_equal_chunks(
    tensor: torch.Tensor, 
    group_size: Optional[int] = None
) -> torch.Tensor:
    """
    Split tensor into equal 1D chunks across processes.
    
    Args:
        tensor: Input tensor
        group_size: Size of the process group
        
    Returns:
        Chunk of the tensor for the current process
    """
    if group_size is None:
        group_size = get_tensor_model_parallel_world_size()
    
    # Total number of elements
    numel = tensor.numel()
    
    # Number of elements per chunk
    numel_per_chunk = divide(numel, group_size)
    
    # Reshape and get the chunk for this rank
    tensor_1d = tensor.view(-1)
    rank = get_tensor_model_parallel_rank()
    
    return tensor_1d[rank * numel_per_chunk: (rank + 1) * numel_per_chunk].clone()

def gather_1d_tensor_chunks(
    tensor: torch.Tensor, 
    tensor_shape: torch.Size, 
    group_size: Optional[int] = None
) -> torch.Tensor:
    """
    Gather 1D tensor chunks from all processes.
    
    Args:
        tensor: Local tensor chunk
        tensor_shape: Shape of the original tensor
        group_size: Size of the process group
        
    Returns:
        Gathered tensor with the original shape
    """
    if group_size is None:
        group_size = get_tensor_model_parallel_world_size()
    
    # If world size is 1, return the tensor as is
    if group_size == 1:
        return tensor.reshape(tensor_shape)
    
    # Create list of tensors to gather
    tensors = [torch.empty_like(tensor) for _ in range(group_size)]
    
    # Get the group to use for all_gather
    group = get_tensor_model_parallel_group()
    
    # Gather chunks from all processes
    torch.distributed.all_gather(tensors, tensor, group=group)
    
    # Concatenate and reshape
    gathered = torch.cat(tensors, dim=0)
    return gathered.reshape(tensor_shape)

def set_tensor_model_parallel_attributes(
    tensor: torch.Tensor,
    is_parallel: bool,
    dim: int,
    stride: int
) -> torch.Tensor:
    """
    Set tensor model parallel attributes.
    
    Args:
        tensor: Input tensor
        is_parallel: Whether tensor is parallelized
        dim: Dimension along which tensor is parallelized
        stride: Stride for parallelization
        
    Returns:
        Tensor with parallel attributes set
    """
    # Add attributes to the tensor
    tensor.is_tensor_parallel = is_parallel
    tensor.tensor_parallel_dim = dim
    tensor.tensor_parallel_stride = stride
    
    return tensor

def copy_tensor_model_parallel_attributes(
    destination_tensor: torch.Tensor,
    source_tensor: torch.Tensor
) -> None:
    """
    Copy tensor model parallel attributes from source to destination.
    
    Args:
        destination_tensor: Destination tensor
        source_tensor: Source tensor with attributes
    """
    if hasattr(source_tensor, 'is_tensor_parallel'):
        destination_tensor.is_tensor_parallel = source_tensor.is_tensor_parallel
    
    if hasattr(source_tensor, 'tensor_parallel_dim'):
        destination_tensor.tensor_parallel_dim = source_tensor.tensor_parallel_dim
    
    if hasattr(source_tensor, 'tensor_parallel_stride'):
        destination_tensor.tensor_parallel_stride = source_tensor.tensor_parallel_stride

def get_parallel_tensor_info(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Get information about a tensor's parallel attributes.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Dictionary with tensor parallel information
    """
    info = {
        'is_parallel': getattr(tensor, 'is_tensor_parallel', False),
        'parallel_dim': getattr(tensor, 'tensor_parallel_dim', None),
        'parallel_stride': getattr(tensor, 'tensor_parallel_stride', None),
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'device': tensor.device
    }
    
    return info

# Additional functions for configuration and analysis

def analyze_model_for_parallelism(model: nn.Module) -> Dict[str, Any]:
    """
    Analyze a PyTorch model to determine optimal parallelism strategies.
    
    This function examines the model architecture to identify which layers
    can benefit from different parallelism techniques (tensor, sequence, pipeline)
    and provides recommendations for partitioning the model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dict containing analysis results and parallelism recommendations
    """
    # Initialize results
    result = {
        "model_size": 0,
        "num_parameters": 0,
        "parameter_distribution": {},
        "layer_counts": {},
        "tensor_parallel_modules": [],
        "sequence_parallel_modules": [],
        "pipeline_stages": [],
        "attention_blocks": [],
        "mlp_blocks": [],
        "embedding_blocks": [],
        "checkpointed_modules": [],
        "recommendations": {}
    }
    
    # Count parameters and identify layer types
    total_params = 0
    param_distribution = {}
    layer_counts = {}
    
    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        
        # Count layer types
        if module_type not in layer_counts:
            layer_counts[module_type] = 0
        layer_counts[module_type] += 1
        
        # Skip container modules for parameter counting
        if len(list(module.children())) > 0:
            continue
            
        # Count parameters in this module
        module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += module_params
        
        # Track parameter distribution
        if module_type not in param_distribution:
            param_distribution[module_type] = 0
        param_distribution[module_type] += module_params
        
        # Identify modules for tensor parallelism
        if isinstance(module, (nn.Linear, nn.Conv2d)) and module_params > 1_000_000:
            result["tensor_parallel_modules"].append(module_type)
            
        # Identify modules for sequence parallelism
        if "attention" in name.lower() or "self_attn" in name.lower():
            result["sequence_parallel_modules"].append(module_type)
            result["attention_blocks"].append(name)
            
        # Identify modules for activation checkpointing
        if ("block" in name.lower() or "layer" in name.lower()) and module_params > 5_000_000:
            result["checkpointed_modules"].append(module_type)
            
        # Identify MLP blocks
        if "mlp" in name.lower() or "ffn" in name.lower():
            result["mlp_blocks"].append(name)
            
        # Identify embedding blocks
        if "embed" in name.lower():
            result["embedding_blocks"].append(name)
    
    # Calculate model size in bytes (assuming float32 by default)
    model_size_bytes = total_params * 4
    
    # Update results
    result["model_size"] = model_size_bytes
    result["num_parameters"] = total_params
    result["parameter_distribution"] = param_distribution
    result["layer_counts"] = layer_counts
    
    # Make recommendations based on analysis
    if total_params > 1_000_000_000:  # > 1B params
        result["recommendations"]["tensor_parallel_size"] = 4
        result["recommendations"]["sequence_parallel_size"] = 2
        result["recommendations"]["pipeline_parallel_size"] = 2
        result["recommendations"]["activation_checkpointing"] = True
    elif total_params > 100_000_000:  # > 100M params
        result["recommendations"]["tensor_parallel_size"] = 2
        result["recommendations"]["sequence_parallel_size"] = 1
        result["recommendations"]["pipeline_parallel_size"] = 1
        result["recommendations"]["activation_checkpointing"] = True
    else:
        result["recommendations"]["tensor_parallel_size"] = 1
        result["recommendations"]["sequence_parallel_size"] = 1
        result["recommendations"]["pipeline_parallel_size"] = 1
        result["recommendations"]["activation_checkpointing"] = False
        
    # Create pipeline stages based on model structure
    if hasattr(model, "layers") or hasattr(model, "blocks"):
        # For models with explicit layer structure
        layers = getattr(model, "layers", None) or getattr(model, "blocks", [])
        num_layers = len(layers)
        
        if num_layers >= 4:
            # Simple splitting strategy for pipeline parallelism
            pipeline_stages = []
            if result["recommendations"]["pipeline_parallel_size"] > 1:
                stage_size = num_layers // result["recommendations"]["pipeline_parallel_size"]
                for i in range(result["recommendations"]["pipeline_parallel_size"]):
                    start = i * stage_size
                    end = (i + 1) * stage_size if i < result["recommendations"]["pipeline_parallel_size"] - 1 else num_layers
                    pipeline_stages.append((start, end))
            
            result["pipeline_stages"] = pipeline_stages
    
    return result

def estimate_memory_requirements(model: nn.Module, batch_size: int, seq_len: int) -> Dict[str, int]:
    """
    Estimate memory requirements for model inference with given batch and sequence length.
    
    Args:
        model: PyTorch model
        batch_size: Batch size for inference
        seq_len: Sequence length for inference
        
    Returns:
        Dict containing memory requirement estimates in bytes
    """
    # Count model parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Estimate model size in bytes (assuming float32 parameters)
    model_size = num_params * 4  # float32 = 4 bytes
    
    # Estimate activation memory
    # This is a simplified heuristic based on model size and input dimensions
    # Real activation memory depends on model architecture and implementation details
    hidden_size = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            hidden_size = max(hidden_size, module.out_features)
    
    if hidden_size == 0:
        # Fallback if we couldn't determine hidden size
        hidden_size = 1024  # Common default for medium-sized models
    
    # Activation memory estimation (simplified)
    # Assumes activations for each layer in the sequence
    activation_size_per_token = hidden_size * 4  # float32 = 4 bytes
    num_layers = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
    activation_memory = batch_size * seq_len * activation_size_per_token * num_layers
    
    # KV cache memory for transformer models (if applicable)
    kv_cache_size = 0
    has_attention = any("attention" in name.lower() for name, _ in model.named_modules())
    if has_attention:
        # Estimate KV cache memory
        # Each token stores key and value vectors for each layer and attention head
        num_layers = sum(1 for _ in model.modules() if "attention" in _.__class__.__name__.lower())
        num_heads = 1  # Default
        for module in model.modules():
            if hasattr(module, "num_heads"):
                num_heads = module.num_heads
                break
        
        head_dim = hidden_size // num_heads
        kv_cache_size = batch_size * seq_len * num_layers * num_heads * head_dim * 2 * 4  # float32 = 4 bytes
    
    # Calculate overhead (workspace, temporary buffers, etc.)
    # This is a rough estimate based on empirical observations
    overhead = int(0.1 * (model_size + activation_memory + kv_cache_size))
    
    # Calculate total memory requirement
    total_memory = model_size + activation_memory + kv_cache_size + overhead
    
    return {
        "model_params": model_size,
        "activations": activation_memory,
        "kv_cache": kv_cache_size,
        "overhead": overhead,
        "total": total_memory
    }

def calculate_communication_overhead(config: "ParallelConfig", model_info: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate communication overhead for different parallelism strategies.
    
    Args:
        config: Parallel configuration
        model_info: Model information from analyze_model_for_parallelism
        
    Returns:
        Dict containing communication overhead estimates
    """
    # Extract relevant information
    model_size = model_info.get("model_size", 0)
    num_params = model_info.get("num_parameters", 0)
    
    # Default parameter sizes
    hidden_size = 1024  # Default hidden size
    
    # Try to determine hidden size from layer counts
    layer_counts = model_info.get("layer_counts", {})
    if "Linear" in layer_counts and layer_counts["Linear"] > 0:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hidden_size = max(hidden_size, module.out_features)
                break
    
    # Tensor parallel communication
    tp_size = config.tensor_parallel_size
    tp_overhead = 0
    if tp_size > 1:
        # Estimate all-reduce communication for tensor parallelism
        # For each forward pass, we need to all-reduce outputs of parallel linear layers
        num_tp_modules = len(model_info.get("tensor_parallel_modules", []))
        avg_tensor_size = hidden_size * hidden_size / tp_size  # Average size of tensor to all-reduce
        
        # Communication volume = number of tp operations * size of tensors * number of processes
        tp_overhead = num_tp_modules * avg_tensor_size * 4 * tp_size  # float32 = 4 bytes
    
    # Sequence parallel communication
    sp_size = config.sequence_parallel_size
    sp_overhead = 0
    if sp_size > 1:
        # Estimate all-gather communication for sequence parallelism
        # For each layer with sequence parallelism, we need all-gather operations
        num_sp_modules = len(model_info.get("sequence_parallel_modules", []))
        avg_sequence_size = hidden_size * (1024 / sp_size)  # Average size of sequence tensor to all-gather
        
        # Communication volume = number of sp operations * size of tensors * number of processes
        sp_overhead = num_sp_modules * avg_sequence_size * 4 * sp_size  # float32 = 4 bytes
    
    # Pipeline parallel communication
    pp_size = config.pipeline_parallel_size
    pp_overhead = 0
    if pp_size > 1:
        # Estimate point-to-point communication for pipeline parallelism
        # For each micro-batch at pipeline boundaries, we need to send activations
        num_pipeline_stages = len(model_info.get("pipeline_stages", []))
        if num_pipeline_stages == 0:
            num_pipeline_stages = pp_size
        
        # Communication volume = number of pipeline boundaries * activation size
        avg_activation_size = hidden_size * 512  # Average activation size
        pp_overhead = (num_pipeline_stages - 1) * avg_activation_size * 4  # float32 = 4 bytes
    
    # Data parallel communication (for completeness, though less relevant for inference)
    dp_size = config.data_parallel_size
    dp_overhead = 0
    if dp_size > 1:
        # For inference, data parallel overhead is minimal
        # Only need to broadcast inputs and gather outputs
        dp_overhead = hidden_size * 4 * dp_size  # float32 = 4 bytes
    
    # Total communication overhead
    total_overhead = tp_overhead + sp_overhead + pp_overhead + dp_overhead
    
    # Calculate overhead percentage relative to model size
    overhead_percentage = (total_overhead / model_size) * 100 if model_size > 0 else 0
    
    return {
        "tensor_parallel_overhead": tp_overhead,
        "sequence_parallel_overhead": sp_overhead,
        "pipeline_parallel_overhead": pp_overhead,
        "data_parallel_overhead": dp_overhead,
        "total_overhead": total_overhead,
        "overhead_percentage": overhead_percentage
    }

def validate_parallel_config(config: "ParallelConfig", world_size: int) -> Tuple[bool, str]:
    """
    Validate that a parallel configuration is valid.
    
    Args:
        config: Parallel configuration to validate
        world_size: Total number of devices
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if product of parallel sizes equals world size
    product = config.tensor_parallel_size * config.pipeline_parallel_size * config.data_parallel_size * config.sequence_parallel_size
    
    if product != world_size:
        return False, f"Product of parallel sizes ({product}) must equal world size ({world_size})"
    
    # Check compatibility between tensor and sequence parallelism
    if config.tensor_parallel_size > 1 and config.sequence_parallel_size > 1:
        # While this is allowed, there are constraints
        if not is_power_of_two(config.tensor_parallel_size) or not is_power_of_two(config.sequence_parallel_size):
            return False, "When using both tensor and sequence parallelism, each size should be a power of 2"
    
    # Check that tensor parallel size is valid
    if config.tensor_parallel_size > 1:
        if not is_power_of_two(config.tensor_parallel_size):
            return False, "Tensor parallel size should be a power of 2"
    
    # Pipeline parallel special checks
    if config.pipeline_parallel_size > 1:
        # Pipeline parallelism typically works best with limited number of stages
        if config.pipeline_parallel_size > 8:
            return False, "Pipeline parallel size > 8 is not recommended"
            
        # Pipeline and sequence parallelism can have complex interactions
        if config.sequence_parallel_size > 1 and config.pipeline_parallel_size > 4:
            return False, "Using both sequence and pipeline parallelism with large sizes can lead to poor performance"
    
    # Check hardware compatibility
    if config.tensor_parallel_size > 1 or config.sequence_parallel_size > 1:
        # These strategies benefit from fast GPU interconnect
        # In a proper implementation, we would check for NVLink or similar
        pass
    
    return True, ""

def initialize_parallel_groups(config: "ParallelConfig") -> Dict[str, dist.ProcessGroup]:
    """
    Initialize process groups for different parallel strategies.
    
    Args:
        config: Parallel configuration
        
    Returns:
        Dict of process groups for each parallelism dimension
    """
    # Check if distributed is initialized
    if not dist.is_initialized():
        raise RuntimeError("Distributed backend must be initialized before creating process groups")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    process_groups = {}
    
    # Create tensor model parallel group
    if config.tensor_parallel_size > 1:
        # Determine the ranks in each tensor parallel group
        tp_size = config.tensor_parallel_size
        tp_groups = []
        
        for i in range(world_size // tp_size):
            start_rank = i * tp_size
            end_rank = (i + 1) * tp_size
            group_ranks = list(range(start_rank, end_rank))
            tp_groups.append(group_ranks)
            
            # Create the tensor parallel group
            group = dist.new_group(ranks=group_ranks)
            
            # Store the group if rank is part of it
            if rank in group_ranks:
                process_groups["tensor"] = group
    
    # Create sequence parallel group
    if config.sequence_parallel_size > 1:
        # Determine the ranks in each sequence parallel group
        sp_size = config.sequence_parallel_size
        sp_groups = []
        
        # Sequence parallel groups are often identical to tensor parallel groups
        # But they could be organized differently if needed
        if config.tensor_parallel_size == config.sequence_parallel_size:
            if "tensor" in process_groups:
                process_groups["sequence"] = process_groups["tensor"]
        else:
            for i in range(world_size // sp_size):
                start_rank = i * sp_size
                end_rank = (i + 1) * sp_size
                group_ranks = list(range(start_rank, end_rank))
                sp_groups.append(group_ranks)
                
                # Create the sequence parallel group
                group = dist.new_group(ranks=group_ranks)
                
                # Store the group if rank is part of it
                if rank in group_ranks:
                    process_groups["sequence"] = group
    
    # Create pipeline parallel group
    if config.pipeline_parallel_size > 1:
        # Determine the ranks in each pipeline parallel group
        pp_size = config.pipeline_parallel_size
        pp_groups = []
        
        # Pipeline parallel groups span across tensor/sequence parallel groups
        tp_size = config.tensor_parallel_size
        sp_size = config.sequence_parallel_size
        dp_size = config.data_parallel_size
        
        # Calculate process grid dimensions
        grid_dim_pp = pp_size
        grid_dim_tp_sp = tp_size * sp_size
        
        for i in range(world_size // pp_size):
            # Pipeline ranks are strided based on tensor/sequence parallelism
            group_ranks = []
            for pp_rank in range(pp_size):
                # Calculate the actual rank in the world
                rank_in_group = i + pp_rank * (world_size // pp_size)
                group_ranks.append(rank_in_group)
            
            # Create the pipeline parallel group
            group = dist.new_group(ranks=group_ranks)
            
            # Store the group if rank is part of it
            if rank in group_ranks:
                process_groups["pipeline"] = group
    
    # Create data parallel group
    if config.data_parallel_size > 1:
        # Determine the ranks in each data parallel group
        dp_size = config.data_parallel_size
        dp_groups = []
        
        pp_size = config.pipeline_parallel_size
        tp_size = config.tensor_parallel_size
        sp_size = config.sequence_parallel_size
        
        # Data parallel groups contain the same pipeline stage across all replicas
        for i in range(world_size // dp_size):
            # Data parallel ranks are determined by the replica position
            group_ranks = []
            for dp_rank in range(dp_size):
                # Calculate the actual rank in the world
                base_offset = i % (pp_size * tp_size * sp_size)
                rank_in_group = base_offset + dp_rank * (pp_size * tp_size * sp_size)
                group_ranks.append(rank_in_group)
            
            # Create the data parallel group
            group = dist.new_group(ranks=group_ranks)
            
            # Store the group if rank is part of it
            if rank in group_ranks:
                process_groups["data"] = group
    
    return process_groups

def get_process_group_for_operation(op_type: str) -> dist.ProcessGroup:
    """
    Get the appropriate process group for a given operation type.
    
    Args:
        op_type: Type of operation ("tensor", "sequence", "pipeline", "data")
        
    Returns:
        Process group for the operation
    """
    # In a real implementation, this would reference the stored process groups
    # Here we use a placeholder implementation
    if not dist.is_initialized():
        return None
    
    # Default to WORLD group if specific group is not defined
    return dist.group.WORLD

def create_communication_streams() -> Dict[str, torch.cuda.Stream]:
    """
    Create CUDA streams for overlapping communication and computation.
    
    Returns:
        Dict of CUDA streams for different communication types
    """
    streams = {}
    
    if torch.cuda.is_available():
        streams["tensor"] = torch.cuda.Stream()
        streams["sequence"] = torch.cuda.Stream()
        streams["pipeline"] = torch.cuda.Stream()
        streams["data"] = torch.cuda.Stream()
    
    return streams

def synchronize_all_streams() -> None:
    """
    Synchronize all CUDA streams.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()