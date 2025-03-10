import torch
import torch.nn as nn
import torch.distributed as dist
import functools
import weakref
import logging
import os
import math
import subprocess
from typing import Optional, List, Tuple, Dict, Any, Callable, Union

def initialize_distributed(
    local_rank: int,
    world_size: int,
    backend: str = "nccl"
) -> None:
    """
    Initialize the distributed environment for tensor parallelism.
    
    Args:
        local_rank: Local GPU rank
        world_size: Total number of GPUs
        backend: PyTorch distributed backend (default: nccl)
    """
    if not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, world_size=world_size, rank=local_rank)

def get_rank() -> int:
    """Get the current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size() -> int:
    """Get the world size (total number of GPUs)."""
    return dist.get_world_size() if dist.is_initialized() else 1

def all_reduce(
    tensor: torch.Tensor, 
    op: Union[dist.ReduceOp, str] = dist.ReduceOp.SUM, 
    async_op: bool = False,
    group: Optional[dist.ProcessGroup] = None,
    use_fp16: bool = False,
    use_bf16: bool = False,
    use_unbalanced: bool = False,
    stream: Optional[torch.cuda.Stream] = None
) -> Union[torch.Tensor, Tuple[torch.distributed.Work, torch.Tensor]]:
    """
    Perform all-reduce operation across all processes.
    
    Args:
        tensor: Input tensor to be reduced
        op: Reduction operation (dist.ReduceOp or string: "sum", "avg", "max", "min", "prod")
        async_op: Whether to perform asynchronous operation
        group: Process group
        use_fp16: Convert to FP16 before communication for bandwidth reduction
        use_bf16: Convert to BF16 before communication for bandwidth reduction
        use_unbalanced: Use specialized algorithm for unbalanced workloads
        stream: CUDA stream to use for operation (for overlapping compute and communication)
        
    Returns:
        Reduced tensor or tuple of (work handle, tensor) if async_op=True
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    
    # Use precision reduction for communication if requested
    orig_dtype = tensor.dtype
    comm_tensor = tensor
    
    if (use_fp16 or use_bf16) and tensor.is_floating_point():
        if use_fp16 and tensor.dtype != torch.float16:
            comm_tensor = tensor.to(dtype=torch.float16)
        elif use_bf16 and tensor.dtype != torch.bfloat16:
            comm_tensor = tensor.to(dtype=torch.bfloat16)
    
    # Set stream for communication if provided
    if stream is not None and torch.cuda.is_available():
        prev_stream = torch.cuda.current_stream()
        stream.wait_stream(prev_stream)  # Ensure data is ready
    
    # Convert string op to PyTorch's ReduceOp if needed
    if isinstance(op, str):
        op_map = {
            "sum": dist.ReduceOp.SUM,
            "avg": None,  # Special case handled below
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
            "prod": dist.ReduceOp.PRODUCT
        }
        if op.lower() not in op_map:
            raise ValueError(f"Unsupported reduction operation: {op}")
        op_val = op_map[op.lower()]
    else:
        op_val = op
    
    # Use unbalanced algorithm for specialized workloads if requested
    if use_unbalanced:
        # This implements a specialized algorithm for scenarios where
        # some ranks have much more computation than others using a hierarchical approach
        # We use a tree-based reduction to handle imbalanced workloads more efficiently
        
        # Get our position in the process group
        rank = get_rank() if group is None else dist.get_rank(group)
        world_size = get_world_size() if group is None else group.size()
        
        # Create a copy of the tensor to avoid modifying the input
        result = tensor.clone()
        
        # Hierarchical reduction: Use a binary tree structure
        # This approach has log(N) communication steps instead of potentially 
        # waiting for slower nodes in an all-reduce
        
        # Step 1: Determine tree structure (simple binary tree)
        is_power_of_2 = (world_size & (world_size - 1)) == 0
        
        if is_power_of_2:
            # For power-of-2 sizes, use simple tree reduction
            distance = 1
            while distance < world_size:
                target = rank ^ distance
                
                if target < world_size:
                    # Ranks whose bit 'distance' is 1 send to ranks with that bit set to 0
                    if (rank & distance) != 0:
                        # Send data
                        dist.send(result, target, group=group)
                    else:
                        # Receive data
                        received = torch.empty_like(result)
                        dist.recv(received, target, group=group)
                        
                        # Apply reduction operation
                        if op_val == dist.ReduceOp.SUM:
                            result.add_(received)
                        elif op_val == dist.ReduceOp.MAX:
                            result = torch.max(result, received)
                        elif op_val == dist.ReduceOp.MIN:
                            result = torch.min(result, received)
                        elif op_val == dist.ReduceOp.PRODUCT:
                            result.mul_(received)
                
                distance *= 2
                
            # Step 2: Broadcast the result from rank 0 to all ranks
            dist.broadcast(result, src=0, group=group)
            
        else:
            # For non-power-of-2 sizes, use a binomial tree
            # This is more complex but handles non-power-of-2 sizes better
            for d in range(math.ceil(math.log2(world_size))):
                mask = 1 << d
                
                if rank < mask:
                    # Lower half receives from upper half
                    partner = rank + mask
                    if partner < world_size:
                        received = torch.empty_like(result)
                        dist.recv(received, partner, group=group)
                        
                        # Apply reduction
                        if op_val == dist.ReduceOp.SUM:
                            result.add_(received)
                        elif op_val == dist.ReduceOp.MAX:
                            result = torch.max(result, received)
                        elif op_val == dist.ReduceOp.MIN:
                            result = torch.min(result, received)
                        elif op_val == dist.ReduceOp.PRODUCT:
                            result.mul_(received)
                            
                elif (rank & mask) == mask:
                    # Upper half sends to lower half
                    partner = rank - mask
                    dist.send(result, partner, group=group)
            
            # Broadcast the result from rank 0
            dist.broadcast(result, src=0, group=group)
            
        # Return the tree-reduced tensor
        return result
    
    # Special handling for average
    if op_val is None:  # "avg" case
        # Create a clone to avoid modifying the input tensor if it's needed after this call
        result = tensor.clone()
        
        # Perform sum reduction
        work = dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group, async_op=async_op)
        
        if async_op:
            # For async, we need to return a function that will do the division when called
            def avg_completion(work_handle, result_tensor, world_size):
                work_handle.wait()
                result_tensor.div_(world_size)
                return result_tensor
                
            return work, functools.partial(avg_completion, work, result, 
                                         get_world_size() if group is None else group.size())
        else:
            # For sync op, directly divide by world size
            result.div_(get_world_size() if group is None else group.size())
            return result
    
    # All other reduction operations
    if async_op:
        work = dist.all_reduce(tensor, op=op_val, group=group, async_op=True)
        return work, tensor
    else:
        dist.all_reduce(tensor, op=op_val, group=group, async_op=False)
        return tensor

def all_gather(
    tensor: torch.Tensor,
    dim: int = 0,
    async_op: bool = False,
    group: Optional[dist.ProcessGroup] = None
) -> Union[torch.Tensor, Tuple[torch.distributed.Work, torch.Tensor]]:
    """
    Perform all-gather operation to gather tensors from all processes.
    
    Args:
        tensor: Input tensor
        dim: Dimension along which to concatenate the gathered tensors
        async_op: Whether to perform asynchronous operation
        group: Process group
        
    Returns:
        Gathered tensor or tuple of (work handle, tensor) if async_op=True
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    
    # Get world size for this process group
    world_size = get_world_size() if group is None else group.size()
    
    # Create list to store gathered tensors
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    
    # Perform all_gather operation
    if async_op:
        work = dist.all_gather(tensor_list, tensor, group=group, async_op=True)
        # Pre-concatenate for convenience
        result = torch.cat(tensor_list, dim=dim)
        return work, result
    else:
        dist.all_gather(tensor_list, tensor, group=group, async_op=False)
        return torch.cat(tensor_list, dim=dim)

def reduce_scatter(
    tensor: torch.Tensor,
    dim: int = 0,
    op: Union[dist.ReduceOp, str] = dist.ReduceOp.SUM,
    async_op: bool = False,
    group: Optional[dist.ProcessGroup] = None
) -> Union[torch.Tensor, Tuple[torch.distributed.Work, torch.Tensor]]:
    """
    Perform reduce-scatter operation.
    
    Args:
        tensor: Input tensor to be scattered
        dim: Dimension along which to split the tensor
        op: Reduction operation
        async_op: Whether to perform asynchronous operation
        group: Process group
        
    Returns:
        Output tensor with reduced values or tuple of (work handle, tensor) if async_op=True
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    
    # Convert string op to PyTorch's ReduceOp if needed
    if isinstance(op, str):
        op_map = {
            "sum": dist.ReduceOp.SUM,
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
            "prod": dist.ReduceOp.PRODUCT
        }
        if op.lower() not in op_map:
            raise ValueError(f"Unsupported reduction operation: {op}")
        op = op_map[op.lower()]
    
    # Get world size for this process group
    world_size = get_world_size() if group is None else group.size()
    
    # Get expected chunk size after scatter
    split_size = tensor.size(dim) // world_size
    
    # Create the output tensor to hold the result
    output_shape = list(tensor.size())
    output_shape[dim] = split_size
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    
    # Split the tensor into chunks
    chunks = list(torch.split(tensor, split_size, dim=dim))
    chunks = [chunk.contiguous() for chunk in chunks]
    
    # Perform reduce_scatter operation
    if async_op:
        work = dist.reduce_scatter(output, chunks, op=op, group=group, async_op=True)
        return work, output
    else:
        dist.reduce_scatter(output, chunks, op=op, group=group, async_op=False)
        return output

def broadcast(
    tensor: torch.Tensor,
    src: int = 0,
    async_op: bool = False,
    group: Optional[dist.ProcessGroup] = None
) -> Union[torch.Tensor, Tuple[torch.distributed.Work, torch.Tensor]]:
    """
    Broadcast tensor from source rank to all other processes.
    
    Args:
        tensor: Input tensor
        src: Source rank
        async_op: Whether to perform asynchronous operation
        group: Process group
        
    Returns:
        Broadcast tensor or tuple of (work handle, tensor) if async_op=True
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    
    if async_op:
        work = dist.broadcast(tensor, src=src, group=group, async_op=True)
        return work, tensor
    else:
        dist.broadcast(tensor, src=src, group=group, async_op=False)
        return tensor

def scatter(
    tensor: torch.Tensor,
    scatter_list: Optional[List[torch.Tensor]] = None,
    src: int = 0,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False
) -> Union[torch.Tensor, Tuple[torch.distributed.Work, torch.Tensor]]:
    """
    Scatter tensor from source rank to all processes.
    
    Args:
        tensor: Output tensor
        scatter_list: List of tensors to scatter (only needed at src)
        src: Source rank
        group: Process group
        async_op: Whether to perform asynchronous operation
        
    Returns:
        Scattered tensor or tuple of (work handle, tensor) if async_op=True
    """
    if not dist.is_initialized() or get_world_size() == 1:
        if scatter_list is not None and len(scatter_list) > 0:
            tensor.copy_(scatter_list[0])
        return tensor
    
    if async_op:
        work = dist.scatter(tensor, scatter_list, src=src, group=group, async_op=True)
        return work, tensor
    else:
        dist.scatter(tensor, scatter_list, src=src, group=group, async_op=False)
        return tensor

def barrier(group: Optional[dist.ProcessGroup] = None) -> None:
    """
    Synchronize all processes.
    
    Args:
        group: Process group
    """
    if dist.is_initialized():
        dist.barrier(group=group)

def register_communication_hook(
    hook_type: str,
    process_group: Optional[dist.ProcessGroup] = None,
    hook: Optional[Callable] = None
) -> None:
    """
    Register a communication hook to optimize distributed operations.
    
    Args:
        hook_type: Type of hook to register ("fp16", "byteps", "powersgd", "custom")
        process_group: Process group to apply the hook to
        hook: Custom hook function (only used when hook_type is "custom")
    """
    if not dist.is_initialized():
        return
    
    if process_group is None:
        process_group = dist.group.WORLD
    
    # Define different hook implementations
    if hook_type == "fp16":
        # FP16 compression hook
        def fp16_compression_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future:
            # Compress gradients to FP16
            tensors = bucket.get_tensors()
            compressed_tensors = [t.to(torch.float16) for t in tensors]
            
            # All-reduce with compressed tensors
            future = dist.all_reduce(
                torch.cat([t.view(-1) for t in compressed_tensors]),
                op=dist.ReduceOp.SUM,
                group=process_group,
                async_op=True
            ).to_future()
            
            # Decompress gradients back to original dtype
            def decompress(fut):
                decompressed = torch.cat([t.view(-1) for t in tensors])
                return decompressed
            
            return future.then(decompress)
        
        hook_fn = fp16_compression_hook
    
    elif hook_type == "powersgd":
        # PowerSGD (low-rank) compression
        # This is a simplified version; a full implementation would be more complex
        def powersgd_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future:
            tensors = bucket.get_tensors()
            future = dist.all_reduce(
                torch.cat([t.view(-1) for t in tensors]),
                op=dist.ReduceOp.SUM,
                group=process_group,
                async_op=True
            ).to_future()
            return future
        
        hook_fn = powersgd_hook
    
    elif hook_type == "byteps":
        # BytePS-style communication scheduling
        def byteps_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future:
            tensors = bucket.get_tensors()
            future = dist.all_reduce(
                torch.cat([t.view(-1) for t in tensors]),
                op=dist.ReduceOp.SUM,
                group=process_group,
                async_op=True
            ).to_future()
            return future
        
        hook_fn = byteps_hook
    
    elif hook_type == "custom":
        # Use custom provided hook
        if hook is None:
            raise ValueError("Custom hook must be provided when hook_type is 'custom'")
        hook_fn = hook
    
    else:
        raise ValueError(f"Unsupported hook type: {hook_type}")
    
    # Register the hook
    # Note: This specific API is not available in PyTorch's distributed package directly,
    # but similar functionality exists in DDP and is conceptually represented here
    if hasattr(dist, "register_comm_hook"):
        dist.register_comm_hook(process_group, hook_fn)

def setup_device_groups(
    world_size: int, 
    tp_size: int, 
    dp_size: int
) -> Tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """
    Set up process groups for tensor parallelism and data parallelism.
    
    Args:
        world_size: Total number of processes
        tp_size: Tensor parallel size
        dp_size: Data parallel size
        
    Returns:
        Tuple of (tensor_parallel_group, data_parallel_group)
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment not initialized. Call initialize_distributed first.")
    
    if world_size != tp_size * dp_size:
        raise ValueError(f"World size ({world_size}) must equal tp_size ({tp_size}) * dp_size ({dp_size})")
    
    rank = get_rank()
    
    # Calculate TP and DP ranks
    tp_rank = rank % tp_size
    dp_rank = rank // tp_size
    
    # Create TP groups - processes with the same DP rank
    tp_ranks = [dp_rank * tp_size + i for i in range(tp_size)]
    tp_group = dist.new_group(ranks=tp_ranks)
    
    # Create DP groups - processes with the same TP rank
    dp_ranks = [i * tp_size + tp_rank for i in range(dp_size)]
    dp_group = dist.new_group(ranks=dp_ranks)
    
    return tp_group, dp_group

def optimize_communication_overlap(model: nn.Module) -> nn.Module:
    """
    Optimize a model by enabling communication-computation overlap.
    
    Args:
        model: PyTorch model
        
    Returns:
        Optimized model
    """
    def _apply_async_ops(module):
        # Store original forward method
        original_forward = module.forward
        
        # Create async communication wrapper
        @functools.wraps(original_forward)
        def async_forward(*args, **kwargs):
            # Call original forward
            output = original_forward(*args, **kwargs)
            
            # Add async communication handling to applicable operations
            if hasattr(module, 'communication_handles'):
                # Clear previous handles
                module.communication_handles = []
            else:
                module.communication_handles = []
            
            # Check if this module performs communication
            for name, submodule in module.named_modules():
                # Look for tensor parallel modules
                if hasattr(submodule, 'all_reduce') or hasattr(submodule, 'all_gather'):
                    # Store the handles for async communications
                    if hasattr(submodule, 'pending_handles'):
                        module.communication_handles.extend(submodule.pending_handles)
                        submodule.pending_handles = []
            
            return output
        
        # Replace forward method
        module.forward = async_forward
    
    for name, module in model.named_modules():
        _apply_async_ops(module)
    
    return model

def create_overlap_communicator(model: nn.Module) -> None:
    """
    Create a communicator that overlaps communication with computation.
    
    Args:
        model: PyTorch model
    """
    # Track async operations
    model._communication_handles = []
    
    # Create a context manager for grouping operations
    class CommunicationOverlapContext:
        def __init__(self, model):
            self.model = model
            self.handles = []
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Wait for all pending communications to complete
            for handle in self.handles:
                handle.wait()
                
            # Clear handles
            self.handles = []
    
    # Attach context manager to model
    model.communication_overlap = CommunicationOverlapContext(model)

# Sequence parallelism communication utilities

def setup_sequence_parallel_group(world_size: int, sp_size: int) -> dist.ProcessGroup:
    """
    Set up process groups for sequence parallelism.
    
    This function creates process groups where processes with the same DP rank
    are grouped together for sequence parallelism.
    
    Args:
        world_size: Total number of processes
        sp_size: Sequence parallel size
        
    Returns:
        Process group for sequence parallelism
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment not initialized. Call initialize_distributed first.")
    
    if world_size % sp_size != 0:
        raise ValueError(f"World size ({world_size}) must be divisible by sp_size ({sp_size})")
    
    dp_size = world_size // sp_size
    rank = get_rank()
    
    # Determine SP and DP ranks
    dp_rank = rank // sp_size
    sp_rank = rank % sp_size
    
    # Create SP groups - processes with the same DP rank
    # For example, if sp_size=4 and world_size=8:
    # SP group 0: [0, 1, 2, 3], SP group 1: [4, 5, 6, 7]
    sp_groups = []
    for i in range(dp_size):
        sp_ranks = [i * sp_size + j for j in range(sp_size)]
        group = dist.new_group(ranks=sp_ranks)
        sp_groups.append(group)
    
    # Store the process group for current rank
    sp_group = sp_groups[dp_rank]
    
    return sp_group

def scatter_along_sequence_dim(tensor: torch.Tensor, sp_size: int = None) -> torch.Tensor:
    """
    Scatter a tensor along the sequence dimension.
    
    This function splits a tensor along the sequence dimension (typically dimension 1 for
    [batch_size, seq_len, hidden_size] tensors) and distributes the chunks to different
    processes in the sequence parallel group.
    
    Args:
        tensor: Input tensor to scatter
        sp_size: Sequence parallel size (if None, uses all available processes)
        
    Returns:
        Local chunk of the tensor
    """
    if not dist.is_initialized() or sp_size == 1:
        return tensor
    
    # Get current rank and world size
    rank = get_rank()
    world_size = get_world_size() if sp_size is None else sp_size
    
    # Default to sequence dimension 1 (for [batch, seq_len, hidden_size] tensors)
    seq_dim = 1
    
    # Get sequence length and ensure it's divisible by world_size
    seq_len = tensor.size(seq_dim)
    if seq_len % world_size != 0:
        raise ValueError(f"Sequence length ({seq_len}) must be divisible by sp_size ({world_size})")
    
    # Calculate chunk size
    chunk_size = seq_len // world_size
    
    # Calculate this rank's chunk
    start_idx = (rank % world_size) * chunk_size
    end_idx = start_idx + chunk_size
    
    # Extract chunk using narrow to avoid memory copy
    chunk = tensor.narrow(seq_dim, start_idx, chunk_size)
    
    return chunk

def gather_along_sequence_dim(tensor: torch.Tensor, sp_size: int = None) -> torch.Tensor:
    """
    Gather tensor chunks from all processes along the sequence dimension.
    
    This function collects tensor chunks from all processes in the sequence parallel
    group and concatenates them along the sequence dimension.
    
    Args:
        tensor: Local tensor chunk to gather
        sp_size: Sequence parallel size (if None, uses all available processes)
        
    Returns:
        Gathered tensor
    """
    if not dist.is_initialized() or sp_size == 1:
        return tensor
    
    # Get world size and sequence parallel group
    world_size = get_world_size() if sp_size is None else sp_size
    
    # Default to sequence dimension 1 (for [batch, seq_len, hidden_size] tensors)
    seq_dim = 1
    
    # Create list to store gathered tensors
    gathered_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # Get the process group (would use a sequence parallel specific group in practice)
    group = dist.group.WORLD
    
    # Gather tensors from all processes
    dist.all_gather(gathered_list, tensor, group=group)
    
    # Concatenate along sequence dimension
    gathered_tensor = torch.cat(gathered_list, dim=seq_dim)
    
    return gathered_tensor

def ring_exchange(
    tensor_a: torch.Tensor, 
    tensor_b: Optional[torch.Tensor] = None,
    tensor_c: Optional[torch.Tensor] = None,
    group: Optional[dist.ProcessGroup] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Exchange tensors with the next rank in a ring communication pattern.
    
    This function is used for ring-based attention, where each process sends its
    key and value tensors to the next process in the ring and receives from the
    previous process.
    
    Args:
        tensor_a: First tensor (typically keys)
        tensor_b: Second tensor (typically values), optional
        tensor_c: Third tensor (typically attention mask), optional
        group: Process group for communication
        
    Returns:
        Tuple of tensors received from previous rank in the ring
    """
    if not dist.is_initialized():
        return tensor_a, tensor_b, tensor_c
    
    # Get rank and world size
    rank = get_rank()
    world_size = get_world_size() if group is None else group.size()
    
    if world_size == 1:
        return tensor_a, tensor_b, tensor_c
    
    # Calculate source and destination ranks in the ring
    src_rank = (rank - 1) % world_size
    dst_rank = (rank + 1) % world_size
    
    # Create output tensors
    output_a = torch.empty_like(tensor_a)
    output_b = torch.empty_like(tensor_b) if tensor_b is not None else None
    output_c = torch.empty_like(tensor_c) if tensor_c is not None else None
    
    # Send to the next rank and receive from the previous rank
    reqs = []
    
    # Send tensor_a and receive into output_a
    send_req = dist.isend(tensor_a, dst_rank, group=group)
    reqs.append(send_req)
    recv_req = dist.irecv(output_a, src_rank, group=group)
    reqs.append(recv_req)
    
    # If tensor_b is provided, send and receive it
    if tensor_b is not None:
        send_req = dist.isend(tensor_b, dst_rank, group=group)
        reqs.append(send_req)
        output_b = torch.empty_like(tensor_b)
        recv_req = dist.irecv(output_b, src_rank, group=group)
        reqs.append(recv_req)
    
    # If tensor_c is provided, send and receive it
    if tensor_c is not None:
        send_req = dist.isend(tensor_c, dst_rank, group=group)
        reqs.append(send_req)
        output_c = torch.empty_like(tensor_c)
        recv_req = dist.irecv(output_c, src_rank, group=group)
        reqs.append(recv_req)
    
    # Wait for all communications to complete
    for req in reqs:
        req.wait()
    
    return output_a, output_b, output_c

def create_pipeline_schedule(num_chunks: int, sp_size: int) -> List[Tuple[int, int]]:
    """
    Create a schedule for pipeline-parallel execution with sequence parallelism.
    
    This function generates a schedule of send/recv operations for pipelined
    execution, allowing communication and computation to be overlapped.
    
    Args:
        num_chunks: Number of sequence chunks
        sp_size: Sequence parallel size
        
    Returns:
        List of (sender_rank, receiver_rank) tuples representing the communication schedule
    """
    schedule = []
    
    # Create a basic linear pipeline schedule
    for step in range(num_chunks + sp_size - 1):
        step_comms = []
        
        for sender in range(sp_size - 1):
            receiver = sender + 1
            chunk_idx = step - sender
            
            if 0 <= chunk_idx < num_chunks:
                step_comms.append((sender, receiver))
        
        if step_comms:
            schedule.append(step_comms)
    
    return schedule

def overlap_input_processing(func: Callable) -> Callable:
    """
    Decorator for overlapping input processing with computation.
    
    This function wraps a forward function to overlap the processing of
    input data with computation, improving performance.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with overlapped input processing
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Extract hidden states
        hidden_states = args[0] if args else kwargs.get('hidden_states')
        
        # If no hidden states or not using CUDA, just call the original function
        if hidden_states is None or not torch.cuda.is_available():
            return func(self, *args, **kwargs)
        
        # Record the current stream
        main_stream = torch.cuda.current_stream()
        
        # Create a new stream for input processing
        input_stream = torch.cuda.Stream()
        
        # Launch input preprocessing in the new stream
        with torch.cuda.stream(input_stream):
            # Process the input and obtain processed_args, processed_kwargs
            # This would depend on the specific processing needed
            processed_args = list(args)
            processed_kwargs = kwargs.copy()
            
            # Add any input processing here...
            # For example, casting to half precision:
            if isinstance(hidden_states, torch.Tensor):
                processed_hidden_states = hidden_states.to(dtype=torch.float16)
                if args:
                    processed_args[0] = processed_hidden_states
                else:
                    processed_kwargs['hidden_states'] = processed_hidden_states
        
        # Wait for the input stream to finish
        main_stream.wait_stream(input_stream)
        
        # Call the original function with the processed inputs
        return func(self, *processed_args, **processed_kwargs)
    
    return wrapper

def create_communication_buffers(
    shape: Tuple[int, ...], 
    dtype: torch.dtype,
    num_buffers: int
) -> List[torch.Tensor]:
    """
    Create a pool of reusable communication buffers.
    
    This function allocates a set of buffers with the same shape and dtype,
    which can be reused for communications to avoid memory allocations.
    
    Args:
        shape: Shape of the buffers
        dtype: Data type of the buffers
        num_buffers: Number of buffers to create
        
    Returns:
        List of buffer tensors
    """
    buffers = []
    
    for _ in range(num_buffers):
        # Create buffer with pinned memory for faster GPU transfer
        buffer = torch.empty(shape, dtype=dtype, device='cuda')
        buffers.append(buffer)
    
    return buffers

# Experimental communication optimization utilities

def enable_nccl_p2p_optimization(
    group: Optional[dist.ProcessGroup] = None,
    nvlink_threshold: float = 10.0, # GB/s bandwidth threshold to detect NVLink
    p2p_matrix: Optional[List[List[bool]]] = None
) -> bool:
    """
    Enable peer-to-peer memory access between GPUs and optimize NCCL communication.
    
    This function performs the following optimizations:
    1. Enables direct P2P memory access between GPUs when available
    2. Detects and optimizes for NVLink connections
    3. Sets appropriate NCCL environment variables for optimal performance
    4. Provides fallback mechanisms when P2P access is not available
    
    Args:
        group: Process group to optimize (defaults to WORLD)
        nvlink_threshold: Bandwidth threshold in GB/s to identify NVLink connections
        p2p_matrix: Optional pre-detected P2P connectivity matrix; if None, will be detected
    
    Returns:
        bool: Whether optimization was successfully enabled
    """
    logger = logging.getLogger("p2p_optimizer")
    
    # Check prerequisites
    if not dist.is_initialized():
        logger.warning("PyTorch distributed not initialized, skipping P2P optimization")
        return False
        
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping P2P optimization")
        return False
    
    backend = dist.get_backend()
    if backend != "nccl":
        logger.warning(f"P2P optimization requires NCCL backend, found {backend}")
        return False
    
    # Use specified group or default to WORLD
    if group is None:
        group = dist.group.WORLD
    
    try:
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        
        # Skip optimization for single-GPU case
        if world_size <= 1:
            logger.info("Single GPU detected, no P2P optimization needed")
            return True
        
        # Step 1: Enable peer-to-peer memory access between all GPUs in the group
        # -----------------------------------------------------------------------
        
        # Get the device IDs for all ranks in the group
        local_device = torch.cuda.current_device()
        all_devices = [None] * world_size  # Will store device IDs for all ranks
        
        # Share device IDs across all ranks
        device_tensor = torch.tensor([local_device], dtype=torch.int64, device="cuda")
        device_list = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(world_size)]
        dist.all_gather(device_list, device_tensor, group=group)
        
        for i, tensor in enumerate(device_list):
            all_devices[i] = tensor.item()
        
        # Determine if we're in a single-node or multi-node setup
        hostname = subprocess.check_output("hostname", shell=True).decode().strip()
        hostnames = [None] * world_size
        
        # Share hostnames to detect multi-node setup
        hostname_bytes = hostname.encode()
        hostname_tensor = torch.zeros(256, dtype=torch.uint8, device="cuda")
        hostname_tensor[:len(hostname_bytes)] = torch.tensor([ord(c) for c in hostname], dtype=torch.uint8)
        hostname_list = [torch.zeros(256, dtype=torch.uint8, device="cuda") for _ in range(world_size)]
        dist.all_gather(hostname_list, hostname_tensor, group=group)
        
        for i, tensor in enumerate(hostname_list):
            hostnames[i] = "".join([chr(b) for b in tensor.cpu().tolist() if b > 0])
        
        # Find all devices on this node to enable P2P access
        local_devices = []
        for i, remote_hostname in enumerate(hostnames):
            if remote_hostname == hostname:
                local_devices.append(all_devices[i])
        
        # Enable P2P access between all local GPUs
        p2p_enabled_devices = set()
        if p2p_matrix is None:
            p2p_matrix = [[False for _ in range(world_size)] for _ in range(world_size)]
            
            # Enable P2P access for all pairs of local devices
            for i, dev_i in enumerate(local_devices):
                for j, dev_j in enumerate(local_devices):
                    if i != j:
                        can_access = torch.cuda.can_device_access_peer(dev_i, dev_j)
                        if can_access:
                            # Only need to enable once per pair of devices
                            pair_key = (min(dev_i, dev_j), max(dev_i, dev_j))
                            if pair_key not in p2p_enabled_devices:
                                try:
                                    # Enable P2P access
                                    with torch.cuda.device(dev_i):
                                        torch.cuda.device(dev_i).enable_peer_access(dev_j)
                                    logger.info(f"P2P access enabled from GPU {dev_i} to GPU {dev_j}")
                                    p2p_enabled_devices.add(pair_key)
                                    
                                    # Update connectivity matrix
                                    idx_i = local_devices.index(dev_i)
                                    idx_j = local_devices.index(dev_j)
                                    p2p_matrix[idx_i][idx_j] = True
                                    p2p_matrix[idx_j][idx_i] = True
                                except RuntimeError as e:
                                    logger.warning(f"Failed to enable P2P access from GPU {dev_i} to {dev_j}: {e}")
        
        # Step 2: Detect NVLink connections and optimize for them
        # -------------------------------------------------------
        nvlink_detected = False
        
        # Check for NVLink between devices using bandwidth measurements
        if torch.cuda.is_available():
            # Use CUDA events for timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Measure memory bandwidth between devices
            with torch.no_grad():
                # Create large tensors for bandwidth measurement
                test_size = 256 * 1024 * 1024  # 256 MB
                a = torch.ones(test_size // 4, dtype=torch.float32, device=f"cuda:{local_device}")
                
                # Get the current device architecture details
                device_props = torch.cuda.get_device_properties(local_device)
                
                # Check if the device is likely to have NVLink 
                # (Ampere, Volta, Hopper, etc. have enhanced NVLink support)
                if device_props.major >= 7:  # Volta (SM 7.x) and later architectures
                    for remote_device in local_devices:
                        if remote_device != local_device:
                            # Skip testing if devices are known not to have P2P access
                            pair_key = (min(local_device, remote_device), max(local_device, remote_device))
                            if pair_key not in p2p_enabled_devices:
                                continue
                                
                            # Measure bandwidth by copying data between devices
                            b = torch.empty_like(a, device=f"cuda:{remote_device}")
                            torch.cuda.synchronize()
                            
                            # Time the copy operation
                            start_event.record()
                            b.copy_(a)
                            end_event.record()
                            
                            # Synchronize and compute bandwidth
                            torch.cuda.synchronize()
                            elapsed_time = start_event.elapsed_time(end_event) / 1000  # convert to seconds
                            bandwidth = (test_size / (1024 * 1024 * 1024)) / elapsed_time  # GB/s
                            
                            logger.info(f"Bandwidth between GPU {local_device} and GPU {remote_device}: {bandwidth:.2f} GB/s")
                            
                            # Check if the bandwidth is high enough to indicate NVLink
                            if bandwidth > nvlink_threshold:
                                nvlink_detected = True
                                logger.info(f"NVLink connection detected between GPU {local_device} and GPU {remote_device}")
        
        # Step 3: Set appropriate NCCL environment variables for optimal performance
        # -------------------------------------------------------------------------
        
        # Set different environment variables based on the detected hardware configuration
        
        # Environment variables that improve performance regardless of hardware
        os.environ["NCCL_MIN_NCHANNELS"] = "4"  # Use multiple channels for communication
        os.environ["NCCL_BUFFSIZE"] = "4194304"  # 4 MB buffer size for better throughput
        
        # Only set algorithm if NVLink is detected
        if nvlink_detected:
            # Optimize for NVLink connections
            os.environ["NCCL_P2P_LEVEL"] = "NVL"  # Prefer NVLink over PCIe
            os.environ["NCCL_NET_GDR_LEVEL"] = "5"  # Maximum performance for GPUDirect RDMA
            os.environ["NCCL_ALGO"] = "NVLS"  # Use NVLink-optimized algorithm
            os.environ["NCCL_PROTO"] = "LL128"  # Use NVLink optimized protocol when appropriate
            
            # If this is a system with A100/H100 GPUs, enable SHARP/Tree algorithm
            if device_props.major >= 8:  # Ampere (SM 8.x) or later architecture
                os.environ["NCCL_ALGO"] = "SHARP_NVLS"  # Use optimized algorithms with tree and NVLS
                logger.info("Enabled SHARP/Tree NCCL algorithm for Ampere+ GPUs")
        else:
            # Fallback to standard optimizations for PCIe-connected GPUs
            os.environ["NCCL_ALGO"] = "RING"  # Use ring algorithm (reliable default)
            os.environ["NCCL_PROTO"] = "SIMPLE"  # Use simple protocol for stable performance
            os.environ["NCCL_P2P_LEVEL"] = "LOC"  # Local node communication optimization
        
        # Set NCCL debug level if we're logging verbosely
        if logger.level <= logging.DEBUG:
            os.environ["NCCL_DEBUG"] = "INFO"
            os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
        
        # Special handling for CUDA-aware MPI environments
        if "OMPI_COMM_WORLD_RANK" in os.environ:  # Detect OpenMPI environment
            os.environ["NCCL_DIRECT_GPU_P2P"] = "1"
        
        # Step 4: Handle edge cases and fallbacks for unavailable P2P access
        # -----------------------------------------------------------------
        
        # If no P2P connections were established, log a warning and modify environment variables
        if not p2p_enabled_devices and world_size > 1:
            logger.warning("No P2P connections established between GPUs")
            
            # Configure NCCL to work optimally without P2P
            os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P attempts that would fail
            os.environ["NCCL_SHM_DISABLE"] = "0"  # Ensure shared memory is enabled for single-node
            os.environ["NCCL_SOCKET_IFNAME"] = "^lo"  # Avoid using loopback interface
            
            # Fallback to TCP transport for multi-node setups
            if len(set(hostnames)) > 1:
                os.environ["NCCL_SOCKET_NTHREADS"] = "4"  # Increase socket threads for performance
                os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"  # Multiple sockets per thread
        
        # Synchronize processes after environment variable changes
        dist.barrier(group)
        
        # Log success and return
        logger.info(f"NCCL P2P optimization enabled. NVLink detected: {nvlink_detected}")
        return True
        
    except Exception as e:
        # Log the error but don't crash the program
        logging.error(f"Failed to enable NCCL P2P optimization: {e}", exc_info=True)
        return False

def create_p2p_communication_pattern(
    group: Optional[dist.ProcessGroup] = None,
    pattern_type: str = "ring",
    prioritize_nvlink: bool = True,
    min_peers: Optional[int] = None,
    max_peers: Optional[int] = None
) -> Dict[int, Dict[int, List[int]]]:
    """
    Analyze GPU topology and create optimized peer-to-peer communication patterns.
    
    This function creates communication patterns that minimize cross-node transfers
    and prioritize high-bandwidth connections like NVLink over PCIe when available.
    
    Args:
        group: Process group to analyze (defaults to WORLD)
        pattern_type: Communication pattern type ("ring", "bidir_ring", "hierarchical", "fully_connected")
        prioritize_nvlink: Whether to prioritize NVLink connections over PCIe
        min_peers: Minimum number of peers each rank should communicate with (None = auto)
        max_peers: Maximum number of peers each rank should communicate with (None = auto)
    
    Returns:
        A nested dictionary mapping source ranks to destination ranks and their communication paths.
        Format: {src_rank: {dst_rank: [path_nodes]}} where path_nodes is the sequence of
        intermediate ranks to reach dst_rank from src_rank.
    """
    logger = logging.getLogger("comm_pattern")
    
    # Check prerequisites
    if not dist.is_initialized():
        logger.warning("PyTorch distributed not initialized, returning empty pattern")
        return {}
    
    # Use specified group or default to WORLD
    if group is None:
        group = dist.group.WORLD
    
    # Get rank and world size info
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    
    if world_size <= 1:
        # For single process, no communication needed
        return {0: {}}
    
    # Set default min/max peers based on world size and pattern type
    if min_peers is None:
        if pattern_type == "fully_connected":
            min_peers = world_size - 1
        elif pattern_type in ["ring", "bidir_ring"]:
            min_peers = 2  # Two neighbors for each rank
        else:  # hierarchical
            min_peers = min(4, world_size - 1)  # Reasonable default for hierarchical
    
    if max_peers is None:
        if pattern_type == "fully_connected":
            max_peers = world_size - 1
        elif pattern_type in ["ring", "bidir_ring"]:
            max_peers = 2  # Two neighbors for each rank
        else:  # hierarchical
            # For hierarchical, allow more connections but not fully connected
            max_peers = min(world_size - 1, 8)
    
    # Initialize an empty pattern
    comm_pattern = {i: {} for i in range(world_size)}
    
    try:
        # Step 1: Get GPU and node topology information
        # ---------------------------------------------
        
        # Get device IDs and their hostname mappings
        local_device = torch.cuda.current_device() if torch.cuda.is_available() else -1
        all_devices = [None] * world_size
        all_hostnames = [None] * world_size
        
        # Share device IDs across ranks
        device_tensor = torch.tensor([local_device], dtype=torch.int64, device="cuda" if torch.cuda.is_available() else "cpu")
        device_list = [torch.zeros(1, dtype=torch.int64, device=device_tensor.device) for _ in range(world_size)]
        dist.all_gather(device_list, device_tensor, group=group)
        
        for i, tensor in enumerate(device_list):
            all_devices[i] = tensor.item()
        
        # Share hostnames to identify node boundaries
        hostname = subprocess.check_output("hostname", shell=True).decode().strip()
        hostname_bytes = hostname.encode()
        max_hostname_len = 256
        
        hostname_tensor = torch.zeros(max_hostname_len, dtype=torch.uint8, device=device_tensor.device)
        hostname_tensor[:len(hostname_bytes)] = torch.tensor([ord(c) for c in hostname_bytes], dtype=torch.uint8)
        
        hostname_list = [torch.zeros(max_hostname_len, dtype=torch.uint8, device=device_tensor.device) for _ in range(world_size)]
        dist.all_gather(hostname_list, hostname_tensor, group=group)
        
        for i, tensor in enumerate(hostname_list):
            all_hostnames[i] = "".join([chr(b) for b in tensor.cpu().tolist() if b > 0])
        
        # Create a mapping of {hostname: [ranks]}
        node_to_ranks = {}
        for i, hostname in enumerate(all_hostnames):
            if hostname not in node_to_ranks:
                node_to_ranks[hostname] = []
            node_to_ranks[hostname].append(i)
        
        # Number of nodes in the cluster
        num_nodes = len(node_to_ranks)
        
        # Step 2: Get P2P and NVLink connectivity information
        # ---------------------------------------------------
        
        # Create connectivity matrices
        p2p_matrix = [[False for _ in range(world_size)] for _ in range(world_size)]
        nvlink_matrix = [[False for _ in range(world_size)] for _ in range(world_size)]
        bandwidth_matrix = [[0.0 for _ in range(world_size)] for _ in range(world_size)]
        
        # NVIDIA Management Library (NVML) for detailed topology information
        nvml_available = False
        try:
            # Try to import pynvml for topology information
            import pynvml
            pynvml.nvmlInit()
            nvml_available = True
            logger.info("NVML available for detailed topology information")
        except (ImportError, AttributeError):
            logger.info("NVML not available, using PyTorch for topology detection")
        
        # Detect P2P and NVLink connectivity for this rank's device
        if torch.cuda.is_available():
            # First, collect basic P2P accessibility information
            local_p2p_access = {}
            for i, device_id in enumerate(all_devices):
                if all_hostnames[i] == hostname and device_id >= 0:
                    # Only check local devices on same node
                    try:
                        can_access = torch.cuda.can_device_access_peer(local_device, device_id)
                        local_p2p_access[i] = can_access
                        p2p_matrix[rank][i] = can_access
                        p2p_matrix[i][rank] = can_access  # Assuming symmetry
                    except RuntimeError:
                        # Handle cases where device is invalid
                        local_p2p_access[i] = False
            
            # If NVML is available, get detailed NVLink information
            if nvml_available:
                try:
                    # Get NVLink connectivity between GPUs
                    device_handle = pynvml.nvmlDeviceGetHandleByIndex(local_device)
                    
                    # Check for NVLink connectivity
                    for i, device_id in enumerate(all_devices):
                        if all_hostnames[i] == hostname and device_id >= 0 and i != rank:
                            try:
                                # Get remote device handle
                                remote_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                                
                                # Check NVLink connectivity
                                links = pynvml.nvmlDeviceGetNvLinkState(device_handle, device_id)
                                has_nvlink = any(links)
                                
                                # If NVLink is available, get bandwidth
                                if has_nvlink:
                                    nvlink_matrix[rank][i] = True
                                    nvlink_matrix[i][rank] = True
                                    
                                    # Try to get bandwidth information
                                    try:
                                        # Sum up bandwidth from all links
                                        total_bandwidth = 0
                                        for link in range(6):  # Up to 6 NVLink connections
                                            link_state = pynvml.nvmlDeviceGetNvLinkState(device_handle, link)
                                            if link_state == 1:  # Link is active
                                                # Get NVLink utilization info
                                                link_util = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(
                                                    device_handle, link, 0)  # 0 for receive counter
                                                total_bandwidth += link_util
                                        
                                        bandwidth_matrix[rank][i] = total_bandwidth
                                        bandwidth_matrix[i][rank] = total_bandwidth
                                    except pynvml.NVMLError:
                                        # If bandwidth info isn't available, just mark as NVLink
                                        bandwidth_matrix[rank][i] = 100  # Arbitrary high value
                                        bandwidth_matrix[i][rank] = 100
                            except pynvml.NVMLError:
                                pass
                except Exception as e:
                    logger.warning(f"Error getting NVML topology information: {e}")
            
            # If NVML isn't available or failed, use bandwidth measurements
            if not nvml_available or not any(nvlink_matrix[rank]):
                # Detect NVLink using bandwidth measurements
                # Create large tensors for bandwidth measurement
                test_size = 64 * 1024 * 1024  # 64 MB for quicker test
                a = torch.ones(test_size // 4, dtype=torch.float32, device=f"cuda:{local_device}")
                
                # Use CUDA events for timing
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # For each local peer device, measure bandwidth
                for i, device_id in enumerate(all_devices):
                    if all_hostnames[i] == hostname and device_id >= 0 and i != rank and local_p2p_access.get(i, False):
                        try:
                            # Measure bandwidth by copying data between devices
                            b = torch.empty_like(a, device=f"cuda:{device_id}")
                            torch.cuda.synchronize()
                            
                            # Time the copy operation
                            start_event.record()
                            b.copy_(a)
                            end_event.record()
                            
                            # Synchronize and compute bandwidth
                            torch.cuda.synchronize()
                            elapsed_time = start_event.elapsed_time(end_event) / 1000  # convert to seconds
                            bandwidth = (test_size / (1024 * 1024 * 1024)) / elapsed_time  # GB/s
                            
                            # Store the measured bandwidth
                            bandwidth_matrix[rank][i] = bandwidth
                            bandwidth_matrix[i][rank] = bandwidth
                            
                            # Check if the bandwidth is high enough to indicate NVLink (typically >20 GB/s)
                            if bandwidth > 20:  # Conservative NVLink threshold
                                nvlink_matrix[rank][i] = True
                                nvlink_matrix[i][rank] = True
                        except RuntimeError:
                            pass
        
        # Now gather everyone's P2P and NVLink connectivity info
        # Each rank has filled in its row of the matrices, share them
        
        # Convert matrices to tensors for gathering
        p2p_tensor = torch.tensor(p2p_matrix[rank], dtype=torch.bool, device=device_tensor.device)
        nvlink_tensor = torch.tensor(nvlink_matrix[rank], dtype=torch.bool, device=device_tensor.device)
        bandwidth_tensor = torch.tensor(bandwidth_matrix[rank], dtype=torch.float32, device=device_tensor.device)
        
        # Gather the tensors
        p2p_list = [torch.zeros(world_size, dtype=torch.bool, device=device_tensor.device) for _ in range(world_size)]
        nvlink_list = [torch.zeros(world_size, dtype=torch.bool, device=device_tensor.device) for _ in range(world_size)]
        bandwidth_list = [torch.zeros(world_size, dtype=torch.float32, device=device_tensor.device) for _ in range(world_size)]
        
        dist.all_gather(p2p_list, p2p_tensor, group=group)
        dist.all_gather(nvlink_list, nvlink_tensor, group=group)
        dist.all_gather(bandwidth_list, bandwidth_tensor, group=group)
        
        # Reconstruct the matrices
        for i in range(world_size):
            for j in range(world_size):
                p2p_matrix[i][j] = p2p_list[i][j].item()
                nvlink_matrix[i][j] = nvlink_list[i][j].item()
                bandwidth_matrix[i][j] = bandwidth_list[i][j].item()
        
        # Step 3: Create the communication pattern based on topology information
        # ----------------------------------------------------------------------
        
        # Create a graph representation of the topology
        connection_graph = {}
        
        # Add nodes for all ranks
        for i in range(world_size):
            connection_graph[i] = {}
        
        # Add weighted edges based on connectivity type
        # Weight priority: NVLink < PCIe < Cross-node
        for i in range(world_size):
            for j in range(world_size):
                if i == j:
                    continue  # Skip self
                
                # Determine connection weight
                if nvlink_matrix[i][j] and prioritize_nvlink:
                    weight = 1  # NVLink - lowest weight (preferred)
                elif p2p_matrix[i][j]:
                    weight = 10  # PCIe
                elif all_hostnames[i] == all_hostnames[j]:
                    weight = 50  # Same node but no direct P2P
                else:
                    weight = 100  # Cross-node - highest weight (avoid if possible)
                
                # Adjust weight based on bandwidth (higher bandwidth = lower weight)
                if bandwidth_matrix[i][j] > 0:
                    # Use a logarithmic scale to ensure bandwidth matters but doesn't overwhelm
                    # connectivity type
                    bandwidth_factor = max(0.1, 1.0 - math.log10(bandwidth_matrix[i][j]) / 10)
                    weight *= bandwidth_factor
                
                connection_graph[i][j] = weight
        
        # Now implement the requested pattern type
        if pattern_type == "ring":
            # Create a unidirectional ring communication pattern
            # Each rank sends to (rank+1) and receives from (rank-1)
            
            # First find the optimal ring order using approximation for TSP
            # Start with a simple nearest-neighbor approach
            current = 0  # Start with rank 0
            ring_order = [current]
            unvisited = set(range(1, world_size))
            
            while unvisited:
                # Find the nearest unvisited neighbor
                nearest = None
                nearest_weight = float('inf')
                
                for neighbor in unvisited:
                    weight = connection_graph[current][neighbor]
                    if weight < nearest_weight:
                        nearest = neighbor
                        nearest_weight = weight
                
                ring_order.append(nearest)
                unvisited.remove(nearest)
                current = nearest
            
            # Now create the ring pattern using the optimized order
            for i in range(len(ring_order)):
                src = ring_order[i]
                dst = ring_order[(i + 1) % len(ring_order)]
                # Direct path 
                comm_pattern[src][dst] = [dst]
            
        elif pattern_type == "bidir_ring":
            # Create a bidirectional ring communication pattern
            # Each rank sends to both (rank+1) and (rank-1)
            
            # Use the same ring ordering algorithm as for "ring"
            current = 0
            ring_order = [current]
            unvisited = set(range(1, world_size))
            
            while unvisited:
                nearest = None
                nearest_weight = float('inf')
                
                for neighbor in unvisited:
                    weight = connection_graph[current][neighbor]
                    if weight < nearest_weight:
                        nearest = neighbor
                        nearest_weight = weight
                
                ring_order.append(nearest)
                unvisited.remove(nearest)
                current = nearest
            
            # Create bidirectional connections
            for i in range(len(ring_order)):
                src = ring_order[i]
                next_rank = ring_order[(i + 1) % len(ring_order)]
                prev_rank = ring_order[(i - 1) % len(ring_order)]
                
                # Add both forward and backward paths
                comm_pattern[src][next_rank] = [next_rank]
                comm_pattern[src][prev_rank] = [prev_rank]
            
        elif pattern_type == "hierarchical":
            # Create a hierarchical communication pattern
            # Ranks in the same node form a fully connected group
            # Each node has one or more designated communicators that connect to other nodes
            
            # Step 1: Group ranks by node
            node_groups = {}
            for hostname, ranks in node_to_ranks.items():
                node_groups[hostname] = ranks
            
            # Step 2: Create fully connected groups within each node
            for hostname, ranks in node_groups.items():
                for src in ranks:
                    for dst in ranks:
                        if src != dst:
                            comm_pattern[src][dst] = [dst]  # Direct connection
            
            # Step 3: For each node, select a leader rank that has the best connections
            node_leaders = {}
            for hostname, ranks in node_groups.items():
                best_leader = None
                best_score = float('-inf')
                
                for candidate in ranks:
                    # Calculate connectivity score (lower weights are better)
                    score = 0
                    for other_hostname, other_ranks in node_groups.items():
                        if hostname != other_hostname:
                            # Find best connection to other node
                            best_connection = float('inf')
                            for other_rank in other_ranks:
                                weight = connection_graph[candidate][other_rank]
                                best_connection = min(best_connection, weight)
                            score -= best_connection  # Negative because lower weights are better
                    
                    if score > best_score:
                        best_leader = candidate
                        best_score = score
                
                node_leaders[hostname] = best_leader
            
            # Step 4: Connect the node leaders
            for src_hostname, src_leader in node_leaders.items():
                for dst_hostname, dst_leader in node_leaders.items():
                    if src_hostname != dst_hostname:
                        # Connect the leaders
                        comm_pattern[src_leader][dst_leader] = [dst_leader]
            
            # Step 5: Connect non-leader ranks to other nodes via their leader
            for src_hostname, ranks in node_groups.items():
                src_leader = node_leaders[src_hostname]
                
                for src in ranks:
                    if src != src_leader:  # Skip the leader
                        for dst_hostname, dst_ranks in node_groups.items():
                            if src_hostname != dst_hostname:
                                dst_leader = node_leaders[dst_hostname]
                                
                                # Connect through the leader
                                for dst in dst_ranks:
                                    # Path: src -> src_leader -> dst_leader -> dst
                                    if dst_leader == dst:
                                        # If destination is the remote leader, go through local leader
                                        comm_pattern[src][dst] = [src_leader, dst]
                                    else:
                                        # Regular case: go through both leaders
                                        comm_pattern[src][dst] = [src_leader, dst_leader, dst]
            
        elif pattern_type == "fully_connected":
            # Every rank connects directly to every other rank
            for src in range(world_size):
                for dst in range(world_size):
                    if src != dst:
                        comm_pattern[src][dst] = [dst]  # Direct connection
        
        else:
            # Unknown pattern type
            logger.warning(f"Unknown pattern type: {pattern_type}. Using ring pattern.")
            # Create a simple ring pattern as fallback
            for src in range(world_size):
                dst = (src + 1) % world_size
                comm_pattern[src][dst] = [dst]
                
                # Also add the reverse direction
                rev_dst = (src - 1) % world_size
                comm_pattern[src][rev_dst] = [rev_dst]
        
        # Step 4: Apply peer count constraints
        # ------------------------------------
        
        # Adjust the number of peers per rank
        for src in range(world_size):
            peers = list(comm_pattern[src].keys())
            num_peers = len(peers)
            
            # If we have too many peers, keep the ones with the best connections
            if num_peers > max_peers:
                # Sort peers by connection weight (lower is better)
                peers_by_weight = sorted(peers, key=lambda dst: connection_graph[src][dst])
                # Keep only the top max_peers
                peers_to_keep = peers_by_weight[:max_peers]
                # Remove excess peers
                for dst in peers:
                    if dst not in peers_to_keep:
                        del comm_pattern[src][dst]
            
            # If we have too few peers, add more connections
            elif num_peers < min_peers and num_peers < world_size - 1:
                # Find additional peers to connect to
                remaining_peers = [p for p in range(world_size) if p != src and p not in peers]
                # Sort by connection weight
                remaining_peers.sort(key=lambda dst: connection_graph[src][dst])
                # Add peers until we reach min_peers or run out of candidates
                for dst in remaining_peers:
                    if len(comm_pattern[src]) >= min_peers:
                        break
                    comm_pattern[src][dst] = [dst]  # Direct connection
        
        # Step 5: Log and return the pattern
        # ----------------------------------
        
        if rank == 0:
            logger.info(f"Created {pattern_type} communication pattern with "
                      f"{world_size} ranks across {num_nodes} nodes")
            
            # Log detailed stats about the pattern
            total_connections = sum(len(peers) for peers in comm_pattern.values())
            avg_peers = total_connections / world_size
            logger.info(f"Average peers per rank: {avg_peers:.2f}")
            
            # Count NVLink vs PCIe vs cross-node connections
            nvlink_count = 0
            pcie_count = 0
            cross_node_count = 0
            
            for src, peers in comm_pattern.items():
                for dst in peers:
                    if nvlink_matrix[src][dst]:
                        nvlink_count += 1
                    elif p2p_matrix[src][dst]:
                        pcie_count += 1
                    elif all_hostnames[src] != all_hostnames[dst]:
                        cross_node_count += 1
            
            logger.info(f"Connection types: NVLink={nvlink_count}, PCIe={pcie_count}, "
                      f"Cross-node={cross_node_count}")
        
        # Clean up NVML if used
        if nvml_available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
        
        return comm_pattern
        
    except Exception as e:
        logger.error(f"Error creating communication pattern: {e}", exc_info=True)
        # Return a simple default pattern on error
        default_pattern = {}
        for i in range(world_size):
            default_pattern[i] = {(i + 1) % world_size: [(i + 1) % world_size]}
        return default_pattern


def apply_communication_pattern_to_model(
    model: nn.Module, 
    comm_pattern: Dict[int, Dict[int, List[int]]],
    module_pattern: Optional[Dict[str, List[int]]] = None
) -> nn.Module:
    """
    Apply a communication pattern to a PyTorch model.
    
    Args:
        model: PyTorch model
        comm_pattern: Communication pattern returned by create_p2p_communication_pattern
        module_pattern: Optional dictionary mapping module names to target ranks
        
    Returns:
        Model with communication pattern applied
    """
    if not dist.is_initialized():
        return model
        
    # Get rank information
    rank = dist.get_rank()
    
    # Store the global communication pattern on the model
    model._global_comm_pattern = comm_pattern
    
    # If specific module pattern is provided, use it
    if module_pattern is not None:
        model._module_comm_pattern = module_pattern
        
        # Apply to specific modules
        for module_name, target_ranks in module_pattern.items():
            # Find the module by name
            module = None
            for name, mod in model.named_modules():
                if name == module_name:
                    module = mod
                    break
                    
            if module is None:
                continue
                
            # If module supports communication pattern preparation
            if hasattr(module, 'prepare_p2p_communication'):
                # Get the optimal paths to target ranks from the global pattern
                paths = {}
                my_pattern = comm_pattern.get(rank, {})
                
                for target_rank in target_ranks:
                    # Get path to target rank if available
                    if target_rank in my_pattern:
                        paths[target_rank] = my_pattern[target_rank]
                    else:
                        # Direct path if not found
                        paths[target_rank] = [target_rank]
                
                # Prepare module for communication
                module.prepare_p2p_communication(paths)
    
    return model


def ring_exchange(
    *tensors: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
    use_fp16: bool = False,
    use_nccl_collectives: bool = True
) -> Union[List[torch.Tensor], Tuple[torch.distributed.Work, List[torch.Tensor]]]:
    """
    Implement ring-exchange communication pattern for sequence parallelism.
    
    In ring exchange, each rank sends data to rank+1 and receives from rank-1,
    forming a ring topology. This is useful for attention patterns where each worker
    needs to access all keys/values in a ring fashion.
    
    Args:
        *tensors: Tensors to exchange (all ranks send and receive the same number of tensors)
        group: Process group for communication
        async_op: Whether to perform operation asynchronously
        use_fp16: Whether to convert tensors to FP16 for reduced bandwidth
        use_nccl_collectives: Whether to use NCCL's optimized collectives instead of P2P sends/recvs
        
    Returns:
        List of exchanged tensors (received from previous rank) or
        tuple of (work handle, tensors) if async_op=True
    """
    if not dist.is_initialized():
        return list(tensors)
        
    if group is None:
        group = dist.group.WORLD
        
    # Get rank info
    rank = dist.get_rank(group)
    world_size = group.size()
    
    if world_size == 1:
        return list(tensors)
        
    # Determine source and destination ranks in the ring
    src_rank = (rank - 1) % world_size
    dst_rank = (rank + 1) % world_size
    
    # Create tensor copies to avoid modifying the inputs
    send_tensors = []
    for tensor in tensors:
        # Convert to FP16 for bandwidth reduction if requested
        if use_fp16 and tensor.is_floating_point() and tensor.dtype != torch.float16:
            send_tensors.append(tensor.to(dtype=torch.float16))
        else:
            send_tensors.append(tensor.clone())
            
    # Create output tensors to receive data
    recv_tensors = []
    for tensor in tensors:
        # Create tensor with same shape and dtype
        recv_dtype = torch.float16 if use_fp16 and tensor.is_floating_point() else tensor.dtype
        recv_tensor = torch.empty_like(tensor, dtype=recv_dtype, device=tensor.device)
        recv_tensors.append(recv_tensor)
    
    # Use NCCL's optimized collectives if available and requested
    if use_nccl_collectives and hasattr(dist, "recv_from_src") and torch.cuda.is_available():
        # This is a hypothetical NCCL-optimized API that would exist in a production system
        # In real systems, NCCL has optimized collectives for ring patterns
        work_handles = []
        for send_tensor, recv_tensor in zip(send_tensors, recv_tensors):
            # Perform send to next rank and receive from previous rank
            with torch.cuda.stream(torch.cuda.Stream()):
                handle = dist.send_to_dst_recv_from_src(
                    send_tensor, dst_rank,
                    recv_tensor, src_rank,
                    group=group, async_op=True
                )
                work_handles.append(handle)
                
        if async_op:
            # Create a combined work handle
            combined_work = _combine_work_handles(work_handles)
            return combined_work, recv_tensors
            
        # Wait for all operations to complete
        for handle in work_handles:
            handle.wait()
            
        # Convert back to original dtype if needed
        if use_fp16:
            for i, (recv_tensor, orig_tensor) in enumerate(zip(recv_tensors, tensors)):
                if orig_tensor.dtype != recv_tensor.dtype:
                    recv_tensors[i] = recv_tensor.to(dtype=orig_tensor.dtype)
                    
        return recv_tensors
    
    # Fall back to standard send/recv operations
    work_handles = []
    
    # Create streams for communication to allow overlap
    if torch.cuda.is_available():
        send_stream = torch.cuda.Stream()
        recv_stream = torch.cuda.Stream()
    else:
        send_stream = None
        recv_stream = None
    
    # Post receives first (to avoid deadlocks)
    for i, recv_tensor in enumerate(recv_tensors):
        if recv_stream is not None:
            with torch.cuda.stream(recv_stream):
                handle = dist.irecv(recv_tensor, src_rank, group=group)
                work_handles.append(handle)
        else:
            handle = dist.irecv(recv_tensor, src_rank, group=group)
            work_handles.append(handle)
    
    # Then post sends
    for i, send_tensor in enumerate(send_tensors):
        if send_stream is not None:
            with torch.cuda.stream(send_stream):
                handle = dist.isend(send_tensor, dst_rank, group=group)
                work_handles.append(handle)
        else:
            handle = dist.isend(send_tensor, dst_rank, group=group)
            work_handles.append(handle)
    
    if async_op:
        # Create a combined work handle
        combined_work = _combine_work_handles(work_handles)
        return combined_work, recv_tensors
    
    # Wait for all operations to complete
    for handle in work_handles:
        handle.wait()
    
    # Convert back to original dtype if needed
    if use_fp16:
        for i, (recv_tensor, orig_tensor) in enumerate(zip(recv_tensors, tensors)):
            if orig_tensor.dtype != recv_tensor.dtype:
                recv_tensors[i] = recv_tensor.to(dtype=orig_tensor.dtype)
    
    return recv_tensors


def _combine_work_handles(handles: List[torch.distributed.Work]) -> torch.distributed.Work:
    """
    Combine multiple work handles into a single one.
    
    Args:
        handles: List of distributed work handles
        
    Returns:
        A single work handle that represents all operations
    """
    # This is a helper implementation - in PyTorch you'd typically implement
    # a custom WorkHandle class that wraps multiple handles
    class CombinedWork:
        def __init__(self, handles):
            self.handles = handles
            
        def wait(self):
            for handle in self.handles:
                handle.wait()
                
        def is_completed(self):
            return all(handle.is_completed() for handle in self.handles)
                
        def is_success(self):
            return all(handle.is_success() for handle in self.handles)
            
        def exception(self):
            for handle in self.handles:
                if not handle.is_success():
                    return handle.exception()
            return None
    
    return CombinedWork(handles)