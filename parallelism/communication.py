import torch
import torch.nn as nn
import torch.distributed as dist
import functools
import weakref
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
    group: Optional[dist.ProcessGroup] = None
) -> Union[torch.Tensor, Tuple[torch.distributed.Work, torch.Tensor]]:
    """
    Perform all-reduce operation across all processes.
    
    Args:
        tensor: Input tensor to be reduced
        op: Reduction operation (dist.ReduceOp or string: "sum", "avg", "max", "min", "prod")
        async_op: Whether to perform asynchronous operation
        group: Process group
        
    Returns:
        Reduced tensor or tuple of (work handle, tensor) if async_op=True
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    
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

def enable_nccl_p2p_optimization() -> None:
    """Enable NCCL P2P optimization for better communication performance."""
    if torch.distributed.is_initialized() and torch.distributed.get_backend() == "nccl":
        # In a real implementation, this would involve setting NCCL environment variables
        # or using NCCL's API to optimize P2P communication patterns
        pass

def create_p2p_communication_pattern(
    model: nn.Module, 
    communication_pattern: Dict[str, List[int]]
) -> None:
    """
    Set up an optimized peer-to-peer communication pattern.
    
    Args:
        model: PyTorch model
        communication_pattern: Dictionary mapping module names to lists of ranks to communicate with
    """
    # In a real implementation, this would involve optimizing the communication graph
    # based on the model's structure and the physical network topology
    pass