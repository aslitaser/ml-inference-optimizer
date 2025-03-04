import os
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Any, Tuple, List, Optional, Union

from .parallel_utils import (
    analyze_model_for_parallelism,
    estimate_memory_requirements,
    calculate_communication_overhead,
    validate_parallel_config,
    initialize_parallel_groups,
    get_process_group_for_operation,
    create_communication_streams,
    synchronize_all_streams
)


class ParallelConfig:
    """
    Configuration class for multi-dimensional parallelism strategies.
    
    This class defines how a model should be partitioned across multiple GPUs
    using various parallelism techniques including tensor parallelism,
    sequence parallelism, data parallelism, and pipeline parallelism.
    
    Attributes:
        world_size (int): Total number of GPUs available
        tensor_parallel_size (int): Number of GPUs for tensor parallelism
        sequence_parallel_size (int): Number of GPUs for sequence parallelism
        data_parallel_size (int): Number of GPUs for data parallelism
        pipeline_parallel_size (int): Number of GPUs for pipeline parallelism
        communication_dtype (torch.dtype): Data type used for communication
        overlap_communication (bool): Whether to overlap communication with computation
        optimize_memory (bool): Whether to optimize for memory efficiency
        activation_checkpointing (bool): Whether to use activation checkpointing
    """
    def __init__(
        self,
        world_size: int,
        tensor_parallel_size: int = 1,
        sequence_parallel_size: int = 1,
        data_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        communication_dtype: torch.dtype = torch.float16,
        overlap_communication: bool = True,
        optimize_memory: bool = True,
        activation_checkpointing: bool = False
    ):
        self.world_size = world_size
        self.tensor_parallel_size = tensor_parallel_size
        self.sequence_parallel_size = sequence_parallel_size
        self.data_parallel_size = data_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.communication_dtype = communication_dtype
        self.overlap_communication = overlap_communication
        self.optimize_memory = optimize_memory
        self.activation_checkpointing = activation_checkpointing
        
    def validate_configuration(self) -> Tuple[bool, str]:
        """
        Validates that the parallel configuration is valid.
        
        Ensures that:
        1. The product of all parallel sizes equals the world size
        2. Parallel sizes are compatible with each other
        3. The configuration is feasible for the available hardware
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        is_valid, error_message = validate_parallel_config(self, self.world_size)
        return is_valid, error_message

    def __str__(self) -> str:
        """
        Returns a string representation of the configuration.
        """
        return (
            f"ParallelConfig(world_size={self.world_size}, "
            f"tensor_parallel_size={self.tensor_parallel_size}, "
            f"sequence_parallel_size={self.sequence_parallel_size}, "
            f"data_parallel_size={self.data_parallel_size}, "
            f"pipeline_parallel_size={self.pipeline_parallel_size}, "
            f"communication_dtype={self.communication_dtype}, "
            f"overlap_communication={self.overlap_communication}, "
            f"optimize_memory={self.optimize_memory}, "
            f"activation_checkpointing={self.activation_checkpointing})"
        )


class InferenceSchedule:
    """
    Defines the execution schedule for parallel inference.
    
    This class encapsulates the logic for scheduling operations across
    multiple GPUs, including communication patterns and synchronization points.
    """
    def __init__(
        self,
        config: ParallelConfig,
        model_info: Dict[str, Any],
        batch_size: int,
        seq_len: int
    ):
        self.config = config
        self.model_info = model_info
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.schedule = self._create_schedule()
        self.communication_streams = create_communication_streams()
        
    def _create_schedule(self) -> List[Dict[str, Any]]:
        """
        Creates a detailed execution schedule based on the model and configuration.
        
        Returns:
            List[Dict[str, Any]]: List of operations in execution order
        """
        schedule = []
        # Implementation depends on model architecture and parallel strategy
        # Simplified example:
        if self.config.pipeline_parallel_size > 1:
            # Create pipeline schedule with micro-batches
            microbatch_size = self.batch_size // self.config.pipeline_parallel_size
            for i in range(self.config.pipeline_parallel_size * 2 - 1):  # Pipeline bubbles
                stage = min(i, self.config.pipeline_parallel_size - 1)
                if i < self.config.pipeline_parallel_size:
                    microbatch = i
                    schedule.append({
                        "type": "forward",
                        "stage": stage,
                        "microbatch": microbatch,
                    })
                else:
                    microbatch = i - self.config.pipeline_parallel_size
                    next_stage = stage + 1
                    schedule.append({
                        "type": "communication",
                        "from_stage": stage,
                        "to_stage": next_stage,
                        "microbatch": microbatch,
                    })
        else:
            # Simple forward pass schedule for non-pipeline configurations
            schedule.append({
                "type": "forward",
                "stage": 0,
                "is_parallel": self.config.tensor_parallel_size > 1 or self.config.sequence_parallel_size > 1,
            })
            
        return schedule
    
    def execute(self, model_components: List[nn.Module], inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Executes the inference schedule on the given model components.
        
        Args:
            model_components: List of model components assigned to this process
            inputs: Input tensors for the model
            
        Returns:
            torch.Tensor: Output of the model
        """
        outputs = None
        current_inputs = inputs
        
        for step in self.schedule:
            if step["type"] == "forward":
                stage = step["stage"]
                if stage == dist.get_rank() // (self.config.tensor_parallel_size * self.config.sequence_parallel_size):
                    outputs = model_components[stage](current_inputs)
            
            elif step["type"] == "communication":
                from_stage = step["from_stage"]
                to_stage = step["to_stage"]
                # Handle communication between pipeline stages
                if from_stage == dist.get_rank() // (self.config.tensor_parallel_size * self.config.sequence_parallel_size):
                    # Send outputs to next stage
                    process_group = get_process_group_for_operation("pipeline")
                    with torch.cuda.stream(self.communication_streams["pipeline"]):
                        dist.send(outputs, to_stage, process_group)
                
                if to_stage == dist.get_rank() // (self.config.tensor_parallel_size * self.config.sequence_parallel_size):
                    # Receive inputs from previous stage
                    process_group = get_process_group_for_operation("pipeline")
                    with torch.cuda.stream(self.communication_streams["pipeline"]):
                        dist.recv(current_inputs, from_stage, process_group)
        
        synchronize_all_streams()
        return outputs


class TensorParallelExecutor:
    """
    Executor for tensor parallel operations.
    
    Handles the distribution and execution of tensor-parallel operations,
    including sharding weights and managing all-reduce communications.
    """
    def __init__(self, config: ParallelConfig, process_groups: Dict[str, dist.ProcessGroup]):
        self.config = config
        self.process_groups = process_groups
        self.tp_group = process_groups.get("tensor")
        self.tp_rank = dist.get_rank(self.tp_group) if self.tp_group else 0
        self.communication_streams = create_communication_streams()
        
    def shard_module(self, module: nn.Module) -> nn.Module:
        """
        Shards a module's parameters across tensor-parallel devices.
        
        Args:
            module: Module to shard
            
        Returns:
            nn.Module: Sharded module
        """
        # Implementation depends on module type
        # Example: shard linear layer weights
        if isinstance(module, nn.Linear):
            # Split output dimension for column parallelism
            out_features = module.out_features
            in_features = module.in_features
            
            shard_size = out_features // self.config.tensor_parallel_size
            start_idx = shard_size * self.tp_rank
            end_idx = start_idx + shard_size if self.tp_rank < self.config.tensor_parallel_size - 1 else out_features
            
            # Create sharded weight and bias
            sharded_weight = nn.Parameter(module.weight[start_idx:end_idx, :].clone())
            sharded_bias = None
            if module.bias is not None:
                sharded_bias = nn.Parameter(module.bias[start_idx:end_idx].clone())
            
            # Create new linear layer with sharded parameters
            sharded_module = nn.Linear(in_features, end_idx - start_idx, bias=module.bias is not None)
            sharded_module.weight = sharded_weight
            if sharded_bias is not None:
                sharded_module.bias = sharded_bias
                
            return sharded_module
            
        return module
    
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs an all-reduce operation across tensor-parallel devices.
        
        Args:
            tensor: Tensor to reduce
            
        Returns:
            torch.Tensor: Reduced tensor
        """
        if self.config.tensor_parallel_size > 1:
            with torch.cuda.stream(self.communication_streams["tensor"]):
                dist.all_reduce(tensor, group=self.tp_group)
        return tensor


class SequenceParallelExecutor:
    """
    Executor for sequence parallel operations.
    
    Handles the distribution and execution of sequence-parallel operations,
    including splitting inputs along the sequence dimension and managing all-gather communications.
    """
    def __init__(self, config: ParallelConfig, process_groups: Dict[str, dist.ProcessGroup]):
        self.config = config
        self.process_groups = process_groups
        self.sp_group = process_groups.get("sequence")
        self.sp_rank = dist.get_rank(self.sp_group) if self.sp_group else 0
        self.communication_streams = create_communication_streams()
        
    def split_sequence_dimension(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Splits a tensor along the sequence dimension for sequence parallelism.
        
        Args:
            tensor: Input tensor (batch_size, seq_len, ...)
            dim: Dimension to split (typically 1 for sequence dimension)
            
        Returns:
            torch.Tensor: Sequence-split tensor
        """
        if self.config.sequence_parallel_size <= 1:
            return tensor
            
        seq_len = tensor.size(dim)
        split_size = seq_len // self.config.sequence_parallel_size
        start_idx = split_size * self.sp_rank
        end_idx = start_idx + split_size if self.sp_rank < self.config.sequence_parallel_size - 1 else seq_len
        
        return tensor.narrow(dim, start_idx, end_idx - start_idx)
    
    def all_gather_sequence_dimension(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Gathers sequence-parallelized tensor outputs along the sequence dimension.
        
        Args:
            tensor: Sequence-split tensor
            dim: Dimension to gather (typically 1 for sequence dimension)
            
        Returns:
            torch.Tensor: Full sequence tensor
        """
        if self.config.sequence_parallel_size <= 1:
            return tensor
            
        # Get tensor sizes on all ranks
        local_size = tensor.size(dim)
        all_sizes = [torch.tensor([local_size], device=tensor.device) for _ in range(self.config.sequence_parallel_size)]
        
        with torch.cuda.stream(self.communication_streams["sequence"]):
            dist.all_gather(all_sizes, all_sizes[self.sp_rank], group=self.sp_group)
            
        # Create output tensor
        total_size = sum(size.item() for size in all_sizes)
        output_shape = list(tensor.size())
        output_shape[dim] = total_size
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        
        # Perform all-gather operation
        gather_list = [torch.empty_like(tensor) for _ in range(self.config.sequence_parallel_size)]
        with torch.cuda.stream(self.communication_streams["sequence"]):
            dist.all_gather(gather_list, tensor, group=self.sp_group)
            
        # Reassemble the full tensor
        offset = 0
        for i, size in enumerate(all_sizes):
            size = size.item()
            slice_indices = [slice(None)] * len(output_shape)
            slice_indices[dim] = slice(offset, offset + size)
            output[slice_indices] = gather_list[i].narrow(dim, 0, size)
            offset += size
            
        return output


class HybridParallelExecutor:
    """
    Executor for hybrid parallel operations.
    
    Combines multiple parallelism strategies for optimal performance,
    handling the interactions between tensor, sequence, and pipeline parallelism.
    """
    def __init__(
        self,
        config: ParallelConfig,
        process_groups: Dict[str, dist.ProcessGroup],
        model_info: Dict[str, Any]
    ):
        self.config = config
        self.process_groups = process_groups
        self.model_info = model_info
        self.tensor_parallel = TensorParallelExecutor(config, process_groups)
        self.sequence_parallel = SequenceParallelExecutor(config, process_groups)
        
    def distribute_module(self, module: nn.Module) -> nn.Module:
        """
        Distributes a module using hybrid parallelism strategies.
        
        Args:
            module: Module to distribute
            
        Returns:
            nn.Module: Distributed module
        """
        # Determine which parallelism strategy to apply based on module type and configuration
        module_name = module.__class__.__name__
        
        if module_name in self.model_info.get("tensor_parallel_modules", []):
            module = self.tensor_parallel.shard_module(module)
            
        # Apply sequence parallelism hooks where needed
        if module_name in self.model_info.get("sequence_parallel_modules", []):
            # Add hooks for sequence parallelism
            def pre_forward_hook(module, input):
                return (self.sequence_parallel.split_sequence_dimension(input[0]),)
                
            def post_forward_hook(module, input, output):
                return self.sequence_parallel.all_gather_sequence_dimension(output)
                
            module.register_forward_pre_hook(pre_forward_hook)
            module.register_forward_hook(post_forward_hook)
            
        return module


class RuntimeAdaptiveExecutor:
    """
    Executor that adapts parallelism strategy at runtime.
    
    Monitors execution metrics and dynamically adjusts parallelism strategy
    to optimize performance based on observed characteristics.
    """
    def __init__(
        self,
        config: ParallelConfig,
        process_groups: Dict[str, dist.ProcessGroup],
        model_info: Dict[str, Any]
    ):
        self.config = config
        self.process_groups = process_groups
        self.model_info = model_info
        self.hybrid_executor = HybridParallelExecutor(config, process_groups, model_info)
        self.performance_metrics = {
            "latency": [],
            "throughput": [],
            "memory_usage": [],
            "communication_overhead": []
        }
        
    def distribute_module(self, module: nn.Module) -> nn.Module:
        """
        Distributes a module with runtime adaptive capabilities.
        
        Args:
            module: Module to distribute
            
        Returns:
            nn.Module: Distributed module with runtime adaptation
        """
        # Apply basic hybrid parallelism
        module = self.hybrid_executor.distribute_module(module)
        
        # Add runtime monitoring hooks
        def runtime_monitoring_hook(module, input, output):
            # Record execution time, memory usage, etc.
            # This data will be used to adapt parallelism strategy
            return output
            
        module.register_forward_hook(runtime_monitoring_hook)
        return module
        
    def adapt_strategy(self, input_size: Tuple[int, ...], memory_pressure: float) -> None:
        """
        Adapts parallelism strategy based on runtime conditions.
        
        Args:
            input_size: Size of input tensors
            memory_pressure: Current memory pressure (0.0-1.0)
        """
        # Example adaptation logic:
        if memory_pressure > 0.9 and self.config.sequence_parallel_size < self.config.world_size:
            # Increase sequence parallelism under high memory pressure
            logging.info("Adapting parallelism strategy: increasing sequence parallelism")
            # Adaptation implementation would reconfigure process groups and update modules
        elif memory_pressure < 0.5 and input_size[1] > 4096:  # Long sequence
            # For long sequences with low memory pressure, optimize for throughput
            logging.info("Adapting parallelism strategy: optimizing for long sequences")
            # Reconfigure for throughput-optimized execution


class ModelParallelWrapper(nn.Module):
    """
    Wrapper for model parallelism that applies multiple parallelism strategies.
    
    This class wraps a PyTorch model and applies the configured parallelism
    strategies for distributed inference across multiple GPUs.
    """
    def __init__(self, model: nn.Module, config: ParallelConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.is_valid, error_message = config.validate_configuration()
        if not self.is_valid:
            raise ValueError(f"Invalid parallel configuration: {error_message}")
            
        # Initialize process groups for communication
        self.process_groups = initialize_parallel_groups(config)
        
        # Analyze model to determine parallelism strategy
        self.model_info = analyze_model_for_parallelism(model)
        
        # Select executor based on configuration
        if config.tensor_parallel_size > 1 and config.sequence_parallel_size > 1:
            self.executor = HybridParallelExecutor(config, self.process_groups, self.model_info)
        elif config.tensor_parallel_size > 1:
            self.executor = TensorParallelExecutor(config, self.process_groups)
        elif config.sequence_parallel_size > 1:
            self.executor = SequenceParallelExecutor(config, self.process_groups)
        else:
            # Data parallel only or pipeline parallel
            self.executor = None
            
        # Apply parallelism to model
        self._parallelize_model()
        
        # Create forward hooks for specialized processing
        self.create_forward_hooks()
        
        # Optimize model for inference if requested
        if config.optimize_memory:
            self.optimize_for_inference()
            
    def _parallelize_model(self) -> None:
        """
        Applies parallelism transformations to the model.
        """
        # For models with distinct modules/layers:
        if hasattr(self.model, "layers") or hasattr(self.model, "blocks"):
            layers = getattr(self.model, "layers", None) or getattr(self.model, "blocks", [])
            
            for i, layer in enumerate(layers):
                if self.executor is not None:
                    layers[i] = self.executor.distribute_module(layer)
                    
        # For other module types, apply parallelism by recursively transforming submodules
        else:
            def _apply_parallelism(module):
                for name, child in module.named_children():
                    if self.executor is not None:
                        transformed_child = self.executor.distribute_module(child)
                        if transformed_child is not child:
                            setattr(module, name, transformed_child)
                    _apply_parallelism(getattr(module, name))
                    
            _apply_parallelism(self.model)
            
    def create_forward_hooks(self) -> None:
        """
        Creates forward hooks for parallel execution.
        """
        # Add hooks for activation checkpointing if enabled
        if self.config.activation_checkpointing:
            from torch.utils.checkpoint import checkpoint
            
            def activation_checkpoint_hook(module, input):
                def custom_forward(*inputs):
                    return module(*inputs)
                
                return (checkpoint(custom_forward, *input),)
                
            for module in self.model.modules():
                if module.__class__.__name__ in self.model_info.get("checkpointed_modules", []):
                    module.register_forward_pre_hook(activation_checkpoint_hook)
    
    def optimize_for_inference(self) -> None:
        """
        Applies inference-specific optimizations to the model.
        """
        # Fuse operations where possible
        # Example: fuse batch norm into conv layers
        for module in self.model.modules():
            if isinstance(module, nn.Sequential):
                # Check for Conv->BatchNorm->ReLU pattern for fusion
                for i in range(len(module) - 2):
                    if (isinstance(module[i], nn.Conv2d) and 
                        isinstance(module[i+1], nn.BatchNorm2d) and 
                        isinstance(module[i+2], nn.ReLU)):
                        # Fuse these layers
                        # Implementation would depend on available fusion APIs
                        pass
        
        # Apply half-precision optimizations if using fp16 communication
        if self.config.communication_dtype == torch.float16:
            # Convert appropriate modules to half precision
            for module in self.model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    module.half()
    
    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass with parallel execution.
        
        Args:
            *args: Positional arguments to the model
            **kwargs: Keyword arguments to the model
            
        Returns:
            Any: Model outputs
        """
        return self._distribute_and_forward(*args, **kwargs)
        
    def _distribute_and_forward(self, *args, **kwargs) -> Any:
        """
        Distributes inputs and executes forward pass with parallel strategies.
        
        Args:
            *args: Positional arguments to the model
            **kwargs: Keyword arguments to the model
            
        Returns:
            Any: Model outputs
        """
        # Create inference schedule for execution
        batch_size = args[0].size(0) if args else kwargs.get("input_ids", kwargs.get("inputs", None)).size(0)
        seq_len = args[0].size(1) if args else kwargs.get("input_ids", kwargs.get("inputs", None)).size(1)
        
        schedule = InferenceSchedule(
            self.config,
            self.model_info,
            batch_size,
            seq_len
        )
        
        # Handle tensor and sequence parallelism
        if self.config.tensor_parallel_size > 1 or self.config.sequence_parallel_size > 1:
            # Process inputs based on parallelism strategy
            processed_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor) and len(arg.shape) >= 2:
                    # Apply sequence parallelism to sequence dimension (typically dim=1)
                    if self.config.sequence_parallel_size > 1 and hasattr(self, "executor") and self.executor is not None:
                        if isinstance(self.executor, SequenceParallelExecutor) or isinstance(self.executor, HybridParallelExecutor):
                            arg = self.executor.sequence_parallel.split_sequence_dimension(arg, dim=1)
                processed_args.append(arg)
                
            processed_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                    # Apply sequence parallelism to sequence dimension
                    if self.config.sequence_parallel_size > 1 and hasattr(self, "executor") and self.executor is not None:
                        if isinstance(self.executor, SequenceParallelExecutor) or isinstance(self.executor, HybridParallelExecutor):
                            value = self.executor.sequence_parallel.split_sequence_dimension(value, dim=1)
                processed_kwargs[key] = value
            
            # Execute forward pass
            outputs = self.model(*processed_args, **processed_kwargs)
            
            # Gather outputs from tensor/sequence parallel execution
            if isinstance(outputs, torch.Tensor) and hasattr(self, "executor") and self.executor is not None:
                if isinstance(self.executor, TensorParallelExecutor) or isinstance(self.executor, HybridParallelExecutor):
                    if self.config.tensor_parallel_size > 1:
                        outputs = self.executor.tensor_parallel.all_reduce(outputs)
                        
                if isinstance(self.executor, SequenceParallelExecutor) or isinstance(self.executor, HybridParallelExecutor):
                    if self.config.sequence_parallel_size > 1:
                        outputs = self.executor.sequence_parallel.all_gather_sequence_dimension(outputs, dim=1)
                        
            return outputs
            
        elif self.config.pipeline_parallel_size > 1:
            # For pipeline parallelism, use the schedule
            return schedule.execute([self.model], {"inputs": args[0] if args else kwargs.get("input_ids", kwargs.get("inputs"))})
            
        else:
            # For data parallelism or single GPU, execute normally
            return self.model(*args, **kwargs)


class ParallelOrchestrator:
    """
    Orchestrates multi-dimensional parallelism strategies for model inference.
    
    This class manages the configuration, setup, and execution of parallel
    strategies across multiple GPUs, optimizing for performance and memory usage.
    """
    def __init__(self, config: ParallelConfig):
        """
        Initializes the parallelism orchestrator with the given configuration.
        
        Args:
            config: Configuration for parallel execution
        """
        self.config = config
        self.is_valid, error_message = config.validate_configuration()
        if not self.is_valid:
            raise ValueError(f"Invalid parallel configuration: {error_message}")
            
        # Initialize process groups
        self.process_groups = self.setup_process_groups()
        
    def setup_process_groups(self) -> Dict[str, dist.ProcessGroup]:
        """
        Sets up process groups for different parallelism dimensions.
        
        Returns:
            Dict[str, dist.ProcessGroup]: Process groups for each parallelism dimension
        """
        # Initialize distributed backend if not already initialized
        if not dist.is_initialized():
            # Get rank and world size from environment variables
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            
            # Initialize distributed backend
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size
            )
            
        return initialize_parallel_groups(self.config)
        
    def configure_model(self, model: nn.Module) -> nn.Module:
        """
        Configures a model for parallel execution.
        
        Args:
            model: PyTorch model to configure
            
        Returns:
            nn.Module: Configured model for parallel execution
        """
        return ModelParallelWrapper(model, self.config)
        
    def partition_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Partitions a batch of inputs for parallel execution.
        
        Args:
            batch: Batch of inputs
            
        Returns:
            Dict[str, torch.Tensor]: Partitioned batch
        """
        # Determine which dimension to partition based on parallelism strategy
        if self.config.data_parallel_size > 1:
            # Partition batch dimension for data parallelism
            dp_rank = dist.get_rank(self.process_groups["data"])
            partitioned_batch = {}
            
            batch_size = None
            for key, tensor in batch.items():
                if tensor.dim() > 0:
                    if batch_size is None:
                        batch_size = tensor.size(0)
                    
                    # Calculate partition indices
                    partition_size = batch_size // self.config.data_parallel_size
                    start_idx = dp_rank * partition_size
                    end_idx = start_idx + partition_size if dp_rank < self.config.data_parallel_size - 1 else batch_size
                    
                    # Partition tensor along batch dimension
                    partitioned_batch[key] = tensor[start_idx:end_idx]
                else:
                    # For scalar tensors, keep as is
                    partitioned_batch[key] = tensor
                    
            return partitioned_batch
            
        elif self.config.sequence_parallel_size > 1:
            # Partition sequence dimension for sequence parallelism
            sp_rank = dist.get_rank(self.process_groups["sequence"])
            partitioned_batch = {}
            
            for key, tensor in batch.items():
                if tensor.dim() > 1:  # Has sequence dimension
                    # Assume sequence dimension is dim=1 (common for [batch, seq_len, ...])
                    seq_len = tensor.size(1)
                    
                    # Calculate partition indices
                    partition_size = seq_len // self.config.sequence_parallel_size
                    start_idx = sp_rank * partition_size
                    end_idx = start_idx + partition_size if sp_rank < self.config.sequence_parallel_size - 1 else seq_len
                    
                    # Partition tensor along sequence dimension
                    partitioned_batch[key] = tensor.narrow(1, start_idx, end_idx - start_idx)
                else:
                    # For tensors without sequence dimension, keep as is
                    partitioned_batch[key] = tensor
                    
            return partitioned_batch
            
        else:
            # No partitioning needed
            return batch
        
    def get_optimal_config(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        max_memory: int
    ) -> ParallelConfig:
        """
        Gets the optimal parallelism configuration for the given model and constraints.
        
        Args:
            model: PyTorch model
            batch_size: Batch size for inference
            seq_len: Sequence length for inference
            max_memory: Maximum memory per GPU in bytes
            
        Returns:
            ParallelConfig: Optimal configuration
        """
        from .auto_config import AutoParallelConfig
        
        # Analyze model to get size information
        model_info = analyze_model_for_parallelism(model)
        
        # Set up constraints
        constraints = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "max_memory_per_gpu": max_memory,
            "throughput_target": None,  # Optional throughput target
            "latency_target": None,     # Optional latency target
        }
        
        # Create auto-config instance
        auto_config = AutoParallelConfig(model, constraints)
        
        # Search for optimal configuration
        return auto_config.search_optimal_config()
        
    def create_inference_schedule(self) -> InferenceSchedule:
        """
        Creates an inference execution schedule.
        
        Returns:
            InferenceSchedule: Execution schedule for inference
        """
        # This is a placeholder implementation.
        # In practice, would need model and input information to create a proper schedule.
        return InferenceSchedule(
            self.config,
            {"model_type": "unknown"},  # Placeholder model info
            batch_size=1,
            seq_len=1
        )
        
    def memory_usage_estimate(self) -> Dict[str, int]:
        """
        Estimates memory usage with the current configuration.
        
        Returns:
            Dict[str, int]: Estimated memory usage by category
        """
        # This is a placeholder implementation.
        # In practice, would need model information for accurate estimates.
        return {
            "model_params": 0,
            "activations": 0,
            "optimizer_states": 0,
            "gradients": 0,
            "total": 0
        }
        
    def throughput_estimate(self) -> float:
        """
        Estimates throughput with the current configuration.
        
        Returns:
            float: Estimated throughput in samples/second
        """
        # This is a placeholder implementation.
        # In practice, would use a performance model based on model and hardware.
        return 0.0