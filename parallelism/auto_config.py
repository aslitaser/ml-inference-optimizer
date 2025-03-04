import torch
import torch.nn as nn
import math
import itertools
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import gc

from .orchestrator import ParallelConfig
from .parallel_utils import (
    analyze_model_for_parallelism,
    estimate_memory_requirements,
    calculate_communication_overhead,
    validate_parallel_config,
    is_power_of_two
)


class ParallelismCostModel:
    """
    Cost model for estimating performance of different parallelism configurations.
    
    This class provides methods to estimate execution time, memory usage, and
    communication overhead for different parallel configurations, enabling
    selection of optimal strategies based on hardware constraints.
    
    Attributes:
        model_info: Dict containing model information from analyze_model_for_parallelism
        hardware_info: Dict containing hardware characteristics like GPU memory, interconnect speed
    """
    def __init__(
        self,
        model_info: Dict[str, Any],
        hardware_info: Dict[str, Any]
    ):
        """
        Initialize the parallelism cost model.
        
        Args:
            model_info: Dict containing model information
            hardware_info: Dict containing hardware characteristics
        """
        self.model_info = model_info
        self.hardware_info = hardware_info
        
        # Extract commonly used information
        self.num_params = model_info.get("num_parameters", 0)
        self.model_size_bytes = model_info.get("model_size", 0)
        self.hidden_size = self._get_hidden_size()
        
        # Hardware characteristics
        self.gpu_flops = hardware_info.get("gpu_flops", 0)  # FLOPS per GPU
        self.gpu_memory = hardware_info.get("gpu_memory", 0)  # Memory per GPU in bytes
        self.link_bandwidth = hardware_info.get("link_bandwidth", 0)  # Interconnect bandwidth in GB/s
        self.link_latency = hardware_info.get("link_latency", 0)  # Interconnect latency in microseconds
        
    def _get_hidden_size(self) -> int:
        """
        Extract or estimate the model's hidden size from model information.
        
        Returns:
            int: Estimated hidden size
        """
        # Try to get hidden size from parameter distribution
        param_dist = self.model_info.get("parameter_distribution", {})
        
        # Default value if we can't determine hidden size
        hidden_size = 1024  # Common default for medium-sized models
        
        # Try to estimate from parameter distribution
        linear_params = param_dist.get("Linear", 0)
        if linear_params > 0 and "layer_counts" in self.model_info:
            linear_count = self.model_info["layer_counts"].get("Linear", 1)
            if linear_count > 0:
                # Rough estimation assuming square weight matrices
                avg_linear_size = math.sqrt(linear_params / linear_count)
                hidden_size = max(hidden_size, int(avg_linear_size))
        
        return hidden_size
    
    def estimate_execution_time(
        self,
        config: ParallelConfig,
        batch_size: int,
        seq_len: int
    ) -> float:
        """
        Estimate execution time for a given parallel configuration.
        
        Args:
            config: Parallel configuration
            batch_size: Batch size for inference
            seq_len: Sequence length for inference
            
        Returns:
            float: Estimated execution time in milliseconds
        """
        # Computation time
        compute_time = self._estimate_compute_time(config, batch_size, seq_len)
        
        # Communication time
        comm_time = self.estimate_communication_time(config, batch_size, seq_len)
        
        # Pipeline bubbles and synchronization overhead
        pipeline_overhead = 0.0
        if config.pipeline_parallel_size > 1:
            # Pipeline bubbles add (pipeline_size - 1) * per_stage_time overhead
            per_stage_time = compute_time / config.pipeline_parallel_size
            pipeline_overhead = (config.pipeline_parallel_size - 1) * per_stage_time
        
        # Tensor/sequence parallelism synchronization overhead
        parallel_sync_overhead = 0.0
        if config.tensor_parallel_size > 1 or config.sequence_parallel_size > 1:
            # Synchronization overhead increases with number of parallel GPUs
            parallel_size = max(config.tensor_parallel_size, config.sequence_parallel_size)
            sync_overhead_factor = 0.05  # 5% overhead per parallel dimension
            parallel_sync_overhead = compute_time * sync_overhead_factor * math.log2(parallel_size)
        
        # Total execution time
        total_time = compute_time + comm_time + pipeline_overhead + parallel_sync_overhead
        
        # Apply efficiency factor based on parallelism strategy
        # More complex strategies typically have lower efficiency
        efficiency_factor = self._calculate_efficiency_factor(config)
        
        return total_time / efficiency_factor
    
    def _estimate_compute_time(
        self,
        config: ParallelConfig,
        batch_size: int,
        seq_len: int
    ) -> float:
        """
        Estimate computation time for a given configuration.
        
        Args:
            config: Parallel configuration
            batch_size: Batch size for inference
            seq_len: Sequence length for inference
            
        Returns:
            float: Estimated computation time in milliseconds
        """
        # Calculate effective batch size per GPU
        effective_batch_size = batch_size / config.data_parallel_size
        
        # Calculate effective sequence length when using sequence parallelism
        effective_seq_len = seq_len
        if config.sequence_parallel_size > 1:
            effective_seq_len = seq_len / config.sequence_parallel_size
        
        # Estimate FLOPs for transformer computation
        # FLOPs per token ≈ 6 * hidden_size^2 for an attention+MLP block
        num_layers = len(self.model_info.get("attention_blocks", []))
        if num_layers == 0:
            # Fallback if we couldn't determine number of layers
            num_layers = self.num_params // (12 * self.hidden_size * self.hidden_size)
            num_layers = max(1, num_layers)  # Ensure at least one layer
        
        flops_per_token = 6 * self.hidden_size * self.hidden_size
        total_flops = effective_batch_size * effective_seq_len * flops_per_token * num_layers
        
        # Adjust FLOPs based on tensor parallelism
        if config.tensor_parallel_size > 1:
            # Tensor parallelism divides computation in attention/MLP
            total_flops /= config.tensor_parallel_size
        
        # Convert FLOPs to time based on GPU performance
        if self.gpu_flops > 0:
            compute_time_ms = (total_flops / self.gpu_flops) * 1000
        else:
            # Fallback if GPU FLOPS not provided
            compute_time_ms = total_flops / 1e9  # Assume 1 TFLOPS as default
        
        return compute_time_ms
    
    def _calculate_efficiency_factor(self, config: ParallelConfig) -> float:
        """
        Calculate efficiency factor for a given configuration.
        
        Args:
            config: Parallel configuration
            
        Returns:
            float: Efficiency factor (0.0-1.0)
        """
        # Base efficiency starts at 1.0 (100% efficient)
        efficiency = 1.0
        
        # Efficiency decreases with more complex parallelism strategies
        if config.tensor_parallel_size > 1:
            # Tensor parallelism has communication overhead
            efficiency *= 0.9  # 10% efficiency loss
            
            # Additional loss for larger tensor parallel sizes
            if config.tensor_parallel_size > 2:
                efficiency *= 0.95  # 5% additional loss
                
        if config.sequence_parallel_size > 1:
            # Sequence parallelism has communication overhead
            efficiency *= 0.9  # 10% efficiency loss
            
            # Additional loss for larger sequence parallel sizes
            if config.sequence_parallel_size > 2:
                efficiency *= 0.95  # 5% additional loss
                
        if config.pipeline_parallel_size > 1:
            # Pipeline parallelism has bubble overhead
            efficiency *= 0.85  # 15% efficiency loss
            
            # Additional loss for larger pipeline parallel sizes
            if config.pipeline_parallel_size > 2:
                efficiency *= 0.9  # 10% additional loss
                
        # Efficiency loss for using multiple parallelism strategies together
        num_strategies = sum([
            config.tensor_parallel_size > 1,
            config.sequence_parallel_size > 1,
            config.pipeline_parallel_size > 1,
            config.data_parallel_size > 1
        ])
        
        if num_strategies > 1:
            # Combined strategies have coordination overhead
            efficiency *= (1.0 - 0.05 * (num_strategies - 1))  # 5% loss per additional strategy
            
        return efficiency
    
    def estimate_memory_usage(
        self,
        config: ParallelConfig,
        batch_size: int,
        seq_len: int
    ) -> Dict[str, int]:
        """
        Estimate memory usage for a given parallel configuration.
        
        Args:
            config: Parallel configuration
            batch_size: Batch size for inference
            seq_len: Sequence length for inference
            
        Returns:
            Dict containing memory usage estimates by category
        """
        # Start with model parameters memory
        model_params_memory = self.model_size_bytes
        
        # Adjust for tensor parallelism
        if config.tensor_parallel_size > 1:
            # Tensor parallelism divides parameters across GPUs
            model_params_memory /= config.tensor_parallel_size
        
        # Calculate activation memory
        # Activation memory scales with batch size and sequence length
        activation_size_per_token = self.hidden_size * 4  # float32 = 4 bytes
        effective_batch_size = batch_size / config.data_parallel_size
        effective_seq_len = seq_len
        
        if config.sequence_parallel_size > 1:
            # Sequence parallelism divides sequence dimension
            effective_seq_len /= config.sequence_parallel_size
        
        # Estimate number of layers that store activations
        num_layers = len(self.model_info.get("attention_blocks", []))
        if num_layers == 0:
            # Fallback estimate based on model size
            num_layers = self.num_params // (12 * self.hidden_size * self.hidden_size)
            num_layers = max(1, num_layers)  # Ensure at least one layer
        
        # Calculate activation memory
        activation_memory = effective_batch_size * effective_seq_len * activation_size_per_token * num_layers
        
        # Adjust for activation checkpointing
        if config.activation_checkpointing:
            # Activation checkpointing trades computation for memory
            # Typically saves ~50% of activation memory
            activation_memory *= 0.5
        
        # Estimate KV cache for transformer models
        kv_cache_memory = 0
        if len(self.model_info.get("attention_blocks", [])) > 0:
            # Estimate number of attention heads
            num_heads = self.hidden_size // 64  # Common head dimension is 64
            head_dim = self.hidden_size // num_heads
            
            # KV cache size (keys and values for each token, each layer)
            # Each key and value vector has size head_dim per head
            kv_cache_memory = effective_batch_size * seq_len * num_layers * num_heads * head_dim * 2 * 4  # float32 = 4 bytes
            
            # Adjust for tensor parallelism
            if config.tensor_parallel_size > 1:
                # Tensor parallelism divides attention heads
                kv_cache_memory /= config.tensor_parallel_size
        
        # Add buffer for optimizer states if training (not applicable for inference)
        optimizer_memory = 0
        
        # Calculate workspace memory (CUDA kernels, temporary buffers)
        # This is a rough estimate based on empirical observations
        workspace_memory = int(0.1 * (model_params_memory + activation_memory + kv_cache_memory))
        
        # Add memory for pipeline parallel buffers if applicable
        pipeline_buffer_memory = 0
        if config.pipeline_parallel_size > 1:
            # Need to store micro-batch outputs at pipeline boundaries
            micro_batch_size = effective_batch_size / config.pipeline_parallel_size
            # Each pipeline stage needs buffer for receiving/sending activations
            pipeline_buffer_memory = micro_batch_size * effective_seq_len * self.hidden_size * 4 * 2  # float32 = 4 bytes, *2 for input/output
        
        # Total memory usage
        total_memory = (
            model_params_memory +
            activation_memory +
            kv_cache_memory +
            optimizer_memory +
            workspace_memory +
            pipeline_buffer_memory
        )
        
        return {
            "model_params": model_params_memory,
            "activations": activation_memory,
            "kv_cache": kv_cache_memory,
            "optimizer_states": optimizer_memory,
            "workspace": workspace_memory,
            "pipeline_buffers": pipeline_buffer_memory,
            "total": total_memory
        }
    
    def estimate_communication_time(
        self,
        config: ParallelConfig,
        batch_size: int,
        seq_len: int
    ) -> float:
        """
        Estimate communication time for a given parallel configuration.
        
        Args:
            config: Parallel configuration
            batch_size: Batch size for inference
            seq_len: Sequence length for inference
            
        Returns:
            float: Estimated communication time in milliseconds
        """
        # Calculate communication volume for different parallelism strategies
        comm_volume = 0.0
        
        # Tensor parallel communication
        if config.tensor_parallel_size > 1:
            # All-reduce operations in tensor parallelism
            # Communication happens after each parallel linear layer
            num_tp_ops = len(self.model_info.get("tensor_parallel_modules", []))
            if num_tp_ops == 0:
                # Fallback estimate based on model architecture
                num_tp_ops = len(self.model_info.get("attention_blocks", [])) * 4  # Q, K, V, output projections
                num_tp_ops += len(self.model_info.get("mlp_blocks", [])) * 2  # Two linear layers per MLP
            
            effective_batch_size = batch_size / config.data_parallel_size
            effective_seq_len = seq_len
            if config.sequence_parallel_size > 1:
                effective_seq_len /= config.sequence_parallel_size
            
            # Size of tensor to all-reduce per operation
            tensor_size = effective_batch_size * effective_seq_len * self.hidden_size * 4  # float32 = 4 bytes
            
            # All-reduce communication volume is 2(N-1)/N times tensor size
            tp_comm_factor = 2 * (config.tensor_parallel_size - 1) / config.tensor_parallel_size
            tp_comm_volume = num_tp_ops * tensor_size * tp_comm_factor
            
            comm_volume += tp_comm_volume
        
        # Sequence parallel communication
        if config.sequence_parallel_size > 1:
            # All-gather operations in sequence parallelism
            # Communication happens at the start/end of sequence parallel regions
            num_sp_ops = len(self.model_info.get("sequence_parallel_modules", []))
            if num_sp_ops == 0:
                # Fallback estimate based on model architecture
                num_sp_ops = len(self.model_info.get("attention_blocks", [])) * 2  # Input and output of attention
            
            effective_batch_size = batch_size / config.data_parallel_size
            
            # Size of tensor to all-gather per operation
            tensor_size = effective_batch_size * (seq_len / config.sequence_parallel_size) * self.hidden_size * 4  # float32 = 4 bytes
            
            # All-gather communication volume
            sp_comm_volume = num_sp_ops * tensor_size * config.sequence_parallel_size
            
            comm_volume += sp_comm_volume
        
        # Pipeline parallel communication
        if config.pipeline_parallel_size > 1:
            # Point-to-point communication in pipeline parallelism
            # Communication happens between pipeline stages for activation passing
            num_pipeline_boundaries = config.pipeline_parallel_size - 1
            
            # Micro-batch size for pipeline parallelism
            micro_batch_size = (batch_size / config.data_parallel_size) / config.pipeline_parallel_size
            
            # Size of activations to communicate
            activation_size = micro_batch_size * (seq_len / config.sequence_parallel_size) * self.hidden_size * 4  # float32 = 4 bytes
            
            # Number of micro-batches to process
            num_micro_batches = config.pipeline_parallel_size  # Minimum number for pipeline
            
            # Pipeline communication volume
            pp_comm_volume = num_pipeline_boundaries * activation_size * num_micro_batches * 2  # *2 for forward and backward
            
            comm_volume += pp_comm_volume
        
        # Data parallel communication (minimal for inference)
        if config.data_parallel_size > 1:
            # Broadcast inputs at start of inference
            input_size = batch_size * seq_len * self.hidden_size * 4  # float32 = 4 bytes
            dp_comm_volume = input_size
            
            comm_volume += dp_comm_volume
        
        # Convert communication volume to time
        # Communication time = latency + volume / bandwidth
        comm_time_ms = 0.0
        
        if comm_volume > 0:
            # Link latency in microseconds
            latency_factor = 0.0
            
            # Tensor parallel operations typically involve collective operations
            if config.tensor_parallel_size > 1:
                latency_factor += math.log2(config.tensor_parallel_size) * num_tp_ops
            
            # Sequence parallel operations also involve collective operations
            if config.sequence_parallel_size > 1:
                latency_factor += math.log2(config.sequence_parallel_size) * num_sp_ops
            
            # Pipeline parallel operations involve point-to-point communication
            if config.pipeline_parallel_size > 1:
                latency_factor += num_pipeline_boundaries * num_micro_batches
            
            # Total latency contribution (microseconds to milliseconds)
            latency_ms = latency_factor * self.link_latency / 1000.0
            
            # Bandwidth contribution (bytes / (GB/s) -> milliseconds)
            if self.link_bandwidth > 0:
                bandwidth_ms = comm_volume / (self.link_bandwidth * 1e9) * 1000
            else:
                # Fallback if bandwidth not provided (assume 50GB/s)
                bandwidth_ms = comm_volume / (50 * 1e9) * 1000
            
            comm_time_ms = latency_ms + bandwidth_ms
            
            # Apply efficiency factor for communication
            # Communication efficiency decreases with more GPUs
            comm_efficiency = 0.8  # 80% efficiency
            if config.tensor_parallel_size > 2 or config.sequence_parallel_size > 2:
                comm_efficiency *= 0.9  # Additional 10% loss for larger sizes
                
            comm_time_ms /= comm_efficiency
        
        return comm_time_ms
    
    def score_configuration(
        self,
        config: ParallelConfig,
        batch_size: int,
        seq_len: int
    ) -> float:
        """
        Score a parallel configuration based on execution time and memory usage.
        
        Args:
            config: Parallel configuration
            batch_size: Batch size for inference
            seq_len: Sequence length for inference
            
        Returns:
            float: Score for the configuration (higher is better)
        """
        # Estimate execution time
        execution_time = self.estimate_execution_time(config, batch_size, seq_len)
        
        # Estimate memory usage
        memory_usage = self.estimate_memory_usage(config, batch_size, seq_len)
        
        # Get peak memory per GPU
        peak_memory = memory_usage["total"]
        
        # Check if memory fits on GPU
        if peak_memory > self.gpu_memory:
            # Configuration exceeds GPU memory
            return 0.0
        
        # Score based on execution time (lower is better)
        time_score = 1000.0 / (execution_time + 1.0)  # +1 to avoid division by zero
        
        # Score based on memory utilization (higher utilization is better, up to a threshold)
        memory_utilization = peak_memory / self.gpu_memory
        
        # Optimal memory utilization is around 80%
        if memory_utilization > 0.9:
            # Penalize configurations that use >90% of GPU memory
            memory_score = 0.5 * (1.0 - (memory_utilization - 0.9) * 10)  # Linear penalty
        elif memory_utilization < 0.3:
            # Penalize configurations that use <30% of GPU memory
            memory_score = 0.5 + 0.5 * (memory_utilization / 0.3)  # Linear scaling
        else:
            # Optimal range: 30-90% utilization
            memory_score = 1.0
        
        # Calculate overall score (weighted average)
        # Execution time is more important than memory utilization
        overall_score = 0.7 * time_score + 0.3 * memory_score
        
        return overall_score


class AutoParallelConfig:
    """
    Automatic discovery of optimal parallel configurations.
    
    This class analyzes model characteristics and hardware constraints
    to automatically find the optimal parallel configuration for inference.
    
    Attributes:
        model: PyTorch model
        constraints: Dict containing constraints like max_memory, throughput_target
        model_info: Dict containing model analysis information
    """
    def __init__(self, model: nn.Module, constraints: Dict[str, Any]):
        """
        Initialize the automatic configuration finder.
        
        Args:
            model: PyTorch model to optimize
            constraints: Dict containing constraints
        """
        self.model = model
        self.constraints = constraints
        
        # Analyze model to get parallel-friendly modules
        self.model_info = analyze_model_for_parallelism(model)
        
        # Extract constraints
        self.batch_size = constraints.get("batch_size", 1)
        self.seq_len = constraints.get("seq_len", 512)
        self.max_memory = constraints.get("max_memory_per_gpu", 16 * 1024 * 1024 * 1024)  # Default 16GB
        self.throughput_target = constraints.get("throughput_target", None)
        self.latency_target = constraints.get("latency_target", None)
        
        # Hardware information
        self.hardware_info = self._get_hardware_info()
        
        # Create cost model
        self.cost_model = ParallelismCostModel(self.model_info, self.hardware_info)
        
    def _get_hardware_info(self) -> Dict[str, Any]:
        """
        Get hardware information for the current system.
        
        Returns:
            Dict containing hardware information
        """
        hardware_info = {}
        
        # Try to get GPU memory
        if torch.cuda.is_available():
            # Get total memory of the first GPU (assuming homogeneous GPUs)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            hardware_info["gpu_memory"] = gpu_memory
            
            # Estimate GPU FLOPs (very rough estimate)
            # This would be better obtained from device properties or a lookup table
            compute_capability = torch.cuda.get_device_capability(0)
            if compute_capability[0] >= 8:  # Ampere or newer
                hardware_info["gpu_flops"] = 19.5e12  # ~19.5 TFLOPS for A100
            elif compute_capability[0] >= 7:  # Volta/Turing
                hardware_info["gpu_flops"] = 14e12  # ~14 TFLOPS for V100
            else:  # Older architectures
                hardware_info["gpu_flops"] = 10e12  # Conservative estimate
        else:
            # Fallback values if no GPU is available
            hardware_info["gpu_memory"] = self.max_memory
            hardware_info["gpu_flops"] = 10e12  # 10 TFLOPS as default
        
        # Estimate interconnect bandwidth
        # Default to reasonable values, ideally this would be measured or provided
        hardware_info["link_bandwidth"] = 50  # 50 GB/s (NVLink-ish)
        hardware_info["link_latency"] = 5  # 5 microseconds
        
        return hardware_info
    
    def search_optimal_config(self) -> ParallelConfig:
        """
        Search for the optimal parallel configuration.
        
        Returns:
            ParallelConfig: Optimal configuration
        """
        # Get the search space of possible configurations
        configs = self.get_configuration_search_space()
        
        # Score each configuration
        scores = []
        valid_configs = []
        
        for config in configs:
            # Check if the configuration is valid
            is_valid, _ = validate_parallel_config(config, config.world_size)
            
            if is_valid:
                # Check if the configuration is compatible with hardware
                if self._validate_hardware_compatibility(config):
                    # Score the configuration
                    score = self.cost_model.score_configuration(
                        config, self.batch_size, self.seq_len
                    )
                    
                    if score > 0:  # Score of 0 means configuration doesn't fit in memory
                        scores.append(score)
                        valid_configs.append(config)
        
        if not valid_configs:
            # No valid configuration found, return default
            logging.warning("No valid parallel configuration found, using default single-GPU config")
            return ParallelConfig(world_size=1)
        
        # Find the configuration with the highest score
        best_idx = scores.index(max(scores))
        return valid_configs[best_idx]
    
    def _validate_hardware_compatibility(self, config: ParallelConfig) -> bool:
        """
        Validate that a configuration is compatible with the hardware.
        
        Args:
            config: Parallel configuration
            
        Returns:
            bool: Whether the configuration is compatible
        """
        # Check if we have enough GPUs
        if not torch.cuda.is_available():
            # No GPUs available, only allow single-GPU configuration
            return config.world_size == 1
        
        # Check if we have enough GPUs
        num_gpus = torch.cuda.device_count()
        if config.world_size > num_gpus:
            return False
        
        # Estimate memory usage
        memory_usage = self.cost_model.estimate_memory_usage(
            config, self.batch_size, self.seq_len
        )
        
        # Check if memory fits on GPU
        if memory_usage["total"] > self.hardware_info["gpu_memory"]:
            return False
        
        # Check tensor and sequence parallel hardware requirements
        if config.tensor_parallel_size > 1 or config.sequence_parallel_size > 1:
            # These strategies benefit from fast GPU interconnect
            # Ideally, we would check for NVLink or similar
            # Here we just accept it as compatible
            pass
        
        return True
    
    def get_configuration_search_space(self) -> List[ParallelConfig]:
        """
        Generate the search space of possible configurations.
        
        Returns:
            List[ParallelConfig]: List of possible configurations
        """
        # Determine maximum world size based on constraints
        model_requirements = self.analyze_model_requirements()
        
        max_world_size = 8  # Default maximum
        if torch.cuda.is_available():
            max_world_size = min(torch.cuda.device_count(), 8)
        
        # For very large models, need more GPUs
        if model_requirements["min_gpus"] > 1:
            max_world_size = max(max_world_size, model_requirements["min_gpus"])
        
        # Generate all valid combinations of parallel sizes
        configs = []
        
        # Iterate over possible world sizes
        for world_size in range(1, max_world_size + 1):
            # Get all valid factorizations of world_size
            valid_factorizations = self._get_valid_factorizations(world_size)
            
            for tp_size, sp_size, pp_size, dp_size in valid_factorizations:
                # Create configuration
                config = ParallelConfig(
                    world_size=world_size,
                    tensor_parallel_size=tp_size,
                    sequence_parallel_size=sp_size,
                    pipeline_parallel_size=pp_size,
                    data_parallel_size=dp_size,
                    # Use model requirements to set other parameters
                    activation_checkpointing=model_requirements["activation_checkpointing"],
                    optimize_memory=True,
                    overlap_communication=True,
                    communication_dtype=torch.float16
                )
                
                configs.append(config)
        
        return configs
    
    def _get_valid_factorizations(self, world_size: int) -> List[Tuple[int, int, int, int]]:
        """
        Get all valid factorizations of world_size into parallel dimensions.
        
        Args:
            world_size: Total number of GPUs
            
        Returns:
            List of tuples (tp_size, sp_size, pp_size, dp_size)
        """
        # Extract model requirements
        model_requirements = self.analyze_model_requirements()
        min_tensor_parallel = model_requirements.get("min_tensor_parallel", 1)
        max_tensor_parallel = model_requirements.get("max_tensor_parallel", world_size)
        min_sequence_parallel = model_requirements.get("min_sequence_parallel", 1)
        max_sequence_parallel = model_requirements.get("max_sequence_parallel", world_size)
        min_pipeline_parallel = model_requirements.get("min_pipeline_parallel", 1)
        max_pipeline_parallel = model_requirements.get("max_pipeline_parallel", world_size)
        
        # Get all factors of world_size
        factors = []
        for i in range(1, int(math.sqrt(world_size)) + 1):
            if world_size % i == 0:
                factors.append(i)
                if i != world_size // i:
                    factors.append(world_size // i)
        
        factors.sort()
        
        # Generate valid combinations
        valid_combinations = []
        
        # Filter factors for each parallel dimension
        tp_factors = [f for f in factors if min_tensor_parallel <= f <= max_tensor_parallel]
        sp_factors = [f for f in factors if min_sequence_parallel <= f <= max_sequence_parallel]
        pp_factors = [f for f in factors if min_pipeline_parallel <= f <= max_pipeline_parallel]
        
        # Prefer power-of-two sizes for tensor and sequence parallelism
        tp_factors = sorted(tp_factors, key=lambda x: (not is_power_of_two(x), x))
        sp_factors = sorted(sp_factors, key=lambda x: (not is_power_of_two(x), x))
        
        # Try all combinations
        for tp_size in tp_factors:
            for sp_size in sp_factors:
                for pp_size in pp_factors:
                    # Check if the combination is valid
                    if world_size % (tp_size * sp_size * pp_size) == 0:
                        # Calculate data parallel size
                        dp_size = world_size // (tp_size * sp_size * pp_size)
                        
                        # Add to valid combinations
                        valid_combinations.append((tp_size, sp_size, pp_size, dp_size))
        
        return valid_combinations
    
    def analyze_model_requirements(self) -> Dict[str, Any]:
        """
        Analyze model requirements for different parallelism strategies.
        
        Returns:
            Dict containing parallelism requirements
        """
        requirements = {
            "min_gpus": 1,
            "min_tensor_parallel": 1,
            "max_tensor_parallel": 8,
            "min_sequence_parallel": 1,
            "max_sequence_parallel": 8,
            "min_pipeline_parallel": 1,
            "max_pipeline_parallel": 8,
            "activation_checkpointing": False
        }
        
        # Get model size and memory requirements
        model_size_bytes = self.model_info["model_size"]
        num_params = self.model_info["num_parameters"]
        
        # Estimate memory required for single-GPU execution
        memory_estimate = estimate_memory_requirements(
            self.model, self.batch_size, self.seq_len
        )
        
        # Check if model fits on a single GPU
        if memory_estimate["total"] > self.max_memory:
            # Model doesn't fit on a single GPU, require model parallelism
            requirements["min_gpus"] = math.ceil(memory_estimate["total"] / self.max_memory)
            
            # Decide which parallelism strategy is most appropriate
            if memory_estimate["model_params"] > 0.7 * memory_estimate["total"]:
                # Model parameters dominate memory usage, tensor parallelism is effective
                requirements["min_tensor_parallel"] = max(2, requirements["min_gpus"] // 2)
            elif memory_estimate["activations"] > 0.5 * memory_estimate["total"]:
                # Activations dominate memory usage, activation checkpointing and pipeline/sequence parallelism
                requirements["activation_checkpointing"] = True
                if self.seq_len > 1024:
                    # Long sequences benefit from sequence parallelism
                    requirements["min_sequence_parallel"] = max(2, requirements["min_gpus"] // 4)
            elif memory_estimate["kv_cache"] > 0.3 * memory_estimate["total"]:
                # KV cache dominates memory usage
                if self.batch_size > 1:
                    # For batch inference, pipeline parallelism is effective
                    requirements["min_pipeline_parallel"] = max(2, requirements["min_gpus"] // 2)
                else:
                    # For single-sample inference, tensor parallelism is effective
                    requirements["min_tensor_parallel"] = max(2, requirements["min_gpus"] // 2)
        
        # Set maximum parallelism sizes based on model architecture
        
        # Tensor parallelism requires model structure that can be easily sharded
        tp_friendly = len(self.model_info["tensor_parallel_modules"]) > 0
        if not tp_friendly:
            requirements["max_tensor_parallel"] = 1
            
        # Sequence parallelism requires attention-based models
        sp_friendly = len(self.model_info["sequence_parallel_modules"]) > 0
        if not sp_friendly:
            requirements["max_sequence_parallel"] = 1
            
        # Pipeline parallelism requires model with sequential layers
        pp_friendly = len(self.model_info["pipeline_stages"]) > 0 or hasattr(self.model, "layers") or hasattr(self.model, "blocks")
        if not pp_friendly:
            requirements["max_pipeline_parallel"] = 1
            
        # Very large models (>10B parameters) benefit from activation checkpointing
        if num_params > 10_000_000_000:
            requirements["activation_checkpointing"] = True
            
        return requirements