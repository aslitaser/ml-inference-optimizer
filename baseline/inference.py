"""
Module for running inference on ML models with benchmarking capabilities.
Includes optimizations for transformer models such as kernel fusion,
FlashAttention, quantization, KV caching, and CUDA graphs.
"""

import time
import gc
import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Union, Callable

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


class FusionPattern:
    """Class representing a pattern of operations that can be fused."""
    
    def __init__(self, 
                 name: str, 
                 pattern: List[Union[type, Tuple[type, ...]]], 
                 fusion_fn: Callable, 
                 description: Optional[str] = None):
        """
        Initialize a fusion pattern.
        
        Args:
            name: Unique name for the fusion pattern
            pattern: List of module types that form the pattern
            fusion_fn: Function that takes a list of modules and returns a fused module
            description: Optional description of what the fusion does
        """
        self.name = name
        self.pattern = pattern
        self.fusion_fn = fusion_fn
        self.description = description or f"Fuses {[p.__name__ for p in pattern]}"
        
    def match(self, modules: List[nn.Module]) -> bool:
        """
        Check if a sequence of modules matches this pattern.
        
        Args:
            modules: List of modules to check against the pattern
            
        Returns:
            Whether the modules match the pattern
        """
        if len(modules) != len(self.pattern):
            return False
            
        return all(isinstance(module, pattern) for module, pattern in zip(modules, self.pattern))
        
    def fuse(self, modules: List[nn.Module]) -> nn.Module:
        """
        Apply the fusion function to the modules.
        
        Args:
            modules: List of modules to fuse
            
        Returns:
            Fused module
        """
        return self.fusion_fn(modules)


class FusionRegistry:
    """Registry of fusion patterns for optimizing models."""
    
    def __init__(self):
        """Initialize the fusion registry."""
        self.patterns: List[FusionPattern] = []
        
    def register_pattern(self, pattern: FusionPattern) -> None:
        """
        Register a fusion pattern.
        
        Args:
            pattern: FusionPattern to register
        """
        self.patterns.append(pattern)
        
    def find_matching_pattern(self, modules: List[nn.Module]) -> Optional[FusionPattern]:
        """
        Find a pattern that matches a sequence of modules.
        
        Args:
            modules: List of modules to match
            
        Returns:
            Matching pattern or None if no match found
        """
        for pattern in self.patterns:
            if pattern.match(modules):
                return pattern
        return None
        
    def fuse_modules(self, model: nn.Module, inplace: bool = False) -> nn.Module:
        """
        Apply fusion patterns to a model.
        
        Args:
            model: PyTorch model to optimize
            inplace: Whether to modify the model in-place
            
        Returns:
            Optimized model with fused modules
        """
        if not inplace:
            model = deepcopy(model)
            
        # Find sequences of modules that can be fused
        fusion_candidates = self._find_fusion_candidates(model)
        
        # Apply fusions (in reverse order to preserve indices)
        for parent_name, parent_module, seq_start, modules in reversed(fusion_candidates):
            # Get matching pattern
            pattern = self.find_matching_pattern(modules)
            if pattern is None:
                continue
                
            # Get child names to be replaced
            child_names = list(dict(parent_module.named_children()).keys())[seq_start:seq_start+len(modules)]
            
            # Create fused module
            fused_module = pattern.fuse(modules)
            
            # Replace sequential modules with fused module
            self._replace_modules(parent_module, child_names, fused_module)
            
        return model
        
    def _find_fusion_candidates(self, model: nn.Module) -> List[Tuple[str, nn.Module, int, List[nn.Module]]]:
        """
        Find candidates for fusion in a model.
        
        Args:
            model: PyTorch model to search
            
        Returns:
            List of (parent_name, parent_module, start_idx, modules) tuples
        """
        candidates = []
        
        # Iterate through all non-leaf modules
        for parent_name, parent_module in model.named_modules():
            if list(parent_module.children()):  # only process container modules
                # Get direct children
                children = list(parent_module.children())
                
                # Try all possible sequences
                max_pattern_len = max(len(p.pattern) for p in self.patterns)
                for i in range(len(children)):
                    for pattern_len in range(2, min(max_pattern_len + 1, len(children) - i + 1)):
                        seq = children[i:i+pattern_len]
                        if self.find_matching_pattern(seq):
                            candidates.append((parent_name, parent_module, i, seq))
        
        return candidates
        
    def _replace_modules(self, parent_module: nn.Module, child_names: List[str], new_module: nn.Module) -> None:
        """
        Replace a sequence of modules in a parent module with a new module.
        
        Args:
            parent_module: Parent module containing the modules to replace
            child_names: Names of children to replace
            new_module: New module to insert
        """
        if isinstance(parent_module, nn.Sequential):
            # For Sequential, create a new Sequential excluding the modules to replace
            new_seq = nn.Sequential()
            old_modules = list(parent_module.children())
            child_indices = [list(dict(parent_module.named_children()).keys()).index(name) for name in child_names]
            
            # Add modules before the fusion
            for i in range(min(child_indices)):
                new_seq.add_module(str(len(new_seq)), old_modules[i])
                
            # Add the fused module
            new_seq.add_module(str(len(new_seq)), new_module)
            
            # Add modules after the fusion
            for i in range(max(child_indices) + 1, len(old_modules)):
                new_seq.add_module(str(len(new_seq)), old_modules[i])
                
            # Replace the modules in the parent
            for key, value in new_seq.named_children():
                parent_module._modules[key] = value
            
            # Remove any unused modules
            for key in set(parent_module._modules.keys()) - set(new_seq._modules.keys()):
                del parent_module._modules[key]
        else:
            # For other module types, directly set the attributes
            # This works for transformer blocks where modules are attributes
            for name in child_names[1:]:
                if hasattr(parent_module, name):
                    delattr(parent_module, name)
            
            if hasattr(parent_module, child_names[0]):
                setattr(parent_module, child_names[0], new_module)


# Create global fusion registry
fusion_registry = FusionRegistry()


# Import common fusion implementations
from copy import deepcopy
try:
    from ..kernels.mlp.fused_mlp import FusedMLPGeluTanh, FusedMLPReLU, FusedMLPSwiGLU
    from ..kernels.attention.flash_attention import FlashAttentionLayer, FlashSelfAttention
    HAS_CUSTOM_KERNELS = True
    
    # Register common fusion patterns
    
    # Fused MLP (Linear + GELU + Linear)
    def fuse_mlp_gelu(modules: List[nn.Module]) -> nn.Module:
        linear1, gelu, linear2 = modules
        return FusedMLPGeluTanh(
            in_features=linear1.in_features,
            hidden_features=linear1.out_features,
            out_features=linear2.out_features,
            bias1=linear1.bias is not None,
            bias2=linear2.bias is not None
        ).copy_weights_from(linear1, linear2)
        
    fusion_registry.register_pattern(FusionPattern(
        name="linear_gelu_linear",
        pattern=[nn.Linear, nn.GELU, nn.Linear],
        fusion_fn=fuse_mlp_gelu,
        description="Fuses Linear + GELU + Linear into a single FusedMLPGeluTanh module"
    ))
    
    # Fused MLP (Linear + ReLU + Linear)
    def fuse_mlp_relu(modules: List[nn.Module]) -> nn.Module:
        linear1, relu, linear2 = modules
        return FusedMLPReLU(
            in_features=linear1.in_features,
            hidden_features=linear1.out_features,
            out_features=linear2.out_features,
            bias1=linear1.bias is not None,
            bias2=linear2.bias is not None
        ).copy_weights_from(linear1, linear2)
        
    fusion_registry.register_pattern(FusionPattern(
        name="linear_relu_linear",
        pattern=[nn.Linear, nn.ReLU, nn.Linear],
        fusion_fn=fuse_mlp_relu,
        description="Fuses Linear + ReLU + Linear into a single FusedMLPReLU module"
    ))
    
    # Fused Attention + Residual + LayerNorm
    def fuse_attention_residual_ln(modules: List[nn.Module]) -> nn.Module:
        # This would need custom implementation based on model architecture
        # This is a placeholder that would need to be customized
        attention, layernorm = modules
        return nn.Sequential(attention, layernorm)  # A proper implementation would create a truly fused module
    
    # Note: This pattern needs to be customized based on the model's specific attention implementation
    # fusion_registry.register_pattern(FusionPattern(
    #     name="attention_layernorm",
    #     pattern=[nn.MultiheadAttention, nn.LayerNorm],
    #     fusion_fn=fuse_attention_residual_ln,
    #     description="Fuses Attention + LayerNorm into a single module with fused ops"
    # ))
    
except ImportError:
    HAS_CUSTOM_KERNELS = False


# Flash Attention utility functions
def convert_to_flash_attention(model: nn.Module) -> nn.Module:
    """
    Convert standard attention modules to Flash Attention.
    
    Args:
        model: PyTorch model to convert
        
    Returns:
        Model with Flash Attention modules
    """
    if not HAS_CUSTOM_KERNELS:
        logging.warning("FlashAttention not available. No conversion performed.")
        return model
        
    try:
        # Try to import and use the ModelConverter from the kernels module
        from ..kernels.attention.flash_attention import ModelConverter
        converter = ModelConverter()
        return converter.convert(model)
    except (ImportError, AttributeError):
        # Fallback to manual conversion
        return _manual_flash_attention_conversion(model)
        
def _manual_flash_attention_conversion(model: nn.Module) -> nn.Module:
    """
    Manually convert attention modules to Flash Attention.
    Used as fallback when automatic conversion fails.
    
    Args:
        model: PyTorch model to convert
        
    Returns:
        Model with Flash Attention modules
    """
    # Deep copy to avoid modifying the original
    model = deepcopy(model)
    
    # Look for attention modules with MultiheadAttention
    for name, module in model.named_modules():
        # Check for PyTorch MultiheadAttention
        if isinstance(module, nn.MultiheadAttention):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                
                # Create Flash Attention replacement
                flash_attn = FlashSelfAttention(
                    embed_dim=module.embed_dim,
                    num_heads=module.num_heads,
                    dropout=module.dropout,
                    causal=True  # Assume causal for safety, would need to be detected in real impl
                )
                
                # Replace the module
                setattr(parent, child_name, flash_attn)
        
        # Check for HuggingFace-style attention
        elif (hasattr(module, 'query') and hasattr(module, 'key') and 
              hasattr(module, 'value') and hasattr(module, 'qk_bmm')):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                
                # Create Flash Attention replacement
                # For HuggingFace architectures, would need to detect parameters
                # This is a simplified placeholder
                try:
                    hidden_size = module.query.in_features
                    num_heads = getattr(module, 'num_heads', hidden_size // 64)
                    dropout = getattr(module, 'dropout', 0.0)
                    
                    flash_attn = FlashAttentionLayer(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        attention_dropout=dropout
                    )
                    
                    # Replace the module
                    setattr(parent, child_name, flash_attn)
                except (AttributeError, ValueError):
                    # Skip if we can't determine parameters
                    pass
    
    return model


class InferenceRunner(ABC):
    """Base class for model inference with performance metrics."""
    
    def __init__(self, model: nn.Module, device: str, precision: str = "fp16"):
        """
        Initialize the inference runner.
        
        Args:
            model: PyTorch model to run inference with
            device: Device to run inference on ('cuda', 'cpu')
            precision: Precision to use for inference ('fp32', 'fp16', 'bf16')
        """
        self.model = model
        self.device = device
        self.precision = precision
        
        # Store original dtype
        self.original_dtype = next(model.parameters()).dtype
        
        # Convert model to specified precision
        self._set_precision(precision)
        
        # Ensure model is on the right device
        if next(model.parameters()).device.type != device:
            self.model = self.model.to(device)
            
        # Inference metrics
        self.metrics: Dict[str, float] = {}
        
    def _set_precision(self, precision: str) -> None:
        """
        Set model precision.
        
        Args:
            precision: Precision to use ('fp32', 'fp16', 'bf16', 'int8', 'int4')
        """
        if precision == "fp32":
            dtype = torch.float32
            self.model = self.model.to(dtype=dtype)
        elif precision == "fp16":
            dtype = torch.float16
            self.model = self.model.to(dtype=dtype)
        elif precision == "bf16":
            dtype = torch.bfloat16
            self.model = self.model.to(dtype=dtype)
        elif precision == "int8":
            self._quantize_to_int8()
        elif precision == "int4":
            self._quantize_to_int4()
        else:
            raise ValueError(f"Unsupported precision: {precision}")
    
    def _quantize_to_int8(self) -> None:
        """
        Quantize model to INT8 precision using PyTorch quantization.
        """
        try:
            import torch.quantization
            from torch.quantization import quantize_dynamic
            from torch.quantization.quantize_fx import prepare_fx, convert_fx
            
            # Prepare for dynamic quantization (weights only)
            qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}
            
            # For more modern quantization with FX Graph
            if hasattr(torch.quantization, 'quantize_fx'):
                try:
                    # Check if FX graph mode quantization is applicable
                    prepared_model = prepare_fx(self.model, qconfig_dict)
                    self.model = convert_fx(prepared_model)
                    logging.info("Model quantized to INT8 using FX Graph mode")
                    return
                except Exception as e:
                    logging.warning(f"FX Graph quantization failed: {e}. Falling back to eager mode.")
            
            # Fallback to eager mode quantization
            quantizable_ops = {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LSTM, nn.GRU}
            dtype = torch.qint8
            
            self.model = quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d},  # Quantize these layer types
                dtype=dtype
            )
            logging.info("Model quantized to INT8 using eager mode")
            
        except (ImportError, RuntimeError, ValueError) as e:
            logging.warning(f"INT8 quantization failed: {e}. Falling back to FP16.")
            self.model = self.model.to(dtype=torch.float16)
            
    def _quantize_to_int4(self) -> None:
        """
        Quantize model to INT4 precision (weights only).
        This is more specialized and requires custom implementation.
        """
        try:
            # Check for bitsandbytes package for INT4 quantization
            import bitsandbytes as bnb
            
            # Convert eligible linear layers to 4-bit precision
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and module.weight.size(0) > 1:
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent = self.model
                        for part in parent_name.split('.'):
                            if not hasattr(parent, part):
                                break
                            parent = getattr(parent, part)
                        
                        # Create 4-bit replacement
                        in_features = module.in_features
                        out_features = module.out_features
                        bias = module.bias is not None
                        
                        # Use bitsandbytes 4-bit linear layer
                        quantized_module = bnb.nn.Linear4bit(
                            in_features, 
                            out_features,
                            bias=bias
                        )
                        
                        # Copy weights with quantization
                        quantized_module.weight = bnb.nn.Params4bit(
                            module.weight.data,
                            requires_grad=False,
                            quant_type="nf4"  # Can be fp4 or nf4
                        )
                        
                        if bias:
                            quantized_module.bias = nn.Parameter(module.bias.data)
                        
                        # Replace the module
                        setattr(parent, child_name, quantized_module)
            
            logging.info("Model quantized to INT4 (weights only)")
            
        except (ImportError, RuntimeError, ValueError) as e:
            logging.warning(f"INT4 quantization failed: {e}. Falling back to FP16.")
            self.model = self.model.to(dtype=torch.float16)
            
    def calibrate_for_quantization(self, calibration_data: List[Any], 
                                 max_samples: int = 100) -> None:
        """
        Calibrate the model for quantization using representative data.
        
        Args:
            calibration_data: List of sample inputs for calibration
            max_samples: Maximum number of samples to use
        """
        try:
            import torch.quantization
            from torch.quantization import prepare
            
            # For static quantization, we need a forward hook on activations
            calibration_samples = calibration_data[:max_samples]
            
            # Create quantization config
            qconfig = torch.quantization.get_default_qconfig('fbgemm')
            qconfig_dict = {'': qconfig}
            
            # Prepare model for calibration
            self.model.eval()
            self.model.qconfig = qconfig
            prepared_model = prepare(self.model, inplace=False)
            
            # Run calibration
            with torch.no_grad():
                for sample in calibration_samples:
                    _ = prepared_model(sample)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(prepared_model)
            self.model = quantized_model
            
            logging.info(f"Model calibrated and quantized using {len(calibration_samples)} samples")
            
        except (ImportError, RuntimeError, ValueError) as e:
            logging.warning(f"Calibration failed: {e}")
            
    def get_quantization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about quantized layers.
        
        Returns:
            Dictionary with quantization statistics
        """
        stats = {
            "total_layers": 0,
            "quantized_layers": 0,
            "quantized_parameters": 0,
            "total_parameters": 0,
            "compression_ratio": 0.0,
            "quantization_methods": set()
        }
        
        # Count parameters and layers
        for name, module in self.model.named_modules():
            is_leaf = len(list(module.children())) == 0
            if not is_leaf:
                continue
                
            stats["total_layers"] += 1
            param_count = sum(p.numel() for p in module.parameters() if p is not None)
            stats["total_parameters"] += param_count
            
            # Check for quantized modules
            if hasattr(module, 'weight_fake_quant'):
                stats["quantized_layers"] += 1
                stats["quantized_parameters"] += param_count
                stats["quantization_methods"].add("PyTorch Quantization")
            
            # Check bitsandbytes quantization
            elif "bitsandbytes" in module.__class__.__module__:
                stats["quantized_layers"] += 1
                stats["quantized_parameters"] += param_count
                stats["quantization_methods"].add("bitsandbytes")
            
            # Check other custom quantization methods
            elif any(x in module.__class__.__name__.lower() for x in ["int8", "int4", "quant", "4bit", "8bit"]):
                stats["quantized_layers"] += 1
                stats["quantized_parameters"] += param_count
                stats["quantization_methods"].add("Custom")
        
        # Calculate compression ratio
        if stats["total_parameters"] > 0:
            # Rough estimate based on precision difference
            fp32_size = stats["total_parameters"] * 4  # bytes per fp32 parameter
            quantized_size = stats["quantized_parameters"] * 1  # bytes per int8 parameter
            fp32_size_remaining = (stats["total_parameters"] - stats["quantized_parameters"]) * 4
            total_size = quantized_size + fp32_size_remaining
            stats["compression_ratio"] = fp32_size / total_size if total_size > 0 else 1.0
        
        stats["quantization_methods"] = list(stats["quantization_methods"])
        
        return stats
    
    def warmup(self, inputs: Any, iterations: int = 10) -> None:
        """
        Warm up the model with multiple iterations.
        
        Args:
            inputs: Sample inputs for warmup
            iterations: Number of warmup iterations
        """
        with torch.no_grad():
            self.model.eval()
            
            # Record initial memory
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            # Run warmup iterations
            for _ in range(iterations):
                self._forward(inputs)
                
            # Synchronize if using CUDA
            if self.device == "cuda":
                torch.cuda.synchronize()
    
    @abstractmethod
    def _forward(self, inputs: Any) -> Any:
        """
        Run forward pass on the model.
        
        Args:
            inputs: Model inputs
            
        Returns:
            Model outputs
        """
        pass
    
    def run_inference(self, inputs: Any, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """
        Run inference and collect performance metrics.
        
        Args:
            inputs: Model inputs
            **kwargs: Additional arguments for the model
            
        Returns:
            Tuple of (model outputs, performance metrics)
        """
        self.model.eval()
        metrics = {}
        
        # Clear GPU cache before inference
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        # Record memory before inference
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            metrics["memory_before_mb"] = memory_before
        
        # Set up CUDA events for timing
        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        # Start timing
        start_time = time.perf_counter()
        
        # Run inference
        with torch.no_grad():
            outputs = self._forward(inputs, **kwargs)
            
        # End timing
        end_time = time.perf_counter()
        
        # Record CUDA event
        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            cuda_time_ms = start_event.elapsed_time(end_event)
            metrics["cuda_time_ms"] = cuda_time_ms
        
        # Record total time
        total_time = (end_time - start_time) * 1000  # Convert to ms
        metrics["total_time_ms"] = total_time
        
        # Record memory after inference
        if self.device == "cuda":
            memory_after = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            metrics["memory_after_mb"] = memory_after
            metrics["peak_memory_mb"] = peak_memory
            metrics["memory_change_mb"] = memory_after - memory_before
        
        self.metrics = metrics
        return outputs, metrics
    
    def run_batch_inference(self, batch_inputs: List[Any], **kwargs) -> List[Tuple[Any, Dict[str, float]]]:
        """
        Run inference on a batch of inputs.
        
        Args:
            batch_inputs: List of model inputs
            **kwargs: Additional arguments for the model
            
        Returns:
            List of tuples, each containing (model outputs, performance metrics)
        """
        results = []
        batch_metrics = {"total_batch_time_ms": 0.0}
        
        batch_start_time = time.perf_counter()
        
        for inputs in batch_inputs:
            outputs, metrics = self.run_inference(inputs, **kwargs)
            results.append((outputs, metrics))
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key in batch_metrics:
                    batch_metrics[key] += value
                else:
                    batch_metrics[key] = value
        
        batch_end_time = time.perf_counter()
        batch_metrics["total_batch_time_ms"] = (batch_end_time - batch_start_time) * 1000
        batch_metrics["avg_inference_time_ms"] = batch_metrics["total_batch_time_ms"] / len(batch_inputs)
        
        return results
    
    def profile_model(self, inputs: Any, use_cuda: bool = True) -> Dict[str, Any]:
        """
        Profile the model to identify bottlenecks.
        
        Args:
            inputs: Sample inputs for profiling
            use_cuda: Whether to profile CUDA operations
            
        Returns:
            Dictionary containing profiling results
        """
        activities = []
        if use_cuda and self.device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        activities.append(ProfilerActivity.CPU)
        
        self.model.eval()
        
        # Run profiling
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("model_inference"):
                with torch.no_grad():
                    _ = self._forward(inputs)
        
        # Process results
        profile_results = {
            "table": prof.key_averages().table(sort_by="cuda_time_total" if use_cuda else "cpu_time_total", row_limit=20),
            "events": prof.events(),
            "key_averages": prof.key_averages(),
        }
        
        return profile_results
    
    def restore_original_precision(self) -> None:
        """Restore the model to its original precision."""
        self.model = self.model.to(dtype=self.original_dtype)


class KVCache:
    """
    Key-Value cache for efficient transformer inference.
    Stores attention key/value tensors to avoid recomputation.
    """
    
    def __init__(self, max_batch_size: int = 1, max_seq_len: int = 2048, 
                use_block_storage: bool = True, block_size: int = 64):
        """
        Initialize KV cache.
        
        Args:
            max_batch_size: Maximum batch size to support
            max_seq_len: Maximum sequence length to support
            use_block_storage: Whether to use block-based storage for memory efficiency
            block_size: Size of blocks for block-based storage
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.use_block_storage = use_block_storage
        self.block_size = block_size
        self.num_layers = 0
        self.num_heads = 0
        self.head_dim = 0
        
        # Cache storage
        self.k_caches = {}  # {layer_idx: tensor}
        self.v_caches = {}  # {layer_idx: tensor}
        
        # For block-based storage
        self.block_tables = {}  # {batch_idx: list of block_idx}
        self.next_block = 0
        
        # Allocation status
        self.is_initialized = False
        self.current_seq_lengths = [0] * max_batch_size
        
    def initialize(self, num_layers: int, num_heads: int, head_dim: int,
                  dtype: torch.dtype = torch.float16, device: str = "cuda") -> None:
        """
        Initialize cache tensors.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            dtype: Data type for cache tensors
            device: Device to store tensors on
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        if self.use_block_storage:
            # Initialize block-based cache
            total_blocks = math.ceil(self.max_batch_size * self.max_seq_len / self.block_size)
            
            # Initialize block tables
            for batch_idx in range(self.max_batch_size):
                self.block_tables[batch_idx] = []
            
            # Preallocate blocks for all layers
            for layer_idx in range(num_layers):
                # [num_blocks, block_size, num_heads, head_dim]
                self.k_caches[layer_idx] = torch.zeros(
                    (total_blocks, self.block_size, num_heads, head_dim),
                    dtype=dtype, device=device
                )
                self.v_caches[layer_idx] = torch.zeros(
                    (total_blocks, self.block_size, num_heads, head_dim),
                    dtype=dtype, device=device
                )
        else:
            # Initialize contiguous cache
            for layer_idx in range(num_layers):
                # [batch_size, max_seq_len, num_heads, head_dim]
                self.k_caches[layer_idx] = torch.zeros(
                    (self.max_batch_size, self.max_seq_len, num_heads, head_dim),
                    dtype=dtype, device=device
                )
                self.v_caches[layer_idx] = torch.zeros(
                    (self.max_batch_size, self.max_seq_len, num_heads, head_dim),
                    dtype=dtype, device=device
                )
        
        self.is_initialized = True
        
    def reset(self) -> None:
        """Reset the cache to empty state, keeping allocation."""
        if not self.is_initialized:
            return
            
        self.current_seq_lengths = [0] * self.max_batch_size
        
        if self.use_block_storage:
            # Reset block tables
            for batch_idx in range(self.max_batch_size):
                self.block_tables[batch_idx] = []
            self.next_block = 0
        
    def clear(self) -> None:
        """Clear the cache, freeing memory."""
        self.k_caches = {}
        self.v_caches = {}
        self.block_tables = {}
        self.next_block = 0
        self.current_seq_lengths = [0] * self.max_batch_size
        self.is_initialized = False
        
    def get_kv_cache(self, layer_idx: int, batch_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get key-value cache for a specific layer and batch item.
        
        Args:
            layer_idx: Layer index to get cache for
            batch_idx: Batch index to get cache for
            
        Returns:
            Tuple of (k_cache, v_cache) for the specific layer and batch
        """
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized. Call initialize() first.")
        
        if self.use_block_storage:
            # For block-based storage, gather blocks based on block table
            block_indices = self.block_tables[batch_idx]
            seq_len = self.current_seq_lengths[batch_idx]
            
            if not block_indices:
                # Empty cache
                return None, None
                
            # Gather blocks into contiguous tensor
            k_cache = torch.zeros(
                (seq_len, self.num_heads, self.head_dim),
                dtype=self.k_caches[layer_idx].dtype,
                device=self.k_caches[layer_idx].device
            )
            v_cache = torch.zeros(
                (seq_len, self.num_heads, self.head_dim),
                dtype=self.v_caches[layer_idx].dtype,
                device=self.v_caches[layer_idx].device
            )
            
            # Fill from blocks
            offset = 0
            for block_idx in block_indices:
                block_size = min(self.block_size, seq_len - offset)
                if block_size <= 0:
                    break
                    
                k_cache[offset:offset+block_size] = self.k_caches[layer_idx][block_idx, :block_size]
                v_cache[offset:offset+block_size] = self.v_caches[layer_idx][block_idx, :block_size]
                offset += block_size
                
            return k_cache, v_cache
        else:
            # For contiguous storage, simply return the slice
            seq_len = self.current_seq_lengths[batch_idx]
            return (
                self.k_caches[layer_idx][batch_idx, :seq_len],
                self.v_caches[layer_idx][batch_idx, :seq_len]
            )
    
    def append(self, layer_idx: int, batch_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Append new key-value pairs to the cache.
        
        Args:
            layer_idx: Layer index to update
            batch_idx: Batch index to update
            k: Key tensor to append, shape [seq_len, num_heads, head_dim]
            v: Value tensor to append, shape [seq_len, num_heads, head_dim]
        """
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized. Call initialize() first.")
            
        seq_len = k.size(0)
        current_len = self.current_seq_lengths[batch_idx]
        new_len = current_len + seq_len
        
        if new_len > self.max_seq_len:
            raise ValueError(f"Sequence length {new_len} exceeds maximum {self.max_seq_len}")
            
        if self.use_block_storage:
            # Compute how many blocks we need
            start_block = current_len // self.block_size
            end_block = (new_len - 1) // self.block_size
            blocks_needed = end_block - start_block + 1
            
            # Allocate new blocks if needed
            for _ in range(blocks_needed):
                if len(self.block_tables[batch_idx]) <= start_block:
                    self.block_tables[batch_idx].append(self.next_block)
                    self.next_block += 1
            
            # Fill blocks
            for i in range(seq_len):
                pos = current_len + i
                block_idx = pos // self.block_size
                block_pos = pos % self.block_size
                
                block_id = self.block_tables[batch_idx][block_idx]
                self.k_caches[layer_idx][block_id, block_pos] = k[i]
                self.v_caches[layer_idx][block_id, block_pos] = v[i]
        else:
            # Simply copy to the right position
            self.k_caches[layer_idx][batch_idx, current_len:new_len] = k
            self.v_caches[layer_idx][batch_idx, current_len:new_len] = v
            
        # Update sequence length
        self.current_seq_lengths[batch_idx] = new_len
        
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics for the cache.
        
        Returns:
            Dictionary with memory usage statistics (in MB)
        """
        if not self.is_initialized:
            return {"total_memory_mb": 0}
            
        k_cache_memory = sum(cache.element_size() * cache.nelement() 
                            for cache in self.k_caches.values())
        v_cache_memory = sum(cache.element_size() * cache.nelement() 
                            for cache in self.v_caches.values())
        
        total_memory = k_cache_memory + v_cache_memory
        
        # Calculate efficiency for block storage
        if self.use_block_storage:
            used_blocks = self.next_block
            total_blocks = sum(cache.size(0) for cache in self.k_caches.values())
            efficiency = used_blocks / total_blocks if total_blocks > 0 else 1.0
        else:
            # For contiguous storage, calculate based on used positions
            used_positions = sum(self.current_seq_lengths)
            total_positions = self.max_batch_size * self.max_seq_len * self.num_layers
            efficiency = used_positions / total_positions if total_positions > 0 else 1.0
            
        return {
            "k_cache_memory_mb": k_cache_memory / (1024 * 1024),
            "v_cache_memory_mb": v_cache_memory / (1024 * 1024),
            "total_memory_mb": total_memory / (1024 * 1024),
            "memory_efficiency": efficiency
        }


# Import our utility function
from .model_utils import add_paged_attention_to_model
import math # For ceil in BlockManager

# Add BlockManager, SequenceMetadata and PagedKVCache classes
class BlockManager:
    """
    Manages the allocation and freeing of physical memory blocks for the KV cache.
    Implements reference counting for block sharing.
    """
    def __init__(self, num_blocks: int, block_size: int, num_layers: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: str):
        """
        Initializes the BlockManager.

        Args:
            num_blocks: Total number of physical blocks available.
            block_size: Size of each block (number of tokens).
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            head_dim: Dimension of each attention head.
            dtype: Data type for cache tensors.
            device: Device to store tensors on.
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Pool of free block indices
        self.free_blocks = list(range(num_blocks))
        # Reference count for each block (block_idx -> count)
        self.ref_counts = torch.zeros(num_blocks, dtype=torch.int32, device=device)

        # Pre-allocate physical K/V cache blocks
        self.gpu_cache_k = torch.zeros(
            (num_blocks, num_layers, block_size, num_heads, head_dim),
            dtype=dtype, device=device
        )
        self.gpu_cache_v = torch.zeros(
            (num_blocks, num_layers, block_size, num_heads, head_dim),
            dtype=dtype, device=device
        )
        
        self.is_initialized = True
        logging.info(f"BlockManager initialized with {num_blocks} blocks.")


    def allocate_block(self) -> int:
        """Allocates a free physical block."""
        if not self.free_blocks:
            raise MemoryError("Out of memory: No free blocks available in KV cache.")
        block_idx = self.free_blocks.pop()
        self.ref_counts[block_idx] = 1 # Initial reference count
        # logging.debug(f"Allocated block {block_idx}. Free blocks remaining: {len(self.free_blocks)}")
        return block_idx

    def free_block(self, block_idx: int) -> None:
        """Decrements the reference count of a block and frees it if count reaches zero."""
        if self.ref_counts[block_idx] <= 0:
             logging.warning(f"Attempting to free block {block_idx} with ref count {self.ref_counts[block_idx]}.")
             return # Already freed or invalid state

        self.ref_counts[block_idx] -= 1
        if self.ref_counts[block_idx] == 0:
            self.free_blocks.append(block_idx)
            # logging.debug(f"Freed block {block_idx}. Free blocks available: {len(self.free_blocks)}")

    def increase_ref_count(self, block_idx: int) -> None:
        """Increases the reference count for a shared block."""
        if self.ref_counts[block_idx] <= 0:
             raise ValueError(f"Cannot increase ref count for unallocated block {block_idx}.")
        self.ref_counts[block_idx] += 1

    def get_num_free_blocks(self) -> int:
        """Returns the number of currently free blocks."""
        return len(self.free_blocks)

    def get_physical_block(self, block_idx: int, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the physical K and V tensors for a given block index and layer."""
        return self.gpu_cache_k[block_idx, layer_idx], self.gpu_cache_v[block_idx, layer_idx]

    def get_physical_caches(self) -> Tuple[torch.Tensor, torch.Tensor]:
         """Returns the entire physical K and V cache tensors."""
         return self.gpu_cache_k, self.gpu_cache_v


class SequenceMetadata:
    """Holds metadata for a single sequence in the PagedAttention KV Cache."""
    def __init__(self, seq_id: int):
        self.seq_id = seq_id
        self.logical_len = 0
        # List of physical block indices allocated to this sequence
        self.block_table: List[int] = []

    def append_block(self, block_idx: int):
        """Adds a block to the sequence's block table."""
        self.block_table.append(block_idx)

    def get_last_block_physical_idx(self) -> Optional[int]:
        """Returns the physical index of the last block in the table."""
        return self.block_table[-1] if self.block_table else None

    def __len__(self) -> int:
        """Returns the number of blocks allocated to this sequence."""
        return len(self.block_table)


class PagedKVCache:
    """
    Paged Key-Value cache using a BlockManager for memory allocation.
    Manages logical block tables for multiple sequences.
    """
    def __init__(self, num_blocks: int, block_size: int, num_layers: int, num_heads: int, head_dim: int,
                 dtype: torch.dtype = torch.float16, device: str = "cuda"):
        """
        Initializes the PagedKVCache.

        Args:
            num_blocks: Total number of physical blocks to manage.
            block_size: Size of each block (number of tokens).
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            head_dim: Dimension of each attention head.
            dtype: Data type for cache tensors.
            device: Device to store tensors on.
        """
        self.block_manager = BlockManager(num_blocks, block_size, num_layers, num_heads, head_dim, dtype, device)
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Maps sequence ID to its metadata (including block table)
        self.sequences: Dict[int, SequenceMetadata] = {}
        # TODO: Implement prefix caching mechanism
        self.prefix_cache: Dict[Tuple[int, ...], List[int]] = {} # prefix_tuple -> list of shared block_indices

        logging.info(f"PagedKVCache initialized. Block size: {block_size}, Num blocks: {num_blocks}")

    def _ensure_sequence_exists(self, seq_id: int):
        """Creates metadata for a sequence if it doesn't exist."""
        if seq_id not in self.sequences:
            self.sequences[seq_id] = SequenceMetadata(seq_id)

    def _get_logical_block_idx(self, token_pos: int) -> int:
        """Calculates the logical block index for a given token position."""
        return token_pos // self.block_size

    def _get_block_offset(self, token_pos: int) -> int:
        """Calculates the offset within a block for a given token position."""
        return token_pos % self.block_size

    def allocate_blocks_for_sequence(self, seq_id: int, num_tokens: int):
        """Allocates initial blocks for a new sequence or extends an existing one."""
        self._ensure_sequence_exists(seq_id)
        seq_meta = self.sequences[seq_id]

        current_num_blocks = len(seq_meta)
        required_num_blocks = math.ceil(num_tokens / self.block_size)

        # TODO: Implement prefix caching check here
        # If prefix matches, reuse blocks and increase ref counts

        # Allocate new blocks if needed
        for _ in range(required_num_blocks - current_num_blocks):
            try:
                new_block_idx = self.block_manager.allocate_block()
                seq_meta.append_block(new_block_idx)
            except MemoryError as e:
                logging.error(f"Failed to allocate block for seq {seq_id}: {e}")
                # TODO: Need strategy for handling OOM (e.g., preemption)
                self.free_sequence(seq_id) # Free whatever was allocated for this seq
                raise e # Re-raise

        seq_meta.logical_len = num_tokens
        logging.debug(f"Allocated blocks for seq {seq_id}. New block table: {seq_meta.block_table}, Logical length: {num_tokens}")


    def append_token(self, seq_id: int) -> None:
         """
         Appends a single token slot to the sequence, allocating a new block if necessary.
         This is typically called before the forward pass for the next token.
         """
         self._ensure_sequence_exists(seq_id)
         seq_meta = self.sequences[seq_id]
         new_logical_len = seq_meta.logical_len + 1

         # Check if the new token position requires a new block
         current_block_idx = self._get_logical_block_idx(seq_meta.logical_len -1 if seq_meta.logical_len > 0 else 0)
         new_block_idx = self._get_logical_block_idx(new_logical_len - 1)

         if new_block_idx > current_block_idx or not seq_meta.block_table:
             # Need to allocate a new block
             try:
                 new_physical_block_idx = self.block_manager.allocate_block()
                 seq_meta.append_block(new_physical_block_idx)
                 # logging.debug(f"Appended block {new_physical_block_idx} to seq {seq_id} for token {new_logical_len}.")
             except MemoryError as e:
                 logging.error(f"Failed to allocate block for seq {seq_id} during append: {e}")
                 self.free_sequence(seq_id)
                 raise e

         seq_meta.logical_len = new_logical_len


    def get_block_table(self, seq_id: int) -> List[int]:
        """Returns the block table (list of physical block indices) for a sequence."""
        if seq_id not in self.sequences:
             raise ValueError(f"Sequence {seq_id} not found in cache.")
        return self.sequences[seq_id].block_table

    def get_sequence_length(self, seq_id: int) -> int:
        """Returns the current logical length of the sequence."""
        if seq_id not in self.sequences:
            return 0
        return self.sequences[seq_id].logical_len

    def free_sequence(self, seq_id: int) -> None:
        """Frees all blocks associated with a completed or preempted sequence."""
        if seq_id in self.sequences:
            seq_meta = self.sequences[seq_id]
            # logging.debug(f"Freeing sequence {seq_id} with block table: {seq_meta.block_table}")
            for block_idx in seq_meta.block_table:
                self.block_manager.free_block(block_idx)
            del self.sequences[seq_id]
        else:
            logging.warning(f"Attempted to free non-existent sequence {seq_id}")

    def get_physical_caches(self) -> Tuple[torch.Tensor, torch.Tensor]:
         """Returns the underlying physical K and V cache tensors."""
         return self.block_manager.get_physical_caches()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics for the cache."""
        total_blocks = self.block_manager.num_blocks
        free_blocks = self.block_manager.get_num_free_blocks()
        used_blocks = total_blocks - free_blocks

        # Calculate physical memory used by the cache tensors
        k_mem = self.block_manager.gpu_cache_k.element_size() * self.block_manager.gpu_cache_k.nelement()
        v_mem = self.block_manager.gpu_cache_v.element_size() * self.block_manager.gpu_cache_v.nelement()
        total_physical_mem_mb = (k_mem + v_mem) / (1024 * 1024)

        # Calculate logical memory represented by allocated blocks
        allocated_tokens = used_blocks * self.block_size
        # Note: This doesn't account for internal fragmentation within the last block of each sequence

        return {
            "total_physical_blocks": total_blocks,
            "free_physical_blocks": free_blocks,
            "used_physical_blocks": used_blocks,
            "block_size": self.block_size,
            "total_physical_memory_mb": total_physical_mem_mb,
            "gpu_cache_k_shape": tuple(self.block_manager.gpu_cache_k.shape),
            "gpu_cache_v_shape": tuple(self.block_manager.gpu_cache_v.shape),
            "memory_efficiency": free_blocks / total_blocks if total_blocks > 0 else 1.0,
            "active_sequences": len(self.sequences)
        }


# Update the TransformerInferenceRunner.__init__ to use PagedKVCache
class TransformerInferenceRunner(InferenceRunner):
    """Specialized inference runner for transformer models."""

    def __init__(self, model: nn.Module, device: str, precision: str = "fp16",
                 is_encoder_decoder: bool = False, use_kv_cache: bool = True,
                 use_cuda_graph: bool = False,
                 # PagedAttention specific parameters
                 use_paged_attention: bool = True,
                 kv_cache_num_gpu_blocks: Optional[int] = None, # Total blocks on GPU
                 kv_cache_block_size: int = 16):
        """
        Initialize transformer inference runner.

        Args:
            model: Transformer model to run inference with
            device: Device to run inference on ('cuda', 'cpu')
            precision: Precision to use for inference ('fp32', 'fp16', 'bf16', 'int8', 'int4')
            is_encoder_decoder: Whether the model is an encoder-decoder architecture
            use_kv_cache: Whether to use KV cache for efficient generation 
            use_cuda_graph: Whether to use CUDA graph for optimized execution
            use_paged_attention: Whether to use PagedAttention (more efficient than standard KV cache)
            kv_cache_num_gpu_blocks: Total number of physical GPU blocks for the KV cache. If None, calculated based on available memory.
            kv_cache_block_size: Size (in tokens) of each block in the KV cache.
        """
        super().__init__(model, device, precision)
        self.is_encoder_decoder = is_encoder_decoder
        self.use_kv_cache = use_kv_cache and device == "cuda"
        self.use_cuda_graph = use_cuda_graph and device == "cuda"
        self.use_paged_attention = use_paged_attention and use_kv_cache and device == "cuda"
        self.kv_cache_block_size = kv_cache_block_size

        try:
            # Import PagedAttention kernel functions to check availability
            from ..kernels.triton.attention_kernels import triton_paged_attention_forward, TRITON_AVAILABLE
            # If importing fails, we'll catch the ImportError and disable PagedAttention
            self.triton_available = TRITON_AVAILABLE
        except ImportError:
            self.triton_available = False
            if self.use_paged_attention:
                logging.warning("Triton not available. PagedAttention will be disabled.")
                self.use_paged_attention = False

        # Initialize KV cache
        self.kv_cache = None  # Legacy KV cache
        self.paged_kv_cache = None  # New paged KV cache
        
        if self.use_kv_cache:
            if self.use_paged_attention and self.triton_available:
                # Placeholder: We initialize the paged_kv_cache in _initialize_kv_cache after detecting model params
                self.kv_cache_num_gpu_blocks = kv_cache_num_gpu_blocks  # Store for later initialization
                
                # Apply the PagedAttention model modifications to the model
                self.model = add_paged_attention_to_model(self.model)
            else:
                # Fallback to legacy KV cache
                self.kv_cache = KVCache(max_batch_size=1, max_seq_len=2048)
            
            # Actual initialization happens in _initialize_kv_cache after model params are known
            self._detect_model_params()  # Try detecting params early
            self._initialize_kv_cache()  # Initialize based on detected params

        # CUDA graph support
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None


    def _detect_model_params(self):
         """Detects model parameters needed for KV cache initialization."""
         # (This logic is moved from the original _initialize_kv_cache)
         try:
            # Try to get parameters from various model types
            num_layers = 0
            num_heads = 0
            head_dim = 0

            # For HuggingFace models
            if hasattr(self.model, 'config'):
                config = self.model.config
                num_layers = getattr(config, 'num_hidden_layers', 0) or getattr(config, 'n_layer', 0)
                num_heads = getattr(config, 'num_attention_heads', 0) or getattr(config, 'n_head', 0)
                hidden_size = getattr(config, 'hidden_size', 0) or getattr(config, 'n_embd', 0)
                head_dim = hidden_size // num_heads if num_heads > 0 else 0
                # Handle grouped query attention (GQA/MQA)
                num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
                if num_kv_heads != num_heads:
                     logging.info(f"Detected Grouped Query Attention (GQA/MQA): num_heads={num_heads}, num_kv_heads={num_kv_heads}")
                     # The cache usually stores K/V for num_kv_heads
                     num_heads = num_kv_heads # Store num_kv_heads in self.num_heads for cache allocation

            # Fallback detection logic (simplified from original)
            if num_layers == 0 or num_heads == 0 or head_dim == 0:
                 logging.warning("Using fallback model parameter detection for KV Cache.")
                 for module in self.model.modules():
                     if isinstance(module, nn.MultiheadAttention):
                         num_heads = getattr(module, 'num_heads', num_heads)
                         embed_dim = getattr(module, 'embed_dim', 0)
                         head_dim = embed_dim // num_heads if num_heads > 0 else head_dim
                         # Assume num_layers based on module depth or name (crude)
                         break
                     elif hasattr(module, 'num_heads') and hasattr(module, 'head_dim'):
                          num_heads = getattr(module, 'num_heads', num_heads)
                          head_dim = getattr(module, 'head_dim', head_dim)
                          num_kv_heads = getattr(module, 'num_key_value_heads', num_heads)
                          if num_kv_heads != num_heads:
                              num_heads = num_kv_heads # GQA/MQA
                          break

            # Count layers if still unknown
            if num_layers == 0:
                layers_count = 0
                # More robust layer detection needed, depends on model structure conventions
                # Example: looking for 'transformer.h' in GPT-NeoX, 'model.layers' in Llama, etc.
                layers_pattern = re.compile(r'(?:transformer\.h|model\.layers|encoder\.layer|decoder\.layer)\.\d+')
                for name, _ in self.model.named_modules():
                    if layers_pattern.search(name):
                        # Extract the number to avoid double counting nested modules
                        try:
                            layer_num = int(name.split('.')[-1])
                            layers_count = max(layers_count, layer_num + 1)
                        except ValueError:
                            pass # Should probably count unique layer containers
                if layers_count > 0:
                     num_layers = layers_count
                else: # Last resort default
                     logging.warning("Could not detect number of layers reliably. Defaulting to 24.")
                     num_layers = 24

            # Set defaults if detection failed
            if num_heads == 0: logging.warning("Could not detect number of heads. Defaulting to 16."); num_heads = 16
            if head_dim == 0: logging.warning("Could not detect head dimension. Defaulting to 64."); head_dim = 64

            self.num_layers = num_layers
            self.num_heads = num_heads # Stores num_kv_heads if GQA/MQA detected
            self.head_dim = head_dim
            logging.info(f"Detected model parameters: Layers={num_layers}, Heads (KV)={num_heads}, Head Dim={head_dim}")

         except Exception as e:
            logging.error(f"Failed to detect model parameters for KV cache: {e}")
            self.use_kv_cache = False # Disable cache if params unknown


    def _calculate_num_gpu_blocks(self) -> int:
        """Calculates the number of GPU blocks based on available memory and model params."""
        if not torch.cuda.is_available():
            return 0

        # Get model memory usage
        model_params = sum(p.numel() for p in self.model.parameters())
        # Estimate model memory (highly dependent on precision, activations etc.)
        # Assuming fp16/bf16
        model_mem_bytes = model_params * 2
        activation_mem_factor = 4 # Rough estimate for activations, depends on sequence length etc.
        estimated_model_mem_bytes = model_mem_bytes * activation_mem_factor

        # Get total and free GPU memory
        torch.cuda.empty_cache() # Clear cache before checking
        total_gpu_mem = torch.cuda.get_device_properties(0).total_memory
        free_gpu_mem = total_gpu_mem - torch.cuda.memory_allocated(0)

        # Calculate memory available for KV cache (leave some headroom)
        headroom_fraction = 0.10 # Reserve 10%
        available_for_cache = free_gpu_mem - estimated_model_mem_bytes
        available_for_cache -= total_gpu_mem * headroom_fraction
        available_for_cache = max(0, available_for_cache) # Ensure non-negative

        # Calculate memory per block
        # Shape: [num_blocks, num_layers, block_size, num_heads, head_dim]
        # Size of one block: 2(K&V) * num_layers * block_size * num_heads * head_dim * element_size
        element_size = 2  # For float16
        mem_per_block = 2 * self.num_layers * self.kv_cache_block_size * self.num_heads * self.head_dim * element_size

        if mem_per_block == 0:
            return 0

        num_gpu_blocks = int(available_for_cache // mem_per_block)

        logging.info(f"GPU Memory Info: Total={total_gpu_mem/1e9:.2f}GB, Free (before cache)={free_gpu_mem/1e9:.2f}GB")
        logging.info(f"Estimated Model Memory: {estimated_model_mem_bytes/1e9:.2f}GB")
        logging.info(f"Memory Available for KV Cache: {available_for_cache/1e9:.2f}GB")
        logging.info(f"Memory Per Block: {mem_per_block/1e6:.2f}MB")
        logging.info(f"Calculated GPU Blocks for KV Cache: {num_gpu_blocks}")

        if num_gpu_blocks <= 0:
             logging.error("Not enough GPU memory available to allocate any KV cache blocks.")
             return 0

        # Safety check - limit blocks if calculation seems excessive (e.g. >80% of total mem)
        max_cache_mem = total_gpu_mem * 0.8
        if num_gpu_blocks * mem_per_block > max_cache_mem:
             num_gpu_blocks = int(max_cache_mem // mem_per_block)
             logging.warning(f"Limiting KV cache blocks to {num_gpu_blocks} to stay within 80% memory usage.")

        return num_gpu_blocks


    def _initialize_kv_cache(self) -> None:
        """Initialize the appropriate KV cache based on settings."""
        if not self.use_kv_cache:
            return

        # Get model parameters
        if not hasattr(self, 'num_layers') or not hasattr(self, 'num_heads') or not hasattr(self, 'head_dim'):
            logging.warning("Model parameters not detected. Cannot initialize KV cache.")
            self.use_kv_cache = False
            return

        # Get the data type from model parameters
        dtype = next(self.model.parameters()).dtype
        device = self.device

        # Initialize the appropriate KV cache
        if self.use_paged_attention and self.triton_available:
            try:
                # Determine number of blocks
                if self.kv_cache_num_gpu_blocks is None:
                     num_gpu_blocks = self._calculate_num_gpu_blocks()
                     if num_gpu_blocks == 0:
                          raise MemoryError("Insufficient GPU memory for PagedKVCache.")
                     self.kv_cache_num_gpu_blocks = num_gpu_blocks
                else:
                     num_gpu_blocks = self.kv_cache_num_gpu_blocks

                # Initialize PagedKVCache
                self.paged_kv_cache = PagedKVCache(
                    num_blocks=num_gpu_blocks,
                    block_size=self.kv_cache_block_size,
                    num_layers=self.num_layers,
                    num_heads=self.num_heads, # This stores num_kv_heads if GQA/MQA
                    head_dim=self.head_dim,
                    dtype=dtype,
                    device=device
                )
                logging.info(f"PagedKVCache initialized successfully with {num_gpu_blocks} blocks.")

                # For models that have a custom method to set the paged KV cache
                if hasattr(self.model, "set_paged_kv_cache"):
                    self.model.set_paged_kv_cache(self.paged_kv_cache)

            except Exception as e:
                logging.error(f"Failed to initialize PagedKVCache: {e}")
                self.use_paged_attention = False
                self.paged_kv_cache = None
                # Fallback to legacy KV cache
                self.kv_cache = KVCache(max_batch_size=1, max_seq_len=2048)
                self.kv_cache.initialize(self.num_layers, self.num_heads, self.head_dim, dtype, device)
        else:
            # Initialize legacy KV cache
            if self.kv_cache is not None and not self.kv_cache.is_initialized:
                self.kv_cache.initialize(self.num_layers, self.num_heads, self.head_dim, dtype, device)


    def get_kv_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the KV cache.
        
        Returns:
            Dictionary with KV cache statistics
        """
        if not self.use_kv_cache:
            return {"kv_cache_enabled": False}

        # Check if we're using PagedKVCache
        if self.use_paged_attention and self.paged_kv_cache and self.paged_kv_cache.block_manager.is_initialized:
            stats = {
                "kv_cache_enabled": True,
                "kv_cache_type": "PagedAttention"
            }
            # Add memory usage stats specific to PagedKVCache
            stats.update(self.paged_kv_cache.get_memory_usage())
            return stats
        
        # Fallback to original KV cache
        elif self.kv_cache and self.kv_cache.is_initialized:
            stats = {
                "kv_cache_enabled": True,
                "kv_cache_type": "Standard"
            }
            # Add memory usage stats
            stats.update(self.kv_cache.get_memory_usage())
            # Add sequence lengths
            stats["current_seq_lengths"] = self.kv_cache.current_seq_lengths
            return stats
        
        # KV cache is not initialized
        return {"kv_cache_enabled": False, "kv_cache_type": "None"}


class DiffusionInferenceRunner(InferenceRunner):
    """Specialized inference runner for diffusion models."""
    
    def __init__(self, model: nn.Module, device: str, precision: str = "fp16"):
        """
        Initialize diffusion inference runner.
        
        Args:
            model: Diffusion model/pipeline to run inference with
            device: Device to run inference on ('cuda', 'cpu')
            precision: Precision to use for inference ('fp32', 'fp16', 'bf16')
        """
        super().__init__(model, device, precision)
        self.step_times = []
        
    def _forward(self, inputs: Any, **kwargs) -> Any:
        """
        Run forward pass on diffusion model.
        
        Args:
            inputs: Model inputs (typically a dict with prompt and params)
            **kwargs: Additional arguments for the model
            
        Returns:
            Generated images
        """
        # Reset step times
        self.step_times = []
        
        # Capture timing for each denoising step
        original_step = getattr(self.model.scheduler, "step", None)
        
        if original_step is not None:
            def step_with_timing(self_scheduler, *args, **kwargs):
                start_time = time.perf_counter()
                result = original_step(*args, **kwargs)
                end_time = time.perf_counter()
                step_time = (end_time - start_time) * 1000  # ms
                self.step_times.append(step_time)
                return result
            
            # Replace scheduler step method with our timed version
            self.model.scheduler.step = step_with_timing.__get__(
                self.model.scheduler, type(self.model.scheduler)
            )
        
        # Combine inputs and kwargs
        if isinstance(inputs, dict):
            combined_inputs = {**inputs, **kwargs}
        else:
            # Assume it's just the prompt string
            combined_inputs = {"prompt": inputs, **kwargs}
        
        # Run inference
        outputs = self.model(**combined_inputs)
        
        # Restore original step method
        if original_step is not None:
            self.model.scheduler.step = original_step
            
        return outputs
    
    def run_inference(self, inputs: Any, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """
        Run inference and collect performance metrics specific to diffusion models.
        
        Args:
            inputs: Model inputs
            **kwargs: Additional arguments for the model
            
        Returns:
            Tuple of (model outputs, performance metrics)
        """
        # Run the standard inference
        outputs, metrics = super().run_inference(inputs, **kwargs)
        
        # Add diffusion-specific metrics
        if self.step_times:
            metrics["steps_count"] = len(self.step_times)
            metrics["avg_step_time_ms"] = sum(self.step_times) / len(self.step_times)
            metrics["min_step_time_ms"] = min(self.step_times)
            metrics["max_step_time_ms"] = max(self.step_times)
            metrics["step_times_ms"] = self.step_times
        
        return outputs, metrics


# Factory function to create the appropriate inference runner
def benchmark_optimization_impact(model: nn.Module, inputs: Any, device: str = "cuda",
                                precision: str = "fp16", model_type: str = "auto",
                                runs: int = 10) -> Dict[str, Dict[str, float]]:
    """
    Benchmark the impact of various optimization techniques.
    
    Args:
        model: PyTorch model to benchmark
        inputs: Inputs to the model for benchmarking
        device: Device to run benchmarks on
        precision: Precision to use for benchmarks
        model_type: Type of model
        runs: Number of runs for each benchmark
        
    Returns:
        Dictionary mapping optimization configurations to performance metrics
    """
    results = {}
    
    # Optimization configurations to benchmark
    configs = [
        # name, flash_attn, kernel_fusion, kv_cache, cuda_graph
        ("baseline", False, False, False, False),
        ("flash_attention", True, False, False, False),
        ("kernel_fusion", False, True, False, False),
        ("flash_attention_kernel_fusion", True, True, False, False),
        ("kv_cache", False, False, True, False),
        ("cuda_graph", False, False, False, True),
        ("all_optimizations", True, True, True, True)
    ]
    
    for name, flash_attn, kernel_fusion, kv_cache, cuda_graph in configs:
        try:
            # Create runner with specific optimizations
            runner = create_inference_runner(
                model=model,
                device=device,
                precision=precision,
                model_type=model_type,
                use_flash_attention=flash_attn,
                use_kernel_fusion=kernel_fusion,
                use_kv_cache=kv_cache,
                use_cuda_graph=cuda_graph
            )
            
            # Warm up
            runner.warmup(inputs)
            
            # Run benchmarks
            times = []
            for _ in range(runs):
                outputs, metrics = runner.run_inference(inputs)
                times.append(metrics["total_time_ms"])
                
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # Collect memory metrics
            memory_metrics = {
                "peak_memory_mb": metrics.get("peak_memory_mb", 0),
                "memory_change_mb": metrics.get("memory_change_mb", 0)
            }
            
            # Get quantization stats if applicable
            if precision in ["int8", "int4"]:
                quant_stats = runner.get_quantization_stats()
                memory_metrics.update(quant_stats)
                
            # Get KV cache stats if applicable
            if kv_cache and hasattr(runner, "get_kv_cache_stats"):
                kv_cache_stats = runner.get_kv_cache_stats()
                memory_metrics.update(kv_cache_stats)
                
            # Store results
            results[name] = {
                "avg_time_ms": avg_time,
                "min_time_ms": min_time,
                "max_time_ms": max_time,
                "optimization": {
                    "flash_attention": flash_attn,
                    "kernel_fusion": kernel_fusion,
                    "kv_cache": kv_cache,
                    "cuda_graph": cuda_graph
                },
                **memory_metrics
            }
            
        except Exception as e:
            # Log error but continue with other configurations
            logging.error(f"Error benchmarking {name}: {e}")
            results[name] = {"error": str(e)}
            
    return results


def create_inference_runner(model: nn.Module, device: str, precision: str = "fp16", 
                           model_type: str = "auto", use_flash_attention: bool = False,
                           use_kernel_fusion: bool = False, use_kv_cache: bool = True,
                           use_cuda_graph: bool = False) -> InferenceRunner:
    """
    Create an appropriate inference runner for the given model.
    
    Args:
        model: PyTorch model to run inference with
        device: Device to run inference on ('cuda', 'cpu')
        precision: Precision to use for inference ('fp32', 'fp16', 'bf16', 'int8', 'int4')
        model_type: Type of model ('transformer', 'diffusion', 'auto')
        use_flash_attention: Whether to convert attention modules to Flash Attention
        use_kernel_fusion: Whether to apply kernel fusion patterns
        use_kv_cache: Whether to use KV cache for efficient generation
        use_cuda_graph: Whether to use CUDA graph for optimized execution
        
    Returns:
        Appropriate inference runner instance
    """
    # Copy the model to avoid modifying the original
    model_copy = None
    if use_flash_attention or use_kernel_fusion:
        model_copy = deepcopy(model)
    else:
        model_copy = model
        
    # Apply optimizations if requested
    if use_flash_attention and device == "cuda" and HAS_CUSTOM_KERNELS:
        model_copy = convert_to_flash_attention(model_copy)
        
    if use_kernel_fusion and device == "cuda" and HAS_CUSTOM_KERNELS:
        model_copy = fusion_registry.fuse_modules(model_copy)
    
    # Detect model type
    if model_type == "auto":
        # Try to automatically detect model type
        if hasattr(model_copy, "unet") and hasattr(model_copy, "scheduler"):
            model_type = "diffusion"
        elif hasattr(model_copy, "generate") or "transformer" in model_copy.__class__.__name__.lower():
            model_type = "transformer"
        else:
            model_type = "base"  # Default to base runner
    
    # Create appropriate runner with the optimized model
    if model_type == "transformer":
        return TransformerInferenceRunner(
            model_copy, device, precision, 
            use_kv_cache=use_kv_cache, 
            use_cuda_graph=use_cuda_graph
        )
    elif model_type == "diffusion":
        return DiffusionInferenceRunner(model_copy, device, precision)
    else:
        # Create a basic runner that works with regular forward calls
        class BasicInferenceRunner(InferenceRunner):
            def _forward(self, inputs: Any, **kwargs) -> Any:
                return self.model(inputs, **kwargs) if kwargs else self.model(inputs)
        
        return BasicInferenceRunner(model_copy, device, precision)