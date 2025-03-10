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


class TransformerInferenceRunner(InferenceRunner):
    """Specialized inference runner for transformer models."""
    
    def __init__(self, model: nn.Module, device: str, precision: str = "fp16", 
                 is_encoder_decoder: bool = False, use_kv_cache: bool = True,
                 use_cuda_graph: bool = False):
        """
        Initialize transformer inference runner.
        
        Args:
            model: Transformer model to run inference with
            device: Device to run inference on ('cuda', 'cpu')
            precision: Precision to use for inference ('fp32', 'fp16', 'bf16', 'int8', 'int4')
            is_encoder_decoder: Whether the model is an encoder-decoder architecture
            use_kv_cache: Whether to use KV cache for efficient generation
            use_cuda_graph: Whether to use CUDA graph for optimized execution
        """
        super().__init__(model, device, precision)
        self.is_encoder_decoder = is_encoder_decoder
        self.use_kv_cache = use_kv_cache and device == "cuda"
        self.use_cuda_graph = use_cuda_graph and device == "cuda"
        
        # Initialize KV cache
        self.kv_cache = None
        if self.use_kv_cache:
            self.kv_cache = KVCache(max_batch_size=1, max_seq_len=2048)
            self._initialize_kv_cache()
            
        # CUDA graph support
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None
        
    def _initialize_kv_cache(self) -> None:
        """Initialize the KV cache with model parameters."""
        if not self.use_kv_cache or self.kv_cache is None:
            return
            
        # Detect model architecture and extract parameters
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
                
            # For PyTorch transformer models
            if num_layers == 0 and hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
                num_layers = len(self.model.encoder.layers)
                # Try to get num_heads from first layer
                if len(self.model.encoder.layers) > 0:
                    if hasattr(self.model.encoder.layers[0], 'self_attn'):
                        if hasattr(self.model.encoder.layers[0].self_attn, 'num_heads'):
                            num_heads = self.model.encoder.layers[0].self_attn.num_heads
                            
            # If still not found, try to examine model structure
            if num_heads == 0 or head_dim == 0:
                # Look for attention modules
                for module in self.model.modules():
                    # PyTorch MultiheadAttention
                    if isinstance(module, nn.MultiheadAttention):
                        num_heads = getattr(module, 'num_heads', 0)
                        embed_dim = getattr(module, 'embed_dim', 0)
                        head_dim = embed_dim // num_heads if num_heads > 0 else 0
                        break
                    # Custom attention with attributes
                    elif hasattr(module, 'num_heads') and hasattr(module, 'head_dim'):
                        num_heads = module.num_heads
                        head_dim = module.head_dim
                        break
                    # Models like T5 with different naming
                    elif hasattr(module, 'n_heads') and hasattr(module, 'd_kv'):
                        num_heads = module.n_heads
                        head_dim = module.d_kv
                        break
                        
            # If layers still not found, try to count layers
            if num_layers == 0:
                layers_count = 0
                layers_pattern = re.compile(r'layer\.\d+$|layers\.\d+$|layer_\d+$|h\.\d+$')
                for name, _ in self.model.named_modules():
                    if layers_pattern.search(name):
                        layers_count += 1
                num_layers = layers_count
                
            # Set defaults if detection failed
            if num_layers == 0:
                logging.warning("Could not detect number of layers. Defaulting to 24.")
                num_layers = 24
            if num_heads == 0:
                logging.warning("Could not detect number of heads. Defaulting to 16.")
                num_heads = 16
            if head_dim == 0:
                logging.warning("Could not detect head dimension. Defaulting to 64.")
                head_dim = 64
                
            # Initialize the cache
            dtype = next(self.model.parameters()).dtype
            self.kv_cache.initialize(num_layers, num_heads, head_dim, dtype, self.device)
            
        except Exception as e:
            logging.warning(f"Failed to initialize KV cache: {e}")
            self.use_kv_cache = False
            
    def _forward(self, inputs: Any, **kwargs) -> Any:
        """
        Run forward pass on transformer model.
        
        Args:
            inputs: Model inputs (typically a dict with input_ids and attention_mask)
            **kwargs: Additional arguments like max_length, num_beams, etc.
            
        Returns:
            Model outputs
        """
        # Check if inputs is a dict with input_ids or just tensor input_ids
        if isinstance(inputs, dict):
            input_dict = inputs
        elif hasattr(inputs, "to_dict") and callable(inputs.to_dict):
            # Handle BatchEncoding objects from tokenizers
            input_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in inputs.to_dict().items()}
        else:
            # Assume it's just input_ids
            input_dict = {"input_ids": inputs}
            if "attention_mask" not in input_dict and "attention_mask" not in kwargs:
                # Create attention mask if not provided
                input_dict["attention_mask"] = torch.ones_like(input_dict["input_ids"])
        
        # Merge kwargs into input_dict for any keys not already present
        for key, value in kwargs.items():
            if key not in input_dict:
                input_dict[key] = value
        
        # Handle generation vs normal forward pass
        generation_args = ["max_length", "min_length", "num_beams", "temperature", 
                         "top_k", "top_p", "repetition_penalty", "do_sample"]
        
        is_generation = any(arg in input_dict for arg in generation_args) or any(arg in kwargs for arg in generation_args)
        
        if is_generation and hasattr(self.model, "generate"):
            # Add KV cache to generation if available
            if self.use_kv_cache and self.kv_cache and self.kv_cache.is_initialized:
                # Reset KV cache for new generation
                self.kv_cache.reset()
                
                # Add KV cache to input_dict for models that support it
                # Note: Different model architectures handle KV cache differently
                # This is a simplified approach - actual implementation depends on model
                input_dict["use_cache"] = True
                
                # For models that support custom KV cache
                if hasattr(self.model, "set_kv_cache"):
                    self.model.set_kv_cache(self.kv_cache)
                    
            # Use CUDA graph for generation if available
            if self.use_cuda_graph and self.cuda_graph is not None and torch.cuda.is_available():
                # Only use CUDA graph for single token generation in a fixed pattern
                # Real implementation would need multiple graphs for different scenarios
                if input_dict.get("max_length", 0) == input_dict["input_ids"].shape[1] + 1:
                    return self._forward_cuda_graph(input_dict)
            
            # Extract generation-specific parameters
            gen_kwargs = {k: v for k, v in input_dict.items() 
                         if k in generation_args or k in ["input_ids", "attention_mask", "use_cache"]}
            outputs = self.model.generate(**gen_kwargs)
        else:
            # Regular forward pass
            outputs = self.model(**input_dict)
            
        return outputs
        
    def _forward_cuda_graph(self, inputs: Dict[str, Any]) -> Any:
        """
        Run forward pass using CUDA graph for optimized execution.
        
        Args:
            inputs: Dictionary of inputs to the model
            
        Returns:
            Model outputs
        """
        if not torch.cuda.is_available():
            return self.model(**inputs)
            
        # Check if we need to capture a graph
        if self.cuda_graph is None:
            # Create static inputs and outputs for graph capture
            static_inputs = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            
            # Warmup before capture
            for _ in range(3):
                with torch.no_grad():
                    _ = self.model(**static_inputs)
            
            # Capture graph
            torch.cuda.synchronize()
            self.static_input = static_inputs
            self.static_output = None
            
            # Use stream for capture
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            
            with torch.cuda.stream(stream):
                self.cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.cuda_graph):
                    self.static_output = self.model(**self.static_input)
                    
            torch.cuda.current_stream().wait_stream(stream)
            
        # Copy inputs to static tensors
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and k in self.static_input:
                self.static_input[k].copy_(v)
                
        # Run the graph
        self.cuda_graph.replay()
        
        # Create a copy of the output to avoid issues with graph replay
        with torch.no_grad():
            if isinstance(self.static_output, torch.Tensor):
                return self.static_output.clone()
            elif isinstance(self.static_output, tuple):
                return tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in self.static_output)
            elif isinstance(self.static_output, dict):
                return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.static_output.items()}
            else:
                return self.static_output
                
    def get_kv_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the KV cache.
        
        Returns:
            Dictionary with KV cache statistics
        """
        if not self.use_kv_cache or self.kv_cache is None:
            return {"kv_cache_enabled": False}
            
        stats = {"kv_cache_enabled": True}
        
        # Add memory usage stats
        stats.update(self.kv_cache.get_memory_usage())
        
        # Add sequence lengths
        stats["current_seq_lengths"] = self.kv_cache.current_seq_lengths
        
        return stats
    
    def run_inference_with_layer_timing(self, inputs: Any, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """
        Run inference with per-layer timing.
        
        Args:
            inputs: Model inputs
            **kwargs: Additional arguments for the model
            
        Returns:
            Tuple of (model outputs, performance metrics with per-layer timing)
        """
        self.model.eval()
        layer_metrics = {}
        
        # Set up hooks for layer timing
        hooks = []
        layer_times = {}
        
        def forward_hook(name):
            def hook(module, input, output):
                # Record start time
                if self.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                # Store the start time for this layer
                layer_times[name] = {"start": start}
            return hook
        
        def backward_hook(name):
            def hook(module, input, output):
                # Record end time
                if self.device == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                # Calculate elapsed time
                if name in layer_times:
                    start = layer_times[name]["start"]
                    elapsed_ms = (end - start) * 1000  # Convert to ms
                    layer_times[name]["time_ms"] = elapsed_ms
            return hook
        
        # Register hooks for all named modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only register for leaf modules
                hooks.append(module.register_forward_pre_hook(forward_hook(name)))
                hooks.append(module.register_forward_hook(backward_hook(name)))
        
        # Run normal inference
        outputs, metrics = self.run_inference(inputs, **kwargs)
        
        # Process layer timing data
        for name, data in layer_times.items():
            if "time_ms" in data:
                layer_metrics[f"layer_time_{name}_ms"] = data["time_ms"]
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Merge layer metrics with regular metrics
        combined_metrics = {**metrics, **layer_metrics}
        
        return outputs, combined_metrics


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