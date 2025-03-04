"""
Utility functions for analyzing and manipulating PyTorch models.
"""

import math
import re
from typing import Dict, List, Tuple, Set, Any, Optional, Union, Callable

import torch
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys


def get_model_size(model: nn.Module) -> Dict[str, int]:
    """
    Calculate the model size in terms of parameters and memory footprint.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary containing parameter count and estimated memory usage
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get parameter dtype to calculate memory usage
    dtype = next(model.parameters()).dtype
    bytes_per_element = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float64: 8,
        torch.int8: 1,
        torch.uint8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.bool: 1,
    }.get(dtype, 4)  # Default to 4 bytes if not found
    
    # Calculate memory footprint
    param_memory_bytes = total_params * bytes_per_element
    
    # Calculate activations memory (rough estimate based on forward pass)
    # This is a simplistic estimate and might not be accurate for all models
    activation_memory_bytes = 0
    for module in model.modules():
        if hasattr(module, 'weight') and hasattr(module, 'bias') and isinstance(module.weight, torch.Tensor):
            # For linear and conv layers, estimate activation size
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                out_features = module.weight.size(0)
                activation_memory_bytes += out_features * bytes_per_element
    
    # Model state dict size (for saving)
    state_dict_memory_bytes = sum(p.numel() * bytes_per_element for p in model.state_dict().values())
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "param_memory_mb": param_memory_bytes / (1024 * 1024),
        "activation_memory_mb": activation_memory_bytes / (1024 * 1024),
        "state_dict_memory_mb": state_dict_memory_bytes / (1024 * 1024),
        "total_estimated_memory_mb": (param_memory_bytes + activation_memory_bytes) / (1024 * 1024),
        "param_dtype": str(dtype),
        "param_bytes_per_element": bytes_per_element
    }


def get_model_layers(model: nn.Module) -> List[nn.Module]:
    """
    Extract model layers in a flat list, focusing on computation-heavy modules.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        List of model layers (modules)
    """
    # Modules to exclude (container modules and simple operations)
    exclude_module_types = (
        nn.Sequential, nn.ModuleList, nn.ModuleDict,
        nn.Identity, nn.Dropout, nn.Flatten
    )
    
    layers = []
    
    def is_significant_module(module: nn.Module) -> bool:
        """Check if a module is a significant computational unit."""
        # Exclude container modules
        if isinstance(module, exclude_module_types):
            return False
        
        # Exclude modules that just have submodules (no direct compute)
        if list(module.children()):
            has_params = any(isinstance(p, nn.Parameter) for p in module.parameters(recurse=False))
            if not has_params:
                return False
        
        return True
    
    # Get all modules that are significant computation units
    for name, module in model.named_modules():
        if is_significant_module(module):
            layers.append(module)
    
    return layers


def get_attention_modules(model: nn.Module) -> List[nn.Module]:
    """
    Extract attention mechanism modules from the model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        List of attention modules
    """
    attention_modules = []
    attention_keywords = [
        'attention', 'attn', 'self_attn', 'self_attention', 
        'mha', 'multihead', 'multi_head'
    ]
    
    for name, module in model.named_modules():
        # Check class name
        class_name = module.__class__.__name__.lower()
        if any(keyword in class_name for keyword in attention_keywords):
            attention_modules.append(module)
            continue
        
        # Check attribute names (for custom implementations)
        if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
            attention_modules.append(module)
            continue
            
        # Check for specific attention implementations
        if (
            isinstance(module, nn.MultiheadAttention) or 
            (hasattr(module, 'num_heads') and hasattr(module, 'head_dim'))
        ):
            attention_modules.append(module)
            
    return attention_modules


def get_mlp_modules(model: nn.Module) -> List[nn.Module]:
    """
    Extract MLP/feedforward modules from the model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        List of MLP modules
    """
    mlp_modules = []
    mlp_keywords = ['mlp', 'feedforward', 'feed_forward', 'ffn', 'fc']
    
    # First pass: find explicitly named MLPs
    for name, module in model.named_modules():
        name_lower = name.lower()
        class_name = module.__class__.__name__.lower()
        
        # Check for MLP naming patterns
        if any(keyword in name_lower for keyword in mlp_keywords):
            mlp_modules.append(module)
            continue
            
        # Check class name
        if any(keyword in class_name for keyword in mlp_keywords):
            mlp_modules.append(module)
            continue
    
    # Second pass: look for MLP patterns (sequence of linear layers)
    for name, module in model.named_modules():
        # Skip if already found
        if module in mlp_modules:
            continue
            
        # Check for sequential modules with linear layers
        if isinstance(module, nn.Sequential):
            linear_count = sum(1 for m in module.children() if isinstance(m, nn.Linear))
            if linear_count >= 2:  # At least 2 linear layers in sequence
                activations = sum(1 for m in module.children() if isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh)))
                if activations > 0:  # Has activation functions
                    mlp_modules.append(module)
                    continue
        
        # Check for transformer-style MLP modules (two linear layers with activation)
        if (
            list(module.children()) and 
            not isinstance(module, nn.Sequential) and
            not module in mlp_modules
        ):
            linear_layers = [m for m in module.children() if isinstance(m, nn.Linear)]
            activation_layers = [m for m in module.children() if isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh))]
            
            if len(linear_layers) >= 2 and len(activation_layers) >= 1:
                mlp_modules.append(module)
    
    return mlp_modules


def convert_precision(model: nn.Module, precision: str) -> nn.Module:
    """
    Convert model to a specific precision.
    
    Args:
        model: PyTorch model to convert
        precision: Target precision ('fp32', 'fp16', 'bf16')
        
    Returns:
        Model converted to the specified precision
    """
    if precision.lower() == 'fp32':
        dtype = torch.float32
    elif precision.lower() == 'fp16':
        dtype = torch.float16
    elif precision.lower() == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
    # Store current device
    device = next(model.parameters()).device
    
    # Convert model
    model_converted = model.to(dtype=dtype)
    
    # Ensure it's on the same device
    model_converted = model_converted.to(device)
    
    return model_converted


def create_random_input(batch_size: int, seq_len: int, hidden_size: int, 
                      dtype: torch.dtype = torch.float32, 
                      device: str = 'cuda') -> torch.Tensor:
    """
    Create random input tensor for model testing.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden dimension size
        dtype: Data type for the tensor
        device: Device to create the tensor on
        
    Returns:
        Random tensor with shape [batch_size, seq_len, hidden_size]
    """
    return torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)


def calculate_theoretical_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    Calculate theoretical FLOPs for a model.
    
    Args:
        model: PyTorch model to analyze
        input_shape: Shape of the input tensor (excluding batch dimension)
        
    Returns:
        Estimated number of FLOPs
    """
    # Add batch dimension of 1 if not present
    full_shape = input_shape if len(input_shape) >= 3 else (1,) + input_shape
    
    # Create dummy input for tracing through the model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    dummy_input = torch.randn(full_shape, dtype=dtype, device=device)
    
    total_flops = 0
    
    # Track shapes through module hooks
    shape_map = {}
    flops_per_module = {}
    
    # Hooks to register for tracking
    def pre_hook(name):
        def hook(module, inp):
            if isinstance(inp, tuple) and inp:
                shape_map[name + '_in'] = inp[0].shape
        return hook
    
    def post_hook(name):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            
            if hasattr(out, 'shape'):
                shape_map[name + '_out'] = out.shape
                
            # Calculate FLOPs for this module
            flops = 0
            
            # Linear layer
            if isinstance(module, nn.Linear):
                # FLOPs = 2 * in_features * out_features (multiply-add)
                flops = 2 * module.in_features * module.out_features
                
                # If we have the input shape, multiply by input batch and sequence dims
                in_shape = shape_map.get(name + '_in')
                if in_shape is not None:
                    batch_dims = int(torch.prod(torch.tensor(in_shape[:-1])))
                    flops *= batch_dims
                    
            # Conv1d
            elif isinstance(module, nn.Conv1d):
                # FLOPs = 2 * kernel_size * in_channels * out_channels * output_width
                out_shape = shape_map.get(name + '_out')
                if out_shape is not None:
                    output_width = out_shape[-1]
                    flops = 2 * module.kernel_size[0] * module.in_channels * module.out_channels * output_width
                    
            # Conv2d
            elif isinstance(module, nn.Conv2d):
                # FLOPs = 2 * kernel_h * kernel_w * in_channels * out_channels * output_h * output_w
                out_shape = shape_map.get(name + '_out')
                if out_shape is not None:
                    output_h, output_w = out_shape[-2], out_shape[-1]
                    flops = 2 * module.kernel_size[0] * module.kernel_size[1] * \
                            module.in_channels * module.out_channels * output_h * output_w
            
            # Layer normalization
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                # FLOPs H 5 * normalized_shape (estimate)
                in_shape = shape_map.get(name + '_in')
                if in_shape is not None:
                    normalized_elements = int(torch.prod(torch.tensor(in_shape)))
                    flops = 5 * normalized_elements
            
            # Softmax (for attention)
            elif isinstance(module, nn.Softmax):
                in_shape = shape_map.get(name + '_in')
                if in_shape is not None:
                    elements = int(torch.prod(torch.tensor(in_shape)))
                    flops = 5 * elements  # Approximate for exp, div, etc.
            
            # Save the calculated FLOPs
            flops_per_module[name] = flops
            nonlocal total_flops
            total_flops += flops
                
        return hook
    
    # Register hooks for all modules
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LayerNorm, 
                             nn.BatchNorm1d, nn.BatchNorm2d, nn.Softmax)):
            hooks.append(module.register_forward_pre_hook(pre_hook(name)))
            hooks.append(module.register_forward_hook(post_hook(name)))
    
    # Run the model to trigger hooks
    with torch.no_grad():
        model.eval()
        _ = model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Adjust for special model types
    model_type = model.__class__.__name__.lower()
    
    # For transformers, add attention-specific FLOPs
    if 'transformer' in model_type or 'gpt' in model_type or 'bert' in model_type:
        # Look for attention modules
        attention_modules = get_attention_modules(model)
        for module in attention_modules:
            if hasattr(module, 'num_heads') and hasattr(module, 'head_dim'):
                # Estimate attention FLOPs: 2 * seq_len^2 * num_heads * head_dim
                seq_len = input_shape[0] if len(input_shape) >= 2 else 1
                attn_flops = 2 * seq_len**2 * module.num_heads * module.head_dim
                total_flops += attn_flops
    
    return total_flops


def get_model_summary(model: nn.Module) -> str:
    """
    Generate a comprehensive model summary.
    
    Args:
        model: PyTorch model to summarize
        
    Returns:
        String containing model summary
    """
    def get_module_params(module: nn.Module) -> int:
        """Get the number of parameters in a module."""
        return sum(p.numel() for p in module.parameters())
    
    # Get model size info
    size_info = get_model_size(model)
    
    # Get layer information
    layer_info = []
    for name, module in model.named_modules():
        if list(module.children()) == []:  # Only leaf modules
            params = get_module_params(module)
            layer_info.append({
                'name': name,
                'type': module.__class__.__name__,
                'params': params,
                'params_percent': params / size_info['total_params'] * 100 if size_info['total_params'] > 0 else 0
            })
    
    # Sort by parameter count (descending)
    layer_info.sort(key=lambda x: x['params'], reverse=True)
    
    # Build summary string
    summary = []
    summary.append("=" * 80)
    summary.append(f"Model Summary: {model.__class__.__name__}")
    summary.append("=" * 80)
    
    # Parameter counts
    summary.append(f"Total parameters:        {size_info['total_params']:,}")
    summary.append(f"Trainable parameters:    {size_info['trainable_params']:,}")
    summary.append(f"Non-trainable parameters: {size_info['non_trainable_params']:,}")
    summary.append(f"Parameter memory:        {size_info['param_memory_mb']:.2f} MB")
    summary.append(f"Activation memory:       {size_info['activation_memory_mb']:.2f} MB (estimated)")
    summary.append(f"Parameter dtype:         {size_info['param_dtype']}")
    summary.append("-" * 80)
    
    # Layer information table
    summary.append(f"{'Layer':<40} {'Type':<20} {'Parameters':<12} {'%':<6}")
    summary.append("-" * 80)
    
    # Show top 20 layers by parameter count
    for info in layer_info[:20]:
        summary.append(
            f"{info['name'][-39:]:<40} {info['type']:<20} {info['params']:<12,} {info['params_percent']:<6.2f}"
        )
    
    if len(layer_info) > 20:
        summary.append(f"... and {len(layer_info) - 20} more layers")
    
    summary.append("=" * 80)
    
    return "\n".join(summary)


def find_modules_by_type(model: nn.Module, module_type: Union[type, Tuple[type, ...]]) -> List[Tuple[str, nn.Module]]:
    """
    Find all modules of specific types in the model.
    
    Args:
        model: PyTorch model to search
        module_type: Type or tuple of types to search for
        
    Returns:
        List of (name, module) tuples matching the types
    """
    return [(name, module) for name, module in model.named_modules() 
            if isinstance(module, module_type)]


def count_ops_for_module(module: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    Count operations for a specific module.
    
    Args:
        module: PyTorch module to analyze
        input_shape: Shape of the input tensor to the module
        
    Returns:
        Number of operations (MACs)
    """
    device = next(module.parameters()).device
    dtype = next(module.parameters()).dtype
    
    dummy_input = torch.randn(input_shape, dtype=dtype, device=device)
    total_ops = 0
    
    # Linear layer
    if isinstance(module, nn.Linear):
        # Multiply-add operations: input_size * output_size
        batch_size = input_shape[0] if len(input_shape) > 1 else 1
        total_ops = batch_size * module.in_features * module.out_features * 2
        
    # Conv1d
    elif isinstance(module, nn.Conv1d):
        batch_size = input_shape[0]
        L_in = input_shape[-1]
        L_out = (L_in + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
        total_ops = batch_size * L_out * module.out_channels * module.in_channels * module.kernel_size[0] * 2
        
    # Conv2d
    elif isinstance(module, nn.Conv2d):
        batch_size = input_shape[0]
        H_in, W_in = input_shape[-2], input_shape[-1]
        H_out = (H_in + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
        W_out = (W_in + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1) // module.stride[1] + 1
        total_ops = batch_size * H_out * W_out * module.out_channels * module.in_channels * module.kernel_size[0] * module.kernel_size[1] * 2
        
    # For other modules, return an estimate or 0
    else:
        # Run through the module to get output shape
        with torch.no_grad():
            module.eval()
            output = module(dummy_input)
            
        if isinstance(output, torch.Tensor):
            # Estimate based on input and output tensor size
            input_size = torch.prod(torch.tensor(input_shape)).item()
            output_size = torch.prod(torch.tensor(output.shape)).item()
            # Rough estimate: 2 ops per input and output element
            total_ops = (input_size + output_size) * 2
    
    return total_ops


def load_partial_weights(model: nn.Module, state_dict: Dict[str, torch.Tensor], 
                       strict: bool = False) -> _IncompatibleKeys:
    """
    Load partial weights into a model, with detailed mismatch information.
    
    Args:
        model: Target PyTorch model
        state_dict: State dict containing weights to load
        strict: Whether to strictly enforce that the keys match
        
    Returns:
        IncompatibleKeys object containing missing and unexpected keys
    """
    # Get model's own state dict for comparison
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    load_keys = set(state_dict.keys())
    
    # Find missing and unexpected keys
    missing_keys = model_keys - load_keys
    unexpected_keys = load_keys - model_keys
    
    # Check shape compatibility for matched keys
    shape_mismatched = []
    for key in model_keys & load_keys:
        if model_state[key].shape != state_dict[key].shape:
            shape_mismatched.append(key)
    
    # Load weights for compatible keys
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                          if k in model_keys and k not in shape_mismatched}
    
    # Actual loading
    incomp_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    if strict and (missing_keys or unexpected_keys or shape_mismatched):
        error_msg = []
        if missing_keys:
            error_msg.append(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            error_msg.append(f"Unexpected keys: {unexpected_keys}")
        if shape_mismatched:
            error_msg.append(f"Shape-mismatched keys: {shape_mismatched}")
        raise RuntimeError("Error(s) in loading state_dict: {}".format("\n".join(error_msg)))
    
    return incomp_keys


def freeze_layers(model: nn.Module, layer_names: List[str]) -> nn.Module:
    """
    Freeze specific layers in the model.
    
    Args:
        model: PyTorch model to modify
        layer_names: List of layer names to freeze (can use glob patterns)
        
    Returns:
        Modified model with frozen layers
    """
    import fnmatch
    
    patterns = [(pattern, re.compile(fnmatch.translate(pattern))) for pattern in layer_names]
    frozen_count = 0
    
    for name, param in model.named_parameters():
        should_freeze = any(pattern[1].match(name) for pattern in patterns)
        if should_freeze:
            param.requires_grad = False
            frozen_count += 1
    
    print(f"Froze {frozen_count} parameters")
    return model