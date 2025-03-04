import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any, Type, Union

# Import Triton kernels if available
try:
    from ..triton.mlp_kernels import fused_mlp_forward, fused_mlp_backward
except ImportError:
    fused_mlp_forward, fused_mlp_backward = None, None


@dataclass
class FusedMLPConfig:
    """Configuration for FusedMLP modules."""
    activation_fn: str = "gelu"  # Activation function type
    dropout_prob: float = 0.0  # Dropout probability
    use_triton: bool = True  # Whether to use Triton kernels
    precision: str = "fp16"  # Computation precision
    fuse_bias_gelu: bool = True  # Whether to fuse bias addition with activation
    recompute_activation: bool = False  # Whether to recompute activation in backward pass
    sequence_parallel: bool = False  # Whether to use sequence parallelism
    tensor_parallel: bool = False  # Whether to use tensor parallelism
    checkpoint_activation: bool = False  # Whether to checkpoint activations


class FusedMLP(nn.Module):
    """
    Fused Multi-Layer Perceptron implementation with various optimizations.
    
    This implementation optionally uses Triton kernels for improved performance
    and supports various forms of model parallelism.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        config: Optional[FusedMLPConfig] = None
    ):
        super().__init__()
        self.config = config or FusedMLPConfig()
        
        # Initialize weights and biases
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
        
        # Initialize dropout if needed
        self.dropout = nn.Dropout(self.config.dropout_prob) if self.config.dropout_prob > 0 else None
        
        # Apply parallelism if configured
        if self.config.sequence_parallel:
            self._apply_sequence_parallel()
        
        if self.config.tensor_parallel:
            self._apply_tensor_parallel(tp_size=1)  # Default size, to be updated during actual parallelization
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FusedMLP module.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        if self.config.use_triton and fused_mlp_forward is not None:
            return self._forward_triton(hidden_states)
        else:
            return self._forward_pytorch(hidden_states)
    
    def _forward_triton(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Triton kernels for acceleration.
        
        Args:
            hidden_states: Input tensor
            
        Returns:
            Output tensor
        """
        # Get weights and biases
        fc1_weight = self.fc1.weight
        fc1_bias = self.fc1.bias
        fc2_weight = self.fc2.weight
        fc2_bias = self.fc2.bias
        
        # Use Triton kernel for fused MLP operations
        output = fused_mlp_forward(
            hidden_states=hidden_states,
            fc1_weight=fc1_weight,
            fc1_bias=fc1_bias,
            fc2_weight=fc2_weight,
            fc2_bias=fc2_bias,
            activation=self.config.activation_fn,
            fuse_bias_gelu=self.config.fuse_bias_gelu,
            dropout_prob=self.config.dropout_prob,
            checkpoint_activation=self.config.checkpoint_activation
        )
        
        return output
    
    def _forward_pytorch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using standard PyTorch operations.
        
        Args:
            hidden_states: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.fc1(hidden_states)
        
        # Apply activation function
        if self.config.activation_fn == "gelu":
            x = F.gelu(x)
        elif self.config.activation_fn == "relu":
            x = F.relu(x)
        elif self.config.activation_fn == "silu":
            x = F.silu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.config.activation_fn}")
        
        # Apply dropout if needed
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Second linear layer
        x = self.fc2(x)
        
        return x
    
    def _apply_sequence_parallel(self) -> None:
        """
        Apply sequence parallelism optimization to the MLP.
        
        This function sets up the module for sequence parallelism,
        which involves partitioning the sequence dimension across devices.
        """
        # This would typically involve setting the appropriate 
        # communication handlers and sharding strategies
        pass
    
    def _apply_tensor_parallel(self, tp_size: int) -> None:
        """
        Apply tensor parallelism optimization to the MLP.
        
        This function partitions the model weights across multiple devices.
        
        Args:
            tp_size: Number of tensor parallel partitions
        """
        # This would typically involve partitioning the fc1 and fc2 weights
        # across the tensor parallel dimension
        pass


class FusedMLPGeluTanh(FusedMLP):
    """
    FusedMLP implementation with the tanh-approximation of the GELU activation function.
    
    This variant uses the tanh-based approximation for GELU which can be faster and more
    memory-efficient than the exact computation.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        config: Optional[FusedMLPConfig] = None
    ):
        config = config or FusedMLPConfig()
        config.activation_fn = "gelu"  # Force GELU activation
        super().__init__(hidden_size, intermediate_size, config)
    
    def _forward_pytorch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Override _forward_pytorch to use tanh-approximated GELU"""
        x = self.fc1(hidden_states)
        
        # Tanh approximation of GELU
        x = 0.5 * x * (1.0 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        x = self.fc2(x)
        return x


class FusedMLPSwiGLU(FusedMLP):
    """
    FusedMLP implementation with SwiGLU activation function.
    
    SwiGLU activation (Swish-Gated Linear Unit) is used in several modern
    transformer architectures for improved performance.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        config: Optional[FusedMLPConfig] = None
    ):
        config = config or FusedMLPConfig()
        config.activation_fn = "swiglu"  # Force SwiGLU activation
        super().__init__(hidden_size, intermediate_size, config)
        
        # For SwiGLU, we need two sets of weights for the first layer
        # Repurpose the existing fc1 and create a gate projection
        self.fc1_gate = nn.Linear(hidden_size, intermediate_size, bias=True)
    
    def _forward_pytorch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Override _forward_pytorch to use SwiGLU activation"""
        # Compute the gate and value projections
        gate = self.fc1_gate(hidden_states)
        value = self.fc1(hidden_states)
        
        # Apply SwiGLU: silu(gate) * value
        x = F.silu(gate) * value
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        x = self.fc2(x)
        return x
    
    def _forward_triton(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Override _forward_triton to handle SwiGLU activation"""
        if fused_mlp_forward is not None and hasattr(fused_mlp_forward, "supports_swiglu") and fused_mlp_forward.supports_swiglu:
            # Use specialized SwiGLU kernel if available
            output = fused_mlp_forward(
                hidden_states=hidden_states,
                fc1_weight=self.fc1.weight,
                fc1_bias=self.fc1.bias,
                fc1_gate_weight=self.fc1_gate.weight,
                fc1_gate_bias=self.fc1_gate.bias,
                fc2_weight=self.fc2.weight,
                fc2_bias=self.fc2.bias,
                activation="swiglu",
                dropout_prob=self.config.dropout_prob,
                checkpoint_activation=self.config.checkpoint_activation
            )
            return output
        else:
            # Fall back to PyTorch implementation
            return self._forward_pytorch(hidden_states)


class FusedMLPReLU(FusedMLP):
    """
    FusedMLP implementation with ReLU activation function.
    
    This variant uses the simple ReLU activation function which can be
    more computationally efficient than GELU or other activations.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        config: Optional[FusedMLPConfig] = None
    ):
        config = config or FusedMLPConfig()
        config.activation_fn = "relu"  # Force ReLU activation
        super().__init__(hidden_size, intermediate_size, config)


class FusedTransformerMLP(nn.Module):
    """
    A drop-in replacement for transformer MLP blocks with fused operations.
    
    This module is designed to match the interface of standard transformer MLP blocks
    while providing optimized execution through fused operations.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_fn: str = "gelu",
        config: Optional[FusedMLPConfig] = None
    ):
        super().__init__()
        
        # Initialize config if not provided
        self.config = config or FusedMLPConfig()
        self.config.activation_fn = activation_fn
        
        # Select the appropriate MLP implementation based on activation function
        if activation_fn == "gelu":
            self.mlp = FusedMLPGeluTanh(hidden_size, intermediate_size, self.config)
        elif activation_fn == "swiglu":
            self.mlp = FusedMLPSwiGLU(hidden_size, intermediate_size, self.config)
        elif activation_fn == "relu":
            self.mlp = FusedMLPReLU(hidden_size, intermediate_size, self.config)
        else:
            # Default to base implementation with requested activation
            self.mlp = FusedMLP(hidden_size, intermediate_size, self.config)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that matches the interface of standard transformer MLP blocks.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        return self.mlp(hidden_states)
    
    def load_from_standard_mlp(self, state_dict: Dict[str, torch.Tensor], prefix: str = "") -> None:
        """
        Load weights from a standard transformer MLP implementation.
        
        Args:
            state_dict: State dictionary containing the weights
            prefix: Prefix used in the state_dict keys
        """
        # Map standard MLP keys to FusedMLP keys
        key_mapping = {
            f"{prefix}dense.weight": "mlp.fc1.weight",
            f"{prefix}dense.bias": "mlp.fc1.bias",
            f"{prefix}output.dense.weight": "mlp.fc2.weight",
            f"{prefix}output.dense.bias": "mlp.fc2.bias"
        }
        
        # For SwiGLU architecture, handle gate projection
        if isinstance(self.mlp, FusedMLPSwiGLU) and f"{prefix}gate_proj.weight" in state_dict:
            # If model has gate projection, use those weights
            key_mapping.update({
                f"{prefix}gate_proj.weight": "mlp.fc1_gate.weight",
                f"{prefix}gate_proj.bias": "mlp.fc1_gate.bias",
                f"{prefix}up_proj.weight": "mlp.fc1.weight",
                f"{prefix}up_proj.bias": "mlp.fc1.bias"
            })
            
        # Load the mapped weights
        own_state = self.state_dict()
        for name, param in own_state.items():
            for source_key, target_key in key_mapping.items():
                if name == target_key and source_key in state_dict:
                    own_state[name].copy_(state_dict[source_key])
                    break
        
        self.load_state_dict(own_state)


class MLPConverter:
    """
    Utility class for converting standard MLP blocks to fused versions.
    
    This class provides methods to scan a model and replace its MLPs with 
    optimized fused implementations.
    """
    
    def __init__(self, config: Optional[FusedMLPConfig] = None):
        """
        Initialize the converter with configuration.
        
        Args:
            config: Configuration for the fused MLPs
        """
        self.config = config or FusedMLPConfig()
        self.activation_map = {
            "gelu": "gelu",
            "relu": "relu",
            "silu": "silu",
            "swish": "silu",
            "swiglu": "swiglu",
            "gelu_new": "gelu"
        }
    
    def _detect_mlp_type(self, module: nn.Module) -> Optional[Dict[str, Any]]:
        """
        Detect if a module is a standard MLP block and determine its parameters.
        
        Args:
            module: Module to analyze
            
        Returns:
            Dictionary with MLP parameters if detected, None otherwise
        """
        # Common MLP architecture patterns
        # Pattern 1: HuggingFace style MLP (dense -> act -> output.dense)
        if (hasattr(module, "dense") and hasattr(module, "output") and 
            hasattr(module.output, "dense")):
            # Extract parameters
            hidden_size = module.dense.in_features
            intermediate_size = module.dense.out_features
            # Try to detect activation function
            activation_fn = "gelu"  # Default
            if hasattr(module, "act"):
                activation_name = module.act.__class__.__name__.lower()
                if "gelu" in activation_name:
                    activation_fn = "gelu"
                elif "relu" in activation_name:
                    activation_fn = "relu"
                elif "silu" in activation_name or "swish" in activation_name:
                    activation_fn = "silu"
            
            return {
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "activation_fn": activation_fn,
                "pattern": "huggingface"
            }
        
        # Pattern 2: PyTorch style MLP (linear1 -> act -> dropout -> linear2)
        elif (hasattr(module, "linear1") and hasattr(module, "linear2")):
            hidden_size = module.linear1.in_features
            intermediate_size = module.linear1.out_features
            
            # Try to detect activation function
            activation_fn = "gelu"  # Default
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, nn.Module) and "activation" in attr_name.lower():
                    activation_name = attr.__class__.__name__.lower()
                    if "gelu" in activation_name:
                        activation_fn = "gelu"
                    elif "relu" in activation_name:
                        activation_fn = "relu"
                    elif "silu" in activation_name or "swish" in activation_name:
                        activation_fn = "silu"
            
            return {
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "activation_fn": activation_fn,
                "pattern": "pytorch"
            }
        
        # Pattern 3: LLaMA style (gate_proj, up_proj, down_proj for SwiGLU)
        elif (hasattr(module, "gate_proj") and hasattr(module, "up_proj") and 
              hasattr(module, "down_proj")):
            hidden_size = module.gate_proj.in_features
            intermediate_size = module.gate_proj.out_features
            
            return {
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "activation_fn": "swiglu",
                "pattern": "llama"
            }
        
        return None
    
    def _create_fused_mlp(self, mlp_params: Dict[str, Any]) -> FusedTransformerMLP:
        """
        Create a fused MLP based on detected parameters.
        
        Args:
            mlp_params: Parameters of the MLP
            
        Returns:
            Fused MLP module
        """
        hidden_size = mlp_params["hidden_size"]
        intermediate_size = mlp_params["intermediate_size"]
        activation_fn = mlp_params["activation_fn"]
        
        # Map to supported activation function
        if activation_fn in self.activation_map:
            activation_fn = self.activation_map[activation_fn]
            
        return FusedTransformerMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_fn=activation_fn,
            config=self.config
        )
    
    def _copy_weights(self, fused_mlp: FusedTransformerMLP, original_mlp: nn.Module, mlp_params: Dict[str, Any]) -> None:
        """
        Copy weights from original MLP to fused MLP.
        
        Args:
            fused_mlp: Target fused MLP
            original_mlp: Source original MLP
            mlp_params: Parameters of the MLP
        """
        # Create state dict from original module
        state_dict = {}
        pattern = mlp_params["pattern"]
        
        if pattern == "huggingface":
            state_dict["dense.weight"] = original_mlp.dense.weight
            state_dict["dense.bias"] = original_mlp.dense.bias
            state_dict["output.dense.weight"] = original_mlp.output.dense.weight
            state_dict["output.dense.bias"] = original_mlp.output.dense.bias
        
        elif pattern == "pytorch":
            state_dict["dense.weight"] = original_mlp.linear1.weight
            state_dict["dense.bias"] = original_mlp.linear1.bias
            state_dict["output.dense.weight"] = original_mlp.linear2.weight
            state_dict["output.dense.bias"] = original_mlp.linear2.bias
            
        elif pattern == "llama":
            state_dict["gate_proj.weight"] = original_mlp.gate_proj.weight
            state_dict["gate_proj.bias"] = original_mlp.gate_proj.bias if hasattr(original_mlp.gate_proj, "bias") else torch.zeros_like(original_mlp.gate_proj.weight[0])
            state_dict["up_proj.weight"] = original_mlp.up_proj.weight
            state_dict["up_proj.bias"] = original_mlp.up_proj.bias if hasattr(original_mlp.up_proj, "bias") else torch.zeros_like(original_mlp.up_proj.weight[0])
            state_dict["down_proj.weight"] = original_mlp.down_proj.weight
            state_dict["down_proj.bias"] = original_mlp.down_proj.bias if hasattr(original_mlp.down_proj, "bias") else torch.zeros_like(original_mlp.down_proj.weight[0])
        
        # Load weights
        fused_mlp.load_from_standard_mlp(state_dict)
    
    def convert_model(self, model: nn.Module, target_class_names: Optional[list] = None) -> nn.Module:
        """
        Convert MLPs in a model to fused versions.
        
        Args:
            model: PyTorch model to convert
            target_class_names: Optional list of class names to target for conversion
            
        Returns:
            Model with converted MLPs
        """
        # If no target classes specified, try to guess common MLP class names
        if target_class_names is None:
            target_class_names = [
                "MLP", "FFN", "FeedForward", "MLPBlock", "GELU_MLP", "SwiGLU", 
                "FeedForwardNetwork", "PositionwiseFeedForward"
            ]
        
        # Keep track of replacements
        replacements = 0
        
        # Helper function to recursively process modules
        def _process_module(module: nn.Module, parent: Optional[nn.Module] = None, name: str = ""):
            nonlocal replacements
            
            # Check if this module should be replaced
            should_replace = False
            if target_class_names:
                module_class_name = module.__class__.__name__
                should_replace = any(target in module_class_name for target in target_class_names)
            
            # If potentially replaceable, analyze it
            if should_replace:
                mlp_params = self._detect_mlp_type(module)
                if mlp_params:
                    # Create replacement
                    fused_mlp = self._create_fused_mlp(mlp_params)
                    
                    # Copy weights
                    self._copy_weights(fused_mlp, module, mlp_params)
                    
                    # Replace in parent
                    if parent is not None:
                        setattr(parent, name, fused_mlp)
                        replacements += 1
                        return
            
            # Recurse into children
            for child_name, child in module.named_children():
                _process_module(child, module, child_name)
        
        # Process model
        _process_module(model)
        
        return model