import torch
import triton
import triton.language as tl
import numpy as np
from typing import Optional, Dict


@triton.jit
def fused_dense_gelu_dense_kernel(
    # Pointers to matrices
    input_ptr, fc1_weight_ptr, fc1_bias_ptr, fc2_weight_ptr, fc2_bias_ptr, output_ptr,
    # Matrix dimensions
    batch_size, seq_len, hidden_size, intermediate_size,
    # Strides
    input_batch_stride, input_row_stride, input_col_stride,
    fc1_weight_row_stride, fc1_weight_col_stride,
    fc2_weight_row_stride, fc2_weight_col_stride,
    output_batch_stride, output_row_stride, output_col_stride,
    # Meta-parameters
    block_size: tl.constexpr,
    block_k: tl.constexpr,
    block_n: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    """
    Fused kernel that computes: FC2(GELU(FC1(x)))
    
    This kernel fuses the entire MLP operation with GELU activation into a single kernel
    to maximize data reuse and minimize memory accesses.
    """
    # Program ID
    pid_batch = tl.program_id(0)
    pid_row = tl.program_id(1)
    pid_col = tl.program_id(2)
    
    # Compute offset for the current block
    batch_offset = pid_batch * input_batch_stride
    hidden_offset = pid_row * block_size
    intermediate_offset_fc1 = pid_col * block_n
    
    # Initialize pointers to input and fc1 weight
    input_block_ptr = input_ptr + batch_offset + hidden_offset * input_col_stride
    fc1_weight_block_ptr = fc1_weight_ptr + intermediate_offset_fc1 * fc1_weight_col_stride + 0 * fc1_weight_row_stride
    
    # Initialize accumulator for intermediate result
    acc_fc1 = tl.zeros([block_size, block_n], dtype=tl.float32)
    
    # Load fc1 bias if using bias
    if USE_BIAS:
        fc1_bias = tl.load(fc1_bias_ptr + intermediate_offset_fc1 + tl.arange(0, block_n))
    
    # Compute FC1: Matmul with input and fc1_weight
    for k in range(0, hidden_size, block_k):
        # Load input block
        k_remaining = min(block_k, hidden_size - k)
        input_k = tl.load(input_block_ptr + k * input_col_stride + tl.arange(0, k_remaining)[None, :] * input_col_stride, 
                          mask=tl.arange(0, block_size)[:, None] < seq_len, other=0.0)
        
        # Load fc1 weight block
        fc1_weight_k = tl.load(fc1_weight_block_ptr + k * fc1_weight_row_stride + tl.arange(0, block_n)[:, None] * fc1_weight_col_stride, 
                               mask=tl.arange(0, k_remaining)[None, :] < hidden_size, other=0.0)
        
        # Compute partial FC1
        acc_fc1 += tl.dot(input_k, tl.trans(fc1_weight_k))
    
    # Add bias and apply GELU activation
    if USE_BIAS:
        intermediate = acc_fc1 + fc1_bias[None, :]
    else:
        intermediate = acc_fc1
    
    # Apply GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = 0.7978845608028654
    one = 1.0
    half = 0.5
    coeff = 0.044715
    
    x_cube = intermediate * intermediate * intermediate
    inner = sqrt_2_pi * (intermediate + coeff * x_cube)
    intermediate = half * intermediate * (one + tl.tanh(inner))
    
    # Initialize pointers for FC2
    fc2_weight_block_ptr = fc2_weight_ptr + 0 * fc2_weight_col_stride
    output_block_ptr = output_ptr + batch_offset + hidden_offset * output_col_stride
    
    # Initialize accumulator for output
    acc_fc2 = tl.zeros([block_size, hidden_size], dtype=tl.float32)
    
    # Compute FC2: Matmul with intermediate and fc2_weight
    for k in range(0, intermediate_size, block_k):
        # Load fc2 weight block
        k_remaining = min(block_k, intermediate_size - k)
        fc2_weight_k = tl.load(fc2_weight_block_ptr + k * fc2_weight_row_stride + tl.arange(0, hidden_size)[None, :] * fc2_weight_col_stride,
                               mask=tl.arange(0, k_remaining)[:, None] < intermediate_size, other=0.0)
        
        # Extract corresponding part of intermediate
        intermediate_k = intermediate[:, k:k+k_remaining]
        
        # Compute partial FC2
        acc_fc2 += tl.dot(intermediate_k, fc2_weight_k)
    
    # Add fc2 bias if using bias
    if USE_BIAS:
        fc2_bias = tl.load(fc2_bias_ptr + tl.arange(0, hidden_size))
        acc_fc2 += fc2_bias
    
    # Store output
    output_mask = (tl.arange(0, block_size)[:, None] < seq_len) & (tl.arange(0, hidden_size)[None, :] < hidden_size)
    tl.store(output_block_ptr + tl.arange(0, hidden_size)[None, :] * output_col_stride, acc_fc2, mask=output_mask)


@triton.jit
def fused_bias_gelu_kernel(
    input_ptr, bias_ptr, output_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    """
    Fused kernel that computes: GELU(x + bias)
    
    This kernel fuses the bias addition with GELU activation for better performance.
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Block start index
    block_start = pid * block_size
    
    # Compute offsets
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    
    # Load input and bias
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offsets % (n_elements // x.shape[0]), mask=mask, other=0.0)
    
    # Add bias
    x_bias = x + b
    
    # Apply GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = 0.7978845608028654
    one = 1.0
    half = 0.5
    coeff = 0.044715
    
    x_cube = x_bias * x_bias * x_bias
    inner = sqrt_2_pi * (x_bias + coeff * x_cube)
    output = half * x_bias * (one + tl.tanh(inner))
    
    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def fused_bias_swiglu_kernel(
    input_ptr, w1_ptr, w2_ptr, b1_ptr, b2_ptr, output_ptr,
    batch_size, seq_len, hidden_size, intermediate_size,
    input_batch_stride, input_row_stride, input_col_stride,
    weight_row_stride, weight_col_stride,
    output_batch_stride, output_row_stride, output_col_stride,
    block_size: tl.constexpr,
    block_k: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    """
    Fused kernel that computes SwiGLU activation: SiLU(Wx1 + b1) * (Wx2 + b2)
    
    This kernel is specialized for the SwiGLU activation used in models like LLaMa.
    """
    # Program ID
    pid_batch = tl.program_id(0)
    pid_row = tl.program_id(1)
    
    # Compute offset for the current block
    batch_offset = pid_batch * input_batch_stride
    hidden_offset = pid_row * block_size
    
    # Initialize pointers to input and weights
    input_block_ptr = input_ptr + batch_offset + hidden_offset * input_col_stride
    
    # Initialize accumulators for gate (x1) and value (x2)
    acc_gate = tl.zeros([block_size, intermediate_size], dtype=tl.float32)
    acc_value = tl.zeros([block_size, intermediate_size], dtype=tl.float32)
    
    # Compute gate and value projections
    for k in range(0, hidden_size, block_k):
        # Load input block
        k_remaining = min(block_k, hidden_size - k)
        input_k = tl.load(input_block_ptr + k * input_col_stride + tl.arange(0, k_remaining)[None, :] * input_col_stride, 
                          mask=tl.arange(0, block_size)[:, None] < seq_len, other=0.0)
        
        # Load weight blocks for gate (w1) and value (w2)
        w1_block_ptr = w1_ptr + k * weight_row_stride
        w2_block_ptr = w2_ptr + k * weight_row_stride
        
        w1_k = tl.load(w1_block_ptr + tl.arange(0, intermediate_size)[None, :] * weight_col_stride, 
                       mask=tl.arange(0, k_remaining)[:, None] < hidden_size, other=0.0)
        
        w2_k = tl.load(w2_block_ptr + tl.arange(0, intermediate_size)[None, :] * weight_col_stride, 
                       mask=tl.arange(0, k_remaining)[:, None] < hidden_size, other=0.0)
        
        # Compute partial projections
        acc_gate += tl.dot(input_k, w1_k)
        acc_value += tl.dot(input_k, w2_k)
    
    # Add biases if using bias
    if USE_BIAS:
        b1 = tl.load(b1_ptr + tl.arange(0, intermediate_size))
        b2 = tl.load(b2_ptr + tl.arange(0, intermediate_size))
        
        acc_gate += b1
        acc_value += b2
    
    # Apply SwiGLU activation: SiLU(gate) * value
    # SiLU(x) = x * sigmoid(x)
    gate_sigmoid = 1.0 / (1.0 + tl.exp(-acc_gate))
    swiglu_result = (acc_gate * gate_sigmoid) * acc_value
    
    # Initialize pointer for output
    output_block_ptr = output_ptr + batch_offset + hidden_offset * output_col_stride
    
    # Store output
    output_mask = (tl.arange(0, block_size)[:, None] < seq_len) & (tl.arange(0, intermediate_size)[None, :] < intermediate_size)
    tl.store(output_block_ptr + tl.arange(0, intermediate_size)[None, :] * output_col_stride, swiglu_result, mask=output_mask)


@triton.jit
def fused_dense_relu_dense_kernel(
    # Pointers to matrices
    input_ptr, fc1_weight_ptr, fc1_bias_ptr, fc2_weight_ptr, fc2_bias_ptr, output_ptr,
    # Matrix dimensions
    batch_size, seq_len, hidden_size, intermediate_size,
    # Strides
    input_batch_stride, input_row_stride, input_col_stride,
    fc1_weight_row_stride, fc1_weight_col_stride,
    fc2_weight_row_stride, fc2_weight_col_stride,
    output_batch_stride, output_row_stride, output_col_stride,
    # Meta-parameters
    block_size: tl.constexpr,
    block_k: tl.constexpr,
    block_n: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    """
    Fused kernel that computes: FC2(ReLU(FC1(x)))
    
    This kernel fuses the entire MLP operation with ReLU activation into a single kernel
    to maximize data reuse and minimize memory accesses.
    """
    # Program ID
    pid_batch = tl.program_id(0)
    pid_row = tl.program_id(1)
    pid_col = tl.program_id(2)
    
    # Compute offset for the current block
    batch_offset = pid_batch * input_batch_stride
    hidden_offset = pid_row * block_size
    intermediate_offset_fc1 = pid_col * block_n
    
    # Initialize pointers to input and fc1 weight
    input_block_ptr = input_ptr + batch_offset + hidden_offset * input_col_stride
    fc1_weight_block_ptr = fc1_weight_ptr + intermediate_offset_fc1 * fc1_weight_col_stride + 0 * fc1_weight_row_stride
    
    # Initialize accumulator for intermediate result
    acc_fc1 = tl.zeros([block_size, block_n], dtype=tl.float32)
    
    # Load fc1 bias if using bias
    if USE_BIAS:
        fc1_bias = tl.load(fc1_bias_ptr + intermediate_offset_fc1 + tl.arange(0, block_n))
    
    # Compute FC1: Matmul with input and fc1_weight
    for k in range(0, hidden_size, block_k):
        # Load input block
        k_remaining = min(block_k, hidden_size - k)
        input_k = tl.load(input_block_ptr + k * input_col_stride + tl.arange(0, k_remaining)[None, :] * input_col_stride, 
                          mask=tl.arange(0, block_size)[:, None] < seq_len, other=0.0)
        
        # Load fc1 weight block
        fc1_weight_k = tl.load(fc1_weight_block_ptr + k * fc1_weight_row_stride + tl.arange(0, block_n)[:, None] * fc1_weight_col_stride, 
                               mask=tl.arange(0, k_remaining)[None, :] < hidden_size, other=0.0)
        
        # Compute partial FC1
        acc_fc1 += tl.dot(input_k, tl.trans(fc1_weight_k))
    
    # Add bias and apply ReLU activation
    if USE_BIAS:
        intermediate = acc_fc1 + fc1_bias[None, :]
    else:
        intermediate = acc_fc1
    
    # Apply ReLU activation: max(0, x)
    intermediate = tl.maximum(0.0, intermediate)
    
    # Initialize pointers for FC2
    fc2_weight_block_ptr = fc2_weight_ptr + 0 * fc2_weight_col_stride
    output_block_ptr = output_ptr + batch_offset + hidden_offset * output_col_stride
    
    # Initialize accumulator for output
    acc_fc2 = tl.zeros([block_size, hidden_size], dtype=tl.float32)
    
    # Compute FC2: Matmul with intermediate and fc2_weight
    for k in range(0, intermediate_size, block_k):
        # Load fc2 weight block
        k_remaining = min(block_k, intermediate_size - k)
        fc2_weight_k = tl.load(fc2_weight_block_ptr + k * fc2_weight_row_stride + tl.arange(0, hidden_size)[None, :] * fc2_weight_col_stride,
                               mask=tl.arange(0, k_remaining)[:, None] < intermediate_size, other=0.0)
        
        # Extract corresponding part of intermediate
        intermediate_k = intermediate[:, k:k+k_remaining]
        
        # Compute partial FC2
        acc_fc2 += tl.dot(intermediate_k, fc2_weight_k)
    
    # Add fc2 bias if using bias
    if USE_BIAS:
        fc2_bias = tl.load(fc2_bias_ptr + tl.arange(0, hidden_size))
        acc_fc2 += fc2_bias
    
    # Store output
    output_mask = (tl.arange(0, block_size)[:, None] < seq_len) & (tl.arange(0, hidden_size)[None, :] < hidden_size)
    tl.store(output_block_ptr + tl.arange(0, hidden_size)[None, :] * output_col_stride, acc_fc2, mask=output_mask)


# Python wrapper functions
def triton_fused_mlp(
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: Optional[torch.Tensor],
    fc2_weight: torch.Tensor,
    fc2_bias: Optional[torch.Tensor],
    activation_fn: str = "gelu"
) -> torch.Tensor:
    """
    Fused MLP implementation using Triton kernels.
    
    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        fc1_weight: Weight tensor for first linear layer
        fc1_bias: Optional bias tensor for first linear layer
        fc2_weight: Weight tensor for second linear layer
        fc2_bias: Optional bias tensor for second linear layer
        activation_fn: Activation function to use ("gelu", "relu", etc.)
        
    Returns:
        Output tensor of shape [batch_size, seq_len, hidden_size]
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    intermediate_size = fc1_weight.shape[0]
    
    # We need to handle both bias and no-bias cases
    use_bias = fc1_bias is not None and fc2_bias is not None
    
    # If bias is None, create zero tensors for the kernel
    if fc1_bias is None:
        fc1_bias = torch.zeros(intermediate_size, device=hidden_states.device, dtype=hidden_states.dtype)
    if fc2_bias is None:
        fc2_bias = torch.zeros(hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
    
    # Initialize output tensor
    output = torch.empty((batch_size, seq_len, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
    
    # Compute meta-parameters for the kernel
    block_size = 16
    block_k = 16
    block_n = 16
    
    # Calculate grid dimensions
    grid = (
        batch_size,
        triton.cdiv(seq_len, block_size),
        triton.cdiv(intermediate_size, block_n)
    )
    
    # Choose the appropriate kernel based on the activation function
    if activation_fn == "gelu":
        fused_dense_gelu_dense_kernel[grid](
            hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias, output,
            batch_size, seq_len, hidden_size, intermediate_size,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            fc1_weight.stride(0), fc1_weight.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            block_size=block_size, block_k=block_k, block_n=block_n,
            USE_BIAS=use_bias,
        )
    elif activation_fn == "relu":
        fused_dense_relu_dense_kernel[grid](
            hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias, output,
            batch_size, seq_len, hidden_size, intermediate_size,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            fc1_weight.stride(0), fc1_weight.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            block_size=block_size, block_k=block_k, block_n=block_n,
            USE_BIAS=use_bias,
        )
    else:
        raise ValueError(f"Unsupported activation function: {activation_fn}")
    
    return output


def triton_fused_bias_gelu(
    hidden_states: torch.Tensor,
    bias: torch.Tensor
) -> torch.Tensor:
    """
    Fused bias addition and GELU activation using Triton kernels.
    
    Args:
        hidden_states: Input tensor
        bias: Bias tensor
        
    Returns:
        Output tensor after bias addition and GELU activation
    """
    # Flatten the input for kernel processing
    input_flat = hidden_states.reshape(-1)
    n_elements = input_flat.numel()
    
    # Prepare output tensor
    output = torch.empty_like(input_flat)
    
    # Compute meta-parameters for the kernel
    block_size = 1024
    
    # Calculate grid dimensions
    grid = (triton.cdiv(n_elements, block_size),)
    
    # Launch kernel
    fused_bias_gelu_kernel[grid](
        input_flat, bias, output,
        n_elements,
        block_size=block_size,
    )
    
    # Reshape output back to original shape
    return output.reshape(hidden_states.shape)


def triton_fused_swiglu(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Fused SwiGLU computation using Triton kernels.
    
    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        w1: Weight tensor for gate projection
        w2: Weight tensor for value projection
        b1: Optional bias tensor for gate projection
        b2: Optional bias tensor for value projection
        
    Returns:
        Output tensor after SwiGLU activation
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    intermediate_size = w1.shape[0]
    
    # We need to handle both bias and no-bias cases
    use_bias = b1 is not None and b2 is not None
    
    # If bias is None, create zero tensors for the kernel
    if b1 is None:
        b1 = torch.zeros(intermediate_size, device=hidden_states.device, dtype=hidden_states.dtype)
    if b2 is None:
        b2 = torch.zeros(intermediate_size, device=hidden_states.device, dtype=hidden_states.dtype)
    
    # Initialize output tensor
    output = torch.empty((batch_size, seq_len, intermediate_size), device=hidden_states.device, dtype=hidden_states.dtype)
    
    # Compute meta-parameters for the kernel
    block_size = 16
    block_k = 16
    
    # Calculate grid dimensions
    grid = (
        batch_size,
        triton.cdiv(seq_len, block_size),
    )
    
    # Launch kernel
    fused_bias_swiglu_kernel[grid](
        hidden_states, w1, w2, b1, b2, output,
        batch_size, seq_len, hidden_size, intermediate_size,
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        w1.stride(0), w1.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        block_size=block_size, block_k=block_k,
        USE_BIAS=use_bias,
    )
    
    return output


# Create the wrapper function for the module
def fused_mlp_forward(
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: Optional[torch.Tensor],
    fc2_weight: torch.Tensor,
    fc2_bias: Optional[torch.Tensor],
    activation: str = "gelu",
    fuse_bias_gelu: bool = True,
    dropout_prob: float = 0.0,
    checkpoint_activation: bool = False,
    fc1_gate_weight: Optional[torch.Tensor] = None,
    fc1_gate_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Main entry point for fused MLP operations in the FusedMLP module.
    
    Args:
        hidden_states: Input tensor
        fc1_weight: Weight tensor for first linear layer
        fc1_bias: Bias tensor for first linear layer
        fc2_weight: Weight tensor for second linear layer
        fc2_bias: Bias tensor for second linear layer
        activation: Activation function to use ("gelu", "relu", "swiglu")
        fuse_bias_gelu: Whether to fuse bias addition with activation
        dropout_prob: Dropout probability (not yet implemented in Triton kernels)
        checkpoint_activation: Whether to checkpoint activations (not yet implemented in Triton kernels)
        fc1_gate_weight: Optional gate weight tensor for SwiGLU
        fc1_gate_bias: Optional gate bias tensor for SwiGLU
        
    Returns:
        Output tensor after MLP transformation
    """
    # Check if we have the FC1 gate projections for SwiGLU
    has_swiglu_gates = fc1_gate_weight is not None
    
    # Handle SwiGLU separately since it has a different structure
    if activation == "swiglu" and has_swiglu_gates:
        # Use specialized SwiGLU implementation
        intermediate = triton_fused_swiglu(
            hidden_states,
            fc1_gate_weight, fc1_weight,
            fc1_gate_bias, fc1_bias
        )
        
        # Apply the output projection manually for now
        # TODO: Fuse this into the SwiGLU kernel
        return torch.nn.functional.linear(intermediate, fc2_weight, fc2_bias)
    
    # For GELU and ReLU, use the fused MLP implementation
    return triton_fused_mlp(
        hidden_states,
        fc1_weight, fc1_bias,
        fc2_weight, fc2_bias,
        activation_fn=activation
    )

# Expose the SwiGLU support flag
fused_mlp_forward.supports_swiglu = True


# Benchmarking functions
def benchmark_fused_mlp(
    batch_size: int = 8,
    seq_len: int = 512,
    hidden_size: int = 768,
    intermediate_size: int = 3072,
    activation_fn: str = "gelu",
    num_warmup: int = 10,
    num_iter: int = 100
) -> Dict[str, float]:
    """
    Benchmark the performance of the fused MLP implementations.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden size
        intermediate_size: Intermediate size
        activation_fn: Activation function to use
        num_warmup: Number of warmup iterations
        num_iter: Number of iterations to measure
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    # Create inputs
    hidden_states = torch.randn((batch_size, seq_len, hidden_size), device="cuda", dtype=torch.float16)
    fc1_weight = torch.randn((intermediate_size, hidden_size), device="cuda", dtype=torch.float16)
    fc1_bias = torch.randn((intermediate_size,), device="cuda", dtype=torch.float16)
    fc2_weight = torch.randn((hidden_size, intermediate_size), device="cuda", dtype=torch.float16)
    fc2_bias = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
    
    # Add gate for SwiGLU if needed
    if activation_fn == "swiglu":
        fc1_gate_weight = torch.randn((intermediate_size, hidden_size), device="cuda", dtype=torch.float16)
        fc1_gate_bias = torch.randn((intermediate_size,), device="cuda", dtype=torch.float16)
    else:
        fc1_gate_weight = None
        fc1_gate_bias = None
    
    # Warmup
    for _ in range(num_warmup):
        _ = fused_mlp_forward(
            hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
            activation=activation_fn, fc1_gate_weight=fc1_gate_weight, fc1_gate_bias=fc1_gate_bias
        )
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Time the fused MLP
    start_time = time.time()
    for _ in range(num_iter):
        _ = fused_mlp_forward(
            hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
            activation=activation_fn, fc1_gate_weight=fc1_gate_weight, fc1_gate_bias=fc1_gate_bias
        )
    torch.cuda.synchronize()
    fused_time = (time.time() - start_time) / num_iter * 1000  # ms
    
    return {
        "fused_mlp_time_ms": fused_time,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "activation_fn": activation_fn,
    }


def compare_with_standard_mlp(
    batch_size: int = 8,
    seq_len: int = 512,
    hidden_size: int = 768,
    intermediate_size: int = 3072,
    activation_fn: str = "gelu",
    num_warmup: int = 10,
    num_iter: int = 100
) -> Dict[str, float]:
    """
    Compare the performance of fused MLP with standard PyTorch implementation.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden size
        intermediate_size: Intermediate size
        activation_fn: Activation function to use
        num_warmup: Number of warmup iterations
        num_iter: Number of iterations to measure
        
    Returns:
        Dictionary with timing results and speedup
    """
    import time
    
    # Create inputs
    hidden_states = torch.randn((batch_size, seq_len, hidden_size), device="cuda", dtype=torch.float16)
    fc1_weight = torch.randn((intermediate_size, hidden_size), device="cuda", dtype=torch.float16)
    fc1_bias = torch.randn((intermediate_size,), device="cuda", dtype=torch.float16)
    fc2_weight = torch.randn((hidden_size, intermediate_size), device="cuda", dtype=torch.float16)
    fc2_bias = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
    
    # Create standard PyTorch implementation
    class StandardMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(hidden_size, intermediate_size)
            self.fc2 = torch.nn.Linear(intermediate_size, hidden_size)
            
            # Initialize with same weights
            self.fc1.weight.data.copy_(fc1_weight)
            self.fc1.bias.data.copy_(fc1_bias)
            self.fc2.weight.data.copy_(fc2_weight)
            self.fc2.bias.data.copy_(fc2_bias)
            
        def forward(self, x):
            if activation_fn == "gelu":
                return self.fc2(torch.nn.functional.gelu(self.fc1(x)))
            elif activation_fn == "relu":
                return self.fc2(torch.nn.functional.relu(self.fc1(x)))
            else:
                raise ValueError(f"Unsupported activation: {activation_fn}")
    
    mlp = StandardMLP().cuda().half()
    
    # Add gate for SwiGLU if needed
    if activation_fn == "swiglu":
        fc1_gate_weight = torch.randn((intermediate_size, hidden_size), device="cuda", dtype=torch.float16)
        fc1_gate_bias = torch.randn((intermediate_size,), device="cuda", dtype=torch.float16)
    else:
        fc1_gate_weight = None
        fc1_gate_bias = None
    
    # Warmup standard implementation
    for _ in range(num_warmup):
        _ = mlp(hidden_states)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Time the standard MLP
    start_time = time.time()
    for _ in range(num_iter):
        _ = mlp(hidden_states)
    torch.cuda.synchronize()
    standard_time = (time.time() - start_time) / num_iter * 1000  # ms
    
    # Warmup fused implementation
    for _ in range(num_warmup):
        _ = fused_mlp_forward(
            hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
            activation=activation_fn, fc1_gate_weight=fc1_gate_weight, fc1_gate_bias=fc1_gate_bias
        )
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Time the fused MLP
    start_time = time.time()
    for _ in range(num_iter):
        _ = fused_mlp_forward(
            hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
            activation=activation_fn, fc1_gate_weight=fc1_gate_weight, fc1_gate_bias=fc1_gate_bias
        )
    torch.cuda.synchronize()
    fused_time = (time.time() - start_time) / num_iter * 1000  # ms
    
    # Calculate speedup
    speedup = standard_time / fused_time
    
    return {
        "standard_mlp_time_ms": standard_time,
        "fused_mlp_time_ms": fused_time,
        "speedup": speedup,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "activation_fn": activation_fn,
    }


def memory_usage_comparison(
    batch_size: int = 8,
    seq_len: int = 512,
    hidden_size: int = 768,
    intermediate_size: int = 3072
) -> Dict[str, int]:
    """
    Compare memory usage between standard and fused MLP implementations.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden size
        intermediate_size: Intermediate size
        
    Returns:
        Dictionary with memory usage statistics
    """
    # Calculate theoretical memory usage
    hidden_states_size = batch_size * seq_len * hidden_size * 2  # 2 bytes for fp16
    intermediate_states_size = batch_size * seq_len * intermediate_size * 2  # 2 bytes for fp16
    
    # Standard implementation memory usage (peak)
    # It needs to store:
    # 1. Input hidden states
    # 2. Intermediate activation after first FC
    # 3. Intermediate activation after activation function
    # 4. Output hidden states
    standard_memory = hidden_states_size + intermediate_states_size * 2 + hidden_states_size
    
    # Fused implementation memory usage (peak)
    # It only needs to store:
    # 1. Input hidden states
    # 2. Output hidden states
    # (Intermediate results are kept in registers/shared memory)
    fused_memory = hidden_states_size + hidden_states_size
    
    # Calculate memory savings
    memory_saved = standard_memory - fused_memory
    memory_reduction_percent = (memory_saved / standard_memory) * 100
    
    return {
        "standard_memory_bytes": standard_memory,
        "fused_memory_bytes": fused_memory,
        "memory_saved_bytes": memory_saved,
        "memory_reduction_percent": memory_reduction_percent,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
    }