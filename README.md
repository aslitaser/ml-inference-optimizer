# ML Inference Optimizer

A high-performance optimization framework for transformer-based inference with a focus on generative media models.

## Overview

ML Inference Optimizer provides tools and implementations to improve inference performance for large language models and diffusion models. 

## Key Features

- **Advanced Attention Mechanisms**
  - Flash Attention 3 implementation with tiled block-based computation
  - Ring Attention for distributed processing of extremely long sequences
  - Memory-efficient attention patterns that reduce complexity from O(N²) to O(N)

- **Fusion Optimizations**
  - FusedMLP implementation that keeps intermediates in fast memory
  - Kernel fusion for reduced memory bandwidth requirements
  - Support for various activation functions (GELU, SwiGLU, etc.)

- **Parallelism Strategies**
  - Tensor parallelism for distributed model weights
  - Sequence parallelism for handling long contexts
  - Multi-dimensional parallelism orchestration

- **Performance Analysis**
  - Profiling system for identifying bottlenecks
  - Memory usage tracking and optimization
  - Visualization dashboard for performance metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/aslitaser/ml-inference-optimizer.git
cd ml-inference-optimizer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ and compatible GPU
- Triton 2.0+ (for optimized kernels)

## Quick Start

```python
from ml_inference_optimizer import Optimizer
from ml_inference_optimizer.baseline import load_model

# Load a transformer model
model = load_model("gpt2")

# Create an optimizer instance
optimizer = Optimizer(model)

# Profile the model to identify bottlenecks
bottlenecks = optimizer.profile()
print(f"Main bottlenecks: {bottlenecks}")

# Apply optimizations
optimized_model = optimizer.optimize(
    use_flash_attention=True,
    use_fused_mlp=True,
    tensor_parallel_size=1  # Set to >1 for multi-GPU
)

# Run optimized inference
output = optimized_model.generate(
    input_text="Hello, world!",
    max_new_tokens=100
)
```

## Optimization Details

### Flash Attention 3

Our FA3 implementation uses tiled block-based computation to reduce memory complexity while maintaining computational efficiency:

- Processes attention in manageable blocks to fit in GPU shared memory
- Implements interleaved matmul and softmax operations
- Supports causal masking for autoregressive generation
- Compatible with both single-GPU and multi-GPU setups

### Ring Attention

Ring Attention enables processing of extremely long sequences by distributing workload in a ring communication pattern:

- Scales memory requirements linearly with number of GPUs
- Enables context lengths of 65K+ tokens with standard hardware
- Optimizes communication patterns to overlap with computation
- Integrates with other optimizations like Flash Attention

### FusedMLP

FusedMLP implementation significantly improves performance for feed-forward networks:

- Combines multiple operations into single kernels
- Keeps intermediate activations in shared memory
- Supports various activation functions including SwiGLU
- Reduces memory bandwidth requirements by 40-60%

## Benchmarks

| Model | Optimization | Throughput Improvement | Latency Reduction | Memory Reduction |
|-------|-------------|------------------------|-------------------|------------------|
| GPT-2 | Baseline    | 1.0x                   | 1.0x              | 1.0x             |
| GPT-2 | FA3         | 2.3x                   | 1.8x              | 3.5x             |
| GPT-2 | FusedMLP    | 1.8x                   | 1.6x              | 1.7x             |
| GPT-2 | Combined    | 3.7x                   | 2.9x              | 4.2x             |

*For long sequences (4096 tokens), improvements are even more*

## Project Structure

```
ml-inference-optimizer/
├── baseline/               # Baseline model implementations
├── benchmarks/             # Performance benchmarking tools
├── config/                 # Configuration system
├── dashboard/              # Performance visualization
├── kernels/                # Optimized kernel implementations
│   ├── attention/          # Attention optimizations
│   ├── mlp/                # MLP optimizations
│   └── triton/             # Triton kernels
├── parallelism/            # Parallel execution strategies
├── profiling/              # Performance analysis tools
└── utils/                  # Common utilities
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project builds on research from multiple papers about FlashAttention, Ring Attention, and others
