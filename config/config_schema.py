from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field


class HardwareConfig(BaseModel):
    """Hardware configuration settings."""
    gpu_count: int = Field(1, description="Number of GPUs to use")
    gpu_type: str = Field("", description="Type of GPU (e.g., 'A100', 'V100')")
    gpu_memory_gb: float = Field(0.0, description="GPU memory in GB")
    cpu_count: int = Field(0, description="Number of CPU cores to use")


class KernelConfig(BaseModel):
    """Configuration for optimized kernels."""
    use_flash_attention: bool = Field(False, description="Use Flash Attention")
    use_ring_attention: bool = Field(False, description="Use Ring Attention")
    use_fused_mlp: bool = Field(False, description="Use fused MLP operations")
    use_custom_layernorm: bool = Field(False, description="Use custom LayerNorm kernel")
    use_triton_kernels: bool = Field(False, description="Use Triton-based GPU kernels")


class ParallelismConfig(BaseModel):
    """Configuration for model parallelism strategies."""
    tensor_parallel_size: int = Field(1, description="Tensor parallelism degree")
    sequence_parallel: bool = Field(False, description="Enable sequence parallelism")
    pipeline_parallel_size: int = Field(1, description="Pipeline parallelism degree")
    data_parallel_size: int = Field(1, description="Data parallelism degree")
    communication_dtype: str = Field("fp32", description="Data type for communication")
    optimization_level: int = Field(0, description="Auto-parallelism optimization level (0-3)")


class ModelConfig(BaseModel):
    """Configuration for the model settings."""
    model_name_or_path: str = Field("", description="Model name or path")
    model_type: str = Field("", description="Model type (e.g., 'transformer', 'diffusion')")
    precision: Literal["fp32", "fp16", "bf16"] = Field("fp32", description="Model precision")
    trust_remote_code: bool = Field(False, description="Trust remote code when loading model")
    max_batch_size: int = Field(1, description="Maximum batch size for inference")
    max_sequence_length: int = Field(512, description="Maximum sequence length")
    use_cache: bool = Field(True, description="Use KV cache for transformer models")
    cpu_offload: bool = Field(False, description="Offload model parts to CPU")


class BenchmarkConfig(BaseModel):
    """Configuration for benchmarking runs."""
    batch_sizes: List[int] = Field([1], description="Batch sizes to benchmark")
    sequence_lengths: List[int] = Field([512], description="Sequence lengths to benchmark")
    num_iterations: int = Field(100, description="Number of iterations for benchmarking")
    warmup_iterations: int = Field(10, description="Number of warmup iterations")
    report_path: str = Field("benchmark_results", description="Path to save benchmark reports")
    metrics: List[str] = Field(["latency", "throughput", "memory"], description="Metrics to collect")
    compare_with_baseline: bool = Field(True, description="Compare with baseline performance")


class ProfilingConfig(BaseModel):
    """Configuration for profiling and bottleneck analysis."""
    enable_profiling: bool = Field(False, description="Enable detailed profiling")
    profile_iterations: int = Field(10, description="Number of iterations to profile")
    save_timeline: bool = Field(False, description="Save profiler timeline")
    memory_profiling: bool = Field(False, description="Track memory usage")
    bottleneck_analysis: bool = Field(False, description="Run bottleneck analysis")
    profiler_output_dir: str = Field("profiling_results", description="Directory to save profiling results")


class DashboardConfig(BaseModel):
    """Configuration for the dashboard."""
    enable_dashboard: bool = Field(False, description="Enable web dashboard")
    port: int = Field(8050, description="Dashboard port")
    host: str = Field("127.0.0.1", description="Dashboard host")
    update_interval_seconds: int = Field(5, description="Dashboard update interval")


class OptimizerConfig(BaseModel):
    """Root configuration for the ML inference optimizer."""
    hardware: HardwareConfig = Field(default_factory=HardwareConfig, description="Hardware configuration")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    kernels: KernelConfig = Field(default_factory=KernelConfig, description="Kernel optimizations")
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig, description="Parallelism strategies")
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig, description="Benchmark settings")
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig, description="Profiling settings")
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig, description="Dashboard settings")
    output_dir: str = Field("output", description="Main output directory")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO", description="Logging level")