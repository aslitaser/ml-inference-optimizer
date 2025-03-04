"""
Benchmark runners for ML inference optimization.

This module provides classes for running benchmarks on ML models with various
optimization techniques. It allows measuring throughput, latency, memory usage,
and scaling efficiency across different configurations.
"""

import os
import time
import json
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict

from benchmarks.metrics import (
    calculate_throughput,
    calculate_latency_statistics,
    calculate_memory_efficiency,
    calculate_gpu_utilization,
    calculate_flops_utilization
)
from utils.gpu_utils import get_gpu_memory_usage
from profiling.torch_profiler import profile_model_execution


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    model_name: str  # Name of the model to benchmark
    batch_sizes: List[int]  # Batch sizes to test
    sequence_lengths: List[int]  # Sequence lengths to test
    optimization_types: List[str]  # Optimizations to benchmark
    num_iterations: int = 100  # Number of iterations per test
    warmup_iterations: int = 10  # Number of warmup iterations
    devices: List[str] = None  # Devices to test on
    precision: str = "fp16"  # Precision for testing
    save_results: bool = True  # Whether to save results
    profiling: bool = False  # Whether to enable profiling
    validate_outputs: bool = True  # Whether to validate outputs against baseline
    
    def __post_init__(self):
        """Set default values for optional fields if not provided."""
        if self.devices is None:
            self.devices = ["cuda:0"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return asdict(self)


class BenchmarkRunner:
    """Base class for running benchmarks on ML models."""
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark runner.
        
        Args:
            config: Configuration for the benchmark.
        """
        self.config = config
        self.results_dir = os.path.join("results", config.model_name)
        
        # Create results directory if it doesn't exist
        if config.save_results and not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)
        
        # Set device
        self.primary_device = torch.device(config.devices[0])
        
        # Set precision
        self.dtype = self._get_dtype_from_precision(config.precision)
    
    def _get_dtype_from_precision(self, precision: str) -> torch.dtype:
        """
        Get PyTorch dtype from precision string.
        
        Args:
            precision: Precision string (e.g., "fp16", "fp32", "bf16").
            
        Returns:
            PyTorch dtype.
        """
        precision_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "int8": torch.int8
        }
        
        if precision not in precision_map:
            raise ValueError(f"Unsupported precision: {precision}")
        
        return precision_map[precision]
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """
        Run benchmarks for all configurations.
        
        Returns:
            Dictionary containing benchmark results.
        """
        results = {
            "config": self.config.to_dict(),
            "timestamp": time.time(),
            "benchmarks": {}
        }
        
        # Setup model variants
        model_variants = self.setup_model_variants()
        
        # Run benchmarks for each configuration
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                inputs = self.generate_test_inputs(batch_size, seq_len)
                
                # Get baseline outputs if validation is enabled
                baseline_outputs = None
                if self.config.validate_outputs and "baseline" in model_variants:
                    with torch.no_grad():
                        baseline_model = model_variants["baseline"].to(self.primary_device)
                        baseline_inputs = {k: v.to(self.primary_device) for k, v in inputs.items()}
                        baseline_outputs = baseline_model(**baseline_inputs)
                
                config_results = {}
                
                # Run benchmarks for each optimization type
                for opt_type, model in model_variants.items():
                    model = model.to(self.primary_device)
                    model_inputs = {k: v.to(self.primary_device) for k, v in inputs.items()}
                    
                    # Run warmup iterations
                    with torch.no_grad():
                        for _ in range(self.config.warmup_iterations):
                            _ = model(**model_inputs)
                    
                    # Measure performance
                    perf_metrics = self.measure_performance(model, model_inputs)
                    
                    # Validate outputs if needed
                    if self.config.validate_outputs and baseline_outputs is not None and opt_type != "baseline":
                        with torch.no_grad():
                            outputs = model(**model_inputs)
                            validation_result = self.validate_model_outputs(baseline_outputs, outputs)
                            perf_metrics["output_validation"] = validation_result
                    
                    config_results[opt_type] = perf_metrics
                
                results["benchmarks"][f"bs{batch_size}_seq{seq_len}"] = config_results
        
        # Save results if enabled
        if self.config.save_results:
            filename = f"{int(time.time())}_{self.config.model_name}.json"
            self.save_benchmark_results(results, filename)
        
        return results
    
    def setup_model_variants(self) -> Dict[str, nn.Module]:
        """
        Set up model variants for benchmarking.
        
        This method should be implemented by subclasses to set up the models
        with different optimization strategies.
        
        Returns:
            Dictionary mapping optimization types to model instances.
        """
        raise NotImplementedError("Subclasses must implement setup_model_variants()")
    
    def generate_test_inputs(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """
        Generate test inputs for benchmarking.
        
        Args:
            batch_size: Batch size to use.
            seq_len: Sequence length to use.
            
        Returns:
            Dictionary of input tensors.
        """
        raise NotImplementedError("Subclasses must implement generate_test_inputs()")
    
    def measure_performance(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Measure model performance.
        
        Args:
            model: Model to benchmark.
            inputs: Model inputs.
            
        Returns:
            Dictionary containing performance metrics.
        """
        model.eval()
        batch_size = inputs.get("input_ids", next(iter(inputs.values()))).shape[0]
        seq_len = inputs.get("input_ids", next(iter(inputs.values()))).shape[1]
        
        # Initialize metrics
        latencies = []
        memory_usage = []
        
        # Enable profiling if requested
        profile_results = None
        if self.config.profiling:
            profile_results = profile_model_execution(model, inputs, self.config.num_iterations // 10)
        
        # Measure performance
        torch.cuda.synchronize()
        with torch.no_grad():
            for _ in range(self.config.num_iterations):
                memory_before = get_gpu_memory_usage(self.primary_device)
                
                start_time = time.time()
                _ = model(**inputs)
                torch.cuda.synchronize()
                end_time = time.time()
                
                memory_after = get_gpu_memory_usage(self.primary_device)
                
                latencies.append(end_time - start_time)
                memory_usage.append(memory_after - memory_before)
        
        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies)
        throughput = calculate_throughput(batch_size, seq_len, avg_latency)
        latency_stats = calculate_latency_statistics(latencies)
        avg_memory = sum(memory_usage) / len(memory_usage)
        
        # Compile results
        results = {
            "avg_latency_ms": avg_latency * 1000,  # Convert to ms
            "throughput_samples_per_sec": throughput,
            "latency_p50_ms": latency_stats["p50"] * 1000,
            "latency_p90_ms": latency_stats["p90"] * 1000,
            "latency_p95_ms": latency_stats["p95"] * 1000,
            "latency_p99_ms": latency_stats["p99"] * 1000,
            "memory_usage_mb": avg_memory,
            "batch_size": batch_size,
            "sequence_length": seq_len
        }
        
        # Add profiling results if available
        if profile_results:
            results["profiling"] = profile_results
        
        return results
    
    def validate_model_outputs(
        self, 
        baseline_outputs: Any, 
        optimized_outputs: Any, 
        rtol: float = 1e-3, 
        atol: float = 1e-3
    ) -> bool:
        """
        Validate model outputs against baseline.
        
        Args:
            baseline_outputs: Outputs from the baseline model.
            optimized_outputs: Outputs from the optimized model.
            rtol: Relative tolerance for validation.
            atol: Absolute tolerance for validation.
            
        Returns:
            True if outputs are valid, False otherwise.
        """
        # Handle different output types
        if isinstance(baseline_outputs, torch.Tensor) and isinstance(optimized_outputs, torch.Tensor):
            return torch.allclose(baseline_outputs, optimized_outputs, rtol=rtol, atol=atol)
        elif isinstance(baseline_outputs, tuple) and isinstance(optimized_outputs, tuple):
            if len(baseline_outputs) != len(optimized_outputs):
                return False
            
            for b_out, o_out in zip(baseline_outputs, optimized_outputs):
                if isinstance(b_out, torch.Tensor) and isinstance(o_out, torch.Tensor):
                    if not torch.allclose(b_out, o_out, rtol=rtol, atol=atol):
                        return False
            
            return True
        elif isinstance(baseline_outputs, dict) and isinstance(optimized_outputs, dict):
            if baseline_outputs.keys() != optimized_outputs.keys():
                return False
            
            for key in baseline_outputs:
                b_out = baseline_outputs[key]
                o_out = optimized_outputs[key]
                
                if isinstance(b_out, torch.Tensor) and isinstance(o_out, torch.Tensor):
                    if not torch.allclose(b_out, o_out, rtol=rtol, atol=atol):
                        return False
            
            return True
        else:
            # Can't validate unknown types, return False for safety
            return False
    
    def save_benchmark_results(self, results: Dict[str, Any], filename: str) -> str:
        """
        Save benchmark results to a file.
        
        Args:
            results: Benchmark results to save.
            filename: Name of the file to save to.
            
        Returns:
            Path to the saved file.
        """
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert tensor values to serializable types
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(i) for i in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_tensors(i) for i in obj)
            else:
                return obj
        
        serializable_results = convert_tensors(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return filepath


class ThroughputBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner focused on measuring model throughput."""
    
    def measure_performance(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Measure model throughput performance.
        
        Extends the base implementation with throughput-specific measurements.
        
        Args:
            model: Model to benchmark.
            inputs: Model inputs.
            
        Returns:
            Dictionary containing performance metrics.
        """
        # Get base performance measurements
        results = super().measure_performance(model, inputs)
        
        # Add throughput-specific metrics
        batch_size = inputs.get("input_ids", next(iter(inputs.values()))).shape[0]
        seq_len = inputs.get("input_ids", next(iter(inputs.values()))).shape[1]
        
        # Calculate tokens per second
        tokens_per_sec = batch_size * seq_len / (results["avg_latency_ms"] / 1000)
        results["tokens_per_second"] = tokens_per_sec
        
        return results


class LatencyBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner focused on measuring model latency."""
    
    def measure_performance(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Measure model latency performance.
        
        Extends the base implementation with latency-specific measurements.
        
        Args:
            model: Model to benchmark.
            inputs: Model inputs.
            
        Returns:
            Dictionary containing performance metrics.
        """
        # Get base performance measurements
        results = super().measure_performance(model, inputs)
        
        # Add more detailed latency metrics
        batch_size = inputs.get("input_ids", next(iter(inputs.values()))).shape[0]
        
        # Run additional single-sample latency tests
        if batch_size > 1:
            single_sample_inputs = {
                k: v[:1] for k, v in inputs.items() if isinstance(v, torch.Tensor)
            }
            
            latencies = []
            torch.cuda.synchronize()
            with torch.no_grad():
                for _ in range(self.config.num_iterations):
                    start_time = time.time()
                    _ = model(**single_sample_inputs)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    latencies.append(end_time - start_time)
            
            latency_stats = calculate_latency_statistics(latencies)
            results["single_sample_latency_ms"] = sum(latencies) / len(latencies) * 1000
            results["single_sample_p99_ms"] = latency_stats["p99"] * 1000
        
        return results


class MemoryBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner focused on measuring memory usage."""
    
    def measure_performance(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Measure model memory usage.
        
        Extends the base implementation with memory-specific measurements.
        
        Args:
            model: Model to benchmark.
            inputs: Model inputs.
            
        Returns:
            Dictionary containing performance metrics.
        """
        # Get base performance measurements
        results = super().measure_performance(model, inputs)
        
        # Calculate model parameters and size
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results["num_parameters"] = num_params
        
        # Calculate parameter memory in MB
        param_memory_bytes = 0
        for param in model.parameters():
            param_bytes = param.nelement() * param.element_size()
            param_memory_bytes += param_bytes
        
        results["parameter_memory_mb"] = param_memory_bytes / (1024 * 1024)
        
        # Calculate memory efficiency
        if "theoretical_memory" in results:
            memory_efficiency = calculate_memory_efficiency(
                results["theoretical_memory"], 
                results["memory_usage_mb"]
            )
            results["memory_efficiency"] = memory_efficiency
        
        return results


class ScalingBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner focused on measuring scaling across multiple GPUs."""
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """
        Run scaling benchmarks across multiple GPUs.
        
        Returns:
            Dictionary containing benchmark results.
        """
        results = {
            "config": self.config.to_dict(),
            "timestamp": time.time(),
            "benchmarks": {}
        }
        
        # Setup model variants
        model_variants = self.setup_model_variants()
        
        # Run single-GPU baseline first
        single_gpu_results = {}
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                inputs = self.generate_test_inputs(batch_size, seq_len)
                
                for opt_type, model in model_variants.items():
                    model = model.to(self.primary_device)
                    model_inputs = {k: v.to(self.primary_device) for k, v in inputs.items()}
                    
                    # Run warmup iterations
                    with torch.no_grad():
                        for _ in range(self.config.warmup_iterations):
                            _ = model(**model_inputs)
                    
                    # Measure performance
                    perf_metrics = self.measure_performance(model, model_inputs)
                    single_gpu_results[f"{opt_type}_bs{batch_size}_seq{seq_len}"] = perf_metrics
        
        results["single_gpu"] = single_gpu_results
        
        # Run multi-GPU benchmarks if multiple devices are specified
        if len(self.config.devices) > 1:
            multi_gpu_results = {}
            
            # Implement distributed model setup and benchmarking
            # This is a simplified version - in practice, this would use
            # torch.distributed or model parallelism techniques
            
            results["multi_gpu"] = multi_gpu_results
        
        # Calculate scaling efficiency
        if len(self.config.devices) > 1:
            scaling_results = {}
            
            for single_key, single_metrics in single_gpu_results.items():
                if single_key in results["multi_gpu"]:
                    multi_metrics = results["multi_gpu"][single_key]
                    
                    # Calculate speedup
                    single_throughput = single_metrics["throughput_samples_per_sec"]
                    multi_throughput = multi_metrics["throughput_samples_per_sec"]
                    
                    scaling_efficiency = (multi_throughput / single_throughput) / len(self.config.devices)
                    
                    scaling_results[single_key] = {
                        "speedup": multi_throughput / single_throughput,
                        "scaling_efficiency": scaling_efficiency,
                        "num_gpus": len(self.config.devices)
                    }
            
            results["scaling_efficiency"] = scaling_results
        
        # Save results if enabled
        if self.config.save_results:
            filename = f"{int(time.time())}_{self.config.model_name}_scaling.json"
            self.save_benchmark_results(results, filename)
        
        return results