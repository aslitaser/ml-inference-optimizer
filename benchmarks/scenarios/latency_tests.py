"""
Latency benchmark scenarios for ML inference optimization.

This module provides specific benchmark scenarios focused on measuring
latency performance for different optimization techniques and configurations.
"""

import os
import time
import threading
import queue
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

from benchmarks.runners import BenchmarkConfig, LatencyBenchmarkRunner
from benchmarks.metrics import calculate_latency_statistics
from baseline.model_loader import load_model
from baseline.inference import create_dummy_inputs

logger = logging.getLogger(__name__)


def test_standard_latency(model_name: str, optimization_types: List[str]) -> Dict[str, Any]:
    """
    Run standard latency benchmark for a model with various optimization techniques.
    
    This benchmark measures the basic latency of the model with different
    optimization techniques, focusing on single-sample latency.
    
    Args:
        model_name: Name of the model to benchmark.
        optimization_types: List of optimization techniques to benchmark.
        
    Returns:
        Dictionary containing benchmark results.
    """
    logger.info(f"Running standard latency benchmark for {model_name}")
    logger.info(f"Optimization types: {optimization_types}")
    
    # Define benchmark configuration
    config = BenchmarkConfig(
        model_name=model_name,
        batch_sizes=[1],  # Focus on single-sample latency
        sequence_lengths=[128, 512, 1024],
        optimization_types=optimization_types,
        num_iterations=100,  # More iterations for stable latency stats
        warmup_iterations=20,
        precision="fp16"
    )
    
    # Create custom runner for latency benchmarks
    class ModelLatencyRunner(LatencyBenchmarkRunner):
        def setup_model_variants(self) -> Dict[str, nn.Module]:
            models = {}
            
            # Load baseline model
            if "baseline" in self.config.optimization_types:
                models["baseline"] = load_model(
                    self.config.model_name, 
                    precision=self.config.precision,
                    device=self.primary_device
                )
            
            # Load optimized model variants
            for opt_type in self.config.optimization_types:
                if opt_type != "baseline":
                    # This would call specialized loading functions for each optimization type
                    models[opt_type] = load_model(
                        self.config.model_name,
                        precision=self.config.precision,
                        device=self.primary_device,
                        optimization=opt_type
                    )
            
            return models
            
        def generate_test_inputs(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
            return create_dummy_inputs(
                model_name=self.config.model_name,
                batch_size=batch_size,
                seq_len=seq_len,
                device=self.primary_device,
                dtype=self.dtype
            )
    
    # Create and run the benchmark
    runner = ModelLatencyRunner(config)
    results = runner.run_benchmarks()
    
    logger.info(f"Completed standard latency benchmark for {model_name}")
    
    return results


def test_tail_latency(model_name: str, percentiles: List[float] = None) -> Dict[str, Any]:
    """
    Test tail latency characteristics with a focus on high percentiles.
    
    This benchmark runs many iterations to get a detailed latency distribution,
    focusing on measuring the tail latency at high percentiles.
    
    Args:
        model_name: Name of the model to benchmark.
        percentiles: List of percentiles to measure (e.g., [50, 90, 95, 99, 99.9]).
        
    Returns:
        Dictionary containing benchmark results.
    """
    if percentiles is None:
        percentiles = [50, 90, 95, 99, 99.9, 99.99]
    
    logger.info(f"Running tail latency benchmark for {model_name}")
    logger.info(f"Measuring percentiles: {percentiles}")
    
    # Define benchmark configuration
    config = BenchmarkConfig(
        model_name=model_name,
        batch_sizes=[1],  # Focus on single-sample latency
        sequence_lengths=[128, 1024],
        optimization_types=["baseline"],  # Usually baseline is enough for tail latency testing
        num_iterations=1000,  # Many iterations for stable high percentiles
        warmup_iterations=100,
        precision="fp16"
    )
    
    # Create custom runner for tail latency benchmarks
    class TailLatencyRunner(LatencyBenchmarkRunner):
        def setup_model_variants(self) -> Dict[str, nn.Module]:
            return {
                "baseline": load_model(
                    self.config.model_name, 
                    precision=self.config.precision,
                    device=self.primary_device
                )
            }
            
        def generate_test_inputs(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
            return create_dummy_inputs(
                model_name=self.config.model_name,
                batch_size=batch_size,
                seq_len=seq_len,
                device=self.primary_device,
                dtype=self.dtype
            )
        
        def measure_performance(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
            # First get standard metrics
            metrics = super().measure_performance(model, inputs)
            
            # Collect all latencies
            model.eval()
            latencies = []
            
            torch.cuda.synchronize()
            with torch.no_grad():
                for _ in range(self.config.num_iterations):
                    start_time = time.time()
                    _ = model(**inputs)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate custom percentiles
            sorted_latencies = sorted(latencies)
            for p in percentiles:
                idx = min(int(len(sorted_latencies) * (p / 100)), len(sorted_latencies) - 1)
                metrics[f"p{p}_latency_ms"] = sorted_latencies[idx]
            
            # Calculate jitter (standard deviation of latencies)
            metrics["jitter_ms"] = np.std(latencies)
            
            return metrics
    
    # Create and run the benchmark
    runner = TailLatencyRunner(config)
    results = runner.run_benchmarks()
    
    # Add detailed percentile information to results
    results["percentiles_measured"] = percentiles
    
    logger.info(f"Completed tail latency benchmark for {model_name}")
    
    return results


def test_latency_under_load(model_name: str, concurrent_requests: int) -> Dict[str, Any]:
    """
    Test latency characteristics when the system is under load.
    
    This benchmark simulates a scenario where multiple concurrent requests
    are being processed by the model, measuring how latency is affected.
    
    Args:
        model_name: Name of the model to benchmark.
        concurrent_requests: Number of concurrent requests to simulate.
        
    Returns:
        Dictionary containing benchmark results.
    """
    logger.info(f"Running latency under load benchmark for {model_name}")
    logger.info(f"Concurrent requests: {concurrent_requests}")
    
    # Define benchmark configuration
    config = BenchmarkConfig(
        model_name=model_name,
        batch_sizes=[1],  # Simulate single-sample requests
        sequence_lengths=[128],
        optimization_types=["baseline"],
        num_iterations=50,
        warmup_iterations=10,
        precision="fp16"
    )
    
    # Load model once to be shared across threads
    device = torch.device("cuda:0")
    model = load_model(
        model_name, 
        precision=config.precision,
        device=device
    )
    model.eval()
    
    # Prepare inputs
    test_inputs = create_dummy_inputs(
        model_name=model_name,
        batch_size=1,
        seq_len=config.sequence_lengths[0],
        device=device,
        dtype=torch.float16 if config.precision == "fp16" else torch.float32
    )
    
    # Function for worker threads
    def worker(request_queue, result_queue):
        with torch.no_grad():
            while True:
                try:
                    # Get request from queue
                    inputs = request_queue.get(timeout=1.0)
                    if inputs is None:
                        break
                    
                    # Process request and measure time
                    start_time = time.time()
                    _ = model(**inputs)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    # Put result in result queue
                    result_queue.put(end_time - start_time)
                    
                    # Mark as done
                    request_queue.task_done()
                    
                except queue.Empty:
                    break
    
    # Create queues for requests and results
    request_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Create and start worker threads
    threads = []
    for _ in range(concurrent_requests):
        t = threading.Thread(target=worker, args=(request_queue, result_queue))
        t.daemon = True
        t.start()
        threads.append(t)
    
    # Add requests to the queue
    num_requests = 200
    for _ in range(num_requests):
        request_queue.put(test_inputs)
    
    # Wait for all requests to be processed
    request_queue.join()
    
    # Tell workers to exit
    for _ in range(concurrent_requests):
        request_queue.put(None)
    
    # Wait for all workers to finish
    for t in threads:
        t.join()
    
    # Collect all latencies
    latencies = []
    while not result_queue.empty():
        latencies.append(result_queue.get() * 1000)  # Convert to ms
    
    # Calculate latency statistics
    latency_stats = calculate_latency_statistics(latencies)
    
    # Create results
    results = {
        "model_name": model_name,
        "concurrent_requests": concurrent_requests,
        "num_requests": num_requests,
        "sequence_length": config.sequence_lengths[0],
        "precision": config.precision,
        "latency_stats": latency_stats,
        "mean_latency_ms": latency_stats["mean"] * 1000,
        "median_latency_ms": latency_stats["median"] * 1000,
        "p90_latency_ms": latency_stats["p90"] * 1000,
        "p95_latency_ms": latency_stats["p95"] * 1000,
        "p99_latency_ms": latency_stats["p99"] * 1000,
    }
    
    logger.info(f"Completed latency under load benchmark for {model_name}")
    
    return results


def test_first_token_latency(model_name: str, optimization_types: List[str]) -> Dict[str, Any]:
    """
    Test latency to generate the first token in sequence generation.
    
    This benchmark focuses on measuring the time it takes for the model to
    generate the first token, which is critical for perceived latency.
    
    Args:
        model_name: Name of the model to benchmark.
        optimization_types: List of optimization techniques to benchmark.
        
    Returns:
        Dictionary containing benchmark results.
    """
    logger.info(f"Running first token latency benchmark for {model_name}")
    logger.info(f"Optimization types: {optimization_types}")
    
    # Define benchmark configuration
    config = BenchmarkConfig(
        model_name=model_name,
        batch_sizes=[1],
        sequence_lengths=[128],
        optimization_types=optimization_types,
        num_iterations=50,
        warmup_iterations=10,
        precision="fp16"
    )
    
    # Create custom runner for first token latency benchmarks
    class FirstTokenLatencyRunner(LatencyBenchmarkRunner):
        def setup_model_variants(self) -> Dict[str, nn.Module]:
            models = {}
            
            # Load baseline model
            if "baseline" in self.config.optimization_types:
                models["baseline"] = load_model(
                    self.config.model_name, 
                    precision=self.config.precision,
                    device=self.primary_device
                )
            
            # Load optimized model variants
            for opt_type in self.config.optimization_types:
                if opt_type != "baseline":
                    models[opt_type] = load_model(
                        self.config.model_name,
                        precision=self.config.precision,
                        device=self.primary_device,
                        optimization=opt_type
                    )
            
            return models
            
        def generate_test_inputs(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
            return create_dummy_inputs(
                model_name=self.config.model_name,
                batch_size=batch_size,
                seq_len=seq_len,
                device=self.primary_device,
                dtype=self.dtype
            )
        
        def measure_performance(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
            # First get standard metrics
            metrics = super().measure_performance(model, inputs)
            
            model.eval()
            
            # Measure first token latency
            first_token_latencies = []
            full_generation_latencies = []
            
            with torch.no_grad():
                for _ in range(self.config.num_iterations):
                    # Setup generation
                    input_ids = inputs.get("input_ids", None)
                    if input_ids is None:
                        # Can't measure first token latency without input_ids
                        continue
                    
                    # Measure time to first token
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    # Forward pass (encoding)
                    outputs = model(**inputs)
                    
                    # Get logits from outputs
                    logits = None
                    if isinstance(outputs, torch.Tensor):
                        logits = outputs
                    elif isinstance(outputs, tuple) and isinstance(outputs[0], torch.Tensor):
                        logits = outputs[0]
                    elif hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
                        logits = outputs.logits
                    else:
                        # Can't get logits, skip this measurement
                        continue
                    
                    # Get next token (usually argmax of last logits)
                    next_token_logits = logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                    
                    torch.cuda.synchronize()
                    first_token_time = time.time()
                    
                    # Measure complete generation (e.g., generate 10 more tokens)
                    num_new_tokens = 10
                    new_input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
                    
                    for _ in range(num_new_tokens - 1):
                        # Forward pass with updated input_ids
                        temp_inputs = inputs.copy()
                        temp_inputs["input_ids"] = new_input_ids
                        
                        # If attention mask is present, extend it
                        if "attention_mask" in temp_inputs:
                            # Extend attention mask to match new input_ids length
                            attn_mask = temp_inputs["attention_mask"]
                            new_attn_mask = torch.cat([
                                attn_mask, 
                                torch.ones((attn_mask.shape[0], 1), device=attn_mask.device)
                            ], dim=-1)
                            temp_inputs["attention_mask"] = new_attn_mask
                        
                        # Forward pass
                        outputs = model(**temp_inputs)
                        
                        # Get logits
                        if isinstance(outputs, torch.Tensor):
                            logits = outputs
                        elif isinstance(outputs, tuple) and isinstance(outputs[0], torch.Tensor):
                            logits = outputs[0]
                        elif hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
                            logits = outputs.logits
                        else:
                            break
                        
                        # Get next token
                        next_token_logits = logits[:, -1, :]
                        next_token_id = torch.argmax(next_token_logits, dim=-1)
                        
                        # Append next token to input_ids
                        new_input_ids = torch.cat([new_input_ids, next_token_id.unsqueeze(-1)], dim=-1)
                    
                    torch.cuda.synchronize()
                    full_generation_time = time.time()
                    
                    # Record latencies
                    first_token_latencies.append((first_token_time - start_time) * 1000)  # Convert to ms
                    full_generation_latencies.append((full_generation_time - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            if first_token_latencies:
                metrics["first_token_latency_ms"] = sum(first_token_latencies) / len(first_token_latencies)
                metrics["first_token_p90_ms"] = sorted(first_token_latencies)[int(0.9 * len(first_token_latencies))]
                metrics["first_token_p99_ms"] = sorted(first_token_latencies)[int(0.99 * len(first_token_latencies))]
            
            if full_generation_latencies:
                metrics["full_generation_latency_ms"] = sum(full_generation_latencies) / len(full_generation_latencies)
                metrics["tokens_per_second"] = num_new_tokens / (metrics["full_generation_latency_ms"] / 1000)
            
            return metrics
    
    # Create and run the benchmark
    runner = FirstTokenLatencyRunner(config)
    results = runner.run_benchmarks()
    
    logger.info(f"Completed first token latency benchmark for {model_name}")
    
    return results