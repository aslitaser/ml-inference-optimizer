#!/usr/bin/env python
"""
Test script for ML inference profiling system validation.

This script loads a medium-sized transformer model (GPT-2 medium) and creates a
synthetic dataset with varying sequence lengths to validate the profiling system
by applying different optimization techniques and bottlenecks.
"""

import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, schedule
from typing import Dict, List, Tuple, Optional, Any

# Import ML inference optimizer components
from profiling.torch_profiler import ProfilerConfig, TorchProfilerWrapper, ProfileResults
from profiling.profile_visualizer import ProfileVisualizer
from profiling.kernel_profiler import KernelProfiler, KernelProfileResults
from utils.gpu_utils import get_gpu_memory_usage

# Conditionally import transformers
try:
    from transformers import GPT2Model, GPT2Config
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers package not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "transformers"])
    from transformers import GPT2Model, GPT2Config


class BottleneckLayer(nn.Module):
    """Layer that artificially introduces a bottleneck for testing."""
    
    def __init__(self, sleep_time: float = 0.01, memory_spike: int = 0):
        """
        Initialize the bottleneck layer.
        
        Args:
            sleep_time: Amount of time to sleep in seconds
            memory_spike: Amount of memory to allocate temporarily in MB
        """
        super().__init__()
        self.sleep_time = sleep_time
        self.memory_spike = memory_spike
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with artificial bottlenecks."""
        # Create artificial compute bottleneck
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)
        
        # Create artificial memory bottleneck
        if self.memory_spike > 0:
            # Allocate temporary tensor to simulate memory spike
            temp_size = int(self.memory_spike * 1024 * 1024 / 4)  # Convert MB to float32 elements
            temp_tensor = torch.zeros(temp_size, device=x.device, dtype=torch.float)
            # Just to make sure the allocation isn't optimized away
            temp_tensor.fill_(0.1)
            # Use the temp tensor in a trivial computation to ensure it's used
            y = x + temp_tensor[0]
            del temp_tensor
            return y
        
        return x


class GPT2WithBottlenecks(nn.Module):
    """GPT-2 model with added bottlenecks for testing profiler efficiency."""
    
    def __init__(
        self, 
        base_model: GPT2Model,
        bottleneck_config: Dict[str, Dict[str, float]]
    ):
        """
        Initialize the model with bottlenecks.
        
        Args:
            base_model: Base GPT-2 model
            bottleneck_config: Configuration for bottlenecks, mapping layer to sleep/memory params
        """
        super().__init__()
        self.base_model = base_model
        self.bottleneck_config = bottleneck_config
        
        # Add bottleneck modules
        self.bottlenecks = nn.ModuleDict()
        for layer_name, config in bottleneck_config.items():
            sleep_time = config.get('sleep_time', 0)
            memory_spike = config.get('memory_spike', 0)
            self.bottlenecks[layer_name] = BottleneckLayer(sleep_time, memory_spike)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with bottlenecks inserted at specified points."""
        # First part of the forward pass
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state
        
        # Apply bottlenecks before returning
        if 'output' in self.bottlenecks:
            hidden_states = self.bottlenecks['output'](hidden_states)
        
        return hidden_states


def create_synthetic_dataset(
    batch_sizes: List[int], 
    seq_lengths: List[int], 
    vocab_size: int = 50257
) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
    """
    Create a synthetic dataset with varying batch sizes and sequence lengths.
    
    Args:
        batch_sizes: List of batch sizes to use
        seq_lengths: List of sequence lengths to use
        vocab_size: Vocabulary size for token generation
        
    Returns:
        Dictionary mapping (batch_size, seq_len) to input dictionaries
    """
    dataset = {}
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            # Create random input ids
            input_ids = torch.randint(
                0, vocab_size, (batch_size, seq_len), dtype=torch.long
            )
            
            # Create attention mask (no padding in this synthetic data)
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
            
            dataset[(batch_size, seq_len)] = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
    
    return dataset


def load_model_with_bottlenecks(
    bottleneck_config: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[nn.Module, int]:
    """
    Load a GPT-2 medium model with optional bottlenecks.
    
    Args:
        bottleneck_config: Configuration for bottlenecks
        
    Returns:
        Tuple of (model, vocab_size)
    """
    print("Loading GPT-2 medium model...")
    
    # Initialize GPT-2 medium model
    config = GPT2Config.from_pretrained('gpt2-medium')
    base_model = GPT2Model(config)
    vocab_size = config.vocab_size
    
    # Add bottlenecks if specified
    if bottleneck_config:
        model = GPT2WithBottlenecks(base_model, bottleneck_config)
    else:
        model = base_model
    
    return model, vocab_size


def profile_model_with_dataset(
    model: nn.Module,
    dataset: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
    device: torch.device,
    output_dir: str
) -> Dict[Tuple[int, int], ProfileResults]:
    """
    Profile model with different dataset configurations.
    
    Args:
        model: Model to profile
        dataset: Dataset with different configurations
        device: Device to run the model on
        output_dir: Directory to save profiling results
        
    Returns:
        Dictionary mapping (batch_size, seq_len) to profiling results
    """
    results = {}
    model.to(device)
    model.eval()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create profiler config
    config = ProfilerConfig(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
        schedule=schedule(wait=1, warmup=1, active=3),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    )
    
    print(f"Profiling model on {len(dataset)} different configurations...")
    
    for (batch_size, seq_len), inputs in dataset.items():
        print(f"Profiling batch_size={batch_size}, seq_len={seq_len}")
        
        # Move inputs to device
        device_inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Create profiler
        profiler = TorchProfilerWrapper(config)
        
        # Run profiling
        with torch.no_grad():
            profile_results = profiler.profile_model(model, device_inputs, iterations=5)
        
        # Store results
        results[(batch_size, seq_len)] = profile_results
        
        # Save results
        output_file = os.path.join(output_dir, f"profile_bs{batch_size}_seq{seq_len}.pkl")
        profile_results.save(output_file)
        
        # Generate visualizations
        visualizer = ProfileVisualizer(profile_results)
        visualization_dir = os.path.join(output_dir, f"visualizations_bs{batch_size}_seq{seq_len}")
        os.makedirs(visualization_dir, exist_ok=True)
        visualizer.save_visualizations(visualization_dir)
        
        # Profile CUDA kernels if available
        if torch.cuda.is_available():
            kernel_profiler = KernelProfiler()
            
            def run_model():
                with torch.no_grad():
                    return model(**device_inputs)
            
            kernel_results = kernel_profiler.profile_kernels(run_model)
            
            # Create kernel efficiency visualization
            kernel_viz = visualizer.create_kernel_efficiency_plot(kernel_results)
            kernel_viz.savefig(os.path.join(visualization_dir, "kernel_efficiency.png"), dpi=300, bbox_inches="tight")
    
    return results


def validate_profiler_accuracy(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    device: torch.device,
    bottleneck_config: Dict[str, Dict[str, float]],
    output_dir: str
) -> Dict[str, Any]:
    """
    Validate profiler accuracy by adding artificial bottlenecks.
    
    Args:
        model: Base model without bottlenecks
        inputs: Input data for validation
        device: Device to run the model on
        bottleneck_config: Configuration for artificial bottlenecks
        output_dir: Directory to save validation results
        
    Returns:
        Dictionary with validation metrics
    """
    print("Validating profiler accuracy...")
    
    # Create models with different bottlenecks
    if isinstance(model, GPT2WithBottlenecks):
        # If already a bottleneck model, use the base model
        base_model = model.base_model
    else:
        base_model = model
    
    # Prepare validation output directory
    validation_dir = os.path.join(output_dir, "validation")
    os.makedirs(validation_dir, exist_ok=True)
    
    # Test different bottleneck configurations
    validation_results = {}
    
    for bottleneck_name, config in bottleneck_config.items():
        print(f"Testing bottleneck: {bottleneck_name}")
        
        # Create a model with this specific bottleneck
        bottleneck_model = GPT2WithBottlenecks(
            base_model, {bottleneck_name: config}
        ).to(device)
        bottleneck_model.eval()
        
        # Move inputs to device
        device_inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Create profiler config
        profiler_config = ProfilerConfig(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True
        )
        
        # Run profiling
        profiler = TorchProfilerWrapper(profiler_config)
        with torch.no_grad():
            profile_results = profiler.profile_model(bottleneck_model, device_inputs, iterations=5)
        
        # Get most time-consuming operations
        time_consuming_ops = profile_results.get_most_time_consuming(top_k=10)
        
        # Check if bottleneck is detected
        bottleneck_detected = any('BottleneckLayer' in op['name'] for op in time_consuming_ops[:3])
        
        # Get memory statistics
        memory_stats = profile_results.get_memory_stats()
        
        # Store validation results
        validation_results[bottleneck_name] = {
            'bottleneck_config': config,
            'bottleneck_detected': bottleneck_detected,
            'top_operations': time_consuming_ops[:3],
            'peak_memory': memory_stats.get('peak_cuda_memory', 0)
        }
        
        # Save results
        output_file = os.path.join(validation_dir, f"validation_{bottleneck_name}.pkl")
        profile_results.save(output_file)
        
        # Generate visualizations
        visualizer = ProfileVisualizer(profile_results)
        visualization_dir = os.path.join(validation_dir, f"viz_{bottleneck_name}")
        os.makedirs(visualization_dir, exist_ok=True)
        visualizer.save_visualizations(visualization_dir)
    
    # Save validation summary
    with open(os.path.join(validation_dir, "validation_summary.json"), 'w') as f:
        # Convert non-serializable items
        serializable_results = {}
        for k, v in validation_results.items():
            serializable_results[k] = {
                'bottleneck_config': v['bottleneck_config'],
                'bottleneck_detected': v['bottleneck_detected'],
                'top_operations': [
                    {k2: str(v2) if isinstance(v2, torch.Tensor) else v2 
                     for k2, v2 in op.items()}
                    for op in v['top_operations']
                ],
                'peak_memory': float(v['peak_memory'])
            }
        json.dump(serializable_results, f, indent=2)
    
    return validation_results


def generate_summary_report(
    profiling_results: Dict[Tuple[int, int], ProfileResults],
    validation_results: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Generate a summary report of all profiling results.
    
    Args:
        profiling_results: Dictionary mapping (batch_size, seq_len) to profiling results
        validation_results: Results from validation tests
        output_dir: Directory to save the report
        
    Returns:
        Dictionary with summary metrics
    """
    print("Generating summary report...")
    
    summary = {
        'configurations': [],
        'validation': {
            'bottlenecks_tested': len(validation_results),
            'bottlenecks_detected': sum(1 for v in validation_results.values() if v['bottleneck_detected']),
            'details': validation_results
        }
    }
    
    # Process each configuration
    for (batch_size, seq_len), profile_result in profiling_results.items():
        # Get key metrics
        time_consuming_ops = profile_result.get_most_time_consuming(top_k=5)
        memory_stats = profile_result.get_memory_stats()
        
        # Calculate total execution time
        total_time = sum(op['cpu_time_total'] for op in time_consuming_ops)
        
        # Store configuration summary
        config_summary = {
            'batch_size': batch_size,
            'sequence_length': seq_len,
            'total_execution_time_ms': total_time / 1000,  # Convert to ms
            'peak_memory_mb': memory_stats.get('peak_cuda_memory', 0) / (1024 * 1024),
            'top_operations': [op['name'] for op in time_consuming_ops]
        }
        
        summary['configurations'].append(config_summary)
    
    # Save summary report
    with open(os.path.join(output_dir, "summary_report.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define bottleneck configurations for validation
    bottleneck_config = {
        'output': {
            'sleep_time': 0.05,  # 50ms sleep
            'memory_spike': 0    # No memory spike
        },
        'memory_hog': {
            'sleep_time': 0,       # No sleep
            'memory_spike': 500    # 500MB memory spike
        },
        'combined': {
            'sleep_time': 0.02,    # 20ms sleep
            'memory_spike': 200    # 200MB memory spike
        }
    }
    
    # Load model
    model, vocab_size = load_model_with_bottlenecks()
    model.to(device)
    model.eval()
    
    # Create synthetic dataset
    batch_sizes = [1, 4, 8] if not args.small else [1, 2]
    seq_lengths = [128, 256, 512, 1024] if not args.small else [128, 256]
    
    print(f"Creating synthetic dataset with batch sizes {batch_sizes} and sequence lengths {seq_lengths}")
    dataset = create_synthetic_dataset(batch_sizes, seq_lengths, vocab_size)
    
    # Profile model with different dataset configurations
    profile_results = profile_model_with_dataset(
        model, dataset, device, args.output_dir
    )
    
    # Validate profiler accuracy with artificial bottlenecks
    validation_results = validate_profiler_accuracy(
        model, 
        dataset[(batch_sizes[0], seq_lengths[0])], 
        device,
        bottleneck_config,
        args.output_dir
    )
    
    # Generate summary report
    summary = generate_summary_report(
        profile_results, validation_results, args.output_dir
    )
    
    print(f"Profiling completed. Results saved to {args.output_dir}")
    print("\nSummary:")
    print(f"- Configurations tested: {len(summary['configurations'])}")
    print(f"- Bottlenecks tested: {summary['validation']['bottlenecks_tested']}")
    print(f"- Bottlenecks correctly detected: {summary['validation']['bottlenecks_detected']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ML inference profiling system")
    parser.add_argument(
        "--output-dir", type=str, default="./profiling_results",
        help="Directory to save profiling results"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Use CPU instead of CUDA"
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Run a smaller test with fewer configurations"
    )
    
    args = parser.parse_args()
    main(args)