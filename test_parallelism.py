#!/usr/bin/env python
"""
Test script for multi-dimensional parallelism implementation.

This script tests tensor, sequence, and combined parallelism implementations by:
1. Initializing a distributed environment
2. Loading a transformer model
3. Applying different parallelism strategies
4. Measuring performance and memory utilization
5. Validating outputs against non-parallelized versions
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict

# Import parallelism modules
from parallelism.tensor_parallel import (
    TensorParallelConfig, 
    ModelParallelConverter as TPModelConverter
)
from parallelism.sequence_parallel import (
    SequenceParallelConfig,
    SequenceParallelConverter
)
from parallelism.orchestrator import (
    ParallelConfig,
    ParallelOrchestrator
)
from parallelism.communication import (
    get_rank, get_world_size, barrier, initialize_distributed
)

# Import utilities
from utils.gpu_utils import get_gpu_memory_usage

# Check for transformers library
try:
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers package not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "transformers"])
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


@dataclass
class TestConfig:
    """Configuration for parallelism tests."""
    model_name: str = "gpt2-medium"  # Model to use for tests
    batch_sizes: List[int] = None  # Batch sizes to test
    seq_lengths: List[int] = None  # Sequence lengths to test
    max_tp_size: int = 4  # Maximum tensor parallel size
    max_sp_size: int = 4  # Maximum sequence parallel size
    log_dir: str = "./logs"  # Directory for logs
    fallback_to_cpu: bool = False  # Use CPU if insufficient GPUs
    simulate_multi_gpu: bool = False  # Simulate multi-GPU on CPU
    test_accuracy: bool = True  # Test output accuracy
    detailed_logging: bool = True  # Enable detailed logging
    
    def __post_init__(self):
        """Set default values for optional fields."""
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8]
        if self.seq_lengths is None:
            self.seq_lengths = [128, 512, 1024, 2048]
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class ParallelismTestSuite:
    """Test suite for multi-dimensional parallelism."""
    
    def __init__(self, config: TestConfig):
        """
        Initialize test suite.
        
        Args:
            config: Test configuration
        """
        self.config = config
        self.world_size = 1
        self.rank = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.distributed = False
        
        # Create log directory
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Initialize distributed environment if multiple GPUs available
        self._setup_distributed()
        
        # Load model
        self.model = None
        self.tokenizer = None
        self.reference_model = None
    
    def _setup_distributed(self) -> None:
        """
        Set up distributed environment for testing.
        """
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                # Initialize distributed with NCCL for multiple GPUs
                if not dist.is_initialized():
                    self.distributed = True
                    self.world_size = n_gpus
                    initialize_distributed()
                    self.rank = dist.get_rank()
                    torch.cuda.set_device(self.rank)
                    self.device = torch.device(f"cuda:{self.rank}")
                    print(f"Initialized process group with rank {self.rank}/{self.world_size}")
            else:
                print("Only one GPU available, running in non-distributed mode")
        else:
            if self.config.fallback_to_cpu:
                print("No GPUs available, falling back to CPU")
            else:
                raise RuntimeError("No GPUs available and fallback_to_cpu=False")
            
        # Simulate multi-GPU environment for testing on CPU
        if self.config.simulate_multi_gpu and not self.distributed:
            # This is a minimal simulation for testing code paths
            # It does NOT actually simulate distributed computation correctly
            # but allows testing the API and basic functionality
            self.world_size = 4  # Simulate 4 GPUs
            self.rank = 0  # Simulate being rank 0
            self.distributed = True
            print("Simulating multi-GPU environment on CPU")
    
    def load_model(self) -> None:
        """
        Load transformer model for testing.
        """
        if self.rank == 0 or not self.distributed:
            print(f"Loading model {self.config.model_name}...")
        
        # Load model configuration
        model_config = AutoConfig.from_pretrained(self.config.model_name)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Load model in evaluation mode
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, 
            config=model_config
        )
        self.model.eval()
        
        # Keep reference model for accuracy validation
        if self.config.test_accuracy:
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, 
                config=model_config
            )
            self.reference_model.eval()
            
            # Move reference model to CPU to save GPU memory
            self.reference_model = self.reference_model.cpu()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Log model details
        if self.rank == 0 or not self.distributed:
            print(f"Model loaded: {self.config.model_name}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
            print(f"Hidden size: {model_config.hidden_size}")
            print(f"Num heads: {model_config.num_attention_heads}")
            print(f"Num layers: {model_config.num_hidden_layers}")
    
    def create_test_inputs(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """
        Create test inputs for model.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Dictionary of input tensors
        """
        # Create random token IDs
        input_ids = torch.randint(
            0, self.tokenizer.vocab_size, (batch_size, seq_len), 
            dtype=torch.long, device=self.device
        )
        
        # Create attention mask (all 1s for simplicity)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=self.device)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def test_tensor_parallelism(self) -> Dict[str, Any]:
        """
        Test tensor parallelism implementation.
        
        Returns:
            Dictionary with test results
        """
        results = {
            "config": self.config.to_dict(),
            "tensor_parallelism": {}
        }
        
        # Skip tests if distributed environment is not available and not simulating
        if not self.distributed and not self.config.simulate_multi_gpu:
            print("Skipping tensor parallelism tests (no distributed environment)")
            results["tensor_parallelism"]["status"] = "skipped"
            return results
        
        if self.rank == 0 or not self.distributed:
            print("\n==== Testing Tensor Parallelism ====")
        
        # Test with different tensor parallel sizes
        for tp_size in range(2, min(self.world_size + 1, self.config.max_tp_size + 1)):
            if tp_size > self.world_size and not self.config.simulate_multi_gpu:
                # Skip configurations that require more GPUs than available
                continue
                
            tp_results = {}
            
            # Create tensor parallel configuration
            tp_config = TensorParallelConfig(
                world_size=self.world_size,
                tp_size=tp_size,
                parallel_dim=-1  # Head dimension
            )
            
            if self.rank == 0 or not self.distributed:
                print(f"\nTesting tensor parallelism with tp_size={tp_size}")
            
            # Convert model to tensor parallel
            model_converter = TPModelConverter(tp_config)
            tp_model = None
            
            try:
                # Create tensor parallel model
                model_copy = type(self.model)(self.model.config)
                model_copy.load_state_dict(self.model.state_dict())
                model_copy = model_copy.to(self.device)
                
                tp_model = model_converter.convert_model(model_copy)
                tp_model.eval()
                
                tp_results["model_conversion"] = "success"
                
                # Test with different batch sizes
                for batch_size in self.config.batch_sizes:
                    batch_results = {}
                    
                    # Test with different sequence lengths
                    for seq_len in self.config.seq_lengths:
                        if self.rank == 0 or not self.distributed:
                            print(f"Testing batch_size={batch_size}, seq_len={seq_len}")
                        
                        # Create inputs
                        inputs = self.create_test_inputs(batch_size, seq_len)
                        
                        # Record memory before inference
                        mem_before = get_gpu_memory_usage(self.device) if torch.cuda.is_available() else 0
                        
                        # Time the inference
                        start_time = time.time()
                        with torch.no_grad():
                            # Warmup
                            for _ in range(3):
                                _ = tp_model(**inputs)
                            
                            # Synchronize before measuring
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            
                            # Actual measurement
                            measure_start = time.time()
                            for _ in range(5):
                                outputs = tp_model(**inputs)
                            
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            measure_end = time.time()
                        
                        # Calculate inference time
                        inference_time = (measure_end - measure_start) / 5
                        
                        # Record memory after inference
                        mem_after = get_gpu_memory_usage(self.device) if torch.cuda.is_available() else 0
                        
                        # Run reference model if testing accuracy
                        if self.config.test_accuracy and (self.rank == 0 or not self.distributed):
                            ref_inputs = {
                                k: v.cpu() for k, v in inputs.items()
                            }
                            
                            with torch.no_grad():
                                ref_outputs = self.reference_model(**ref_inputs)
                            
                            # Check output accuracy
                            outputs_cpu = outputs.logits.cpu() if hasattr(outputs, "logits") else outputs.cpu()
                            ref_outputs_cpu = ref_outputs.logits.cpu() if hasattr(ref_outputs, "logits") else ref_outputs.cpu()
                            
                            max_diff = (outputs_cpu - ref_outputs_cpu).abs().max().item()
                            avg_diff = (outputs_cpu - ref_outputs_cpu).abs().mean().item()
                            
                            accuracy_ok = max_diff < 0.1  # Reasonable threshold for FP16/BF16
                        else:
                            max_diff = 0.0
                            avg_diff = 0.0
                            accuracy_ok = True
                        
                        # Record results
                        seq_results = {
                            "inference_time": inference_time,
                            "memory_used": mem_after - mem_before if torch.cuda.is_available() else 0,
                            "max_diff": max_diff,
                            "avg_diff": avg_diff,
                            "accuracy_ok": accuracy_ok
                        }
                        
                        # Add communication stats if available
                        if hasattr(tp_model, "last_communication_time"):
                            seq_results["communication_time"] = tp_model.last_communication_time
                            seq_results["communication_bytes"] = tp_model.last_communication_bytes
                        
                        batch_results[f"seq_len_{seq_len}"] = seq_results
                    
                    tp_results[f"batch_size_{batch_size}"] = batch_results
                
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                tp_results["model_conversion"] = "failed"
                tp_results["error"] = str(e)
                tp_results["traceback"] = error_msg
                if self.rank == 0 or not self.distributed:
                    print(f"Error testing tensor parallelism with tp_size={tp_size}: {e}")
                    print(error_msg)
            
            finally:
                # Clean up to save memory
                if tp_model is not None:
                    del tp_model
                torch.cuda.empty_cache()
            
            results["tensor_parallelism"][f"tp_size_{tp_size}"] = tp_results
        
        return results
    
    def test_sequence_parallelism(self) -> Dict[str, Any]:
        """
        Test sequence parallelism implementation.
        
        Returns:
            Dictionary with test results
        """
        results = {
            "config": self.config.to_dict(),
            "sequence_parallelism": {}
        }
        
        # Skip tests if distributed environment is not available and not simulating
        if not self.distributed and not self.config.simulate_multi_gpu:
            print("Skipping sequence parallelism tests (no distributed environment)")
            results["sequence_parallelism"]["status"] = "skipped"
            return results
        
        if self.rank == 0 or not self.distributed:
            print("\n==== Testing Sequence Parallelism ====")
        
        # Test with different sequence parallel sizes
        for sp_size in range(2, min(self.world_size + 1, self.config.max_sp_size + 1)):
            if sp_size > self.world_size and not self.config.simulate_multi_gpu:
                # Skip configurations that require more GPUs than available
                continue
                
            sp_results = {}
            
            # Create sequence parallel configuration
            sp_config = SequenceParallelConfig(
                world_size=self.world_size,
                sp_size=sp_size,
                overlap_communication=True,
                attention_handling="ring"  # Test ring attention strategy
            )
            
            if self.rank == 0 or not self.distributed:
                print(f"\nTesting sequence parallelism with sp_size={sp_size}")
            
            # Convert model to sequence parallel
            sp_converter = SequenceParallelConverter(sp_config)
            sp_model = None
            
            try:
                # Create sequence parallel model
                model_copy = type(self.model)(self.model.config)
                model_copy.load_state_dict(self.model.state_dict())
                model_copy = model_copy.to(self.device)
                
                sp_model = sp_converter.convert_model(model_copy)
                sp_model.eval()
                
                sp_results["model_conversion"] = "success"
                
                # Only test with longer sequences
                test_seq_lengths = [length for length in self.config.seq_lengths if length >= 512]
                
                # Test with different batch sizes
                for batch_size in self.config.batch_sizes:
                    batch_results = {}
                    
                    # Test with different sequence lengths
                    for seq_len in test_seq_lengths:
                        if self.rank == 0 or not self.distributed:
                            print(f"Testing batch_size={batch_size}, seq_len={seq_len}")
                        
                        # Create inputs
                        inputs = self.create_test_inputs(batch_size, seq_len)
                        
                        # Record memory before inference
                        mem_before = get_gpu_memory_usage(self.device) if torch.cuda.is_available() else 0
                        
                        # Time the inference
                        start_time = time.time()
                        with torch.no_grad():
                            # Warmup
                            for _ in range(3):
                                _ = sp_model(**inputs)
                            
                            # Synchronize before measuring
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            
                            # Actual measurement
                            measure_start = time.time()
                            for _ in range(5):
                                outputs = sp_model(**inputs)
                            
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            measure_end = time.time()
                        
                        # Calculate inference time
                        inference_time = (measure_end - measure_start) / 5
                        
                        # Record memory after inference
                        mem_after = get_gpu_memory_usage(self.device) if torch.cuda.is_available() else 0
                        
                        # Run reference model if testing accuracy
                        if self.config.test_accuracy and (self.rank == 0 or not self.distributed):
                            ref_inputs = {
                                k: v.cpu() for k, v in inputs.items()
                            }
                            
                            with torch.no_grad():
                                ref_outputs = self.reference_model(**ref_inputs)
                            
                            # Check output accuracy
                            outputs_cpu = outputs.logits.cpu() if hasattr(outputs, "logits") else outputs.cpu()
                            ref_outputs_cpu = ref_outputs.logits.cpu() if hasattr(ref_outputs, "logits") else ref_outputs.cpu()
                            
                            max_diff = (outputs_cpu - ref_outputs_cpu).abs().max().item()
                            avg_diff = (outputs_cpu - ref_outputs_cpu).abs().mean().item()
                            
                            accuracy_ok = max_diff < 0.1  # Reasonable threshold for FP16/BF16
                        else:
                            max_diff = 0.0
                            avg_diff = 0.0
                            accuracy_ok = True
                        
                        # Record results
                        seq_results = {
                            "inference_time": inference_time,
                            "memory_used": mem_after - mem_before if torch.cuda.is_available() else 0,
                            "max_diff": max_diff,
                            "avg_diff": avg_diff,
                            "accuracy_ok": accuracy_ok
                        }
                        
                        # Add communication stats if available
                        if hasattr(sp_model, "last_communication_time"):
                            seq_results["communication_time"] = sp_model.last_communication_time
                            seq_results["communication_bytes"] = sp_model.last_communication_bytes
                        
                        batch_results[f"seq_len_{seq_len}"] = seq_results
                    
                    sp_results[f"batch_size_{batch_size}"] = batch_results
                
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                sp_results["model_conversion"] = "failed"
                sp_results["error"] = str(e)
                sp_results["traceback"] = error_msg
                if self.rank == 0 or not self.distributed:
                    print(f"Error testing sequence parallelism with sp_size={sp_size}: {e}")
                    print(error_msg)
            
            finally:
                # Clean up to save memory
                if sp_model is not None:
                    del sp_model
                torch.cuda.empty_cache()
            
            results["sequence_parallelism"][f"sp_size_{sp_size}"] = sp_results
        
        return results
    
    def test_combined_parallelism(self) -> Dict[str, Any]:
        """
        Test combined tensor and sequence parallelism.
        
        Returns:
            Dictionary with test results
        """
        results = {
            "config": self.config.to_dict(),
            "combined_parallelism": {}
        }
        
        # Skip tests if distributed environment is not available and not simulating
        if not self.distributed and not self.config.simulate_multi_gpu:
            print("Skipping combined parallelism tests (no distributed environment)")
            results["combined_parallelism"]["status"] = "skipped"
            return results
        
        # Need at least 4 GPUs for meaningful combined parallelism tests
        if self.world_size < 4 and not self.config.simulate_multi_gpu:
            print("Skipping combined parallelism tests (need at least 4 GPUs)")
            results["combined_parallelism"]["status"] = "skipped"
            return results
        
        if self.rank == 0 or not self.distributed:
            print("\n==== Testing Combined Parallelism ====")
        
        # Test with different combinations of TP and SP sizes
        # For example: (tp=2, sp=2), (tp=2, sp=4), (tp=4, sp=2)
        test_combinations = [
            (2, 2),  # tp_size=2, sp_size=2
        ]
        
        if self.world_size >= 8 or self.config.simulate_multi_gpu:
            test_combinations.extend([
                (2, 4),  # tp_size=2, sp_size=4
                (4, 2),  # tp_size=4, sp_size=2
            ])
        
        for tp_size, sp_size in test_combinations:
            if tp_size * sp_size > self.world_size and not self.config.simulate_multi_gpu:
                # Skip configurations that require more GPUs than available
                continue
                
            combined_results = {}
            
            # Create combined parallel configuration
            combined_config = ParallelConfig(
                world_size=self.world_size,
                tensor_parallel_size=tp_size,
                sequence_parallel_size=sp_size,
                communication_dtype=torch.float16,
                overlap_communication=True
            )
            
            if self.rank == 0 or not self.distributed:
                print(f"\nTesting combined parallelism with tp_size={tp_size}, sp_size={sp_size}")
            
            # Create orchestrator for combined parallelism
            orchestrator = ParallelOrchestrator(combined_config)
            combined_model = None
            
            try:
                # Create combined parallel model
                model_copy = type(self.model)(self.model.config)
                model_copy.load_state_dict(self.model.state_dict())
                model_copy = model_copy.to(self.device)
                
                combined_model = orchestrator.configure_model(model_copy)
                combined_model.eval()
                
                combined_results["model_conversion"] = "success"
                
                # Only test with longer sequences
                test_seq_lengths = [length for length in self.config.seq_lengths if length >= 512]
                
                # Test with different batch sizes
                for batch_size in self.config.batch_sizes:
                    batch_results = {}
                    
                    # Test with different sequence lengths
                    for seq_len in test_seq_lengths:
                        if self.rank == 0 or not self.distributed:
                            print(f"Testing batch_size={batch_size}, seq_len={seq_len}")
                        
                        # Create inputs
                        inputs = self.create_test_inputs(batch_size, seq_len)
                        
                        # Record memory before inference
                        mem_before = get_gpu_memory_usage(self.device) if torch.cuda.is_available() else 0
                        
                        # Time the inference
                        start_time = time.time()
                        with torch.no_grad():
                            # Warmup
                            for _ in range(3):
                                _ = combined_model(**inputs)
                            
                            # Synchronize before measuring
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            
                            # Actual measurement
                            measure_start = time.time()
                            for _ in range(5):
                                outputs = combined_model(**inputs)
                            
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            measure_end = time.time()
                        
                        # Calculate inference time
                        inference_time = (measure_end - measure_start) / 5
                        
                        # Record memory after inference
                        mem_after = get_gpu_memory_usage(self.device) if torch.cuda.is_available() else 0
                        
                        # Run reference model if testing accuracy
                        if self.config.test_accuracy and (self.rank == 0 or not self.distributed):
                            ref_inputs = {
                                k: v.cpu() for k, v in inputs.items()
                            }
                            
                            with torch.no_grad():
                                ref_outputs = self.reference_model(**ref_inputs)
                            
                            # Check output accuracy
                            outputs_cpu = outputs.logits.cpu() if hasattr(outputs, "logits") else outputs.cpu()
                            ref_outputs_cpu = ref_outputs.logits.cpu() if hasattr(ref_outputs, "logits") else ref_outputs.cpu()
                            
                            max_diff = (outputs_cpu - ref_outputs_cpu).abs().max().item()
                            avg_diff = (outputs_cpu - ref_outputs_cpu).abs().mean().item()
                            
                            accuracy_ok = max_diff < 0.1  # Reasonable threshold for FP16/BF16
                        else:
                            max_diff = 0.0
                            avg_diff = 0.0
                            accuracy_ok = True
                        
                        # Record results
                        seq_results = {
                            "inference_time": inference_time,
                            "memory_used": mem_after - mem_before if torch.cuda.is_available() else 0,
                            "max_diff": max_diff,
                            "avg_diff": avg_diff,
                            "accuracy_ok": accuracy_ok
                        }
                        
                        batch_results[f"seq_len_{seq_len}"] = seq_results
                    
                    combined_results[f"batch_size_{batch_size}"] = batch_results
                
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                combined_results["model_conversion"] = "failed"
                combined_results["error"] = str(e)
                combined_results["traceback"] = error_msg
                if self.rank == 0 or not self.distributed:
                    print(f"Error testing combined parallelism with tp_size={tp_size}, sp_size={sp_size}: {e}")
                    print(error_msg)
            
            finally:
                # Clean up to save memory
                if combined_model is not None:
                    del combined_model
                torch.cuda.empty_cache()
            
            results["combined_parallelism"][f"tp_size_{tp_size}_sp_size_{sp_size}"] = combined_results
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all parallelism tests.
        
        Returns:
            Dictionary with all test results
        """
        # Load model
        self.load_model()
        
        # Run all tests
        tensor_results = self.test_tensor_parallelism()
        sequence_results = self.test_sequence_parallelism()
        combined_results = self.test_combined_parallelism()
        
        # Combine results
        all_results = {
            "config": self.config.to_dict(),
            "tensor_parallelism": tensor_results["tensor_parallelism"],
            "sequence_parallelism": sequence_results["sequence_parallelism"],
            "combined_parallelism": combined_results["combined_parallelism"]
        }
        
        # Save results to disk (only rank 0 in distributed setting)
        if self.rank == 0 or not self.distributed:
            results_file = os.path.join(self.config.log_dir, "parallelism_test_results.json")
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"Results saved to {results_file}")
        
        return all_results
    
    def run_basic_functionality_tests(self) -> None:
        """
        Run basic functionality tests for tensor and sequence parallelism.
        
        This is a simplified test that checks the basic functionality without
        comprehensive benchmarking.
        """
        # Load model
        self.load_model()
        
        if self.rank == 0 or not self.distributed:
            print("\n==== Basic Functionality Tests ====")
        
        # Test tensor parallelism
        if self.rank == 0 or not self.distributed:
            print("\nTesting basic tensor parallelism functionality")
        
        tp_config = TensorParallelConfig(
            world_size=self.world_size,
            tp_size=min(self.world_size, 2) if self.distributed else 1
        )
        
        model_converter = TPModelConverter(tp_config)
        
        try:
            # Create and test tensor parallel model
            model_copy = type(self.model)(self.model.config)
            model_copy.load_state_dict(self.model.state_dict())
            model_copy = model_copy.to(self.device)
            
            tp_model = model_converter.convert_model(model_copy)
            
            # Test inference
            inputs = self.create_test_inputs(1, 128)
            with torch.no_grad():
                outputs = tp_model(**inputs)
            
            if self.rank == 0 or not self.distributed:
                print("Tensor parallelism basic test PASSED")
        except Exception as e:
            if self.rank == 0 or not self.distributed:
                print(f"Tensor parallelism basic test FAILED: {e}")
        
        # Test sequence parallelism
        if self.rank == 0 or not self.distributed:
            print("\nTesting basic sequence parallelism functionality")
        
        sp_config = SequenceParallelConfig(
            world_size=self.world_size,
            sp_size=min(self.world_size, 2) if self.distributed else 1
        )
        
        sp_converter = SequenceParallelConverter(sp_config)
        
        try:
            # Create and test sequence parallel model
            model_copy = type(self.model)(self.model.config)
            model_copy.load_state_dict(self.model.state_dict())
            model_copy = model_copy.to(self.device)
            
            sp_model = sp_converter.convert_model(model_copy)
            
            # Test inference
            inputs = self.create_test_inputs(1, 512)
            with torch.no_grad():
                outputs = sp_model(**inputs)
            
            if self.rank == 0 or not self.distributed:
                print("Sequence parallelism basic test PASSED")
        except Exception as e:
            if self.rank == 0 or not self.distributed:
                print(f"Sequence parallelism basic test FAILED: {e}")
        
        # Clean up
        torch.cuda.empty_cache()


class SimulatedDistributedEnv:
    """
    Simulated distributed environment for testing on a single device.
    
    This class provides a minimal simulation of a multi-GPU environment
    for testing the functionality of parallelism code paths. It does NOT
    actually simulate correct distributed computation.
    """
    
    def __init__(self, world_size: int = 4):
        """
        Initialize simulated distributed environment.
        
        Args:
            world_size: Number of simulated processes
        """
        self.world_size = world_size
        self.rank = 0
        
        # Override distributed functions
        self._original_get_rank = None
        self._original_get_world_size = None
        self._original_all_reduce = None
        self._original_all_gather = None
        self._original_barrier = None
        
    def __enter__(self):
        """Set up simulated environment."""
        from parallelism.communication import (
            get_rank, get_world_size, all_reduce, all_gather, barrier
        )
        
        # Save original functions
        self._original_get_rank = get_rank
        self._original_get_world_size = get_world_size
        self._original_all_reduce = all_reduce
        self._original_all_gather = all_gather
        self._original_barrier = barrier
        
        # Override with simulated functions
        parallelism.communication.get_rank = lambda: self.rank
        parallelism.communication.get_world_size = lambda: self.world_size
        
        # Simplified all_reduce just returns the tensor
        def simulated_all_reduce(tensor, op=None, group=None):
            return tensor
        
        # Simplified all_gather just duplicates the tensor
        def simulated_all_gather(tensor, dim=0, world_size=None):
            if world_size is None:
                world_size = self.world_size
            return torch.cat([tensor] * world_size, dim=dim)
        
        # No-op barrier
        def simulated_barrier():
            pass
        
        parallelism.communication.all_reduce = simulated_all_reduce
        parallelism.communication.all_gather = simulated_all_gather
        parallelism.communication.barrier = simulated_barrier
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original environment."""
        # Restore original functions
        if self._original_get_rank is not None:
            parallelism.communication.get_rank = self._original_get_rank
        
        if self._original_get_world_size is not None:
            parallelism.communication.get_world_size = self._original_get_world_size
        
        if self._original_all_reduce is not None:
            parallelism.communication.all_reduce = self._original_all_reduce
        
        if self._original_all_gather is not None:
            parallelism.communication.all_gather = self._original_all_gather
        
        if self._original_barrier is not None:
            parallelism.communication.barrier = self._original_barrier


def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description="Test multi-dimensional parallelism")
    parser.add_argument(
        "--model", type=str, default="gpt2-medium",
        help="Model name to test"
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 8],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--seq-lengths", type=int, nargs="+", default=[128, 512, 1024, 2048],
        help="Sequence lengths to test"
    )
    parser.add_argument(
        "--log-dir", type=str, default="./parallelism_logs",
        help="Directory for logs"
    )
    parser.add_argument(
        "--fallback-to-cpu", action="store_true",
        help="Use CPU if no GPUs are available"
    )
    parser.add_argument(
        "--simulate-multi-gpu", action="store_true",
        help="Simulate multi-GPU environment on CPU or single GPU"
    )
    parser.add_argument(
        "--basic-test", action="store_true",
        help="Run only basic functionality tests"
    )
    parser.add_argument(
        "--no-accuracy", action="store_true",
        help="Skip accuracy validation"
    )
    
    args = parser.parse_args()
    
    # Create test configuration
    config = TestConfig(
        model_name=args.model,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        log_dir=args.log_dir,
        fallback_to_cpu=args.fallback_to_cpu,
        simulate_multi_gpu=args.simulate_multi_gpu,
        test_accuracy=not args.no_accuracy
    )
    
    # Create test suite
    test_suite = ParallelismTestSuite(config)
    
    # Run appropriate tests
    if args.basic_test:
        test_suite.run_basic_functionality_tests()
    else:
        test_suite.run_all_tests()


if __name__ == "__main__":
    # Handle simulated environment if requested
    if "--simulate-multi-gpu" in sys.argv:
        with SimulatedDistributedEnv(world_size=4):
            main()
    else:
        main()