#!/usr/bin/env python3
"""
Verification script for baseline model implementation.

This script loads a small transformer model, runs inference using the baseline implementation,
and verifies the outputs against the original HuggingFace implementation.
"""

import os
import time
import logging
import argparse
from typing import Dict, Any, Tuple, List, Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline.model_loader import load_model, HuggingFaceModelLoader
from baseline.inference import create_inference_runner, TransformerInferenceRunner
from baseline.model_utils import get_model_size, get_model_summary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("baseline_verification")


def verify_model_parameters(model: torch.nn.Module, hf_model: torch.nn.Module) -> bool:
    """
    Verify that all model parameters are loaded correctly.
    
    Args:
        model: Model loaded via baseline loader
        hf_model: Model loaded directly via HuggingFace
        
    Returns:
        True if all parameters match, False otherwise
    """
    logger.info("Verifying model parameters...")
    
    # Get parameter dictionaries
    base_params = dict(model.named_parameters())
    hf_params = dict(hf_model.named_parameters())
    
    # Check parameter count
    if len(base_params) != len(hf_params):
        logger.error(f"Parameter count mismatch: {len(base_params)} vs {len(hf_params)}")
        return False
    
    # Check parameter values
    mismatches = []
    
    for name, param in base_params.items():
        if name not in hf_params:
            logger.warning(f"Parameter {name} not found in HuggingFace model")
            mismatches.append(name)
            continue
            
        hf_param = hf_params[name]
        
        # Check shapes
        if param.shape != hf_param.shape:
            logger.warning(f"Shape mismatch for {name}: {param.shape} vs {hf_param.shape}")
            mismatches.append(name)
            continue
            
        # Check values
        if not torch.allclose(param, hf_param, rtol=1e-3, atol=1e-5):
            logger.warning(f"Value mismatch for {name}")
            mismatches.append(name)
    
    if mismatches:
        logger.error(f"Found {len(mismatches)} mismatched parameters out of {len(base_params)}")
        return False
    
    logger.info("All model parameters verified successfully")
    return True


def verify_model_outputs(
    model: torch.nn.Module, 
    inference_runner: TransformerInferenceRunner,
    hf_model: torch.nn.Module,
    sample_inputs: Dict[str, torch.Tensor]
) -> bool:
    """
    Verify model outputs against the original HuggingFace implementation.
    
    Args:
        model: Model loaded via baseline loader
        inference_runner: Inference runner for the baseline model
        hf_model: Model loaded directly via HuggingFace
        sample_inputs: Sample inputs to use for verification
        
    Returns:
        True if outputs match, False otherwise
    """
    logger.info("Verifying model outputs...")
    
    # Run inference with baseline implementation
    model.eval()
    with torch.no_grad():
        baseline_outputs, _ = inference_runner.run_inference(sample_inputs)
    
    # Run inference with HuggingFace implementation
    hf_model.eval()
    with torch.no_grad():
        hf_outputs = hf_model(**sample_inputs)
    
    # Compare outputs
    if hasattr(baseline_outputs, "logits") and hasattr(hf_outputs, "logits"):
        # Compare logits
        baseline_logits = baseline_outputs.logits
        hf_logits = hf_outputs.logits
        
        if baseline_logits.shape != hf_logits.shape:
            logger.error(f"Logits shape mismatch: {baseline_logits.shape} vs {hf_logits.shape}")
            return False
        
        # Check if logits are close (allowing some small differences due to precision)
        if not torch.allclose(baseline_logits, hf_logits, rtol=1e-2, atol=1e-2):
            logger.error("Logits values do not match")
            # Calculate max difference for debugging
            max_diff = torch.max(torch.abs(baseline_logits - hf_logits)).item()
            logger.error(f"Maximum difference in logits: {max_diff}")
            return False
    else:
        # If we don't have logits, check the raw outputs
        # This is a simplified comparison that might need adjustment based on model type
        if type(baseline_outputs) != type(hf_outputs):
            logger.error(f"Output type mismatch: {type(baseline_outputs)} vs {type(hf_outputs)}")
            return False
    
    logger.info("Model outputs verified successfully")
    return True


def compare_performance(
    baseline_metrics: Dict[str, float], 
    hf_metrics: Dict[str, float]
) -> Tuple[Dict[str, float], bool]:
    """
    Compare performance metrics between baseline and HuggingFace implementations.
    
    Args:
        baseline_metrics: Performance metrics from baseline implementation
        hf_metrics: Performance metrics from HuggingFace implementation
        
    Returns:
        Tuple of (comparison metrics, passed flag)
    """
    logger.info("Comparing performance metrics...")
    
    comparison = {}
    passed = True
    
    # Compare timing
    if "total_time_ms" in baseline_metrics and "total_time_ms" in hf_metrics:
        baseline_time = baseline_metrics["total_time_ms"]
        hf_time = hf_metrics["total_time_ms"]
        time_ratio = baseline_time / hf_time if hf_time > 0 else float('inf')
        comparison["time_ratio"] = time_ratio
        comparison["baseline_time_ms"] = baseline_time
        comparison["hf_time_ms"] = hf_time
        
        # If baseline is more than 50% slower, consider it a failure
        if time_ratio > 1.5:
            logger.warning(f"Baseline implementation is {time_ratio:.2f}x slower than HuggingFace")
            passed = False
    
    # Compare memory usage if available
    if "peak_memory_mb" in baseline_metrics and "peak_memory_mb" in hf_metrics:
        baseline_memory = baseline_metrics["peak_memory_mb"]
        hf_memory = hf_metrics["peak_memory_mb"]
        memory_ratio = baseline_memory / hf_memory if hf_memory > 0 else float('inf')
        comparison["memory_ratio"] = memory_ratio
        comparison["baseline_memory_mb"] = baseline_memory
        comparison["hf_memory_mb"] = hf_memory
        
        # If baseline uses more than 20% more memory, flag it
        if memory_ratio > 1.2:
            logger.warning(f"Baseline implementation uses {memory_ratio:.2f}x more memory than HuggingFace")
            # Don't fail on memory, just warn
    
    logger.info(f"Performance comparison: {comparison}")
    return comparison, passed


def run_inference_benchmark(
    inference_runner: TransformerInferenceRunner,
    loader: HuggingFaceModelLoader,
    batch_sizes: List[int],
    seq_lengths: List[int],
    runs_per_config: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run inference benchmark with different batch sizes and sequence lengths.
    
    Args:
        inference_runner: Inference runner for the model
        loader: Model loader with tokenizer
        batch_sizes: List of batch sizes to test
        seq_lengths: List of sequence lengths to test
        runs_per_config: Number of times to run each configuration
        
    Returns:
        Dictionary of benchmark results
    """
    logger.info("Running inference benchmark with different configurations...")
    
    results = []
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            logger.info(f"Testing with batch_size={batch_size}, seq_len={seq_len}")
            
            # Generate sample inputs
            sample_inputs = loader.get_sample_input(batch_size, seq_len)
            
            # Warm up
            inference_runner.warmup(sample_inputs, iterations=3)
            
            # Run multiple times and collect metrics
            config_metrics = []
            for run in range(runs_per_config):
                _, metrics = inference_runner.run_inference(sample_inputs)
                config_metrics.append(metrics)
            
            # Calculate average metrics
            avg_metrics = {}
            for key in config_metrics[0].keys():
                avg_metrics[key] = sum(m[key] for m in config_metrics) / len(config_metrics)
            
            results.append({
                "batch_size": batch_size,
                "seq_len": seq_len,
                "metrics": avg_metrics
            })
    
    return results


def run_and_verify_generation(
    model: torch.nn.Module,
    inference_runner: TransformerInferenceRunner,
    tokenizer: Any,
    device: str,
    prompt: str = "Hello, my name is",
    max_length: int = 30
) -> bool:
    """
    Run text generation and verify it produces valid output.
    
    Args:
        model: The model to test
        inference_runner: Inference runner for the model
        tokenizer: The tokenizer for encoding/decoding
        device: The device to run on
        prompt: Text prompt for generation
        max_length: Maximum generation length
        
    Returns:
        True if generation succeeds, False otherwise
    """
    logger.info(f"Testing text generation with prompt: '{prompt}'")
    
    try:
        # Encode prompt and convert BatchEncoding to a dictionary of tensors
        tokenizer_output = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in tokenizer_output.items()}
        
        # Generate with our inference runner
        generation_args = {
            "max_length": max_length,
            "num_beams": 1,
            "do_sample": False
        }
        
        # Add generation args to inputs
        for k, v in generation_args.items():
            inputs[k] = v
        
        # Run generation
        outputs, metrics = inference_runner.run_inference(inputs)
        
        # Decode generated text
        if outputs is None or not hasattr(outputs, "shape"):
            logger.error("Generation failed to produce valid outputs")
            return False
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Generation successful, output: '{generated_text}'")
        logger.info(f"Generation time: {metrics.get('total_time_ms', 0):.2f} ms")
        
        return True
    
    except Exception as e:
        logger.error(f"Generation failed with error: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Baseline Model Verification")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on")
    parser.add_argument("--precision", type=str, default="fp16", 
                        choices=["fp32", "fp16", "bf16"], help="Precision to use")
    args = parser.parse_args()

    logger.info(f"Starting verification for model {args.model} on {args.device} with {args.precision} precision")
    
    try:
        # 1. Load model using baseline loader
        logger.info("Loading model using baseline loader...")
        torch_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }[args.precision]
        
        model, loader = load_model(
            args.model, 
            device=args.device, 
            dtype=torch_dtype
        )
        
        # 2. Load the same model directly with HuggingFace for comparison
        logger.info("Loading model directly with HuggingFace for comparison...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype
        ).to(args.device)
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # 3. Verify model parameters
        params_match = verify_model_parameters(model, hf_model)
        
        # 4. Create inference runner
        logger.info("Creating inference runner...")
        inference_runner = create_inference_runner(
            model, 
            device=args.device, 
            precision=args.precision,
            model_type="transformer"
        )
        
        # For HuggingFace model, create another runner for comparison
        hf_inference_runner = create_inference_runner(
            hf_model, 
            device=args.device, 
            precision=args.precision,
            model_type="transformer"
        )
        
        # 5. Generate sample inputs
        logger.info("Generating sample inputs...")
        batch_size, seq_len = 2, 32
        sample_inputs = loader.get_sample_input(batch_size, seq_len)
        
        # 6. Verify outputs
        outputs_match = verify_model_outputs(model, inference_runner, hf_model, sample_inputs)
        
        # 7. Run inference benchmarks with different configurations
        batch_sizes = [1, 2, 4]
        seq_lengths = [16, 32, 64]
        benchmark_results = run_inference_benchmark(
            inference_runner, 
            loader, 
            batch_sizes, 
            seq_lengths
        )
        
        # 8. Compare performance against HuggingFace
        # First, warm up both implementations
        inference_runner.warmup(sample_inputs, iterations=5)
        hf_inference_runner.warmup(sample_inputs, iterations=5)
        
        # Run inference and collect metrics
        _, baseline_metrics = inference_runner.run_inference(sample_inputs)
        _, hf_metrics = hf_inference_runner.run_inference(sample_inputs)
        
        performance_comparison, perf_pass = compare_performance(baseline_metrics, hf_metrics)
        
        # 9. Test text generation
        generation_pass = run_and_verify_generation(
            model,
            inference_runner,
            loader.tokenizer,
            args.device
        )
        
        # 10. Print verification results
        logger.info("=" * 50)
        logger.info("VERIFICATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"Model Parameters Check: {'PASS' if params_match else 'FAIL'}")
        logger.info(f"Model Outputs Check: {'PASS' if outputs_match else 'FAIL'}")
        logger.info(f"Performance Check: {'PASS' if perf_pass else 'FAIL'}")
        logger.info(f"Text Generation Check: {'PASS' if generation_pass else 'FAIL'}")
        
        # Overall pass/fail
        overall_pass = params_match and outputs_match and perf_pass and generation_pass
        logger.info(f"Overall Verification: {'PASS' if overall_pass else 'FAIL'}")
        
        # Log performance metrics
        logger.info("\nPerformance Summary:")
        logger.info(f"Baseline Inference Time: {baseline_metrics.get('total_time_ms', 0):.2f} ms")
        logger.info(f"HuggingFace Inference Time: {hf_metrics.get('total_time_ms', 0):.2f} ms")
        if "time_ratio" in performance_comparison:
            logger.info(f"Time Ratio (Baseline/HF): {performance_comparison['time_ratio']:.2f}x")
        
        if "peak_memory_mb" in baseline_metrics:
            logger.info(f"Baseline Peak Memory: {baseline_metrics['peak_memory_mb']:.2f} MB")
        if "peak_memory_mb" in hf_metrics:
            logger.info(f"HuggingFace Peak Memory: {hf_metrics['peak_memory_mb']:.2f} MB")
        
        # Log benchmark results
        logger.info("\nBenchmark Results:")
        for result in benchmark_results:
            logger.info(f"Batch Size: {result['batch_size']}, Seq Length: {result['seq_len']}")
            logger.info(f"  Inference Time: {result['metrics'].get('total_time_ms', 0):.2f} ms")
            if "peak_memory_mb" in result['metrics']:
                logger.info(f"  Peak Memory: {result['metrics']['peak_memory_mb']:.2f} MB")
        
        return 0 if overall_pass else 1
        
    except Exception as e:
        logger.error(f"Verification failed with error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())