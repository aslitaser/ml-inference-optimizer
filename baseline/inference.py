"""
Module for running inference on ML models with benchmarking capabilities.
"""

import time
import gc
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Union, Callable

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity


class InferenceRunner(ABC):
    """Base class for model inference with performance metrics."""
    
    def __init__(self, model: nn.Module, device: str, precision: str = "fp16"):
        """
        Initialize the inference runner.
        
        Args:
            model: PyTorch model to run inference with
            device: Device to run inference on ('cuda', 'cpu')
            precision: Precision to use for inference ('fp32', 'fp16', 'bf16')
        """
        self.model = model
        self.device = device
        self.precision = precision
        
        # Store original dtype
        self.original_dtype = next(model.parameters()).dtype
        
        # Convert model to specified precision
        self._set_precision(precision)
        
        # Ensure model is on the right device
        if next(model.parameters()).device.type != device:
            self.model = self.model.to(device)
            
        # Inference metrics
        self.metrics: Dict[str, float] = {}
        
    def _set_precision(self, precision: str) -> None:
        """
        Set model precision.
        
        Args:
            precision: Precision to use ('fp32', 'fp16', 'bf16')
        """
        if precision == "fp32":
            dtype = torch.float32
        elif precision == "fp16":
            dtype = torch.float16
        elif precision == "bf16":
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported precision: {precision}")
        
        self.model = self.model.to(dtype=dtype)
    
    def warmup(self, inputs: Any, iterations: int = 10) -> None:
        """
        Warm up the model with multiple iterations.
        
        Args:
            inputs: Sample inputs for warmup
            iterations: Number of warmup iterations
        """
        with torch.no_grad():
            self.model.eval()
            
            # Record initial memory
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            # Run warmup iterations
            for _ in range(iterations):
                self._forward(inputs)
                
            # Synchronize if using CUDA
            if self.device == "cuda":
                torch.cuda.synchronize()
    
    @abstractmethod
    def _forward(self, inputs: Any) -> Any:
        """
        Run forward pass on the model.
        
        Args:
            inputs: Model inputs
            
        Returns:
            Model outputs
        """
        pass
    
    def run_inference(self, inputs: Any, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """
        Run inference and collect performance metrics.
        
        Args:
            inputs: Model inputs
            **kwargs: Additional arguments for the model
            
        Returns:
            Tuple of (model outputs, performance metrics)
        """
        self.model.eval()
        metrics = {}
        
        # Clear GPU cache before inference
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        # Record memory before inference
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            metrics["memory_before_mb"] = memory_before
        
        # Set up CUDA events for timing
        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        # Start timing
        start_time = time.perf_counter()
        
        # Run inference
        with torch.no_grad():
            outputs = self._forward(inputs, **kwargs)
            
        # End timing
        end_time = time.perf_counter()
        
        # Record CUDA event
        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            cuda_time_ms = start_event.elapsed_time(end_event)
            metrics["cuda_time_ms"] = cuda_time_ms
        
        # Record total time
        total_time = (end_time - start_time) * 1000  # Convert to ms
        metrics["total_time_ms"] = total_time
        
        # Record memory after inference
        if self.device == "cuda":
            memory_after = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            metrics["memory_after_mb"] = memory_after
            metrics["peak_memory_mb"] = peak_memory
            metrics["memory_change_mb"] = memory_after - memory_before
        
        self.metrics = metrics
        return outputs, metrics
    
    def run_batch_inference(self, batch_inputs: List[Any], **kwargs) -> List[Tuple[Any, Dict[str, float]]]:
        """
        Run inference on a batch of inputs.
        
        Args:
            batch_inputs: List of model inputs
            **kwargs: Additional arguments for the model
            
        Returns:
            List of tuples, each containing (model outputs, performance metrics)
        """
        results = []
        batch_metrics = {"total_batch_time_ms": 0.0}
        
        batch_start_time = time.perf_counter()
        
        for inputs in batch_inputs:
            outputs, metrics = self.run_inference(inputs, **kwargs)
            results.append((outputs, metrics))
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key in batch_metrics:
                    batch_metrics[key] += value
                else:
                    batch_metrics[key] = value
        
        batch_end_time = time.perf_counter()
        batch_metrics["total_batch_time_ms"] = (batch_end_time - batch_start_time) * 1000
        batch_metrics["avg_inference_time_ms"] = batch_metrics["total_batch_time_ms"] / len(batch_inputs)
        
        return results
    
    def profile_model(self, inputs: Any, use_cuda: bool = True) -> Dict[str, Any]:
        """
        Profile the model to identify bottlenecks.
        
        Args:
            inputs: Sample inputs for profiling
            use_cuda: Whether to profile CUDA operations
            
        Returns:
            Dictionary containing profiling results
        """
        activities = []
        if use_cuda and self.device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        activities.append(ProfilerActivity.CPU)
        
        self.model.eval()
        
        # Run profiling
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("model_inference"):
                with torch.no_grad():
                    _ = self._forward(inputs)
        
        # Process results
        profile_results = {
            "table": prof.key_averages().table(sort_by="cuda_time_total" if use_cuda else "cpu_time_total", row_limit=20),
            "events": prof.events(),
            "key_averages": prof.key_averages(),
        }
        
        return profile_results
    
    def restore_original_precision(self) -> None:
        """Restore the model to its original precision."""
        self.model = self.model.to(dtype=self.original_dtype)


class TransformerInferenceRunner(InferenceRunner):
    """Specialized inference runner for transformer models."""
    
    def __init__(self, model: nn.Module, device: str, precision: str = "fp16", 
                 is_encoder_decoder: bool = False):
        """
        Initialize transformer inference runner.
        
        Args:
            model: Transformer model to run inference with
            device: Device to run inference on ('cuda', 'cpu')
            precision: Precision to use for inference ('fp32', 'fp16', 'bf16')
            is_encoder_decoder: Whether the model is an encoder-decoder architecture
        """
        super().__init__(model, device, precision)
        self.is_encoder_decoder = is_encoder_decoder
        
    def _forward(self, inputs: Any, **kwargs) -> Any:
        """
        Run forward pass on transformer model.
        
        Args:
            inputs: Model inputs (typically a dict with input_ids and attention_mask)
            **kwargs: Additional arguments like max_length, num_beams, etc.
            
        Returns:
            Model outputs
        """
        # Check if inputs is a dict with input_ids or just tensor input_ids
        if isinstance(inputs, dict):
            input_dict = inputs
        elif hasattr(inputs, "to_dict") and callable(inputs.to_dict):
            # Handle BatchEncoding objects from tokenizers
            input_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in inputs.to_dict().items()}
        else:
            # Assume it's just input_ids
            input_dict = {"input_ids": inputs}
            if "attention_mask" not in input_dict and "attention_mask" not in kwargs:
                # Create attention mask if not provided
                input_dict["attention_mask"] = torch.ones_like(input_dict["input_ids"])
        
        # Merge kwargs into input_dict for any keys not already present
        for key, value in kwargs.items():
            if key not in input_dict:
                input_dict[key] = value
        
        # Handle generation vs normal forward pass
        generation_args = ["max_length", "min_length", "num_beams", "temperature", 
                         "top_k", "top_p", "repetition_penalty", "do_sample"]
        
        is_generation = any(arg in input_dict for arg in generation_args) or any(arg in kwargs for arg in generation_args)
        
        if is_generation and hasattr(self.model, "generate"):
            # Extract generation-specific parameters
            gen_kwargs = {k: v for k, v in input_dict.items() 
                         if k in generation_args or k in ["input_ids", "attention_mask"]}
            outputs = self.model.generate(**gen_kwargs)
        else:
            # Regular forward pass
            outputs = self.model(**input_dict)
            
        return outputs
    
    def run_inference_with_layer_timing(self, inputs: Any, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """
        Run inference with per-layer timing.
        
        Args:
            inputs: Model inputs
            **kwargs: Additional arguments for the model
            
        Returns:
            Tuple of (model outputs, performance metrics with per-layer timing)
        """
        self.model.eval()
        layer_metrics = {}
        
        # Set up hooks for layer timing
        hooks = []
        layer_times = {}
        
        def forward_hook(name):
            def hook(module, input, output):
                # Record start time
                if self.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                # Store the start time for this layer
                layer_times[name] = {"start": start}
            return hook
        
        def backward_hook(name):
            def hook(module, input, output):
                # Record end time
                if self.device == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                # Calculate elapsed time
                if name in layer_times:
                    start = layer_times[name]["start"]
                    elapsed_ms = (end - start) * 1000  # Convert to ms
                    layer_times[name]["time_ms"] = elapsed_ms
            return hook
        
        # Register hooks for all named modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only register for leaf modules
                hooks.append(module.register_forward_pre_hook(forward_hook(name)))
                hooks.append(module.register_forward_hook(backward_hook(name)))
        
        # Run normal inference
        outputs, metrics = self.run_inference(inputs, **kwargs)
        
        # Process layer timing data
        for name, data in layer_times.items():
            if "time_ms" in data:
                layer_metrics[f"layer_time_{name}_ms"] = data["time_ms"]
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Merge layer metrics with regular metrics
        combined_metrics = {**metrics, **layer_metrics}
        
        return outputs, combined_metrics


class DiffusionInferenceRunner(InferenceRunner):
    """Specialized inference runner for diffusion models."""
    
    def __init__(self, model: nn.Module, device: str, precision: str = "fp16"):
        """
        Initialize diffusion inference runner.
        
        Args:
            model: Diffusion model/pipeline to run inference with
            device: Device to run inference on ('cuda', 'cpu')
            precision: Precision to use for inference ('fp32', 'fp16', 'bf16')
        """
        super().__init__(model, device, precision)
        self.step_times = []
        
    def _forward(self, inputs: Any, **kwargs) -> Any:
        """
        Run forward pass on diffusion model.
        
        Args:
            inputs: Model inputs (typically a dict with prompt and params)
            **kwargs: Additional arguments for the model
            
        Returns:
            Generated images
        """
        # Reset step times
        self.step_times = []
        
        # Capture timing for each denoising step
        original_step = getattr(self.model.scheduler, "step", None)
        
        if original_step is not None:
            def step_with_timing(self_scheduler, *args, **kwargs):
                start_time = time.perf_counter()
                result = original_step(*args, **kwargs)
                end_time = time.perf_counter()
                step_time = (end_time - start_time) * 1000  # ms
                self.step_times.append(step_time)
                return result
            
            # Replace scheduler step method with our timed version
            self.model.scheduler.step = step_with_timing.__get__(
                self.model.scheduler, type(self.model.scheduler)
            )
        
        # Combine inputs and kwargs
        if isinstance(inputs, dict):
            combined_inputs = {**inputs, **kwargs}
        else:
            # Assume it's just the prompt string
            combined_inputs = {"prompt": inputs, **kwargs}
        
        # Run inference
        outputs = self.model(**combined_inputs)
        
        # Restore original step method
        if original_step is not None:
            self.model.scheduler.step = original_step
            
        return outputs
    
    def run_inference(self, inputs: Any, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """
        Run inference and collect performance metrics specific to diffusion models.
        
        Args:
            inputs: Model inputs
            **kwargs: Additional arguments for the model
            
        Returns:
            Tuple of (model outputs, performance metrics)
        """
        # Run the standard inference
        outputs, metrics = super().run_inference(inputs, **kwargs)
        
        # Add diffusion-specific metrics
        if self.step_times:
            metrics["steps_count"] = len(self.step_times)
            metrics["avg_step_time_ms"] = sum(self.step_times) / len(self.step_times)
            metrics["min_step_time_ms"] = min(self.step_times)
            metrics["max_step_time_ms"] = max(self.step_times)
            metrics["step_times_ms"] = self.step_times
        
        return outputs, metrics


# Factory function to create the appropriate inference runner
def create_inference_runner(model: nn.Module, device: str, precision: str = "fp16", 
                           model_type: str = "auto") -> InferenceRunner:
    """
    Create an appropriate inference runner for the given model.
    
    Args:
        model: PyTorch model to run inference with
        device: Device to run inference on ('cuda', 'cpu')
        precision: Precision to use for inference ('fp32', 'fp16', 'bf16')
        model_type: Type of model ('transformer', 'diffusion', 'auto')
        
    Returns:
        Appropriate inference runner instance
    """
    if model_type == "auto":
        # Try to automatically detect model type
        if hasattr(model, "unet") and hasattr(model, "scheduler"):
            model_type = "diffusion"
        elif hasattr(model, "generate") or "transformer" in model.__class__.__name__.lower():
            model_type = "transformer"
        else:
            model_type = "base"  # Default to base runner
    
    # Create appropriate runner
    if model_type == "transformer":
        return TransformerInferenceRunner(model, device, precision)
    elif model_type == "diffusion":
        return DiffusionInferenceRunner(model, device, precision)
    else:
        # Create a basic runner that works with regular forward calls
        class BasicInferenceRunner(InferenceRunner):
            def _forward(self, inputs: Any, **kwargs) -> Any:
                return self.model(inputs, **kwargs) if kwargs else self.model(inputs)
        
        return BasicInferenceRunner(model, device, precision)