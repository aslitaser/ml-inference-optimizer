"""
PyTorch Profiler module for ML Inference Optimizer.

This module provides wrapper classes and utilities for the PyTorch profiler,
allowing easy profiling of models and functions with configurable options.
"""

import json
import pickle
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile


class ProfilerConfig:
    """Configuration class for the PyTorch profiler with customizable settings."""

    def __init__(
        self,
        activities: List[ProfilerActivity] = None,
        schedule: Optional[Callable[[int], torch.profiler.ProfilerAction]] = None,
        record_shapes: bool = False,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = False,
        experimental_config: Dict[str, Any] = None,
    ):
        """
        Initialize profiler configuration.

        Args:
            activities: List of activities to profile (CPU, CUDA)
            schedule: Callable schedule for the profiler
            record_shapes: Whether to record tensor shapes
            profile_memory: Whether to profile memory usage
            with_stack: Whether to record stack traces
            with_flops: Whether to estimate FLOPS (experimental)
            experimental_config: Additional experimental configuration options
        """
        self.activities = activities or [ProfilerActivity.CPU]
        if torch.cuda.is_available() and ProfilerActivity.CUDA not in self.activities:
            self.activities.append(ProfilerActivity.CUDA)
        
        self.schedule = schedule
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.experimental_config = experimental_config or {}


class ProfileResults:
    """Class to store, process and visualize profiling results."""

    def __init__(self, prof: torch.profiler.profile):
        """
        Initialize with raw profiler results.

        Args:
            prof: PyTorch profiler results
        """
        self.raw_prof = prof
        self._events = None
        self._process_events()
    
    def _process_events(self):
        """Process and extract events from profiler results."""
        # Extract events from profiler
        self._events = self.raw_prof.key_averages()
    
    def table(self) -> pd.DataFrame:
        """
        Convert profiling data to a pandas DataFrame.

        Returns:
            DataFrame containing profiling data
        """
        data = []
        for evt in self._events:
            row = {
                "name": evt.key,
                "cpu_time_total": evt.cpu_time_total,
                "cpu_time": evt.cpu_time,
                "cuda_time": evt.cuda_time if hasattr(evt, "cuda_time") else 0,
                "self_cpu_time": evt.self_cpu_time,
                "self_cuda_time": evt.self_cuda_time if hasattr(evt, "self_cuda_time") else 0,
                "count": evt.count,
            }
            
            # Add shape if available
            if hasattr(evt, "input_shapes") and evt.input_shapes:
                row["input_shapes"] = str(evt.input_shapes)
            
            # Add memory info if available
            if hasattr(evt, "cpu_memory_usage"):
                row["cpu_memory_usage"] = evt.cpu_memory_usage
            if hasattr(evt, "cuda_memory_usage"):
                row["cuda_memory_usage"] = evt.cuda_memory_usage
                
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_most_time_consuming(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most time-consuming operations.

        Args:
            top_k: Number of top operations to return

        Returns:
            List of dictionaries with operation details
        """
        df = self.table().sort_values(by="cpu_time_total", ascending=False)
        
        result = []
        for _, row in df.head(top_k).iterrows():
            result.append(row.to_dict())
        
        return result
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory usage statistics
        """
        memory_stats = {
            "total_cpu_memory": 0,
            "total_cuda_memory": 0,
            "peak_cpu_memory": 0,
            "peak_cuda_memory": 0,
            "operations_by_memory": []
        }
        
        df = self.table()
        
        # Calculate totals if available
        if "cpu_memory_usage" in df.columns:
            memory_stats["total_cpu_memory"] = df["cpu_memory_usage"].sum()
            memory_stats["peak_cpu_memory"] = df["cpu_memory_usage"].max()
            
        if "cuda_memory_usage" in df.columns:
            memory_stats["total_cuda_memory"] = df["cuda_memory_usage"].sum()
            memory_stats["peak_cuda_memory"] = df["cuda_memory_usage"].max()
            
        # Get top memory operations
        if "cpu_memory_usage" in df.columns or "cuda_memory_usage" in df.columns:
            memory_col = "cuda_memory_usage" if "cuda_memory_usage" in df.columns else "cpu_memory_usage"
            top_memory_ops = df.sort_values(by=memory_col, ascending=False).head(10)
            
            for _, row in top_memory_ops.iterrows():
                memory_stats["operations_by_memory"].append({
                    "name": row["name"],
                    "memory_usage": row[memory_col],
                    "count": row["count"]
                })
        
        return memory_stats
    
    def save(self, filepath: str) -> None:
        """
        Save profiling results to a file.

        Args:
            filepath: Path to save the results
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> "ProfileResults":
        """
        Load profiling results from a file.

        Args:
            filepath: Path to load the results from

        Returns:
            ProfileResults object
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)


class TorchProfilerWrapper:
    """Wrapper class for the PyTorch profiler with additional functionality."""

    def __init__(self, config: ProfilerConfig = None):
        """
        Initialize the profiler wrapper.

        Args:
            config: Configuration object for the profiler
        """
        self.config = config or ProfilerConfig()
        self._profiler = None
    
    def profile_model(
        self, 
        model: nn.Module, 
        inputs: Any, 
        iterations: int = 10
    ) -> ProfileResults:
        """
        Profile a PyTorch model with given inputs.

        Args:
            model: PyTorch model to profile
            inputs: Input data for the model
            iterations: Number of iterations to run

        Returns:
            ProfileResults containing profiling data
        """
        # Define a function to profile
        def run_model():
            nonlocal model, inputs
            # Handle different input types
            if isinstance(inputs, tuple):
                return model(*inputs)
            elif isinstance(inputs, dict):
                return model(**inputs)
            else:
                return model(inputs)
        
        # Run profiling on the function
        return self.profile_function(run_model, iterations=iterations)
    
    def profile_function(
        self, 
        func: Callable, 
        *args, 
        iterations: int = 10, 
        **kwargs
    ) -> ProfileResults:
        """
        Profile a function with given arguments.

        Args:
            func: Function to profile
            *args: Arguments to pass to the function
            iterations: Number of iterations to run
            **kwargs: Keyword arguments to pass to the function

        Returns:
            ProfileResults containing profiling data
        """
        # Prepare the wrapped function if args/kwargs were provided
        if args or kwargs:
            wrapped_func = lambda: func(*args, **kwargs)
        else:
            wrapped_func = func
        
        # Warm-up (outside of profiling)
        for _ in range(min(3, iterations)):
            wrapped_func()
        
        # Create profiler with config
        with profile(
            activities=self.config.activities,
            schedule=self.config.schedule,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            **self.config.experimental_config
        ) as prof:
            for _ in range(iterations):
                wrapped_func()
                prof.step()
        
        return ProfileResults(prof)
    
    def __enter__(self):
        """Enter the context manager, starting profiling."""
        self._profiler = profile(
            activities=self.config.activities,
            schedule=self.config.schedule,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            **self.config.experimental_config
        )
        self._profiler.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, stopping profiling."""
        self._profiler.__exit__(exc_type, exc_val, exc_tb)
        self.results = ProfileResults(self._profiler)