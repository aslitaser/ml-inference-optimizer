"""
Bottleneck analysis for ML model inference optimization.

This module provides tools for analyzing performance bottlenecks
in machine learning models based on profiling data.
"""

import json
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from profiling.kernel_profiler import KernelProfileResults
from profiling.torch_profiler import ProfileResults


class BottleneckType(str, Enum):
    """Types of performance bottlenecks in ML models."""
    
    COMPUTE_BOUND = "compute-bound"
    MEMORY_BOUND = "memory-bound"
    COMMUNICATION_BOUND = "communication-bound"
    IO_BOUND = "io-bound"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class BottleneckReport:
    """Class to represent and format bottleneck analysis results."""
    
    def __init__(
        self,
        primary_bottleneck_type: str,
        bottleneck_operations: List[Dict[str, Any]],
        suggested_optimizations: List[str],
        bottleneck_scores: Optional[Dict[str, float]] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the bottleneck report.
        
        Args:
            primary_bottleneck_type: Main bottleneck type 
                (compute-bound, memory-bound, communication-bound, etc.)
            bottleneck_operations: List of operations identified as bottlenecks
            suggested_optimizations: List of suggested optimizations
            bottleneck_scores: Optional scores for different bottleneck types
            additional_metrics: Optional additional metrics for analysis
        """
        self.primary_bottleneck_type = primary_bottleneck_type
        self.bottleneck_operations = bottleneck_operations
        self.suggested_optimizations = suggested_optimizations
        self.bottleneck_scores = bottleneck_scores or {}
        self.additional_metrics = additional_metrics or {}
    
    def format_report(self) -> str:
        """
        Format the bottleneck report as a readable string.
        
        Returns:
            Formatted report string
        """
        report = []
        
        # Add header
        report.append("=" * 80)
        report.append("PERFORMANCE BOTTLENECK ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Add primary bottleneck info
        report.append(f"\nPRIMARY BOTTLENECK: {self.primary_bottleneck_type.upper()}")
        
        # Add bottleneck scores if available
        if self.bottleneck_scores:
            report.append("\nBOTTLENECK SCORES:")
            for bottleneck_type, score in sorted(
                self.bottleneck_scores.items(), key=lambda x: x[1], reverse=True
            ):
                report.append(f"  - {bottleneck_type}: {score:.2f}")
        
        # Add top bottleneck operations
        report.append("\nTOP BOTTLENECK OPERATIONS:")
        for i, op in enumerate(self.bottleneck_operations[:10], 1):
            name = op.get("name", "Unknown")
            bottleneck_type = op.get("bottleneck_type", "Unknown")
            time_ms = op.get("time_ms", 0)
            
            # Format additional info based on bottleneck type
            additional_info = ""
            if bottleneck_type == BottleneckType.COMPUTE_BOUND:
                additional_info = f"FLOPs: {op.get('flops', 'N/A')}"
            elif bottleneck_type == BottleneckType.MEMORY_BOUND:
                additional_info = f"Memory: {op.get('memory_bytes', 0) / (1024*1024):.2f} MB"
            
            report.append(f"  {i}. {name} ({bottleneck_type}) - {time_ms:.2f} ms {additional_info}")
        
        # Add suggested optimizations
        report.append("\nSUGGESTED OPTIMIZATIONS:")
        for i, suggestion in enumerate(self.suggested_optimizations, 1):
            report.append(f"  {i}. {suggestion}")
        
        # Add additional metrics if available
        if self.additional_metrics:
            report.append("\nADDITIONAL METRICS:")
            for metric, value in self.additional_metrics.items():
                if isinstance(value, float):
                    report.append(f"  - {metric}: {value:.4f}")
                else:
                    report.append(f"  - {metric}: {value}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the report to a dictionary format.
        
        Returns:
            Dictionary representation of the report
        """
        return {
            "primary_bottleneck_type": self.primary_bottleneck_type,
            "bottleneck_scores": self.bottleneck_scores,
            "bottleneck_operations": self.bottleneck_operations,
            "suggested_optimizations": self.suggested_optimizations,
            "additional_metrics": self.additional_metrics
        }
    
    def to_json(self, filepath: Optional[str] = None) -> Optional[str]:
        """
        Convert the report to JSON format.
        
        Args:
            filepath: Optional path to save the JSON report
            
        Returns:
            JSON string if filepath is None, otherwise None
        """
        report_dict = self.to_dict()
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2)
            return None
        else:
            return json.dumps(report_dict, indent=2)


class BottleneckAnalyzer:
    """
    Analyzes performance bottlenecks in ML models based on profiling data.
    
    This class combines data from PyTorch profiler and kernel profiler
    to identify performance bottlenecks and suggest optimizations.
    """
    
    def __init__(
        self, 
        profile_results: ProfileResults, 
        kernel_results: Optional[KernelProfileResults] = None
    ):
        """
        Initialize the bottleneck analyzer.
        
        Args:
            profile_results: Results from PyTorch profiler
            kernel_results: Optional results from kernel profiler
        """
        self.profile_results = profile_results
        self.kernel_results = kernel_results
        
        # Extract profiling data
        self.profile_df = profile_results.table()
        self.kernel_df = kernel_results.get_kernel_stats() if kernel_results else None
        
        # Prepare memory stats if available
        self.memory_stats = profile_results.get_memory_stats()
        
        # Thresholds for analysis
        self.compute_bound_threshold = 0.7  # Ratio of compute time to memory time
        self.memory_bound_threshold = 0.7  # Ratio of memory time to compute time
    
    def analyze(self) -> BottleneckReport:
        """
        Perform comprehensive bottleneck analysis.
        
        Returns:
            BottleneckReport with analysis results
        """
        # Identify compute-bound and memory-bound operations
        compute_bound_ops = self.identify_compute_bound_ops()
        memory_bound_ops = self.identify_memory_bound_ops()
        
        # Calculate bottleneck scores
        bottleneck_scores = self._calculate_bottleneck_scores()
        
        # Determine primary bottleneck type
        primary_bottleneck_type = max(bottleneck_scores.items(), key=lambda x: x[1])[0]
        
        # Combine bottleneck operations, prioritizing the primary type
        if primary_bottleneck_type == BottleneckType.COMPUTE_BOUND:
            bottleneck_ops = compute_bound_ops + memory_bound_ops
        else:
            bottleneck_ops = memory_bound_ops + compute_bound_ops
        
        # Get suggested optimizations
        suggested_optimizations = self.suggest_optimizations()
        
        # Calculate additional metrics
        additional_metrics = self._calculate_additional_metrics()
        
        # Create and return report
        return BottleneckReport(
            primary_bottleneck_type=primary_bottleneck_type,
            bottleneck_operations=bottleneck_ops,
            suggested_optimizations=suggested_optimizations,
            bottleneck_scores=bottleneck_scores,
            additional_metrics=additional_metrics
        )
    
    def _calculate_bottleneck_scores(self) -> Dict[str, float]:
        """
        Calculate scores for different types of bottlenecks.
        
        Returns:
            Dictionary with bottleneck types and their scores
        """
        scores = {
            BottleneckType.COMPUTE_BOUND: 0.0,
            BottleneckType.MEMORY_BOUND: 0.0,
            BottleneckType.COMMUNICATION_BOUND: 0.0,
            BottleneckType.IO_BOUND: 0.0,
        }
        
        # Analyze top operations for bottleneck indicators
        top_ops = self.profile_df.head(20)
        
        # Compute bound indicators: high FLOPS, high computational intensity
        if "cuda_time" in top_ops.columns and "cpu_time" in top_ops.columns:
            cuda_time_ratio = top_ops["cuda_time"].sum() / max(1e-10, top_ops["cpu_time"].sum())
            scores[BottleneckType.COMPUTE_BOUND] += min(1.0, cuda_time_ratio * 0.5)
        
        # Check for compute kernels in operations
        compute_kernel_patterns = [
            'gemm', 'conv', 'matmul', 'bmm', 'addmm', 'mm', 
            'cudnn', 'relu', 'sigmoid', 'tanh', 'gelu'
        ]
        compute_ops_ratio = 0.0
        for pattern in compute_kernel_patterns:
            pattern_ops = top_ops[top_ops["name"].str.contains(pattern, case=False, na=False)]
            if len(top_ops) > 0:
                compute_ops_ratio += len(pattern_ops) / len(top_ops)
        scores[BottleneckType.COMPUTE_BOUND] += min(1.0, compute_ops_ratio)
        
        # Memory bound indicators: transfers, allocations, high memory usage
        memory_kernel_patterns = [
            'memcpy', 'transfer', 'allocation', 'copy', 'cat', 'concat',
            'gather', 'scatter', 'index', 'slice', 'view', 'permute'
        ]
        memory_ops_ratio = 0.0
        for pattern in memory_kernel_patterns:
            pattern_ops = top_ops[top_ops["name"].str.contains(pattern, case=False, na=False)]
            if len(top_ops) > 0:
                memory_ops_ratio += len(pattern_ops) / len(top_ops)
        scores[BottleneckType.MEMORY_BOUND] += min(1.0, memory_ops_ratio * 1.5)  # Weight memory ops higher
        
        # If we have memory stats, use them for memory bound score
        if self.memory_stats and "peak_cpu_memory" in self.memory_stats and "peak_cuda_memory" in self.memory_stats:
            available_memory = 16 * 1024 * 1024 * 1024  # Assuming 16GB GPU memory
            memory_utilization = self.memory_stats["peak_cuda_memory"] / available_memory
            scores[BottleneckType.MEMORY_BOUND] += min(1.0, memory_utilization * 2)
        
        # Communication bound indicators (distributed training, data loading)
        comm_kernel_patterns = ['nccl', 'broadcast', 'reduce', 'all_reduce', 'gather', 'scatter', 'dataloader']
        for pattern in comm_kernel_patterns:
            pattern_ops = top_ops[top_ops["name"].str.contains(pattern, case=False, na=False)]
            if len(pattern_ops) > 0:
                scores[BottleneckType.COMMUNICATION_BOUND] += 0.5
                
        # IO bound indicators
        io_kernel_patterns = ['read', 'write', 'load', 'save', 'open', 'file', 'disk', 'dataloader']
        for pattern in io_kernel_patterns:
            pattern_ops = top_ops[top_ops["name"].str.contains(pattern, case=False, na=False)]
            if len(pattern_ops) > 0:
                scores[BottleneckType.IO_BOUND] += 0.3
                
        # Normalize scores (0.0 to 1.0 range)
        max_score = max(scores.values())
        if max_score > 0:
            for key in scores:
                scores[key] /= max_score
                
        return scores
    
    def _calculate_additional_metrics(self) -> Dict[str, Any]:
        """
        Calculate additional metrics for the report.
        
        Returns:
            Dictionary with additional metrics
        """
        metrics = {}
        
        # Total execution time
        if "cpu_time_total" in self.profile_df.columns:
            metrics["total_cpu_time_ms"] = self.profile_df["cpu_time_total"].sum()
        
        if "cuda_time" in self.profile_df.columns:
            metrics["total_cuda_time_ms"] = self.profile_df["cuda_time"].sum()
        
        # Memory metrics if available
        if self.memory_stats:
            if "peak_cpu_memory" in self.memory_stats:
                metrics["peak_cpu_memory_mb"] = self.memory_stats["peak_cpu_memory"] / (1024 * 1024)
            if "peak_cuda_memory" in self.memory_stats:
                metrics["peak_cuda_memory_mb"] = self.memory_stats["peak_cuda_memory"] / (1024 * 1024)
        
        # Kernel metrics if available
        if self.kernel_df is not None:
            metrics["total_kernels"] = len(self.kernel_df)
            if "occupancy" in self.kernel_df.columns:
                metrics["average_kernel_occupancy"] = self.kernel_df["occupancy"].mean()
        
        return metrics
    
    def identify_compute_bound_ops(self) -> List[Dict[str, Any]]:
        """
        Identify compute-bound operations in the model.
        
        Returns:
            List of compute-bound operations with details
        """
        compute_bound_ops = []
        
        # Get top operations by CPU or CUDA time
        top_ops = self.profile_df.copy()
        
        # Filter for compute-bound operations based on name patterns
        compute_patterns = [
            'gemm', 'conv', 'matmul', 'bmm', 'addmm', 'mm', 
            'cudnn', 'relu', 'sigmoid', 'tanh', 'gelu'
        ]
        
        # Create pattern matching condition
        pattern_condition = False
        for pattern in compute_patterns:
            pattern_condition |= top_ops["name"].str.contains(pattern, case=False, na=False)
        
        # Filter operations
        compute_ops = top_ops[pattern_condition].copy()
        
        # Sort by time
        if "cuda_time" in compute_ops.columns:
            compute_ops = compute_ops.sort_values(by="cuda_time", ascending=False)
            time_col = "cuda_time"
        else:
            compute_ops = compute_ops.sort_values(by="cpu_time_total", ascending=False)
            time_col = "cpu_time_total"
        
        # Convert to list of dictionaries
        for _, row in compute_ops.iterrows():
            op_info = {
                "name": row["name"],
                "bottleneck_type": BottleneckType.COMPUTE_BOUND,
                "time_ms": row[time_col],
            }
            
            # Add additional info if available
            if "input_shapes" in row and not pd.isna(row["input_shapes"]):
                op_info["input_shapes"] = row["input_shapes"]
            
            # Estimate FLOPs if we can determine it
            if "conv" in row["name"].lower():
                op_info["flops"] = "High (Convolution)"
            elif "gemm" in row["name"].lower() or "matmul" in row["name"].lower():
                op_info["flops"] = "High (Matrix Multiplication)"
                
            compute_bound_ops.append(op_info)
        
        return compute_bound_ops
    
    def identify_memory_bound_ops(self) -> List[Dict[str, Any]]:
        """
        Identify memory-bound operations in the model.
        
        Returns:
            List of memory-bound operations with details
        """
        memory_bound_ops = []
        
        # Get top operations by CPU or CUDA time
        top_ops = self.profile_df.copy()
        
        # Filter for memory-bound operations based on name patterns
        memory_patterns = [
            'memcpy', 'transfer', 'allocation', 'copy', 'cat', 'concat',
            'gather', 'scatter', 'index', 'slice', 'view', 'permute'
        ]
        
        # Create pattern matching condition
        pattern_condition = False
        for pattern in memory_patterns:
            pattern_condition |= top_ops["name"].str.contains(pattern, case=False, na=False)
        
        # Filter operations and add memory usage if available
        memory_ops = top_ops[pattern_condition].copy()
        
        # Sort by time
        if "cuda_time" in memory_ops.columns:
            memory_ops = memory_ops.sort_values(by="cuda_time", ascending=False)
            time_col = "cuda_time"
        else:
            memory_ops = memory_ops.sort_values(by="cpu_time_total", ascending=False)
            time_col = "cpu_time_total"
        
        # Convert to list of dictionaries
        for _, row in memory_ops.iterrows():
            op_info = {
                "name": row["name"],
                "bottleneck_type": BottleneckType.MEMORY_BOUND,
                "time_ms": row[time_col],
            }
            
            # Add memory usage if available
            if "cpu_memory_usage" in row and not pd.isna(row["cpu_memory_usage"]):
                op_info["memory_bytes"] = row["cpu_memory_usage"]
            elif "cuda_memory_usage" in row and not pd.isna(row["cuda_memory_usage"]):
                op_info["memory_bytes"] = row["cuda_memory_usage"]
            
            # Add input shapes if available
            if "input_shapes" in row and not pd.isna(row["input_shapes"]):
                op_info["input_shapes"] = row["input_shapes"]
                
            memory_bound_ops.append(op_info)
        
        return memory_bound_ops
    
    def suggest_optimizations(self) -> List[str]:
        """
        Suggest optimization strategies based on identified bottlenecks.
        
        Returns:
            List of suggested optimization strategies
        """
        suggestions = []
        bottleneck_scores = self._calculate_bottleneck_scores()
        
        # Get the primary bottleneck type
        primary_bottleneck = max(bottleneck_scores.items(), key=lambda x: x[1])[0]
        
        # Compute-bound optimizations
        if primary_bottleneck == BottleneckType.COMPUTE_BOUND or bottleneck_scores[BottleneckType.COMPUTE_BOUND] > 0.5:
            suggestions.extend([
                "Use lower precision (FP16 or INT8) to increase throughput",
                "Apply operation fusion to reduce kernel launch overhead",
                "Investigate using tensor cores through torch.cuda.amp",
                "Consider using optimized kernels from libraries like NVIDIA TensorRT",
                "Explore model pruning to reduce computational demands",
                "Profile layers and replace costly operations with approximations"
            ])
            
            # Add kernel-specific optimizations if we have kernel results
            if self.kernel_df is not None and "occupancy" in self.kernel_df.columns:
                # Check for low occupancy kernels
                low_occupancy = self.kernel_df[self.kernel_df["occupancy"] < 0.3]
                if len(low_occupancy) > 0:
                    suggestions.append(
                        "Optimize low-occupancy kernels to better utilize GPU compute capacity"
                    )
        
        # Memory-bound optimizations
        if primary_bottleneck == BottleneckType.MEMORY_BOUND or bottleneck_scores[BottleneckType.MEMORY_BOUND] > 0.5:
            suggestions.extend([
                "Reduce memory transfers between CPU and GPU",
                "Use in-place operations where possible",
                "Implement gradient checkpointing to reduce memory usage",
                "Consider smaller batch sizes or model partitioning",
                "Minimize tensor format changes (e.g., permute, view, transpose)",
                "Investigate memory-efficient attention implementations"
            ])
            
            # Check memory usage patterns in profile results
            memory_ops = self.identify_memory_bound_ops()
            cat_ops = [op for op in memory_ops if "cat" in op["name"].lower() or "concat" in op["name"].lower()]
            if len(cat_ops) > 2:
                suggestions.append(
                    "Optimize tensor concatenation operations which are consuming significant memory"
                )
        
        # Communication-bound optimizations
        if primary_bottleneck == BottleneckType.COMMUNICATION_BOUND or bottleneck_scores[BottleneckType.COMMUNICATION_BOUND] > 0.5:
            suggestions.extend([
                "Implement gradient compression for distributed training",
                "Use mixed-precision to reduce communication volume",
                "Optimize data loading pipeline with prefetching",
                "Consider using NVIDIA NCCL for faster communication",
                "Explore asynchronous or pipeline parallelism"
            ])
        
        # IO-bound optimizations
        if primary_bottleneck == BottleneckType.IO_BOUND or bottleneck_scores[BottleneckType.IO_BOUND] > 0.5:
            suggestions.extend([
                "Use memory-mapped files for large datasets",
                "Implement data prefetching with multiple workers",
                "Consider caching frequently accessed data in memory",
                "Use more efficient data formats (e.g., parquet, binary)",
                "Optimize data preprocessing and augmentation pipeline"
            ])
        
        # Add always-helpful general optimizations
        suggestions.extend([
            "Profile with larger batch sizes to amortize kernel launch overhead",
            "Analyze and reduce model activation memory usage",
            "Consider model distillation to reduce overall model size and complexity"
        ])
        
        return suggestions