#!/usr/bin/env python3
"""
ML Inference Optimizer

A framework for optimizing and benchmarking ML inference with various optimization techniques.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from config.config_loader import load_config, save_config, get_optimized_config
from config.config_schema import OptimizerConfig

# Initialize components dynamically based on configuration
def init_components(config: OptimizerConfig):
    """
    Initialize all components based on the configuration.
    
    Args:
        config: OptimizerConfig object with configuration
    
    Returns:
        Dictionary of initialized components
    """
    components = {}
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(config.output_dir, "optimizer.log"), mode='w')
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"ML Inference Optimizer starting with {config.hardware.gpu_count} GPUs")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize model
    if config.model.model_name_or_path:
        try:
            from baseline.model_loader import load_model
            logger.info(f"Loading model: {config.model.model_name_or_path}")
            model_info = load_model(
                model_name_or_path=config.model.model_name_or_path,
                model_type=config.model.model_type,
                precision=config.model.precision,
                trust_remote_code=config.model.trust_remote_code
            )
            components["model"] = model_info["model"]
            components["tokenizer"] = model_info.get("tokenizer")
            components["processor"] = model_info.get("processor")
            logger.info(f"Model loaded successfully: {type(components['model']).__name__}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    # Initialize parallelism if enabled
    if any([config.parallelism.tensor_parallel_size > 1, 
            config.parallelism.sequence_parallel,
            config.parallelism.pipeline_parallel_size > 1]):
        try:
            from parallelism.orchestrator import ParallelizationOrchestrator
            logger.info("Initializing parallelization orchestrator")
            orchestrator = ParallelizationOrchestrator(
                model=components.get("model"),
                tensor_parallel_size=config.parallelism.tensor_parallel_size,
                sequence_parallel=config.parallelism.sequence_parallel,
                pipeline_parallel_size=config.parallelism.pipeline_parallel_size,
                data_parallel_size=config.parallelism.data_parallel_size,
                communication_dtype=config.parallelism.communication_dtype
            )
            components["orchestrator"] = orchestrator
            # Update model with parallelized version if model was loaded
            if "model" in components:
                components["model"] = orchestrator.parallelize_model()
                logger.info("Model parallelized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize parallelism: {str(e)}")
            raise
    
    # Initialize optimized kernels if enabled
    if any([config.kernels.use_flash_attention, 
            config.kernels.use_fused_mlp,
            config.kernels.use_triton_kernels]):
        try:
            logger.info("Initializing optimized kernels")
            if config.kernels.use_flash_attention:
                from kernels.attention.flash_attention import apply_flash_attention
                if "model" in components:
                    components["model"] = apply_flash_attention(components["model"])
                    logger.info("Flash Attention applied to model")
            
            if config.kernels.use_fused_mlp:
                from kernels.mlp.fused_mlp import apply_fused_mlp
                if "model" in components:
                    components["model"] = apply_fused_mlp(components["model"])
                    logger.info("Fused MLP applied to model")
                    
            if config.kernels.use_triton_kernels:
                # TODO: Initialize Triton kernels
                pass
        except Exception as e:
            logger.error(f"Failed to initialize optimized kernels: {str(e)}")
            raise
    
    # Initialize inference runner
    try:
        from baseline.inference import InferenceRunner
        logger.info("Initializing inference runner")
        components["inference_runner"] = InferenceRunner(
            model=components.get("model"),
            tokenizer=components.get("tokenizer"),
            processor=components.get("processor"),
            batch_size=config.model.max_batch_size,
            sequence_length=config.model.max_sequence_length,
            use_cache=config.model.use_cache
        )
        logger.info("Inference runner initialized")
    except Exception as e:
        logger.error(f"Failed to initialize inference runner: {str(e)}")
        raise
    
    # Initialize profiler if enabled
    if config.profiling.enable_profiling:
        try:
            from profiling.torch_profiler import TorchProfilerWrapper
            from profiling.bottleneck_analyzer import BottleneckAnalyzer
            
            logger.info("Initializing profiler")
            profiler = TorchProfilerWrapper(
                save_dir=os.path.join(config.profiling.profiler_output_dir, "torch_profile"),
                num_iterations=config.profiling.profile_iterations,
                save_timeline=config.profiling.save_timeline
            )
            components["profiler"] = profiler
            
            if config.profiling.bottleneck_analysis:
                analyzer = BottleneckAnalyzer(
                    model=components.get("model"),
                    profiler=profiler,
                    output_dir=os.path.join(config.profiling.profiler_output_dir, "bottleneck_analysis")
                )
                components["bottleneck_analyzer"] = analyzer
                logger.info("Bottleneck analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize profiler: {str(e)}")
            raise
    
    # Initialize benchmarker
    try:
        from benchmarks.runners import BenchmarkRunner
        logger.info("Initializing benchmark runner")
        components["benchmark_runner"] = BenchmarkRunner(
            inference_runner=components.get("inference_runner"),
            batch_sizes=config.benchmark.batch_sizes,
            sequence_lengths=config.benchmark.sequence_lengths,
            num_iterations=config.benchmark.num_iterations,
            warmup_iterations=config.benchmark.warmup_iterations,
            metrics=config.benchmark.metrics,
            output_dir=os.path.join(config.output_dir, config.benchmark.report_path)
        )
        logger.info("Benchmark runner initialized")
    except Exception as e:
        logger.error(f"Failed to initialize benchmark runner: {str(e)}")
        raise
    
    # Initialize dashboard if enabled
    if config.dashboard.enable_dashboard:
        try:
            from dashboard.app import create_dashboard
            logger.info("Initializing dashboard")
            components["dashboard"] = create_dashboard(
                config=config,
                host=config.dashboard.host,
                port=config.dashboard.port,
                update_interval=config.dashboard.update_interval_seconds
            )
            logger.info(f"Dashboard initialized at http://{config.dashboard.host}:{config.dashboard.port}")
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {str(e)}")
            # Non-critical component, continue without dashboard
    
    return components


def run_optimizer(config: OptimizerConfig, components: Dict[str, Any], 
                  profiling_only: bool = False, benchmark_only: bool = False):
    """
    Run the ML inference optimizer with the given configuration and components.
    
    Args:
        config: OptimizerConfig object with configuration
        components: Dictionary of initialized components
        profiling_only: If True, only run profiling without optimization
        benchmark_only: If True, only run benchmarking without optimization
    """
    logger = logging.getLogger(__name__)
    
    # Save the current configuration
    save_config(config, os.path.join(config.output_dir, "used_config.yaml"))
    
    # Run profiling if enabled
    if config.profiling.enable_profiling or profiling_only:
        logger.info("Running profiling")
        if "profiler" in components and "inference_runner" in components:
            profiler = components["profiler"]
            inference_runner = components["inference_runner"]
            
            # Profile model inference
            with profiler:
                for _ in range(config.profiling.profile_iterations):
                    inference_runner.run_inference(
                        batch_size=config.model.max_batch_size,
                        sequence_length=config.model.max_sequence_length
                    )
            
            # Run bottleneck analysis if enabled
            if config.profiling.bottleneck_analysis and "bottleneck_analyzer" in components:
                analyzer = components["bottleneck_analyzer"]
                bottlenecks = analyzer.analyze()
                logger.info(f"Bottlenecks identified: {bottlenecks}")
                
                # Output recommendations based on bottleneck analysis
                from dashboard.recommendation import generate_optimization_recommendations
                recommendations = generate_optimization_recommendations(bottlenecks, config)
                
                # Save recommendations
                recommendations_path = os.path.join(config.output_dir, "optimization_recommendations.txt")
                with open(recommendations_path, "w") as f:
                    f.write("\n".join(recommendations))
                logger.info(f"Optimization recommendations saved to {recommendations_path}")
        
        # Stop here if only profiling
        if profiling_only:
            return
    
    # Run benchmarking
    if "benchmark_runner" in components:
        logger.info("Running benchmarks")
        benchmark_runner = components["benchmark_runner"]
        results = benchmark_runner.run_benchmarks()
        
        # Generate report
        from benchmarks.reporting import generate_report
        report_path = os.path.join(config.output_dir, "benchmark_report.html")
        generate_report(results, report_path)
        logger.info(f"Benchmark report generated: {report_path}")
    
    # Start dashboard if enabled
    if config.dashboard.enable_dashboard and "dashboard" in components:
        logger.info(f"Starting dashboard at http://{config.dashboard.host}:{config.dashboard.port}")
        dashboard = components["dashboard"]
        dashboard.run_server(
            debug=False,
            host=config.dashboard.host,
            port=config.dashboard.port
        )


def main():
    """Main entry point for the ML inference optimizer."""
    parser = argparse.ArgumentParser(description="ML Inference Optimizer")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--profiling-only", action="store_true", help="Run only profiling without optimization")
    parser.add_argument("--benchmark-only", action="store_true", help="Run only benchmarking without optimization")
    parser.add_argument("--gpus", type=int, help="Number of GPUs to use")
    parser.add_argument("--batch-size", type=int, help="Maximum batch size")
    parser.add_argument("--sequence-length", type=int, help="Maximum sequence length")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], help="Model precision")
    parser.add_argument("--dashboard", action="store_true", help="Enable dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8050, help="Dashboard port")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.model:
        # Generate optimized config for the specified model
        hardware_config = {"gpu_count": args.gpus} if args.gpus else None
        config = get_optimized_config(args.model, hardware_config)
    else:
        # Load from config file or use default
        config = load_config(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.gpus:
        config.hardware.gpu_count = args.gpus
    if args.batch_size:
        config.model.max_batch_size = args.batch_size
    if args.sequence_length:
        config.model.max_sequence_length = args.sequence_length
    if args.precision:
        config.model.precision = args.precision
    if args.dashboard:
        config.dashboard.enable_dashboard = True
    if args.dashboard_port:
        config.dashboard.port = args.dashboard_port
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize components
    components = init_components(config)
    
    # Run optimizer
    run_optimizer(
        config=config,
        components=components,
        profiling_only=args.profiling_only,
        benchmark_only=args.benchmark_only
    )


if __name__ == "__main__":
    main()