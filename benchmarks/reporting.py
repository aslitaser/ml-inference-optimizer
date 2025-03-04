"""
Reporting utilities for ML inference optimization benchmarks.

This module provides classes for generating reports from benchmark results,
including summary statistics, comparison tables, and visualizations.
"""

import os
import json
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from matplotlib.figure import Figure
from benchmarks.runners import BenchmarkConfig


class BenchmarkReport:
    """Class for generating reports from benchmark results."""
    
    def __init__(self, results: Dict[str, Any], config: BenchmarkConfig):
        """
        Initialize the benchmark report.
        
        Args:
            results: Dictionary containing benchmark results.
            config: Configuration used for the benchmark.
        """
        self.results = results
        self.config = config
        self.timestamp = results.get("timestamp", time.time())
        self.formatted_time = datetime.datetime.fromtimestamp(
            self.timestamp
        ).strftime('%Y-%m-%d %H:%M:%S')
    
    def generate_summary(self) -> str:
        """
        Generate a summary of the benchmark results.
        
        Returns:
            String containing the summary.
        """
        summary = []
        summary.append(f"# Benchmark Summary for {self.config.model_name}")
        summary.append(f"Run at: {self.formatted_time}")
        summary.append("")
        summary.append("## Configuration")
        summary.append(f"- Batch sizes: {', '.join(map(str, self.config.batch_sizes))}")
        summary.append(f"- Sequence lengths: {', '.join(map(str, self.config.sequence_lengths))}")
        summary.append(f"- Optimization types: {', '.join(self.config.optimization_types)}")
        summary.append(f"- Precision: {self.config.precision}")
        summary.append(f"- Devices: {', '.join(self.config.devices)}")
        summary.append(f"- Iterations: {self.config.num_iterations} (with {self.config.warmup_iterations} warmup)")
        summary.append("")
        
        summary.append("## Performance Summary")
        
        # Extract best results for each optimization type
        best_results = {}
        for config_key, config_results in self.results.get("benchmarks", {}).items():
            for opt_type, metrics in config_results.items():
                if opt_type not in best_results:
                    best_results[opt_type] = metrics.copy()
                    best_results[opt_type]["config"] = config_key
                elif metrics["throughput_samples_per_sec"] > best_results[opt_type]["throughput_samples_per_sec"]:
                    best_results[opt_type] = metrics.copy()
                    best_results[opt_type]["config"] = config_key
        
        # Add table headers
        summary.append("| Optimization | Best Config | Throughput (samples/s) | Avg Latency (ms) | P99 Latency (ms) | Memory (MB) |")
        summary.append("| ------------ | ----------- | ---------------------- | ---------------- | ---------------- | ----------- |")
        
        # Add results for each optimization type
        for opt_type, metrics in best_results.items():
            config_key = metrics.get("config", "N/A")
            throughput = f"{metrics.get('throughput_samples_per_sec', 0):.2f}"
            avg_latency = f"{metrics.get('avg_latency_ms', 0):.2f}"
            p99_latency = f"{metrics.get('latency_p99_ms', 0):.2f}"
            memory = f"{metrics.get('memory_usage_mb', 0):.2f}"
            
            summary.append(f"| {opt_type} | {config_key} | {throughput} | {avg_latency} | {p99_latency} | {memory} |")
        
        summary.append("")
        
        # Add comparison to baseline if available
        if "baseline" in best_results:
            baseline_metrics = best_results["baseline"]
            summary.append("## Comparison to Baseline")
            summary.append("| Optimization | Speedup | Memory Reduction (%) |")
            summary.append("| ------------ | ------- | -------------------- |")
            
            baseline_latency = baseline_metrics.get("avg_latency_ms", 0)
            baseline_memory = baseline_metrics.get("memory_usage_mb", 0)
            
            for opt_type, metrics in best_results.items():
                if opt_type != "baseline":
                    opt_latency = metrics.get("avg_latency_ms", 0)
                    opt_memory = metrics.get("memory_usage_mb", 0)
                    
                    speedup = baseline_latency / opt_latency if opt_latency > 0 else float('inf')
                    memory_reduction = ((baseline_memory - opt_memory) / baseline_memory * 100) if baseline_memory > 0 else 0
                    
                    summary.append(f"| {opt_type} | {speedup:.2f}x | {memory_reduction:.2f}% |")
            
            summary.append("")
        
        # Add validation results if available
        validation_results = {}
        for config_key, config_results in self.results.get("benchmarks", {}).items():
            for opt_type, metrics in config_results.items():
                if opt_type != "baseline" and "output_validation" in metrics:
                    if opt_type not in validation_results:
                        validation_results[opt_type] = []
                    validation_results[opt_type].append(metrics["output_validation"])
        
        if validation_results:
            summary.append("## Validation Results")
            summary.append("| Optimization | Valid Outputs |")
            summary.append("| ------------ | ------------- |")
            
            for opt_type, validations in validation_results.items():
                valid_count = sum(1 for v in validations if v)
                total_count = len(validations)
                validity = f"{valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)"
                
                summary.append(f"| {opt_type} | {validity} |")
        
        return "\n".join(summary)
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create a comparison table of the benchmark results.
        
        Returns:
            Pandas DataFrame containing the comparison table.
        """
        # Initialize data structure
        data = []
        
        # Extract results for each configuration and optimization type
        for config_key, config_results in self.results.get("benchmarks", {}).items():
            # Parse batch size and sequence length from config key
            parts = config_key.split("_")
            batch_size = int(parts[0].replace("bs", ""))
            seq_len = int(parts[1].replace("seq", ""))
            
            for opt_type, metrics in config_results.items():
                row = {
                    "optimization": opt_type,
                    "batch_size": batch_size,
                    "sequence_length": seq_len,
                    "throughput": metrics.get("throughput_samples_per_sec", 0),
                    "avg_latency_ms": metrics.get("avg_latency_ms", 0),
                    "p50_latency_ms": metrics.get("latency_p50_ms", 0),
                    "p90_latency_ms": metrics.get("latency_p90_ms", 0),
                    "p95_latency_ms": metrics.get("latency_p95_ms", 0),
                    "p99_latency_ms": metrics.get("latency_p99_ms", 0),
                    "memory_mb": metrics.get("memory_usage_mb", 0),
                }
                
                # Add tokens per second if available
                if "tokens_per_second" in metrics:
                    row["tokens_per_second"] = metrics["tokens_per_second"]
                
                data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate speedup and memory reduction compared to baseline if available
        baseline_df = df[df["optimization"] == "baseline"].copy()
        if not baseline_df.empty:
            # Add baseline columns for reference
            baseline_lookup = baseline_df.set_index(["batch_size", "sequence_length"])
            
            # Function to calculate speedup
            def calculate_row_speedup(row):
                try:
                    baseline_latency = baseline_lookup.loc[
                        (row["batch_size"], row["sequence_length"]), "avg_latency_ms"
                    ]
                    return baseline_latency / row["avg_latency_ms"] if row["avg_latency_ms"] > 0 else float('inf')
                except (KeyError, TypeError):
                    return None
            
            # Function to calculate memory reduction
            def calculate_row_memory_reduction(row):
                try:
                    baseline_memory = baseline_lookup.loc[
                        (row["batch_size"], row["sequence_length"]), "memory_mb"
                    ]
                    return ((baseline_memory - row["memory_mb"]) / baseline_memory * 100) if baseline_memory > 0 else 0
                except (KeyError, TypeError):
                    return None
            
            # Apply calculations to non-baseline rows
            non_baseline_df = df[df["optimization"] != "baseline"]
            if not non_baseline_df.empty:
                df.loc[df["optimization"] != "baseline", "speedup"] = non_baseline_df.apply(calculate_row_speedup, axis=1)
                df.loc[df["optimization"] != "baseline", "memory_reduction_pct"] = non_baseline_df.apply(calculate_row_memory_reduction, axis=1)
        
        return df
    
    def plot_scaling_results(self) -> Figure:
        """
        Plot scaling results across different batch sizes.
        
        Returns:
            Matplotlib Figure containing the plot.
        """
        # Get comparison table
        df = self.create_comparison_table()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot throughput vs batch size for each optimization type
        for opt_type in df["optimization"].unique():
            opt_df = df[df["optimization"] == opt_type]
            ax.plot(opt_df["batch_size"], opt_df["throughput"], marker='o', label=opt_type)
        
        # Add labels and title
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (samples/s)")
        ax.set_title(f"Throughput Scaling for {self.config.model_name}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-ticks to match batch sizes
        ax.set_xticks(df["batch_size"].unique())
        
        plt.tight_layout()
        return fig
    
    def plot_latency_distribution(self) -> Figure:
        """
        Plot latency distribution for different optimization types.
        
        Returns:
            Matplotlib Figure containing the plot.
        """
        # Get comparison table
        df = self.create_comparison_table()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for box plot
        plot_data = []
        labels = []
        
        # Extract latency percentiles for each optimization type
        for opt_type in df["optimization"].unique():
            opt_df = df[df["optimization"] == opt_type]
            
            # Get average values across all batch sizes and sequence lengths
            avg_p50 = opt_df["p50_latency_ms"].mean()
            avg_p90 = opt_df["p90_latency_ms"].mean()
            avg_p95 = opt_df["p95_latency_ms"].mean()
            avg_p99 = opt_df["p99_latency_ms"].mean()
            
            # Create box plot data
            plot_data.append([avg_p50, avg_p90, avg_p95, avg_p99])
            labels.append(opt_type)
        
        # Create box plot
        ax.boxplot(plot_data, labels=labels, showfliers=False)
        
        # Add labels and title
        ax.set_xlabel("Optimization Type")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Latency Distribution for {self.config.model_name}")
        
        # Add percentile labels
        ax.text(0.02, 0.95, "Box represents p50, p90, p95, p99 latencies", 
                transform=ax.transAxes, fontsize=9)
        
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_throughput_vs_batch_size(self) -> Figure:
        """
        Plot throughput vs batch size for different optimization types.
        
        Returns:
            Matplotlib Figure containing the plot.
        """
        # Get comparison table
        df = self.create_comparison_table()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot throughput vs batch size for each optimization type
        for opt_type in df["optimization"].unique():
            opt_df = df[df["optimization"] == opt_type]
            
            # Group by batch size and calculate mean throughput
            grouped = opt_df.groupby("batch_size")["throughput"].mean().reset_index()
            
            ax.plot(grouped["batch_size"], grouped["throughput"], marker='o', label=opt_type)
        
        # Add labels and title
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (samples/s)")
        ax.set_title(f"Throughput vs Batch Size for {self.config.model_name}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-ticks to match batch sizes
        ax.set_xticks(df["batch_size"].unique())
        
        plt.tight_layout()
        return fig
    
    def plot_memory_usage(self) -> Figure:
        """
        Plot memory usage for different optimization types.
        
        Returns:
            Matplotlib Figure containing the plot.
        """
        # Get comparison table
        df = self.create_comparison_table()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate average memory usage for each optimization type and batch size
        pivot_df = df.pivot_table(
            index="batch_size", 
            columns="optimization", 
            values="memory_mb",
            aggfunc="mean"
        )
        
        # Plot memory usage
        pivot_df.plot(kind="bar", ax=ax)
        
        # Add labels and title
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Memory Usage (MB)")
        ax.set_title(f"Memory Usage for {self.config.model_name}")
        ax.legend(title="Optimization")
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.tight_layout()
        return fig
    
    def export_to_markdown(self, filepath: str) -> None:
        """
        Export benchmark results to a Markdown file.
        
        Args:
            filepath: Path to the output file.
        """
        with open(filepath, "w") as f:
            f.write(self.generate_summary())
            
            # Add plots if available
            f.write("\n\n## Plots\n\n")
            f.write("Plots are available in the 'plots' directory.\n")
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.dirname(filepath) + "/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save plots
        scaling_fig = self.plot_scaling_results()
        scaling_fig.savefig(f"{plots_dir}/scaling_results.png", dpi=300)
        plt.close(scaling_fig)
        
        latency_fig = self.plot_latency_distribution()
        latency_fig.savefig(f"{plots_dir}/latency_distribution.png", dpi=300)
        plt.close(latency_fig)
        
        throughput_fig = self.plot_throughput_vs_batch_size()
        throughput_fig.savefig(f"{plots_dir}/throughput_vs_batch_size.png", dpi=300)
        plt.close(throughput_fig)
        
        memory_fig = self.plot_memory_usage()
        memory_fig.savefig(f"{plots_dir}/memory_usage.png", dpi=300)
        plt.close(memory_fig)
    
    def export_to_html(self, filepath: str) -> None:
        """
        Export benchmark results to an HTML file.
        
        Args:
            filepath: Path to the output file.
        """
        # Create comparison table
        df = self.create_comparison_table()
        
        # Convert summary to HTML
        summary_html = self.generate_summary().replace("\n", "<br>")
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benchmark Report - {self.config.model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                h1, h2, h3 {{ color: #333; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .plot img {{ max-width: 800px; }}
            </style>
        </head>
        <body>
            <h1>Benchmark Report - {self.config.model_name}</h1>
            <p>Generated at: {self.formatted_time}</p>
            
            <h2>Summary</h2>
            <div>{summary_html}</div>
            
            <h2>Detailed Results</h2>
            {df.to_html(index=False)}
            
            <h2>Plots</h2>
            <div class="plots">
                <div class="plot">
                    <h3>Throughput Scaling</h3>
                    <img src="plots/scaling_results.png" alt="Scaling Results">
                </div>
                
                <div class="plot">
                    <h3>Latency Distribution</h3>
                    <img src="plots/latency_distribution.png" alt="Latency Distribution">
                </div>
                
                <div class="plot">
                    <h3>Throughput vs Batch Size</h3>
                    <img src="plots/throughput_vs_batch_size.png" alt="Throughput vs Batch Size">
                </div>
                
                <div class="plot">
                    <h3>Memory Usage</h3>
                    <img src="plots/memory_usage.png" alt="Memory Usage">
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.dirname(filepath) + "/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save plots
        scaling_fig = self.plot_scaling_results()
        scaling_fig.savefig(f"{plots_dir}/scaling_results.png", dpi=300)
        plt.close(scaling_fig)
        
        latency_fig = self.plot_latency_distribution()
        latency_fig.savefig(f"{plots_dir}/latency_distribution.png", dpi=300)
        plt.close(latency_fig)
        
        throughput_fig = self.plot_throughput_vs_batch_size()
        throughput_fig.savefig(f"{plots_dir}/throughput_vs_batch_size.png", dpi=300)
        plt.close(throughput_fig)
        
        memory_fig = self.plot_memory_usage()
        memory_fig.savefig(f"{plots_dir}/memory_usage.png", dpi=300)
        plt.close(memory_fig)
        
        # Write HTML to file
        with open(filepath, "w") as f:
            f.write(html_content)
    
    def export_to_json(self, filepath: str) -> None:
        """
        Export benchmark results to a JSON file.
        
        Args:
            filepath: Path to the output file.
        """
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)


class ReportGenerator:
    """Class for generating specialized reports."""
    
    def __init__(self, results_dir: str):
        """
        Initialize the report generator.
        
        Args:
            results_dir: Directory containing benchmark results.
        """
        self.results_dir = results_dir
        
        # Check if directory exists
        if not os.path.exists(results_dir):
            raise ValueError(f"Results directory {results_dir} does not exist")
    
    def _load_result(self, result_id: str) -> Dict[str, Any]:
        """
        Load a benchmark result by ID.
        
        Args:
            result_id: Identifier for the benchmark result.
            
        Returns:
            Dictionary containing the benchmark result.
        """
        filepath = os.path.join(self.results_dir, f"{result_id}.json")
        
        if not os.path.exists(filepath):
            raise ValueError(f"Result file {filepath} does not exist")
        
        with open(filepath, "r") as f:
            return json.load(f)
    
    def generate_optimization_report(self, baseline_id: str, optimized_id: str) -> str:
        """
        Generate a report comparing baseline and optimized results.
        
        Args:
            baseline_id: Identifier for the baseline benchmark.
            optimized_id: Identifier for the optimized benchmark.
            
        Returns:
            Generated report as a string.
        """
        # Load benchmark results
        baseline_results = self._load_result(baseline_id)
        optimized_results = self._load_result(optimized_id)
        
        # Create report
        report = []
        report.append(f"# Optimization Report: {baseline_id} vs {optimized_id}")
        report.append("")
        
        # Add model and configuration information
        baseline_config = baseline_results.get("config", {})
        optimized_config = optimized_results.get("config", {})
        
        report.append("## Configuration")
        report.append(f"- Model: {baseline_config.get('model_name', 'Unknown')}")
        report.append(f"- Baseline Optimization: {baseline_config.get('optimization_types', ['Unknown'])[0]}")
        report.append(f"- Optimized Optimization: {optimized_config.get('optimization_types', ['Unknown'])[0]}")
        report.append(f"- Precision: {baseline_config.get('precision', 'Unknown')}")
        report.append(f"- Batch sizes: {', '.join(map(str, baseline_config.get('batch_sizes', [])))}")
        report.append(f"- Sequence lengths: {', '.join(map(str, baseline_config.get('sequence_lengths', [])))}")
        report.append("")
        
        # Extract performance metrics
        baseline_benchmarks = baseline_results.get("benchmarks", {})
        optimized_benchmarks = optimized_results.get("benchmarks", {})
        
        # Compare performance for each configuration
        report.append("## Performance Comparison")
        report.append("| Configuration | Metric | Baseline | Optimized | Improvement |")
        report.append("| ------------- | ------ | -------- | --------- | ----------- |")
        
        for config_key in baseline_benchmarks.keys():
            if config_key in optimized_benchmarks:
                baseline_metrics = baseline_benchmarks[config_key].get(baseline_config.get('optimization_types', ['baseline'])[0], {})
                optimized_metrics = optimized_benchmarks[config_key].get(optimized_config.get('optimization_types', ['optimized'])[0], {})
                
                # Throughput
                baseline_throughput = baseline_metrics.get("throughput_samples_per_sec", 0)
                optimized_throughput = optimized_metrics.get("throughput_samples_per_sec", 0)
                throughput_improvement = (optimized_throughput / baseline_throughput - 1) * 100 if baseline_throughput > 0 else float('inf')
                
                report.append(f"| {config_key} | Throughput (samples/s) | {baseline_throughput:.2f} | {optimized_throughput:.2f} | {throughput_improvement:+.2f}% |")
                
                # Latency
                baseline_latency = baseline_metrics.get("avg_latency_ms", 0)
                optimized_latency = optimized_metrics.get("avg_latency_ms", 0)
                latency_improvement = (1 - optimized_latency / baseline_latency) * 100 if baseline_latency > 0 else float('inf')
                
                report.append(f"| {config_key} | Avg Latency (ms) | {baseline_latency:.2f} | {optimized_latency:.2f} | {latency_improvement:+.2f}% |")
                
                # P99 Latency
                baseline_p99 = baseline_metrics.get("latency_p99_ms", 0)
                optimized_p99 = optimized_metrics.get("latency_p99_ms", 0)
                p99_improvement = (1 - optimized_p99 / baseline_p99) * 100 if baseline_p99 > 0 else float('inf')
                
                report.append(f"| {config_key} | P99 Latency (ms) | {baseline_p99:.2f} | {optimized_p99:.2f} | {p99_improvement:+.2f}% |")
                
                # Memory Usage
                baseline_memory = baseline_metrics.get("memory_usage_mb", 0)
                optimized_memory = optimized_metrics.get("memory_usage_mb", 0)
                memory_improvement = (1 - optimized_memory / baseline_memory) * 100 if baseline_memory > 0 else float('inf')
                
                report.append(f"| {config_key} | Memory Usage (MB) | {baseline_memory:.2f} | {optimized_memory:.2f} | {memory_improvement:+.2f}% |")
        
        return "\n".join(report)
    
    def generate_scaling_report(self, scaling_results: Dict[str, Any]) -> str:
        """
        Generate a report analyzing scaling efficiency.
        
        Args:
            scaling_results: Dictionary containing scaling benchmark results.
            
        Returns:
            Generated report as a string.
        """
        # Create report
        report = []
        
        config = scaling_results.get("config", {})
        model_name = config.get("model_name", "Unknown")
        
        report.append(f"# Scaling Report for {model_name}")
        report.append("")
        
        # Add configuration information
        report.append("## Configuration")
        report.append(f"- Model: {model_name}")
        report.append(f"- Devices: {', '.join(config.get('devices', []))}")
        report.append(f"- Optimization types: {', '.join(config.get('optimization_types', []))}")
        report.append(f"- Batch sizes: {', '.join(map(str, config.get('batch_sizes', [])))}")
        report.append(f"- Precision: {config.get('precision', 'Unknown')}")
        report.append("")
        
        # Add scaling efficiency information
        if "scaling_efficiency" in scaling_results:
            report.append("## Scaling Efficiency")
            report.append("| Configuration | Optimization | Speedup | Scaling Efficiency |")
            report.append("| ------------- | ------------ | ------- | ------------------ |")
            
            for config_key, metrics in scaling_results["scaling_efficiency"].items():
                parts = config_key.split("_")
                opt_type = parts[0]
                config_info = "_".join(parts[1:])
                
                speedup = metrics.get("speedup", 0)
                efficiency = metrics.get("scaling_efficiency", 0)
                num_gpus = metrics.get("num_gpus", 0)
                
                report.append(f"| {config_info} | {opt_type} | {speedup:.2f}x | {efficiency:.2f}% |")
            
            report.append("")
            
            # Calculate average scaling efficiency
            efficiencies = [metrics.get("scaling_efficiency", 0) for metrics in scaling_results["scaling_efficiency"].values()]
            avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0
            
            report.append(f"Average scaling efficiency: {avg_efficiency:.2f}%")
            report.append("")
            
            # Interpret results
            report.append("## Interpretation")
            
            if avg_efficiency >= 90:
                report.append("- Excellent scaling efficiency (>= 90%)")
                report.append("- The model scales very well across multiple GPUs")
                report.append("- Communication overhead is minimal")
            elif avg_efficiency >= 70:
                report.append("- Good scaling efficiency (70% - 90%)")
                report.append("- The model scales reasonably well across multiple GPUs")
                report.append("- Some communication overhead is present but not significant")
            else:
                report.append("- Poor scaling efficiency (< 70%)")
                report.append("- The model does not scale well across multiple GPUs")
                report.append("- Significant communication overhead is present")
                report.append("- Consider optimizing communication patterns or using a different parallelism strategy")
        
        return "\n".join(report)
    
    def generate_comparative_report(self, result_ids: List[str]) -> str:
        """
        Generate a report comparing multiple benchmark results.
        
        Args:
            result_ids: List of result IDs to compare.
            
        Returns:
            Generated report as a string.
        """
        if not result_ids:
            return "No results to compare"
        
        # Load all results
        results = []
        for result_id in result_ids:
            try:
                result = self._load_result(result_id)
                result["id"] = result_id
                results.append(result)
            except ValueError as e:
                print(f"Error loading result {result_id}: {e}")
        
        if not results:
            return "Failed to load any results"
        
        # Create report
        report = []
        report.append("# Comparative Benchmark Report")
        report.append("")
        
        # Add model and configuration information
        report.append("## Configurations")
        report.append("| Result ID | Model | Optimization | Precision | Batch Sizes |")
        report.append("| --------- | ----- | ------------ | --------- | ----------- |")
        
        for result in results:
            result_id = result.get("id", "Unknown")
            config = result.get("config", {})
            model_name = config.get("model_name", "Unknown")
            opt_types = ", ".join(config.get("optimization_types", ["Unknown"]))
            precision = config.get("precision", "Unknown")
            batch_sizes = ", ".join(map(str, config.get("batch_sizes", [])))
            
            report.append(f"| {result_id} | {model_name} | {opt_types} | {precision} | {batch_sizes} |")
        
        report.append("")
        
        # Extract common configurations for comparison
        common_configs = set()
        for result in results:
            configs = set(result.get("benchmarks", {}).keys())
            if not common_configs:
                common_configs = configs
            else:
                common_configs &= configs
        
        if not common_configs:
            report.append("No common configurations found for comparison")
            return "\n".join(report)
        
        # Compare throughput
        report.append("## Throughput Comparison (samples/s)")
        throughput_table = ["| Configuration |"]
        separator = ["| ------------- |"]
        
        # Add result IDs to header
        for result in results:
            result_id = result.get("id", "Unknown")
            throughput_table[0] += f" {result_id} |"
            separator[0] += " --- |"
        
        throughput_table.append(separator[0])
        
        # Add throughput data
        for config in sorted(common_configs):
            row = f"| {config} |"
            
            for result in results:
                config_results = result.get("benchmarks", {}).get(config, {})
                
                # Get the first optimization type's results
                opt_type = result.get("config", {}).get("optimization_types", ["Unknown"])[0]
                metrics = config_results.get(opt_type, {})
                
                throughput = metrics.get("throughput_samples_per_sec", 0)
                row += f" {throughput:.2f} |"
            
            throughput_table.append(row)
        
        report.extend(throughput_table)
        report.append("")
        
        # Compare latency
        report.append("## Latency Comparison (ms)")
        latency_table = ["| Configuration |"]
        separator = ["| ------------- |"]
        
        # Add result IDs to header
        for result in results:
            result_id = result.get("id", "Unknown")
            latency_table[0] += f" {result_id} |"
            separator[0] += " --- |"
        
        latency_table.append(separator[0])
        
        # Add latency data
        for config in sorted(common_configs):
            row = f"| {config} |"
            
            for result in results:
                config_results = result.get("benchmarks", {}).get(config, {})
                
                # Get the first optimization type's results
                opt_type = result.get("config", {}).get("optimization_types", ["Unknown"])[0]
                metrics = config_results.get(opt_type, {})
                
                latency = metrics.get("avg_latency_ms", 0)
                row += f" {latency:.2f} |"
            
            latency_table.append(row)
        
        report.extend(latency_table)
        report.append("")
        
        # Compare memory usage
        report.append("## Memory Usage Comparison (MB)")
        memory_table = ["| Configuration |"]
        separator = ["| ------------- |"]
        
        # Add result IDs to header
        for result in results:
            result_id = result.get("id", "Unknown")
            memory_table[0] += f" {result_id} |"
            separator[0] += " --- |"
        
        memory_table.append(separator[0])
        
        # Add memory data
        for config in sorted(common_configs):
            row = f"| {config} |"
            
            for result in results:
                config_results = result.get("benchmarks", {}).get(config, {})
                
                # Get the first optimization type's results
                opt_type = result.get("config", {}).get("optimization_types", ["Unknown"])[0]
                metrics = config_results.get(opt_type, {})
                
                memory = metrics.get("memory_usage_mb", 0)
                row += f" {memory:.2f} |"
            
            memory_table.append(row)
        
        report.extend(memory_table)
        
        return "\n".join(report)
    
    def create_github_readme(self, result_summary: Dict[str, Any], filepath: str) -> None:
        """
        Create a GitHub README.md with benchmark results.
        
        Args:
            result_summary: Dictionary containing summarized benchmark results.
            filepath: Path to the output file.
        """
        with open(filepath, "w") as f:
            f.write("# ML Inference Optimization Benchmark Results\n\n")
            
            f.write("## Overview\n\n")
            f.write("This repository contains benchmark results for various ML inference optimization techniques.\n")
            f.write("The benchmarks measure throughput, latency, and memory usage for different model configurations.\n\n")
            
            f.write("## Models Tested\n\n")
            for model_name in result_summary.get("models", []):
                f.write(f"- {model_name}\n")
            f.write("\n")
            
            f.write("## Optimization Techniques\n\n")
            for opt_type in result_summary.get("optimization_types", []):
                f.write(f"- {opt_type}\n")
            f.write("\n")
            
            f.write("## Key Findings\n\n")
            
            # Add best performing configurations
            if "best_configs" in result_summary:
                f.write("### Best Performing Configurations\n\n")
                f.write("| Model | Optimization | Batch Size | Sequence Length | Throughput (samples/s) | Avg Latency (ms) |\n")
                f.write("| ----- | ------------ | ---------- | --------------- | ---------------------- | ---------------- |\n")
                
                for model, configs in result_summary.get("best_configs", {}).items():
                    for opt_type, metrics in configs.items():
                        batch_size = metrics.get("batch_size", "N/A")
                        seq_len = metrics.get("sequence_length", "N/A")
                        throughput = metrics.get("throughput_samples_per_sec", 0)
                        latency = metrics.get("avg_latency_ms", 0)
                        
                        f.write(f"| {model} | {opt_type} | {batch_size} | {seq_len} | {throughput:.2f} | {latency:.2f} |\n")
                
                f.write("\n")
            
            # Add speedup summary
            if "speedups" in result_summary:
                f.write("### Optimization Speedups\n\n")
                f.write("| Model | Optimization | Speedup vs Baseline | Memory Reduction |\n")
                f.write("| ----- | ------------ | ------------------- | --------------- |\n")
                
                for model, speedups in result_summary.get("speedups", {}).items():
                    for opt_type, metrics in speedups.items():
                        speedup = metrics.get("speedup", 0)
                        memory_reduction = metrics.get("memory_reduction", 0)
                        
                        f.write(f"| {model} | {opt_type} | {speedup:.2f}x | {memory_reduction:.2f}% |\n")
                
                f.write("\n")
            
            f.write("## Detailed Reports\n\n")
            f.write("Detailed benchmark reports are available in the `reports` directory.\n")
            f.write("Each report includes comprehensive performance metrics and visualizations.\n\n")
            
            f.write("## How to Run Benchmarks\n\n")
            f.write("To run the benchmarks yourself, follow these steps:\n\n")
            f.write("1. Clone this repository\n")
            f.write("2. Install dependencies: `pip install -r requirements.txt`\n")
            f.write("3. Run the benchmarks: `python -m benchmarks.run`\n")