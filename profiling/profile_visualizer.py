"""
Visualization utilities for ML model profiling results.

This module provides tools for creating visualizations of profiling data,
including timeline plots, operation breakdowns, and interactive dashboards.
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import html, dcc

# Import from our package
from profiling.kernel_profiler import KernelProfileResults
from profiling.torch_profiler import ProfileResults


class ProfileVisualizer:
    """
    Creates visualizations based on profiling results.
    
    This class provides methods to generate various plots and visualizations
    to help analyze and understand profiling data.
    """
    
    def __init__(self, profile_results: ProfileResults):
        """
        Initialize the profile visualizer.
        
        Args:
            profile_results: ProfileResults object containing profiling data
        """
        self.profile_results = profile_results
        self.profile_df = profile_results.table()
        self.memory_stats = profile_results.get_memory_stats()
        self.top_ops = self._get_top_operations()
    
    def _get_top_operations(self, top_k: int = 20) -> pd.DataFrame:
        """
        Get the top operations by time.
        
        Args:
            top_k: Number of top operations to return
            
        Returns:
            DataFrame with top operations
        """
        # Choose the time column to sort by (prefer CUDA time if available)
        if "cuda_time" in self.profile_df.columns:
            time_col = "cuda_time"
        else:
            time_col = "cpu_time_total"
        
        # Get top operations by time
        df = self.profile_df.sort_values(by=time_col, ascending=False).head(top_k).copy()
        
        # Calculate percentage of total time
        total_time = self.profile_df[time_col].sum()
        df["percentage"] = (df[time_col] / total_time * 100) if total_time > 0 else 0
        
        return df
    
    def create_timeline_plot(self) -> Figure:
        """
        Create a timeline visualization of operations.
        
        Returns:
            Matplotlib Figure object with timeline plot
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Check if we have necessary data for a timeline
        if "ts" not in self.profile_df.columns and "start_time" not in self.profile_df.columns:
            # Create a simple bar chart instead
            top_ops = self.top_ops.sort_values(by="percentage", ascending=True)
            
            # Create horizontal bar plot
            bars = ax.barh(
                y=np.arange(len(top_ops)),
                width=top_ops["percentage"],
                height=0.7
            )
            
            # Add operation names and percentages
            ax.set_yticks(np.arange(len(top_ops)))
            ax.set_yticklabels(top_ops["name"])
            
            # Add percentage labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x_pos = width * 1.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                        f"{width:.1f}%", va='center')
            
            # Add titles and labels
            ax.set_xlabel("Percentage of Total Time (%)")
            ax.set_title("Top Operations by Execution Time")
            
        else:
            # Get time column name
            time_col = "ts" if "ts" in self.profile_df.columns else "start_time"
            duration_col = "dur" if "dur" in self.profile_df.columns else "cpu_time"
            
            # Get operations sorted by start time
            timeline_df = self.profile_df.sort_values(by=time_col)
            
            # Normalize times to start from 0
            min_time = timeline_df[time_col].min()
            timeline_df["time_normalized"] = timeline_df[time_col] - min_time
            
            # Get unique operation names for y-axis (limit to top operations for readability)
            unique_ops = timeline_df["name"].unique()[:30]  # Limit to 30 unique operations
            op_to_idx = {op: i for i, op in enumerate(unique_ops)}
            
            # Plot each operation as a horizontal line
            for _, row in timeline_df.iterrows():
                if row["name"] in op_to_idx:
                    op_idx = op_to_idx[row["name"]]
                    start_time = row["time_normalized"]
                    duration = row[duration_col]
                    
                    # Plot the operation execution
                    ax.broken_barh(
                        [(start_time, duration)],
                        (op_idx - 0.4, 0.8),
                        facecolors='tab:blue',
                        alpha=0.7
                    )
            
            # Set y-ticks to operation names
            ax.set_yticks(range(len(unique_ops)))
            ax.set_yticklabels(unique_ops)
            
            # Set labels
            ax.set_xlabel("Time (ms)")
            ax.set_title("Operation Execution Timeline")
        
        plt.tight_layout()
        return fig
    
    def create_operation_breakdown(self) -> Figure:
        """
        Create a breakdown of operations by time.
        
        Returns:
            Matplotlib Figure object with operation breakdown
        """
        # Create figure with two subplots (pie chart and bar chart)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Choose the time column to use
        if "cuda_time" in self.profile_df.columns:
            time_col = "cuda_time"
        else:
            time_col = "cpu_time_total"
        
        # Group operations by category (use first part of the operation name as category)
        self.profile_df["category"] = self.profile_df["name"].apply(
            lambda x: x.split("::")[0] if "::" in x else x.split("_")[0]
        )
        
        # Aggregate by category
        category_df = self.profile_df.groupby("category")[time_col].sum().reset_index()
        category_df["percentage"] = category_df[time_col] / category_df[time_col].sum() * 100
        category_df = category_df.sort_values(by="percentage", ascending=False)
        
        # Create pie chart for top categories
        top_categories = category_df.head(8).copy()
        other_percentage = 100 - top_categories["percentage"].sum()
        
        if other_percentage > 0:
            other_row = pd.DataFrame({
                "category": ["Other"],
                time_col: [category_df[time_col].sum() - top_categories[time_col].sum()],
                "percentage": [other_percentage]
            })
            plot_df = pd.concat([top_categories, other_row])
        else:
            plot_df = top_categories
        
        # Create pie chart
        wedges, texts, autotexts = ax1.pie(
            plot_df["percentage"],
            labels=plot_df["category"],
            autopct='%1.1f%%',
            startangle=90,
            shadow=False,
        )
        
        # Make the percentage text more readable
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_weight('bold')
        
        ax1.set_title("Operation Categories by Time")
        
        # Create bar chart for top individual operations
        top_ops = self.top_ops.head(10).copy()
        bars = ax2.barh(
            y=np.arange(len(top_ops)),
            width=top_ops["percentage"],
            height=0.7
        )
        
        # Add operation names and percentages
        ax2.set_yticks(np.arange(len(top_ops)))
        ax2.set_yticklabels(top_ops["name"])
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width * 1.01
            ax2.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f"{width:.1f}%", va='center')
        
        ax2.set_xlabel("Percentage of Total Time (%)")
        ax2.set_title("Top Individual Operations")
        
        plt.tight_layout()
        return fig
    
    def create_memory_usage_plot(self) -> Figure:
        """
        Create a visualization of memory usage.
        
        Returns:
            Matplotlib Figure object with memory usage plots
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Check if we have memory trace data
        memory_trace = None
        if self.memory_stats and "memory_trace" in self.memory_stats:
            memory_trace = self.memory_stats["memory_trace"]
        
        # First subplot: Memory usage over time (if available)
        if memory_trace:
            # Convert memory trace to arrays
            times, memory_values = zip(*memory_trace)
            
            # Normalize times to start from 0
            times = np.array(times) - times[0]
            memory_values = np.array(memory_values) / (1024 * 1024)  # Convert to MB
            
            # Plot memory over time
            ax1.plot(times, memory_values, '-o', markersize=3)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Memory Usage (MB)")
            ax1.set_title("Memory Usage Over Time")
            ax1.grid(True, linestyle='--', alpha=0.7)
        else:
            # Show a message if memory trace is not available
            ax1.text(0.5, 0.5, "Memory trace data not available", 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Memory Usage Over Time")
        
        # Second subplot: Memory usage by operation (if available)
        if "cpu_memory_usage" in self.profile_df.columns or "cuda_memory_usage" in self.profile_df.columns:
            # Choose memory column
            memory_col = "cuda_memory_usage" if "cuda_memory_usage" in self.profile_df.columns else "cpu_memory_usage"
            
            # Get top memory-consuming operations
            memory_ops = self.profile_df.sort_values(by=memory_col, ascending=False).head(10).copy()
            memory_ops[memory_col + "_mb"] = memory_ops[memory_col] / (1024 * 1024)
            
            # Create horizontal bar chart
            bars = ax2.barh(
                y=np.arange(len(memory_ops)),
                width=memory_ops[memory_col + "_mb"],
                height=0.7
            )
            
            # Add operation names and memory values
            ax2.set_yticks(np.arange(len(memory_ops)))
            ax2.set_yticklabels(memory_ops["name"])
            
            # Add memory labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x_pos = width * 1.01
                ax2.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                        f"{width:.1f} MB", va='center')
            
            ax2.set_xlabel("Memory Usage (MB)")
            ax2.set_title("Top Operations by Memory Usage")
        else:
            # Show a message if memory data is not available
            ax2.text(0.5, 0.5, "Memory usage by operation not available", 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("Memory Usage by Operation")
        
        plt.tight_layout()
        return fig
    
    def create_kernel_efficiency_plot(self, kernel_results: KernelProfileResults) -> Figure:
        """
        Create a visualization of kernel efficiency.
        
        Args:
            kernel_results: KernelProfileResults object
            
        Returns:
            Matplotlib Figure object with kernel efficiency plots
        """
        # Get kernel data
        kernel_df = kernel_results.get_kernel_stats()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # First subplot: Kernel occupancy distribution
        if "occupancy" in kernel_df.columns:
            # Create histogram of occupancy values
            ax1.hist(
                kernel_df["occupancy"],
                bins=10,
                range=(0, 1),
                alpha=0.7,
                edgecolor='black'
            )
            
            # Add vertical line for average occupancy
            avg_occupancy = kernel_df["occupancy"].mean()
            ax1.axvline(x=avg_occupancy, color='red', linestyle='--')
            ax1.text(
                avg_occupancy + 0.02, ax1.get_ylim()[1] * 0.9,
                f"Avg: {avg_occupancy:.2f}",
                color='red',
                fontweight='bold'
            )
            
            ax1.set_xlabel("Kernel Occupancy")
            ax1.set_ylabel("Number of Kernels")
            ax1.set_title("Kernel Occupancy Distribution")
        else:
            # Show a message if occupancy data is not available
            ax1.text(0.5, 0.5, "Kernel occupancy data not available", 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Kernel Occupancy Distribution")
        
        # Second subplot: Top kernels by duration
        top_kernels = kernel_df.sort_values(by="duration_ms", ascending=False).head(10).copy()
        
        # Create horizontal bar chart
        bars = ax2.barh(
            y=np.arange(len(top_kernels)),
            width=top_kernels["duration_ms"],
            height=0.7
        )
        
        # Add kernel names and durations
        ax2.set_yticks(np.arange(len(top_kernels)))
        ax2.set_yticklabels(top_kernels["kernel_name"])
        
        # Add duration labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width * 1.01
            ax2.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f"{width:.2f} ms", va='center')
        
        ax2.set_xlabel("Duration (ms)")
        ax2.set_title("Top Kernels by Duration")
        
        plt.tight_layout()
        return fig
    
    def save_visualizations(self, directory: str) -> None:
        """
        Save all visualizations to the specified directory.
        
        Args:
            directory: Directory path to save visualizations
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Create and save timeline plot
        timeline_fig = self.create_timeline_plot()
        timeline_fig.savefig(os.path.join(directory, "timeline_plot.png"), dpi=300, bbox_inches="tight")
        plt.close(timeline_fig)
        
        # Create and save operation breakdown
        breakdown_fig = self.create_operation_breakdown()
        breakdown_fig.savefig(os.path.join(directory, "operation_breakdown.png"), dpi=300, bbox_inches="tight")
        plt.close(breakdown_fig)
        
        # Create and save memory usage plot
        memory_fig = self.create_memory_usage_plot()
        memory_fig.savefig(os.path.join(directory, "memory_usage.png"), dpi=300, bbox_inches="tight")
        plt.close(memory_fig)


def create_interactive_dashboard(profile_results: ProfileResults) -> html.Div:
    """
    Create an interactive dashboard using Dash and Plotly.
    
    Args:
        profile_results: ProfileResults object
        
    Returns:
        Dash HTML div containing the dashboard
    """
    # Extract data from profile results
    profile_df = profile_results.table()
    memory_stats = profile_results.get_memory_stats()
    
    # Choose the time column to use
    if "cuda_time" in profile_df.columns:
        time_col = "cuda_time"
    else:
        time_col = "cpu_time_total"
    
    # Calculate percentages
    total_time = profile_df[time_col].sum()
    profile_df["percentage"] = (profile_df[time_col] / total_time * 100) if total_time > 0 else 0
    
    # Get top operations
    top_ops = profile_df.sort_values(by=time_col, ascending=False).head(20)
    
    # Create operation breakdown figure
    operation_fig = px.bar(
        top_ops,
        y="name",
        x="percentage",
        orientation="h",
        title="Top Operations by Time",
        labels={"percentage": "Percentage of Total Time (%)", "name": "Operation"},
        color="percentage",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    # Add more details on hover
    operation_fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Time: %{x:.2f}%<br>Duration: %{customdata[0]:.2f} ms",
        customdata=top_ops[[time_col]]
    )
    
    # If we have memory data, create memory figure
    if "cpu_memory_usage" in profile_df.columns or "cuda_memory_usage" in profile_df.columns:
        memory_col = "cuda_memory_usage" if "cuda_memory_usage" in profile_df.columns else "cpu_memory_usage"
        memory_ops = profile_df.sort_values(by=memory_col, ascending=False).head(20)
        memory_ops[memory_col + "_mb"] = memory_ops[memory_col] / (1024 * 1024)
        
        memory_fig = px.bar(
            memory_ops,
            y="name",
            x=memory_col + "_mb",
            orientation="h",
            title="Top Operations by Memory Usage",
            labels={memory_col + "_mb": "Memory Usage (MB)", "name": "Operation"},
            color=memory_col + "_mb",
            color_continuous_scale=px.colors.sequential.Plasma
        )
        
        # Add more details on hover
        memory_fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Memory: %{x:.2f} MB"
        )
    else:
        # Create empty figure with message
        memory_fig = go.Figure()
        memory_fig.add_annotation(
            text="Memory usage data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        memory_fig.update_layout(title="Memory Usage")
    
    # Create a third figure showing operation types
    profile_df["category"] = profile_df["name"].apply(
        lambda x: x.split("::")[0] if "::" in x else x.split("_")[0]
    )
    category_df = profile_df.groupby("category")[time_col].sum().reset_index()
    category_df["percentage"] = category_df[time_col] / category_df[time_col].sum() * 100
    category_df = category_df.sort_values(by="percentage", ascending=False)
    
    category_fig = px.pie(
        category_df.head(8),
        names="category",
        values="percentage",
        title="Operation Categories by Time",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    category_fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    # Build the dashboard layout
    dashboard = html.Div([
        html.H1("PyTorch Profiler Dashboard", style={"textAlign": "center"}),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=operation_fig)
            ], style={"width": "50%", "display": "inline-block"}),
            
            html.Div([
                dcc.Graph(figure=category_fig)
            ], style={"width": "50%", "display": "inline-block"})
        ]),
        
        html.Div([
            dcc.Graph(figure=memory_fig)
        ])
    ])
    
    return dashboard


def create_comparative_visualization(results_list: List[ProfileResults]) -> html.Div:
    """
    Create a comparative visualization of multiple profiling results.
    
    Args:
        results_list: List of ProfileResults objects
        
    Returns:
        Dash HTML div containing comparative visualizations
    """
    # Check if we have labels for the results
    if hasattr(results_list[0], "label"):
        labels = [result.label for result in results_list]
    else:
        labels = [f"Model {i+1}" for i in range(len(results_list))]
    
    # Extract data from results
    time_data = []
    memory_data = []
    
    for i, result in enumerate(results_list):
        df = result.table()
        
        # Choose time column
        if "cuda_time" in df.columns:
            time_col = "cuda_time"
        else:
            time_col = "cpu_time_total"
        
        # Get total time
        total_time = df[time_col].sum()
        time_data.append({"model": labels[i], "total_time_ms": total_time})
        
        # Get memory stats if available
        memory_stats = result.get_memory_stats()
        if memory_stats and "peak_cuda_memory" in memory_stats:
            peak_memory = memory_stats["peak_cuda_memory"] / (1024 * 1024)  # Convert to MB
            memory_data.append({"model": labels[i], "peak_memory_mb": peak_memory})
    
    # Create time comparison figure
    time_df = pd.DataFrame(time_data)
    time_fig = px.bar(
        time_df,
        x="model",
        y="total_time_ms",
        title="Total Execution Time Comparison",
        labels={"model": "Model", "total_time_ms": "Total Time (ms)"},
        color="model",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    time_fig.update_layout(showlegend=False)
    
    # Create memory comparison figure if we have memory data
    if memory_data:
        memory_df = pd.DataFrame(memory_data)
        memory_fig = px.bar(
            memory_df,
            x="model",
            y="peak_memory_mb",
            title="Peak Memory Usage Comparison",
            labels={"model": "Model", "peak_memory_mb": "Peak Memory (MB)"},
            color="model",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        memory_fig.update_layout(showlegend=False)
    else:
        # Create empty figure with message
        memory_fig = go.Figure()
        memory_fig.add_annotation(
            text="Memory usage data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        memory_fig.update_layout(title="Memory Usage Comparison")
    
    # Create a third comparative figure for top operations
    top_ops_comparison = make_subplots(
        rows=len(results_list), 
        cols=1,
        subplot_titles=[f"Top Operations for {label}" for label in labels],
        vertical_spacing=0.1
    )
    
    for i, result in enumerate(results_list):
        df = result.table()
        
        # Choose time column
        if "cuda_time" in df.columns:
            time_col = "cuda_time"
        else:
            time_col = "cpu_time_total"
        
        # Calculate percentages
        total_time = df[time_col].sum()
        df["percentage"] = (df[time_col] / total_time * 100) if total_time > 0 else 0
        
        # Get top operations
        top_ops = df.sort_values(by=time_col, ascending=False).head(5)
        
        # Add bar trace for this result
        top_ops_comparison.add_trace(
            go.Bar(
                y=top_ops["name"],
                x=top_ops["percentage"],
                orientation="h",
                marker_color=px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)],
                name=labels[i],
                showlegend=False
            ),
            row=i+1, col=1
        )
    
    top_ops_comparison.update_layout(
        title="Top Operations Comparison",
        height=300 * len(results_list)
    )
    
    # Build the comparative dashboard layout
    comparison = html.Div([
        html.H1("Profiling Results Comparison", style={"textAlign": "center"}),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=time_fig)
            ], style={"width": "50%", "display": "inline-block"}),
            
            html.Div([
                dcc.Graph(figure=memory_fig)
            ], style={"width": "50%", "display": "inline-block"})
        ]),
        
        html.Div([
            dcc.Graph(figure=top_ops_comparison)
        ])
    ])
    
    return comparison