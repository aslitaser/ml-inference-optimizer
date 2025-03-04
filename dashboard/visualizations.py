"""
ML Inference Optimizer Visualization Components

This module implements various visualization components for the ML Inference Optimizer
dashboard, including timeline visualization, operation breakdown, memory usage plots,
kernel efficiency plots, and comparison visualizations.
"""

from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np


def create_timeline_visualization(profile_data: Dict[str, Any]) -> dcc.Graph:
    """
    Create a timeline visualization of model execution
    
    Args:
        profile_data: Profiling data dictionary containing operation timing
        
    Returns:
        Dash Graph component with timeline visualization
    """
    if not profile_data or 'operations' not in profile_data:
        return dcc.Graph(figure=go.Figure().update_layout(title='No timeline data available'))
        
    operations = profile_data.get('operations', [])
    
    # Extract operation data
    op_names = []
    start_times = []
    end_times = []
    categories = []
    device_types = []
    
    for op in operations:
        op_names.append(op.get('name', 'Unknown'))
        start_times.append(op.get('start_time', 0))
        end_times.append(op.get('end_time', 0))
        categories.append(op.get('category', 'Other'))
        device_types.append(op.get('device', 'CPU'))
    
    # Create a color map for operation categories
    unique_categories = list(set(categories))
    color_map = {cat: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                 for i, cat in enumerate(unique_categories)}
    
    # Create figure
    fig = go.Figure()
    
    # Add operation bars
    for i, op_name in enumerate(op_names):
        fig.add_trace(go.Bar(
            x=[end_times[i] - start_times[i]],
            y=[op_name],
            orientation='h',
            base=start_times[i],
            marker_color=color_map[categories[i]],
            customdata=[[categories[i], device_types[i], end_times[i] - start_times[i]]],
            hovertemplate='<b>%{y}</b><br>Category: %{customdata[0]}<br>Device: %{customdata[1]}<br>Duration: %{customdata[2]:.3f} ms<extra></extra>',
            name=categories[i]
        ))
    
    # Add legend for categories
    for cat in unique_categories:
        fig.add_trace(go.Bar(
            x=[0], 
            y=[''],
            orientation='h',
            marker_color=color_map[cat],
            name=cat,
            showlegend=True,
            hoverinfo='none'
        ))
    
    # Update layout
    fig.update_layout(
        title='Operation Timeline',
        xaxis_title='Time (ms)',
        yaxis_title='Operations',
        barmode='stack',
        legend_title='Operation Category',
        height=max(500, 20 * len(op_names) + 150),
        margin=dict(l=150, r=20, t=40, b=40),
        hovermode='closest'
    )
    
    return dcc.Graph(figure=fig, id='timeline-graph')


def create_operation_breakdown(profile_data: Dict[str, Any]) -> dcc.Graph:
    """
    Create an operation breakdown visualization
    
    Args:
        profile_data: Profiling data dictionary containing operation timing
        
    Returns:
        Dash Graph component with operation breakdown
    """
    if not profile_data or 'operations' not in profile_data:
        return dcc.Graph(figure=go.Figure().update_layout(title='No operation data available'))
        
    operations = profile_data.get('operations', [])
    
    # Group operations by category
    categories = {}
    for op in operations:
        cat = op.get('category', 'Other')
        duration = op.get('end_time', 0) - op.get('start_time', 0)
        
        if cat in categories:
            categories[cat] += duration
        else:
            categories[cat] = duration
    
    # Convert to list for plotting
    cat_names = list(categories.keys())
    cat_durations = list(categories.values())
    
    # Calculate percentages
    total_time = sum(cat_durations)
    percentages = [duration / total_time * 100 for duration in cat_durations]
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'Category': cat_names,
        'Duration': cat_durations,
        'Percentage': percentages
    })
    
    # Sort by duration
    df = df.sort_values('Duration', ascending=False)
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Category'],
        y=df['Duration'],
        marker_color=px.colors.qualitative.Plotly[:len(df)],
        customdata=df['Percentage'],
        hovertemplate='<b>%{x}</b><br>Duration: %{y:.3f} ms<br>Percentage: %{customdata:.2f}%<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Operation Breakdown by Category',
        xaxis_title='Operation Category',
        yaxis_title='Total Duration (ms)',
        height=500,
        margin=dict(l=50, r=20, t=40, b=100),
        hovermode='closest'
    )
    
    # Add percentage text
    for i, row in df.iterrows():
        fig.add_annotation(
            x=row['Category'],
            y=row['Duration'],
            text=f"{row['Percentage']:.1f}%",
            showarrow=False,
            yshift=10
        )
    
    return dcc.Graph(figure=fig, id='operation-breakdown-graph')


def create_memory_usage_plot(profile_data: Dict[str, Any]) -> dcc.Graph:
    """
    Create a memory usage plot
    
    Args:
        profile_data: Profiling data dictionary containing memory usage information
        
    Returns:
        Dash Graph component with memory usage plot
    """
    if not profile_data or 'memory_timeline' not in profile_data:
        return dcc.Graph(figure=go.Figure().update_layout(title='No memory data available'))
        
    memory_timeline = profile_data.get('memory_timeline', [])
    
    # Extract memory data
    timestamps = []
    memory_usage = []
    
    for entry in memory_timeline:
        timestamps.append(entry.get('timestamp', 0))
        memory_usage.append(entry.get('memory_usage', 0) / (1024 * 1024))  # Convert to MB
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=memory_usage,
        mode='lines',
        line=dict(width=2, color='#1f77b4'),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
        hovertemplate='<b>Time: %{x:.3f} ms</b><br>Memory Usage: %{y:.2f} MB<extra></extra>'
    ))
    
    # Mark peak memory usage
    peak_memory = max(memory_usage)
    peak_idx = memory_usage.index(peak_memory)
    
    fig.add_trace(go.Scatter(
        x=[timestamps[peak_idx]],
        y=[peak_memory],
        mode='markers',
        marker=dict(size=10, color='red'),
        hovertemplate='<b>Peak Memory</b><br>Time: %{x:.3f} ms<br>Memory Usage: %{y:.2f} MB<extra></extra>',
        name='Peak Memory'
    ))
    
    # Update layout
    fig.update_layout(
        title='Memory Usage Timeline',
        xaxis_title='Time (ms)',
        yaxis_title='Memory Usage (MB)',
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        hovermode='closest',
        legend_title_text='Memory Usage'
    )
    
    # Add peak memory annotation
    fig.add_annotation(
        x=timestamps[peak_idx],
        y=peak_memory,
        text=f"Peak: {peak_memory:.2f} MB",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    return dcc.Graph(figure=fig, id='memory-usage-graph')


def create_kernel_efficiency_plot(profile_data: Dict[str, Any]) -> dcc.Graph:
    """
    Create a kernel efficiency plot
    
    Args:
        profile_data: Profiling data dictionary containing kernel performance data
        
    Returns:
        Dash Graph component with kernel efficiency plot
    """
    if not profile_data or 'kernels' not in profile_data:
        return dcc.Graph(figure=go.Figure().update_layout(title='No kernel data available'))
        
    kernels = profile_data.get('kernels', [])
    
    # Extract kernel data
    kernel_names = []
    theoretical_flops = []
    achieved_flops = []
    efficiency = []
    
    for kernel in kernels:
        kernel_names.append(kernel.get('name', 'Unknown'))
        theoretical = kernel.get('theoretical_flops', 0)
        achieved = kernel.get('achieved_flops', 0)
        
        theoretical_flops.append(theoretical)
        achieved_flops.append(achieved)
        
        eff = (achieved / theoretical * 100) if theoretical > 0 else 0
        efficiency.append(eff)
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'Kernel': kernel_names,
        'Theoretical FLOPS': theoretical_flops,
        'Achieved FLOPS': achieved_flops,
        'Efficiency': efficiency
    })
    
    # Sort by efficiency
    df = df.sort_values('Efficiency', ascending=False)
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Kernel'],
        y=df['Efficiency'],
        marker_color=df['Efficiency'],
        marker_colorscale='RdYlGn',
        marker_cmin=0,
        marker_cmax=100,
        customdata=list(zip(df['Theoretical FLOPS'], df['Achieved FLOPS'])),
        hovertemplate='<b>%{x}</b><br>Efficiency: %{y:.2f}%<br>Theoretical: %{customdata[0]:.2e} FLOPS<br>Achieved: %{customdata[1]:.2e} FLOPS<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Kernel Efficiency (% of Theoretical Peak)',
        xaxis_title='Kernel',
        yaxis_title='Efficiency (%)',
        height=500,
        margin=dict(l=50, r=20, t=40, b=100),
        hovermode='closest',
        xaxis_tickangle=-45
    )
    
    return dcc.Graph(figure=fig, id='kernel-efficiency-graph')


def create_parallel_scaling_plot(scaling_data: Dict[str, Any]) -> dcc.Graph:
    """
    Create a parallel scaling plot
    
    Args:
        scaling_data: Dictionary containing parallel scaling data
        
    Returns:
        Dash Graph component with parallel scaling plot
    """
    if not scaling_data or 'num_processes' not in scaling_data:
        return dcc.Graph(figure=go.Figure().update_layout(title='No scaling data available'))
        
    num_processes = scaling_data.get('num_processes', [])
    latency = scaling_data.get('latency', [])
    throughput = scaling_data.get('throughput', [])
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=num_processes,
        y=latency,
        mode='lines+markers',
        name='Latency',
        line=dict(width=2, color='#1f77b4'),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=num_processes,
        y=throughput,
        mode='lines+markers',
        name='Throughput',
        line=dict(width=2, color='#ff7f0e'),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    # Add ideal scaling line for throughput
    ideal_throughput = [throughput[0] * n / num_processes[0] for n in num_processes]
    
    fig.add_trace(go.Scatter(
        x=num_processes,
        y=ideal_throughput,
        mode='lines',
        line=dict(dash='dash', color='#ff7f0e', width=1),
        name='Ideal Throughput Scaling',
        yaxis='y2'
    ))
    
    # Update layout with secondary y-axis
    fig.update_layout(
        title='Parallel Scaling Performance',
        xaxis_title='Number of Processes',
        yaxis_title='Latency (ms)',
        yaxis2=dict(
            title='Throughput (samples/sec)',
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.02, y=0.98),
        height=400,
        margin=dict(l=50, r=70, t=40, b=40),
        hovermode='x unified'
    )
    
    return dcc.Graph(figure=fig, id='parallel-scaling-graph')


def create_comparative_timeline(profile_data_list: List[Dict[str, Any]], labels: List[str]) -> dcc.Graph:
    """
    Create a comparative timeline visualization
    
    Args:
        profile_data_list: List of profiling data dictionaries
        labels: Labels for each profile data
        
    Returns:
        Dash Graph component with comparative timeline
    """
    if not profile_data_list:
        return dcc.Graph(figure=go.Figure().update_layout(title='No timeline data available'))
    
    # Create figure
    fig = go.Figure()
    
    # Process each profile data
    for i, (profile_data, label) in enumerate(zip(profile_data_list, labels)):
        if 'operations' not in profile_data:
            continue
            
        operations = profile_data.get('operations', [])
        
        # Extract operation data and group by category
        categories = {}
        for op in operations:
            cat = op.get('category', 'Other')
            duration = op.get('end_time', 0) - op.get('start_time', 0)
            
            if cat in categories:
                categories[cat] += duration
            else:
                categories[cat] = duration
        
        # Convert to list for plotting
        cat_names = list(categories.keys())
        cat_durations = list(categories.values())
        
        # Create color map
        unique_categories = cat_names
        color_map = {cat: px.colors.qualitative.Plotly[j % len(px.colors.qualitative.Plotly)] 
                     for j, cat in enumerate(unique_categories)}
        
        # Add stacked bar for each category
        for cat, duration in categories.items():
            fig.add_trace(go.Bar(
                x=[duration],
                y=[label],
                orientation='h',
                name=cat,
                marker_color=color_map[cat],
                hovertemplate=f'<b>{label}</b><br>{cat}: %{{x:.3f}} ms<extra></extra>',
                customdata=[[cat, duration]]
            ))
    
    # Update layout
    fig.update_layout(
        title='Comparative Timeline',
        xaxis_title='Duration (ms)',
        yaxis_title='',
        barmode='stack',
        legend_title='Operation Category',
        height=300,
        margin=dict(l=100, r=20, t=40, b=40),
        hovermode='closest'
    )
    
    return dcc.Graph(figure=fig, id='comparative-timeline-graph')


def create_speedup_comparison(baseline_data: Dict[str, Any], optimized_data: Dict[str, Any]) -> dcc.Graph:
    """
    Create a speedup comparison visualization
    
    Args:
        baseline_data: Baseline profiling data
        optimized_data: Optimized profiling data
        
    Returns:
        Dash Graph component with speedup comparison
    """
    # Extract key metrics
    baseline_latency = baseline_data.get('avg_latency', 0)
    optimized_latency = optimized_data.get('avg_latency', 0)
    
    baseline_throughput = baseline_data.get('throughput', 0)
    optimized_throughput = optimized_data.get('throughput', 0)
    
    # Calculate speedup ratios
    latency_speedup = baseline_latency / optimized_latency if optimized_latency > 0 else 0
    throughput_speedup = optimized_throughput / baseline_throughput if baseline_throughput > 0 else 0
    
    # Create figure
    fig = go.Figure()
    
    metrics = ['Latency Reduction', 'Throughput Improvement']
    speedups = [latency_speedup, throughput_speedup]
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=speedups,
        marker_color=['#1f77b4', '#ff7f0e'],
        text=[f"{s:.2f}x" for s in speedups],
        textposition='auto'
    ))
    
    # Add baseline reference line
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=1,
        x1=1.5,
        y1=1,
        line=dict(
            color="gray",
            width=2,
            dash="dash",
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Performance Improvement',
        xaxis_title='Metric',
        yaxis_title='Speedup Factor (×)',
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        hovermode='closest'
    )
    
    # Add annotations with absolute values
    fig.add_annotation(
        x=0,
        y=latency_speedup,
        text=f"{baseline_latency:.2f}ms ’ {optimized_latency:.2f}ms",
        showarrow=False,
        yshift=20
    )
    
    fig.add_annotation(
        x=1,
        y=throughput_speedup,
        text=f"{baseline_throughput:.2f} ’ {optimized_throughput:.2f} samples/sec",
        showarrow=False,
        yshift=20
    )
    
    return dcc.Graph(figure=fig, id='speedup-comparison-graph')


def create_memory_comparison(baseline_data: Dict[str, Any], optimized_data: Dict[str, Any]) -> dcc.Graph:
    """
    Create a memory usage comparison visualization
    
    Args:
        baseline_data: Baseline profiling data
        optimized_data: Optimized profiling data
        
    Returns:
        Dash Graph component with memory comparison
    """
    # Extract memory data
    baseline_memory = baseline_data.get('memory_timeline', [])
    optimized_memory = optimized_data.get('memory_timeline', [])
    
    baseline_timestamps = [entry.get('timestamp', 0) for entry in baseline_memory]
    baseline_usage = [entry.get('memory_usage', 0) / (1024 * 1024) for entry in baseline_memory]  # Convert to MB
    
    optimized_timestamps = [entry.get('timestamp', 0) for entry in optimized_memory]
    optimized_usage = [entry.get('memory_usage', 0) / (1024 * 1024) for entry in optimized_memory]  # Convert to MB
    
    # Get peak memory for both
    baseline_peak = max(baseline_usage) if baseline_usage else 0
    optimized_peak = max(optimized_usage) if optimized_usage else 0
    
    # Create figure
    fig = go.Figure()
    
    # Add baseline memory trace
    if baseline_timestamps and baseline_usage:
        fig.add_trace(go.Scatter(
            x=baseline_timestamps,
            y=baseline_usage,
            mode='lines',
            name='Baseline',
            line=dict(width=2, color='#1f77b4'),
            hovertemplate='<b>Baseline</b><br>Time: %{x:.3f} ms<br>Memory: %{y:.2f} MB<extra></extra>'
        ))
    
    # Add optimized memory trace
    if optimized_timestamps and optimized_usage:
        fig.add_trace(go.Scatter(
            x=optimized_timestamps,
            y=optimized_usage,
            mode='lines',
            name='Optimized',
            line=dict(width=2, color='#ff7f0e'),
            hovertemplate='<b>Optimized</b><br>Time: %{x:.3f} ms<br>Memory: %{y:.2f} MB<extra></extra>'
        ))
    
    # Mark peak memory for baseline
    if baseline_timestamps and baseline_usage:
        peak_idx = baseline_usage.index(baseline_peak)
        fig.add_trace(go.Scatter(
            x=[baseline_timestamps[peak_idx]],
            y=[baseline_peak],
            mode='markers',
            marker=dict(size=10, color='#1f77b4'),
            hovertemplate='<b>Baseline Peak</b><br>Time: %{x:.3f} ms<br>Memory: %{y:.2f} MB<extra></extra>',
            name='Baseline Peak'
        ))
    
    # Mark peak memory for optimized
    if optimized_timestamps and optimized_usage:
        peak_idx = optimized_usage.index(optimized_peak)
        fig.add_trace(go.Scatter(
            x=[optimized_timestamps[peak_idx]],
            y=[optimized_peak],
            mode='markers',
            marker=dict(size=10, color='#ff7f0e'),
            hovertemplate='<b>Optimized Peak</b><br>Time: %{x:.3f} ms<br>Memory: %{y:.2f} MB<extra></extra>',
            name='Optimized Peak'
        ))
    
    # Update layout
    fig.update_layout(
        title='Memory Usage Comparison',
        xaxis_title='Time (ms)',
        yaxis_title='Memory Usage (MB)',
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        hovermode='closest',
        legend_title_text='Memory Usage'
    )
    
    # Add memory savings annotation
    memory_savings = ((baseline_peak - optimized_peak) / baseline_peak * 100) if baseline_peak > 0 else 0
    
    fig.add_annotation(
        x=0.5,
        y=1.05,
        xref='paper',
        yref='paper',
        text=f"Memory Reduction: {memory_savings:.1f}% ({baseline_peak:.2f}MB ’ {optimized_peak:.2f}MB)",
        showarrow=False,
        font=dict(size=14)
    )
    
    return dcc.Graph(figure=fig, id='memory-comparison-graph')


def create_breakdown_comparison(results_list: List[Dict[str, Any]], labels: List[str]) -> dcc.Graph:
    """
    Create a breakdown comparison visualization
    
    Args:
        results_list: List of profiling data dictionaries
        labels: Labels for each profile data
        
    Returns:
        Dash Graph component with breakdown comparison
    """
    if not results_list:
        return dcc.Graph(figure=go.Figure().update_layout(title='No breakdown data available'))
    
    # Process results to get category breakdowns
    all_categories = set()
    breakdowns = []
    
    for profile_data in results_list:
        if 'operations' not in profile_data:
            continue
            
        operations = profile_data.get('operations', [])
        
        # Group operations by category
        categories = {}
        for op in operations:
            cat = op.get('category', 'Other')
            duration = op.get('end_time', 0) - op.get('start_time', 0)
            
            if cat in categories:
                categories[cat] += duration
            else:
                categories[cat] = duration
            
            all_categories.add(cat)
            
        breakdowns.append(categories)
    
    # Create figure
    fig = go.Figure()
    
    # Create a bar for each result
    all_categories = sorted(list(all_categories))
    
    for i, (breakdown, label) in enumerate(zip(breakdowns, labels)):
        # Ensure all categories are in the breakdown
        for cat in all_categories:
            if cat not in breakdown:
                breakdown[cat] = 0
        
        # Create a bar for each category
        for j, cat in enumerate(all_categories):
            fig.add_trace(go.Bar(
                x=[label],
                y=[breakdown[cat]],
                name=cat,
                marker_color=px.colors.qualitative.Plotly[j % len(px.colors.qualitative.Plotly)],
                hovertemplate=f'<b>{label}</b><br>{cat}: %{{y:.3f}} ms<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title='Operation Breakdown Comparison',
        xaxis_title='',
        yaxis_title='Duration (ms)',
        barmode='stack',
        legend_title='Operation Category',
        height=500,
        margin=dict(l=50, r=20, t=40, b=40),
        hovermode='closest'
    )
    
    return dcc.Graph(figure=fig, id='breakdown-comparison-graph')


def create_interactive_operation_explorer(profile_data: Dict[str, Any]) -> html.Div:
    """
    Create an interactive operation explorer
    
    Args:
        profile_data: Profiling data dictionary containing operation information
        
    Returns:
        Dash Div component with interactive operation explorer
    """
    if not profile_data or 'operations' not in profile_data:
        return html.Div([
            html.H4("No operation data available"),
        ])
        
    operations = profile_data.get('operations', [])
    
    # Extract operation data
    op_data = []
    for op in operations:
        op_data.append({
            'name': op.get('name', 'Unknown'),
            'category': op.get('category', 'Other'),
            'device': op.get('device', 'CPU'),
            'duration': op.get('end_time', 0) - op.get('start_time', 0),
            'memory_used': op.get('memory_used', 0) / (1024 * 1024),  # Convert to MB
            'flops': op.get('flops', 0),
            'input_shapes': op.get('input_shapes', []),
            'output_shapes': op.get('output_shapes', [])
        })
    
    # Create a dataframe
    df = pd.DataFrame(op_data)
    
    # Create a table figure
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Name', 'Category', 'Device', 'Duration (ms)', 'Memory (MB)', 'FLOPS'],
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[
                df['name'],
                df['category'],
                df['device'],
                df['duration'].round(3),
                df['memory_used'].round(2),
                df['flops']
            ],
            fill_color='lavender',
            align='left'
        )
    )])
    
    fig.update_layout(
        title='Operation Details',
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Create the interactive components
    return html.Div([
        html.H4("Operation Explorer"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Filter by Category"),
                dcc.Dropdown(
                    id='category-filter',
                    options=[{'label': cat, 'value': cat} for cat in sorted(df['category'].unique())],
                    multi=True,
                    placeholder="Select categories..."
                )
            ], width=6),
            
            dbc.Col([
                html.Label("Filter by Device"),
                dcc.Dropdown(
                    id='device-filter',
                    options=[{'label': dev, 'value': dev} for dev in sorted(df['device'].unique())],
                    multi=True,
                    placeholder="Select devices..."
                )
            ], width=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Sort by"),
                dcc.Dropdown(
                    id='sort-by',
                    options=[
                        {'label': 'Duration (Descending)', 'value': 'duration-desc'},
                        {'label': 'Duration (Ascending)', 'value': 'duration-asc'},
                        {'label': 'Memory Usage (Descending)', 'value': 'memory-desc'},
                        {'label': 'Memory Usage (Ascending)', 'value': 'memory-asc'},
                        {'label': 'FLOPS (Descending)', 'value': 'flops-desc'},
                        {'label': 'FLOPS (Ascending)', 'value': 'flops-asc'}
                    ],
                    value='duration-desc'
                )
            ], width=6),
            
            dbc.Col([
                html.Label("Top N Operations"),
                dcc.Slider(
                    id='top-n-slider',
                    min=5,
                    max=50,
                    step=5,
                    value=20,
                    marks={i: str(i) for i in range(5, 51, 5)}
                )
            ], width=6)
        ], className="mb-4"),
        
        dcc.Graph(figure=fig, id='operation-table'),
        
        html.Div(id='operation-details', className="mt-4")
    ])


def create_bottleneck_visualization(bottleneck_data: Dict[str, Any]) -> html.Div:
    """
    Create a bottleneck visualization
    
    Args:
        bottleneck_data: Dictionary containing bottleneck analysis
        
    Returns:
        Dash Div component with bottleneck visualization
    """
    if not bottleneck_data or 'bottlenecks' not in bottleneck_data:
        return html.Div([
            html.H4("No bottleneck data available"),
        ])
        
    bottlenecks = bottleneck_data.get('bottlenecks', [])
    
    # Create a figure for the bottleneck tree
    fig = go.Figure(go.Treemap(
        labels=[b.get('operation_name', 'Unknown') for b in bottlenecks],
        parents=[''] + [b.get('parent', '') for b in bottlenecks[1:]],
        values=[b.get('impact_score', 1) for b in bottlenecks],
        text=[f"Category: {b.get('category', 'Unknown')}<br>Impact: {b.get('impact_score', 0):.2f}" for b in bottlenecks],
        hovertemplate='<b>%{label}</b><br>%{text}<br>Value: %{value:.2f}<extra></extra>',
        marker=dict(
            colorscale='RdBu',
            cmid=0.5
        )
    ))
    
    fig.update_layout(
        title='Bottleneck Analysis (Treemap)',
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Create recommendation cards
    recommendation_cards = []
    for i, bottleneck in enumerate(bottlenecks[:5]):  # Show top 5 bottlenecks
        recommendation = bottleneck.get('recommendation', '')
        potential_gain = bottleneck.get('potential_gain', 0)
        
        card = dbc.Card([
            dbc.CardHeader(bottleneck.get('operation_name', 'Unknown')),
            dbc.CardBody([
                html.H5(f"Category: {bottleneck.get('category', 'Unknown')}"),
                html.P(f"Impact Score: {bottleneck.get('impact_score', 0):.2f}"),
                html.P(f"Potential Gain: {potential_gain:.1f}%"),
                html.P(recommendation)
            ])
        ], className="mb-3")
        
        recommendation_cards.append(card)
    
    # Create the bottleneck visualization
    return html.Div([
        html.H4("Bottleneck Analysis"),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig)
            ], width=12)
        ], className="mb-4"),
        
        html.H4("Top Bottlenecks and Recommendations"),
        
        dbc.Row([
            dbc.Col(recommendation_cards, width=12)
        ])
    ])


def create_parameter_sensitivity_plot(sensitivity_data: Dict[str, Any]) -> dcc.Graph:
    """
    Create a parameter sensitivity plot
    
    Args:
        sensitivity_data: Dictionary containing parameter sensitivity analysis
        
    Returns:
        Dash Graph component with parameter sensitivity plot
    """
    if not sensitivity_data or 'parameters' not in sensitivity_data:
        return dcc.Graph(figure=go.Figure().update_layout(title='No sensitivity data available'))
        
    parameters = sensitivity_data.get('parameters', [])
    
    # Extract parameter data
    param_names = []
    sensitivity_scores = []
    
    for param in parameters:
        param_names.append(param.get('name', 'Unknown'))
        sensitivity_scores.append(param.get('sensitivity', 0))
    
    # Sort by sensitivity score
    sorted_indices = np.argsort(sensitivity_scores)[::-1]
    param_names = [param_names[i] for i in sorted_indices]
    sensitivity_scores = [sensitivity_scores[i] for i in sorted_indices]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=param_names,
        y=sensitivity_scores,
        marker_color=px.colors.sequential.Blues[3:][:(len(param_names))],
        hovertemplate='<b>%{x}</b><br>Sensitivity Score: %{y:.3f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Parameter Sensitivity Analysis',
        xaxis_title='Parameter',
        yaxis_title='Sensitivity Score',
        height=400,
        margin=dict(l=50, r=20, t=40, b=100),
        hovermode='closest',
        xaxis_tickangle=-45
    )
    
    return dcc.Graph(figure=fig, id='parameter-sensitivity-graph')


def create_optimization_impact_visualization(before_after_data: Dict[str, Any]) -> html.Div:
    """
    Create an optimization impact visualization
    
    Args:
        before_after_data: Dictionary containing before and after optimization data
        
    Returns:
        Dash Div component with optimization impact visualization
    """
    if not before_after_data:
        return html.Div([
            html.H4("No optimization impact data available"),
        ])
    
    # Extract optimization data
    optimizations = before_after_data.get('optimizations', [])
    
    # Create a table of optimizations
    optimization_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Optimization"),
            html.Th("Before"),
            html.Th("After"),
            html.Th("Improvement")
        ])),
        html.Tbody([
            html.Tr([
                html.Td(opt.get('name', 'Unknown')),
                html.Td(f"{opt.get('before_value', 0):.3f} {opt.get('unit', '')}"),
                html.Td(f"{opt.get('after_value', 0):.3f} {opt.get('unit', '')}"),
                html.Td([
                    html.Span(
                        f"{opt.get('improvement_pct', 0):.1f}%",
                        style={
                            'color': 'green' if opt.get('improvement_pct', 0) > 0 else 'red',
                            'font-weight': 'bold'
                        }
                    )
                ])
            ]) for opt in optimizations
        ])
    ], bordered=True, hover=True, striped=True, className="mb-4")
    
    # Create a radar chart for overall improvement
    categories = [opt.get('name', 'Unknown') for opt in optimizations]
    before_values = [opt.get('before_normalized', 0) for opt in optimizations]
    after_values = [opt.get('after_normalized', 0) for opt in optimizations]
    
    # Add the first point again to close the polygon
    categories.append(categories[0])
    before_values.append(before_values[0])
    after_values.append(after_values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=before_values,
        theta=categories,
        fill='toself',
        name='Before Optimization',
        line_color='#1f77b4'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=after_values,
        theta=categories,
        fill='toself',
        name='After Optimization',
        line_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title='Optimization Impact (Radar Chart)',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        height=500
    )
    
    # Create the visualization layout
    return html.Div([
        html.H4("Optimization Impact Analysis"),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig)
            ], width=12, lg=6),
            
            dbc.Col([
                html.H5("Optimization Details"),
                optimization_table
            ], width=12, lg=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("Overall Improvement"),
                    html.H3(f"{before_after_data.get('overall_improvement', 0):.1f}%", 
                            style={'color': 'green', 'text-align': 'center'})
                ], className="text-center p-4 border rounded")
            ], width=12)
        ], className="mt-4")
    ])