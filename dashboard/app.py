"""
ML Inference Optimization Dashboard Application

This module implements a Flask/Dash application for visualizing and analyzing 
ML model optimization results, providing insights into performance bottlenecks,
and generating optimization recommendations.
"""

import os
import json
from typing import Dict, Any, List, Optional
import logging

from flask import Flask, Response, request, jsonify, render_template
import dash
from dash import dcc, html

from dashboard.visualizations import (
    create_timeline_visualization,
    create_operation_breakdown,
    create_memory_usage_plot,
    create_kernel_efficiency_plot,
    create_parallel_scaling_plot,
    create_comparative_timeline,
    create_speedup_comparison,
    create_memory_comparison,
    create_breakdown_comparison,
    create_interactive_operation_explorer,
    create_bottleneck_visualization,
    create_parameter_sensitivity_plot,
    create_optimization_impact_visualization
)
from dashboard.recommendation import OptimizationRecommender

logger = logging.getLogger(__name__)

class Dashboard:
    """ML Inference Optimization Dashboard Application"""
    
    def __init__(self, results_dir: str = "results", host: str = "0.0.0.0", port: int = 8050):
        """
        Initialize the dashboard application
        
        Args:
            results_dir: Directory to store and read benchmark results
            host: Host to run the server on
            port: Port to run the server on
        """
        self.results_dir = results_dir
        self.host = host
        self.port = port
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize Flask server
        self.server = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, server=self.server, routes_pathname_prefix='/dash/')
        
        # Storage for registered models and benchmark results
        self.models = {}
        self.benchmark_results = {}
        
        # Setup routes and API endpoints
        self.setup_routes()
        
    def setup_routes(self) -> None:
        """Setup Flask routes and API endpoints"""
        
        # Main dashboard route
        @self.server.route('/')
        def index():
            return render_template('index.html')
        
        # API endpoints
        @self.server.route('/api/models', methods=['GET'])
        def get_models():
            return jsonify(self.models)
            
        @self.server.route('/api/results', methods=['GET'])
        def get_results():
            return jsonify(self.benchmark_results)
            
        @self.server.route('/api/profile/<result_id>', methods=['GET'])
        def get_profile(result_id):
            if result_id in self.benchmark_results:
                return jsonify(self.benchmark_results[result_id])
            return jsonify({"error": "Result not found"}), 404
            
        @self.server.route('/api/compare', methods=['POST'])
        def compare_results():
            data = request.json
            if not data or 'result_ids' not in data:
                return jsonify({"error": "Missing result_ids in request"}), 400
                
            result_ids = data['result_ids']
            comparison_data = self.get_comparison_data(result_ids)
            return jsonify(comparison_data)
            
        @self.server.route('/api/recommendations/<result_id>', methods=['GET'])
        def get_recommendations(result_id):
            if result_id not in self.benchmark_results:
                return jsonify({"error": "Result not found"}), 404
                
            profile_data = self.benchmark_results[result_id]
            model_info = {}
            for model_name, model_config in self.models.items():
                if model_name == profile_data.get('model_name'):
                    model_info = model_config
                    break
                    
            recommender = OptimizationRecommender(profile_data, model_info)
            recommendations = recommender.generate_recommendations()
            return jsonify(recommendations)
        
        # Specific route handlers
        self.server.route('/api/upload', methods=['POST'])(self.handle_upload_results)
        self.server.route('/profiling/<result_id>')(self.handle_profiling_view)
        self.server.route('/compare')(self.handle_comparison_view)
        self.server.route('/recommendations/<result_id>')(self.handle_optimization_recommendations)
        self.server.route('/monitoring')(self.handle_live_monitoring)

    def run(self) -> None:
        """Run the dashboard application"""
        self.app.run_server(host=self.host, port=self.port, debug=True)
        
    def register_model(self, model_name: str, model_config: Dict[str, Any]) -> None:
        """
        Register a model with the dashboard
        
        Args:
            model_name: Name of the model
            model_config: Configuration of the model
        """
        self.models[model_name] = model_config
        logger.info(f"Registered model: {model_name}")
        
        # Save models to disk
        with open(os.path.join(self.results_dir, 'models.json'), 'w') as f:
            json.dump(self.models, f)
            
    def register_benchmark_result(self, result_id: str, result_data: Dict[str, Any]) -> None:
        """
        Register a benchmark result with the dashboard
        
        Args:
            result_id: Unique identifier for the result
            result_data: Benchmark result data
        """
        self.benchmark_results[result_id] = result_data
        logger.info(f"Registered benchmark result: {result_id}")
        
        # Save result to disk
        result_file = os.path.join(self.results_dir, f"{result_id}.json")
        with open(result_file, 'w') as f:
            json.dump(result_data, f)
            
    def get_comparison_data(self, result_ids: List[str]) -> Dict[str, Any]:
        """
        Get data for comparing multiple benchmark results
        
        Args:
            result_ids: List of result IDs to compare
            
        Returns:
            Dictionary with comparison data
        """
        comparison_data = {
            "results": {},
            "summary": {}
        }
        
        # Get data for each result
        for result_id in result_ids:
            if result_id in self.benchmark_results:
                comparison_data["results"][result_id] = self.benchmark_results[result_id]
                
        # Calculate summary statistics
        if comparison_data["results"]:
            # Get baseline (first result)
            baseline_id = result_ids[0]
            baseline_data = comparison_data["results"].get(baseline_id, {})
            
            # Calculate relative improvements for each result
            for result_id, result_data in comparison_data["results"].items():
                if result_id != baseline_id:
                    # Calculate latency improvement
                    baseline_latency = baseline_data.get("avg_latency", 0)
                    result_latency = result_data.get("avg_latency", 0)
                    if baseline_latency > 0:
                        latency_improvement = (baseline_latency - result_latency) / baseline_latency * 100
                    else:
                        latency_improvement = 0
                        
                    # Calculate throughput improvement
                    baseline_throughput = baseline_data.get("throughput", 0)
                    result_throughput = result_data.get("throughput", 0)
                    if baseline_throughput > 0:
                        throughput_improvement = (result_throughput - baseline_throughput) / baseline_throughput * 100
                    else:
                        throughput_improvement = 0
                        
                    # Calculate memory usage improvement
                    baseline_memory = baseline_data.get("peak_memory", 0)
                    result_memory = result_data.get("peak_memory", 0)
                    if baseline_memory > 0:
                        memory_improvement = (baseline_memory - result_memory) / baseline_memory * 100
                    else:
                        memory_improvement = 0
                        
                    comparison_data["summary"][result_id] = {
                        "latency_improvement": latency_improvement,
                        "throughput_improvement": throughput_improvement,
                        "memory_improvement": memory_improvement
                    }
        
        return comparison_data
        
    def handle_upload_results(self) -> Response:
        """
        Handle upload of benchmark results
        
        Returns:
            Flask response
        """
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if file:
            try:
                # Read and parse JSON file
                result_data = json.load(file)
                
                # Validate result data (basic validation)
                if 'result_id' not in result_data:
                    return jsonify({"error": "Missing result_id in result data"}), 400
                    
                # Register result
                result_id = result_data['result_id']
                self.register_benchmark_result(result_id, result_data)
                
                return jsonify({"success": True, "result_id": result_id})
            except Exception as e:
                return jsonify({"error": f"Error processing file: {str(e)}"}), 500
                
        return jsonify({"error": "Unknown error"}), 500
        
    def handle_profiling_view(self, result_id: str) -> Response:
        """
        Handle profiling view for a specific result
        
        Args:
            result_id: ID of the result to view
            
        Returns:
            Flask response
        """
        if result_id not in self.benchmark_results:
            return render_template('error.html', error="Result not found"), 404
            
        profile_data = self.benchmark_results[result_id]
        return render_template('profiling.html', result_id=result_id, profile_data=profile_data)
        
    def handle_comparison_view(self) -> Response:
        """
        Handle comparison view for multiple results
        
        Returns:
            Flask response
        """
        return render_template('compare.html')
        
    def handle_optimization_recommendations(self, result_id: str) -> Response:
        """
        Handle optimization recommendations for a specific result
        
        Args:
            result_id: ID of the result to generate recommendations for
            
        Returns:
            Flask response
        """
        if result_id not in self.benchmark_results:
            return render_template('error.html', error="Result not found"), 404
            
        profile_data = self.benchmark_results[result_id]
        model_info = {}
        for model_name, model_config in self.models.items():
            if model_name == profile_data.get('model_name'):
                model_info = model_config
                break
                
        recommender = OptimizationRecommender(profile_data, model_info)
        recommendations = recommender.generate_recommendations()
        
        return render_template('recommendations.html', 
                               result_id=result_id, 
                               recommendations=recommendations)
        
    def handle_live_monitoring(self) -> Response:
        """
        Handle live monitoring view
        
        Returns:
            Flask response
        """
        return render_template('monitoring.html')


def create_dashboard(results_dir: str = "results", host: str = "0.0.0.0", port: int = 8050) -> Dashboard:
    """
    Create and initialize the dashboard application
    
    Args:
        results_dir: Directory to store and read benchmark results
        host: Host to run the server on
        port: Port to run the server on
        
    Returns:
        Initialized Dashboard instance
    """
    dashboard = Dashboard(results_dir=results_dir, host=host, port=port)
    
    # Load existing models and results
    try:
        models_file = os.path.join(results_dir, 'models.json')
        if os.path.exists(models_file):
            with open(models_file, 'r') as f:
                dashboard.models = json.load(f)
                
        # Load result files
        for filename in os.listdir(results_dir):
            if filename.endswith('.json') and filename != 'models.json':
                result_id = os.path.splitext(filename)[0]
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result_data = json.load(f)
                    dashboard.benchmark_results[result_id] = result_data
    except Exception as e:
        logger.error(f"Error loading existing data: {str(e)}")
        
    return dashboard