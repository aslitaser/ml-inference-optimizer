"""
ML Inference Optimization Recommendation System

This module implements a recommendation system that analyzes profiling data
to identify bottlenecks and suggest optimization strategies for improving
ML inference performance.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dash import html
import dash_bootstrap_components as dbc


class OptimizationRecommender:
    """Base class for optimization recommendation"""
    
    def __init__(self, profile_data: Dict[str, Any], model_info: Dict[str, Any]):
        """
        Initialize the optimization recommender
        
        Args:
            profile_data: Profiling data dictionary
            model_info: Model information dictionary
        """
        self.profile_data = profile_data
        self.model_info = model_info
        self.recommendations = []
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate optimization recommendations
        
        Returns:
            List of recommendation dictionaries
        """
        # Identify optimization opportunities
        opportunities = self.identify_optimization_opportunities()
        
        # Estimate improvement potential for each opportunity
        for opportunity in opportunities:
            improvement = self.estimate_improvement_potential(opportunity['type'])
            opportunity.update(improvement)
        
        # Prioritize recommendations
        self.recommendations = self.prioritize_recommendations()
        
        return self.recommendations
    
    def identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """
        Identify optimization opportunities based on profiling data
        
        Returns:
            List of optimization opportunities
        """
        opportunities = []
        
        # Create specific recommenders
        recommenders = [
            ParallelismRecommender(self.profile_data, self.model_info),
            KernelOptimizationRecommender(self.profile_data, self.model_info),
            MemoryOptimizationRecommender(self.profile_data, self.model_info),
            OperationFusionRecommender(self.profile_data, self.model_info)
        ]
        
        # Get recommendations from each recommender
        for recommender in recommenders:
            recommender_opportunities = recommender.identify_optimization_opportunities()
            opportunities.extend(recommender_opportunities)
        
        return opportunities
    
    def estimate_improvement_potential(self, optimization_type: str) -> Dict[str, float]:
        """
        Estimate potential improvement for a given optimization
        
        Args:
            optimization_type: Type of optimization
            
        Returns:
            Dictionary with estimated improvements for latency, throughput, and memory
        """
        # Default improvement estimates
        improvement = {
            'latency_improvement': 0.0,
            'throughput_improvement': 0.0,
            'memory_improvement': 0.0,
            'confidence': 0.5  # Default confidence
        }
        
        # Let specific recommenders estimate improvements
        if optimization_type.startswith('parallel_'):
            recommender = ParallelismRecommender(self.profile_data, self.model_info)
            improvement = recommender.estimate_improvement_potential(optimization_type)
        elif optimization_type.startswith('kernel_'):
            recommender = KernelOptimizationRecommender(self.profile_data, self.model_info)
            improvement = recommender.estimate_improvement_potential(optimization_type)
        elif optimization_type.startswith('memory_'):
            recommender = MemoryOptimizationRecommender(self.profile_data, self.model_info)
            improvement = recommender.estimate_improvement_potential(optimization_type)
        elif optimization_type.startswith('fusion_'):
            recommender = OperationFusionRecommender(self.profile_data, self.model_info)
            improvement = recommender.estimate_improvement_potential(optimization_type)
        
        return improvement
    
    def prioritize_recommendations(self) -> List[Dict[str, Any]]:
        """
        Prioritize recommendations based on improvement potential and confidence
        
        Returns:
            Prioritized list of recommendations
        """
        # Score each recommendation
        for rec in self.recommendations:
            # Consider latency, throughput, and memory improvements
            latency_score = rec.get('latency_improvement', 0) * 0.4
            throughput_score = rec.get('throughput_improvement', 0) * 0.4
            memory_score = rec.get('memory_improvement', 0) * 0.2
            
            # Weight by confidence
            confidence = rec.get('confidence', 0.5)
            
            # Calculate total score
            total_score = (latency_score + throughput_score + memory_score) * confidence
            rec['priority_score'] = total_score
        
        # Sort by priority score
        sorted_recommendations = sorted(
            self.recommendations, 
            key=lambda x: x.get('priority_score', 0), 
            reverse=True
        )
        
        return sorted_recommendations
    
    def generate_recommendation_report(self) -> html.Div:
        """
        Generate a recommendation report as a Dash HTML component
        
        Returns:
            Dash HTML Div with recommendation report
        """
        if not self.recommendations:
            self.generate_recommendations()
        
        # Create recommendation cards
        recommendation_cards = []
        for i, rec in enumerate(self.recommendations[:5]):  # Show top 5 recommendations
            card = dbc.Card([
                dbc.CardHeader([
                    html.H5(rec.get('title', 'Optimization Recommendation'), className="card-title"),
                    html.H6(f"Type: {rec.get('type', 'Unknown')}", className="card-subtitle text-muted")
                ]),
                dbc.CardBody([
                    html.P(rec.get('description', '')),
                    html.Div([
                        html.Span("Estimated Improvements:", className="font-weight-bold"),
                        html.Ul([
                            html.Li(f"Latency: {rec.get('latency_improvement', 0):.1f}%"),
                            html.Li(f"Throughput: {rec.get('throughput_improvement', 0):.1f}%"),
                            html.Li(f"Memory: {rec.get('memory_improvement', 0):.1f}%")
                        ])
                    ]),
                    html.P(f"Confidence: {rec.get('confidence', 0.5):.2f}", className="text-muted")
                ]),
                dbc.CardFooter([
                    html.H6("Implementation:", className="font-weight-bold"),
                    html.P(rec.get('implementation', ''))
                ])
            ], className="mb-3")
            
            recommendation_cards.append(card)
        
        # Create overall summary
        if self.recommendations:
            avg_latency = np.mean([rec.get('latency_improvement', 0) for rec in self.recommendations[:3]])
            avg_throughput = np.mean([rec.get('throughput_improvement', 0) for rec in self.recommendations[:3]])
            avg_memory = np.mean([rec.get('memory_improvement', 0) for rec in self.recommendations[:3]])
            
            summary = dbc.Alert([
                html.H4("Optimization Potential Summary"),
                html.P("Implementing the top recommendations could result in:"),
                html.Ul([
                    html.Li(f"Up to {avg_latency:.1f}% reduction in latency"),
                    html.Li(f"Up to {avg_throughput:.1f}% improvement in throughput"),
                    html.Li(f"Up to {avg_memory:.1f}% reduction in memory usage")
                ])
            ], color="info")
        else:
            summary = dbc.Alert("No optimization recommendations available.", color="warning")
        
        # Create the recommendation report
        return html.Div([
            summary,
            html.H4("Top Recommendations"),
            html.Div(recommendation_cards)
        ])


class ParallelismRecommender(OptimizationRecommender):
    """Recommends optimal parallelism strategies"""
    
    def identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """
        Identify parallelism optimization opportunities
        
        Returns:
            List of parallelism optimization opportunities
        """
        opportunities = []
        
        # Get model and hardware information
        model_size = self.model_info.get('model_size', 0)  # Size in millions of parameters
        num_gpus = self.model_info.get('num_gpus', 1)
        gpu_memory = self.model_info.get('gpu_memory', 0)  # GPU memory in GB
        
        # Check for tensor parallelism opportunity
        if model_size > 1000 and num_gpus > 1:
            opportunities.append({
                'type': 'parallel_tensor',
                'title': 'Tensor Parallelism',
                'description': 'Distribute tensor computations across multiple GPUs to reduce memory per device and enable parallel computation.',
                'implementation': 'Implement tensor parallelism using the parallelism.tensor_parallel module. Split attention heads and MLP layers across GPUs.'
            })
        
        # Check for sequence parallelism opportunity
        if model_size > 5000 and num_gpus > 1:
            opportunities.append({
                'type': 'parallel_sequence',
                'title': 'Sequence Parallelism',
                'description': 'Partition sequence/context length across multiple GPUs to process different parts of the sequence in parallel.',
                'implementation': 'Implement sequence parallelism using the parallelism.sequence_parallel module. Split attention computation along the sequence dimension.'
            })
        
        # Check for data parallelism opportunity (for batch processing)
        batch_size = self.profile_data.get('batch_size', 1)
        if batch_size > 1 and num_gpus > 1:
            opportunities.append({
                'type': 'parallel_data',
                'title': 'Data Parallelism',
                'description': 'Process different batches on different GPUs to increase throughput for batch processing workloads.',
                'implementation': 'Implement data parallelism to process multiple batches in parallel and synchronize gradients if training.'
            })
        
        # Check for pipeline parallelism
        if model_size > 10000 and num_gpus > 2:
            opportunities.append({
                'type': 'parallel_pipeline',
                'title': 'Pipeline Parallelism',
                'description': 'Split the model across multiple GPUs and process different microbatches in a pipelined fashion.',
                'implementation': 'Implement pipeline parallelism by assigning different transformer layers to different GPUs and coordinating the forward pass with the parallelism.orchestrator module.'
            })
        
        return opportunities
    
    def estimate_improvement_potential(self, optimization_type: str) -> Dict[str, float]:
        """
        Estimate potential improvement for parallelism optimizations
        
        Args:
            optimization_type: Type of parallelism optimization
            
        Returns:
            Dictionary with estimated improvements
        """
        # Default improvement estimates
        improvement = {
            'latency_improvement': 0.0,
            'throughput_improvement': 0.0,
            'memory_improvement': 0.0,
            'confidence': 0.6
        }
        
        num_gpus = self.model_info.get('num_gpus', 1)
        
        if optimization_type == 'parallel_tensor':
            # Tensor parallelism typically reduces memory usage and can marginally improve latency
            improvement['latency_improvement'] = 5.0 + (num_gpus - 1) * 3.0  # Modest latency improvement
            improvement['throughput_improvement'] = 10.0 + (num_gpus - 1) * 5.0
            improvement['memory_improvement'] = 30.0 + (num_gpus - 1) * 10.0  # Significant memory improvement
            improvement['confidence'] = 0.8
            
        elif optimization_type == 'parallel_sequence':
            # Sequence parallelism can significantly improve latency for long sequences
            seq_length = self.profile_data.get('sequence_length', 512)
            if seq_length > 1024:
                improvement['latency_improvement'] = 20.0 + (num_gpus - 1) * 10.0
                improvement['throughput_improvement'] = 25.0 + (num_gpus - 1) * 10.0
                improvement['memory_improvement'] = 15.0 + (num_gpus - 1) * 5.0
                improvement['confidence'] = 0.7
            else:
                # Less effective for shorter sequences
                improvement['latency_improvement'] = 5.0
                improvement['throughput_improvement'] = 10.0
                improvement['memory_improvement'] = 5.0
                improvement['confidence'] = 0.6
                
        elif optimization_type == 'parallel_data':
            # Data parallelism significantly improves throughput but not latency
            improvement['latency_improvement'] = 0.0  # No latency improvement
            improvement['throughput_improvement'] = 80.0 + (num_gpus - 1) * 10.0  # Nearly linear scaling
            improvement['memory_improvement'] = 0.0  # No memory improvement per device
            improvement['confidence'] = 0.9
            
        elif optimization_type == 'parallel_pipeline':
            # Pipeline parallelism can improve both throughput and memory usage
            improvement['latency_improvement'] = 0.0  # Might even increase latency slightly
            improvement['throughput_improvement'] = 50.0 + (num_gpus - 2) * 15.0
            improvement['memory_improvement'] = 40.0 + (num_gpus - 2) * 10.0
            improvement['confidence'] = 0.7
            
        return improvement


class KernelOptimizationRecommender(OptimizationRecommender):
    """Recommends kernel optimizations"""
    
    def identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """
        Identify kernel optimization opportunities
        
        Returns:
            List of kernel optimization opportunities
        """
        opportunities = []
        
        # Get kernel information
        kernels = self.profile_data.get('kernels', [])
        
        # Check for attention optimization opportunities
        attention_kernels = [k for k in kernels if 'attention' in k.get('name', '').lower()]
        if attention_kernels:
            # Check if flash attention can be applied
            can_use_flash = True  # Simplified check, in practice would check hardware compatibility
            if can_use_flash:
                opportunities.append({
                    'type': 'kernel_flash_attention',
                    'title': 'Flash Attention',
                    'description': 'Optimize attention computation using flash attention algorithm to reduce memory usage and improve speed.',
                    'implementation': 'Replace standard attention implementation with flash attention using kernels.attention.flash_attention module.'
                })
            
            # Check if ring attention can be applied
            if self.model_info.get('num_gpus', 1) > 1:
                opportunities.append({
                    'type': 'kernel_ring_attention',
                    'title': 'Ring Attention',
                    'description': 'Use ring attention to optimize attention computation across multiple GPUs.',
                    'implementation': 'Implement ring attention using kernels.attention.ring_attention module for multi-GPU setups.'
                })
        
        # Check for MLP optimization opportunities
        mlp_kernels = [k for k in kernels if 'mlp' in k.get('name', '').lower() or 'feedforward' in k.get('name', '').lower()]
        if mlp_kernels:
            opportunities.append({
                'type': 'kernel_fused_mlp',
                'title': 'Fused MLP Operations',
                'description': 'Fuse multiple MLP operations into a single kernel to reduce memory traffic and kernel launch overhead.',
                'implementation': 'Replace separate MLP operations with fused implementation from kernels.mlp.fused_mlp module.'
            })
        
        # Check for layernorm optimization
        layernorm_kernels = [k for k in kernels if 'layernorm' in k.get('name', '').lower() or 'layer_norm' in k.get('name', '').lower()]
        if layernorm_kernels:
            opportunities.append({
                'type': 'kernel_fused_layernorm',
                'title': 'Fused LayerNorm',
                'description': 'Use fused LayerNorm implementation to reduce memory bandwidth requirements.',
                'implementation': 'Replace standard LayerNorm with optimized implementation from kernels.triton.layernorm_kernels module.'
            })
        
        # Check if Triton kernels can be applied
        if 'triton' not in ' '.join([k.get('name', '') for k in kernels]).lower():
            opportunities.append({
                'type': 'kernel_triton',
                'title': 'Triton Kernel Optimization',
                'description': 'Use Triton to generate optimized CUDA kernels for common operations.',
                'implementation': 'Replace standard operations with Triton-optimized kernels from the kernels.triton module.'
            })
        
        return opportunities
    
    def estimate_improvement_potential(self, optimization_type: str) -> Dict[str, float]:
        """
        Estimate potential improvement for kernel optimizations
        
        Args:
            optimization_type: Type of kernel optimization
            
        Returns:
            Dictionary with estimated improvements
        """
        # Default improvement estimates
        improvement = {
            'latency_improvement': 0.0,
            'throughput_improvement': 0.0,
            'memory_improvement': 0.0,
            'confidence': 0.7
        }
        
        # Get operation timing information
        operations = self.profile_data.get('operations', [])
        
        if optimization_type == 'kernel_flash_attention':
            # Estimate improvement from flash attention
            attention_ops = [op for op in operations if 'attention' in op.get('name', '').lower()]
            if attention_ops:
                # Calculate attention time percentage
                attention_time = sum(op.get('end_time', 0) - op.get('start_time', 0) for op in attention_ops)
                total_time = self.profile_data.get('total_time', 1.0)
                attention_pct = attention_time / total_time * 100 if total_time > 0 else 0
                
                # Flash attention typically gives 2-3x speedup for attention
                improvement['latency_improvement'] = attention_pct * 0.5  # 50% improvement on attention part
                improvement['throughput_improvement'] = attention_pct * 0.5
                improvement['memory_improvement'] = 20.0  # Significant memory reduction
                improvement['confidence'] = 0.8
                
        elif optimization_type == 'kernel_ring_attention':
            # Ring attention mainly improves multi-GPU scenarios
            if self.model_info.get('num_gpus', 1) > 1:
                improvement['latency_improvement'] = 10.0
                improvement['throughput_improvement'] = 15.0
                improvement['memory_improvement'] = 5.0
                improvement['confidence'] = 0.6
                
        elif optimization_type == 'kernel_fused_mlp':
            # Estimate improvement from fused MLP operations
            mlp_ops = [op for op in operations if 'mlp' in op.get('name', '').lower() or 'feedforward' in op.get('name', '').lower()]
            if mlp_ops:
                # Calculate MLP time percentage
                mlp_time = sum(op.get('end_time', 0) - op.get('start_time', 0) for op in mlp_ops)
                total_time = self.profile_data.get('total_time', 1.0)
                mlp_pct = mlp_time / total_time * 100 if total_time > 0 else 0
                
                # Fused MLP can give 20-30% speedup for MLP operations
                improvement['latency_improvement'] = mlp_pct * 0.25  # 25% improvement on MLP part
                improvement['throughput_improvement'] = mlp_pct * 0.25
                improvement['memory_improvement'] = 10.0
                improvement['confidence'] = 0.7
                
        elif optimization_type == 'kernel_fused_layernorm':
            # Estimate improvement from fused LayerNorm
            layernorm_ops = [op for op in operations if 'layernorm' in op.get('name', '').lower() or 'layer_norm' in op.get('name', '').lower()]
            if layernorm_ops:
                # Calculate LayerNorm time percentage
                layernorm_time = sum(op.get('end_time', 0) - op.get('start_time', 0) for op in layernorm_ops)
                total_time = self.profile_data.get('total_time', 1.0)
                layernorm_pct = layernorm_time / total_time * 100 if total_time > 0 else 0
                
                # Fused LayerNorm typically gives 30-40% speedup for LayerNorm operations
                improvement['latency_improvement'] = layernorm_pct * 0.35  # 35% improvement on LayerNorm part
                improvement['throughput_improvement'] = layernorm_pct * 0.35
                improvement['memory_improvement'] = 5.0
                improvement['confidence'] = 0.8
                
        elif optimization_type == 'kernel_triton':
            # Triton can provide broad improvements
            improvement['latency_improvement'] = 15.0
            improvement['throughput_improvement'] = 20.0
            improvement['memory_improvement'] = 5.0
            improvement['confidence'] = 0.6  # More variable as it depends on specific operations
            
        return improvement


class MemoryOptimizationRecommender(OptimizationRecommender):
    """Recommends memory optimizations"""
    
    def identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """
        Identify memory optimization opportunities
        
        Returns:
            List of memory optimization opportunities
        """
        opportunities = []
        
        # Get memory usage information
        memory_timeline = self.profile_data.get('memory_timeline', [])
        peak_memory = max([entry.get('memory_usage', 0) for entry in memory_timeline]) if memory_timeline else 0
        
        # Get model size and batch size
        model_size = self.model_info.get('model_size', 0)  # Size in millions of parameters
        batch_size = self.profile_data.get('batch_size', 1)
        sequence_length = self.profile_data.get('sequence_length', 512)
        
        # Check for activation checkpointing opportunity
        if model_size > 1000:
            opportunities.append({
                'type': 'memory_activation_checkpointing',
                'title': 'Activation Checkpointing',
                'description': 'Save memory by discarding intermediate activations and recomputing them during the backward pass.',
                'implementation': 'Apply activation checkpointing to transformer layers to reduce memory usage at the cost of additional computation.'
            })
        
        # Check for mixed precision opportunity
        if 'gpu' in self.profile_data.get('device', '').lower():
            opportunities.append({
                'type': 'memory_mixed_precision',
                'title': 'Mixed Precision Training/Inference',
                'description': 'Use FP16 or BF16 precision to reduce memory usage and improve performance on modern GPUs.',
                'implementation': 'Implement mixed precision using torch.cuda.amp for PyTorch or appropriate libraries for other frameworks.'
            })
        
        # Check for attention memory optimization
        if sequence_length > 1024:
            opportunities.append({
                'type': 'memory_attention_optimization',
                'title': 'Memory-Efficient Attention',
                'description': 'Use memory-efficient attention implementations to reduce the quadratic memory scaling with sequence length.',
                'implementation': 'Replace standard attention with memory-efficient variants from kernels.attention module, such as flash attention.'
            })
        
        # Check for weight quantization opportunity
        if model_size > 100:
            opportunities.append({
                'type': 'memory_weight_quantization',
                'title': 'Weight Quantization',
                'description': 'Quantize model weights to lower precision (INT8, INT4) to reduce memory footprint.',
                'implementation': 'Apply post-training quantization to model weights using appropriate quantization libraries.'
            })
        
        # Check for gradient accumulation (if training)
        if self.profile_data.get('is_training', False) and batch_size > 1:
            opportunities.append({
                'type': 'memory_gradient_accumulation',
                'title': 'Gradient Accumulation',
                'description': 'Accumulate gradients over multiple smaller batches to reduce memory usage during training.',
                'implementation': 'Implement gradient accumulation by processing smaller batches and updating weights less frequently.'
            })
        
        return opportunities
    
    def estimate_improvement_potential(self, optimization_type: str) -> Dict[str, float]:
        """
        Estimate potential improvement for memory optimizations
        
        Args:
            optimization_type: Type of memory optimization
            
        Returns:
            Dictionary with estimated improvements
        """
        # Default improvement estimates
        improvement = {
            'latency_improvement': 0.0,
            'throughput_improvement': 0.0,
            'memory_improvement': 0.0,
            'confidence': 0.7
        }
        
        if optimization_type == 'memory_activation_checkpointing':
            # Activation checkpointing trades compute for memory
            improvement['latency_improvement'] = -10.0  # Slight negative impact on latency
            improvement['throughput_improvement'] = -5.0  # Slight negative impact on throughput
            improvement['memory_improvement'] = 30.0  # Significant memory improvement
            improvement['confidence'] = 0.8
            
        elif optimization_type == 'memory_mixed_precision':
            # Mixed precision improves both memory and performance on compatible hardware
            improvement['latency_improvement'] = 20.0
            improvement['throughput_improvement'] = 30.0
            improvement['memory_improvement'] = 40.0
            improvement['confidence'] = 0.9
            
        elif optimization_type == 'memory_attention_optimization':
            # Memory-efficient attention mainly helps with long sequences
            sequence_length = self.profile_data.get('sequence_length', 512)
            if sequence_length > 1024:
                memory_benefit = min(60.0, 20.0 + sequence_length / 1024.0 * 10.0)  # More benefit for longer sequences
                improvement['latency_improvement'] = 10.0
                improvement['throughput_improvement'] = 15.0
                improvement['memory_improvement'] = memory_benefit
                improvement['confidence'] = 0.8
            else:
                improvement['latency_improvement'] = 5.0
                improvement['throughput_improvement'] = 10.0
                improvement['memory_improvement'] = 15.0
                improvement['confidence'] = 0.7
                
        elif optimization_type == 'memory_weight_quantization':
            # Weight quantization significantly reduces memory with potential performance impact
            improvement['latency_improvement'] = 5.0  # Can be faster due to smaller memory footprint
            improvement['throughput_improvement'] = 10.0
            improvement['memory_improvement'] = 50.0  # Significant memory reduction
            improvement['confidence'] = 0.7
            
        elif optimization_type == 'memory_gradient_accumulation':
            # Gradient accumulation helps with memory during training
            if self.profile_data.get('is_training', False):
                improvement['latency_improvement'] = 0.0  # No impact on per-sample latency
                improvement['throughput_improvement'] = -5.0  # Slight negative impact due to synchronization
                improvement['memory_improvement'] = 40.0  # Significant memory improvement
                improvement['confidence'] = 0.9
                
        return improvement


class OperationFusionRecommender(OptimizationRecommender):
    """Recommends operation fusion opportunities"""
    
    def identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """
        Identify operation fusion opportunities
        
        Returns:
            List of operation fusion opportunities
        """
        opportunities = []
        
        # Get operation information
        operations = self.profile_data.get('operations', [])
        
        # Check for common fusion patterns
        
        # Check for attention + dropout fusion
        has_attention = any('attention' in op.get('name', '').lower() for op in operations)
        has_dropout = any('dropout' in op.get('name', '').lower() for op in operations)
        
        if has_attention and has_dropout:
            opportunities.append({
                'type': 'fusion_attention_dropout',
                'title': 'Attention-Dropout Fusion',
                'description': 'Fuse attention and dropout operations to reduce memory traffic and kernel launches.',
                'implementation': 'Use fused attention implementation from kernels.attention module that includes dropout in the same kernel.'
            })
        
        # Check for layernorm + linear fusion
        has_layernorm = any('layernorm' in op.get('name', '').lower() or 'layer_norm' in op.get('name', '').lower() for op in operations)
        has_linear = any('linear' in op.get('name', '').lower() or 'matmul' in op.get('name', '').lower() for op in operations)
        
        if has_layernorm and has_linear:
            opportunities.append({
                'type': 'fusion_layernorm_linear',
                'title': 'LayerNorm-Linear Fusion',
                'description': 'Fuse LayerNorm and following Linear operation to reduce memory bandwidth requirements.',
                'implementation': 'Implement custom fused LayerNorm+Linear operation using Triton or CUDA kernels from kernels.triton module.'
            })
        
        # Check for linear + bias + activation fusion
        has_bias = any('bias' in op.get('name', '').lower() for op in operations)
        has_activation = any(act in ' '.join([op.get('name', '') for op in operations]).lower() 
                             for act in ['relu', 'gelu', 'swish', 'silu'])
        
        if has_linear and has_bias and has_activation:
            opportunities.append({
                'type': 'fusion_linear_bias_activation',
                'title': 'Linear-Bias-Activation Fusion',
                'description': 'Fuse Linear, Bias add, and Activation functions into a single kernel to reduce memory traffic.',
                'implementation': 'Use fused implementations from kernels.mlp.fused_mlp module or implement custom CUDA kernels.'
            })
        
        # Check for multi-head attention fusion opportunity
        has_multiple_attention_parts = sum(1 for op in operations if 'attention' in op.get('name', '').lower()) > 1
        
        if has_multiple_attention_parts:
            opportunities.append({
                'type': 'fusion_multihead_attention',
                'title': 'Multi-Head Attention Fusion',
                'description': 'Fuse all parts of multi-head attention (Q/K/V projections, attention, output projection) into a single operation.',
                'implementation': 'Replace separate attention operations with a fully fused implementation from kernels.attention module.'
            })
        
        return opportunities
    
    def estimate_improvement_potential(self, optimization_type: str) -> Dict[str, float]:
        """
        Estimate potential improvement for operation fusion
        
        Args:
            optimization_type: Type of operation fusion
            
        Returns:
            Dictionary with estimated improvements
        """
        # Default improvement estimates
        improvement = {
            'latency_improvement': 0.0,
            'throughput_improvement': 0.0,
            'memory_improvement': 0.0,
            'confidence': 0.6  # Operation fusion benefits can be more variable
        }
        
        # Get operation timing information
        operations = self.profile_data.get('operations', [])
        
        if optimization_type == 'fusion_attention_dropout':
            # Estimate improvement from attention-dropout fusion
            attention_ops = [op for op in operations if 'attention' in op.get('name', '').lower()]
            dropout_ops = [op for op in operations if 'dropout' in op.get('name', '').lower()]
            
            if attention_ops and dropout_ops:
                # Calculate combined time percentage
                attention_time = sum(op.get('end_time', 0) - op.get('start_time', 0) for op in attention_ops)
                dropout_time = sum(op.get('end_time', 0) - op.get('start_time', 0) for op in dropout_ops)
                total_time = self.profile_data.get('total_time', 1.0)
                
                combined_pct = (attention_time + dropout_time) / total_time * 100 if total_time > 0 else 0
                
                # Fusion typically saves 10-20% of the combined operation time
                improvement['latency_improvement'] = combined_pct * 0.15  # 15% improvement on the combined ops
                improvement['throughput_improvement'] = combined_pct * 0.15
                improvement['memory_improvement'] = 5.0  # Modest memory improvement
                improvement['confidence'] = 0.7
                
        elif optimization_type == 'fusion_layernorm_linear':
            # Estimate improvement from layernorm-linear fusion
            layernorm_ops = [op for op in operations if 'layernorm' in op.get('name', '').lower() or 'layer_norm' in op.get('name', '').lower()]
            linear_ops = [op for op in operations if 'linear' in op.get('name', '').lower() or 'matmul' in op.get('name', '').lower()]
            
            if layernorm_ops and linear_ops:
                # Calculate combined time percentage
                layernorm_time = sum(op.get('end_time', 0) - op.get('start_time', 0) for op in layernorm_ops)
                linear_time = sum(op.get('end_time', 0) - op.get('start_time', 0) for op in linear_ops)
                total_time = self.profile_data.get('total_time', 1.0)
                
                combined_pct = (layernorm_time + linear_time) / total_time * 100 if total_time > 0 else 0
                
                improvement['latency_improvement'] = combined_pct * 0.2  # 20% improvement on the combined ops
                improvement['throughput_improvement'] = combined_pct * 0.2
                improvement['memory_improvement'] = 8.0
                improvement['confidence'] = 0.6
                
        elif optimization_type == 'fusion_linear_bias_activation':
            # Estimate improvement from linear-bias-activation fusion
            linear_ops = [op for op in operations if 'linear' in op.get('name', '').lower() or 'matmul' in op.get('name', '').lower()]
            bias_ops = [op for op in operations if 'bias' in op.get('name', '').lower()]
            activation_ops = [op for op in operations if any(act in op.get('name', '').lower() 
                                                         for act in ['relu', 'gelu', 'swish', 'silu'])]
            
            if linear_ops and bias_ops and activation_ops:
                # Calculate combined time percentage
                linear_time = sum(op.get('end_time', 0) - op.get('start_time', 0) for op in linear_ops)
                bias_time = sum(op.get('end_time', 0) - op.get('start_time', 0) for op in bias_ops)
                activation_time = sum(op.get('end_time', 0) - op.get('start_time', 0) for op in activation_ops)
                total_time = self.profile_data.get('total_time', 1.0)
                
                combined_pct = (linear_time + bias_time + activation_time) / total_time * 100 if total_time > 0 else 0
                
                improvement['latency_improvement'] = combined_pct * 0.25  # 25% improvement on the combined ops
                improvement['throughput_improvement'] = combined_pct * 0.25
                improvement['memory_improvement'] = 10.0
                improvement['confidence'] = 0.7
                
        elif optimization_type == 'fusion_multihead_attention':
            # Estimate improvement from fully fused multi-head attention
            attention_ops = [op for op in operations if 'attention' in op.get('name', '').lower()]
            
            if len(attention_ops) > 1:
                # Calculate combined time percentage
                attention_time = sum(op.get('end_time', 0) - op.get('start_time', 0) for op in attention_ops)
                total_time = self.profile_data.get('total_time', 1.0)
                
                attention_pct = attention_time / total_time * 100 if total_time > 0 else 0
                
                improvement['latency_improvement'] = attention_pct * 0.3  # 30% improvement on attention part
                improvement['throughput_improvement'] = attention_pct * 0.3
                improvement['memory_improvement'] = 15.0
                improvement['confidence'] = 0.7
                
        return improvement