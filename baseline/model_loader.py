"""
Module for loading ML models from various sources.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, Type, List, Callable

import torch
import torch.nn as nn


class BaseModelLoader(ABC):
    """Abstract base class for model loaders."""
    
    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> nn.Module:
        """
        Load a model by name with optional parameters.
        
        Args:
            model_name: Name or path of the model to load
            **kwargs: Additional model-specific parameters
            
        Returns:
            Loaded PyTorch model
        """
        pass
    
    @abstractmethod
    def get_sample_input(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Generate sample input for the model.
        
        Args:
            batch_size: Batch size for the sample input
            seq_len: Sequence length for the sample input
            
        Returns:
            Tensor of sample input
        """
        pass
    
    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the loaded model.
        
        Returns:
            Dictionary of model configuration
        """
        pass


class HuggingFaceModelLoader(BaseModelLoader):
    """Model loader for Hugging Face transformers models."""
    
    def __init__(self, device: str = "cuda", dtype: Optional[torch.dtype] = None):
        """
        Initialize the Hugging Face model loader.
        
        Args:
            device: Device to load the model on ('cuda', 'cpu')
            dtype: Data type for model parameters (e.g., torch.float16)
        """
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self.config = None
    
    def load_model(self, model_name: str, **kwargs) -> nn.Module:
        """
        Load a Hugging Face model.
        
        Args:
            model_name: Model name or path from Hugging Face
            **kwargs: Additional parameters to pass to from_pretrained()
            
        Returns:
            Loaded PyTorch model
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        except ImportError:
            raise ImportError("transformers package is required for HuggingFaceModelLoader")
        
        torch_dtype = kwargs.pop("torch_dtype", self.dtype)
        
        # Load configuration
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        load_kwargs = {
            "device_map": "auto" if self.device == "cuda" else None,
            "torch_dtype": torch_dtype,
            **kwargs
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=self.config,
            **load_kwargs
        )
        
        # Move to device if needed
        if self.device != "cuda" or "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)
        
        return self.model
    
    def get_sample_input(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Generate sample input for the model.
        
        Args:
            batch_size: Batch size for the sample input
            seq_len: Sequence length for the sample input
            
        Returns:
            Dict of input_ids and attention_mask tensors
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model first.")
        
        # Generate random token IDs for the sample input
        vocab_size = self.tokenizer.vocab_size
        input_ids = torch.randint(
            low=0, high=vocab_size, size=(batch_size, seq_len), 
            device=self.device
        )
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the loaded model.
        
        Returns:
            Dictionary of model configuration
        """
        if self.config is None:
            raise ValueError("Config not loaded. Call load_model first.")
        
        return self.config.to_dict()


class DiffusersModelLoader(BaseModelLoader):
    """Model loader for Stable Diffusion and other diffusion models."""
    
    def __init__(self, device: str = "cuda", dtype: Optional[torch.dtype] = None):
        """
        Initialize the diffusers model loader.
        
        Args:
            device: Device to load the model on ('cuda', 'cpu')
            dtype: Data type for model parameters (e.g., torch.float16)
        """
        self.device = device
        self.dtype = dtype
        self.pipeline = None
        self.model_config = {}
    
    def load_model(self, model_name: str, **kwargs) -> nn.Module:
        """
        Load a diffusion model.
        
        Args:
            model_name: Model name or path
            **kwargs: Additional parameters to pass to from_pretrained()
            
        Returns:
            Loaded diffusion pipeline
        """
        try:
            from diffusers import StableDiffusionPipeline, DiffusionPipeline
        except ImportError:
            raise ImportError("diffusers package is required for DiffusersModelLoader")
        
        # Determine pipeline type
        pipeline_cls = kwargs.pop("pipeline_cls", StableDiffusionPipeline)
        if isinstance(pipeline_cls, str):
            from diffusers import pipelines
            if hasattr(pipelines, pipeline_cls):
                pipeline_cls = getattr(pipelines, pipeline_cls)
            else:
                pipeline_cls = StableDiffusionPipeline
        
        torch_dtype = kwargs.pop("torch_dtype", self.dtype)
        
        # Load pipeline
        self.pipeline = pipeline_cls.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            **kwargs
        )
        
        # Move to device if needed
        if self.device != "cuda" or "device_map" not in kwargs:
            self.pipeline = self.pipeline.to(self.device)
        
        # Extract model configuration
        self.model_config = {
            "model_type": pipeline_cls.__name__,
            "unet_config": self.pipeline.unet.config if hasattr(self.pipeline, "unet") else {},
            "scheduler_config": self.pipeline.scheduler.config if hasattr(self.pipeline, "scheduler") else {},
            "text_encoder_config": self.pipeline.text_encoder.config if hasattr(self.pipeline, "text_encoder") else {}
        }
        
        return self.pipeline
    
    def get_sample_input(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Generate sample input for the model.
        
        Args:
            batch_size: Number of prompts to generate
            seq_len: Not used for diffusion models, kept for API consistency
            
        Returns:
            Dictionary with prompt and other parameters
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not loaded. Call load_model first.")
        
        # For diffusion models, a typical input is a text prompt
        return {
            "prompt": ["Sample prompt"] * batch_size,
            "num_inference_steps": 20,
            "guidance_scale": 7.5
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the loaded model.
        
        Returns:
            Dictionary of model configuration
        """
        if not self.model_config:
            raise ValueError("Model not loaded. Call load_model first.")
        
        return self.model_config


class TorchModelLoader(BaseModelLoader):
    """Model loader for local PyTorch models."""
    
    def __init__(self, device: str = "cuda", dtype: Optional[torch.dtype] = None):
        """
        Initialize the PyTorch model loader.
        
        Args:
            device: Device to load the model on ('cuda', 'cpu')
            dtype: Data type for model parameters (e.g., torch.float16)
        """
        self.device = device
        self.dtype = dtype
        self.model = None
        self.input_shape = None
        self.model_config = {}
    
    def load_model(self, model_name: str, **kwargs) -> nn.Module:
        """
        Load a PyTorch model from a checkpoint.
        
        Args:
            model_name: Path to the model checkpoint
            **kwargs: Additional parameters including:
                - model_cls: Model class to instantiate
                - model_args: Arguments to pass to the model constructor
                - input_shape: Shape of the input tensor (excluding batch dimension)
                
        Returns:
            Loaded PyTorch model
        """
        model_cls = kwargs.pop("model_cls", None)
        model_args = kwargs.pop("model_args", {})
        self.input_shape = kwargs.pop("input_shape", None)
        
        if model_cls is None:
            raise ValueError("model_cls must be provided for TorchModelLoader")
        
        if self.input_shape is None:
            raise ValueError("input_shape must be provided for TorchModelLoader")
        
        # Instantiate the model
        self.model = model_cls(**model_args)
        
        # Load weights if the path exists
        if os.path.exists(model_name):
            state_dict = torch.load(model_name, map_location="cpu")
            # Handle different checkpoint formats
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict)
        
        # Convert to specified dtype
        if self.dtype is not None:
            self.model = self.model.to(dtype=self.dtype)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Store model configuration
        self.model_config = {
            "model_type": model_cls.__name__,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "input_shape": self.input_shape,
            **model_args
        }
        
        return self.model
    
    def get_sample_input(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Generate sample input for the model.
        
        Args:
            batch_size: Batch size for the sample input
            seq_len: Used only if the model expects sequence input
            
        Returns:
            Tensor of sample input
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        if self.input_shape is None:
            raise ValueError("input_shape not specified")
        
        # Adjust input shape based on seq_len if needed
        shape = list(self.input_shape)
        if len(shape) >= 2 and seq_len > 0:
            shape[0] = seq_len  # Assume first dim is sequence length
        
        # Create random input tensor
        sample_input = torch.randn(batch_size, *shape, device=self.device)
        
        # Convert to expected dtype
        if self.dtype is not None:
            sample_input = sample_input.to(dtype=self.dtype)
        
        return sample_input
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the loaded model.
        
        Returns:
            Dictionary of model configuration
        """
        if not self.model_config:
            raise ValueError("Model not loaded. Call load_model first.")
        
        return self.model_config


class ModelRegistry:
    """
    Registry system that maps model names to appropriate loaders.
    Determines the correct loader to use based on model name patterns or explicit mapping.
    """
    
    def __init__(self):
        """Initialize the model registry."""
        self.loaders: Dict[str, Type[BaseModelLoader]] = {}
        self.patterns: List[Tuple[str, Type[BaseModelLoader]]] = []
        self.default_loader = None
        
        # Register built-in loaders
        self.register_loader("huggingface", HuggingFaceModelLoader)
        self.register_loader("diffusers", DiffusersModelLoader)
        self.register_loader("torch", TorchModelLoader)
        
        # Register default patterns
        self.register_pattern(r"^(facebook|meta|google|bigscience|EleutherAI|mistral|microsoft|anthropic)/.*$", 
                             HuggingFaceModelLoader)
        self.register_pattern(r".*stable-diffusion.*", DiffusersModelLoader)
        self.register_pattern(r".*diffusion.*", DiffusersModelLoader)
        self.register_pattern(r".*\.pt$|.*\.pth$", TorchModelLoader)
        
        # Set default loader
        self.set_default_loader(HuggingFaceModelLoader)
    
    def register_loader(self, name: str, loader_cls: Type[BaseModelLoader]) -> None:
        """
        Register a loader class with a name.
        
        Args:
            name: Name to register the loader with
            loader_cls: Loader class to register
        """
        self.loaders[name] = loader_cls
    
    def register_pattern(self, pattern: str, loader_cls: Type[BaseModelLoader]) -> None:
        """
        Register a regex pattern to match model names with a loader.
        
        Args:
            pattern: Regex pattern to match model names
            loader_cls: Loader class to use for matching models
        """
        self.patterns.append((pattern, loader_cls))
    
    def set_default_loader(self, loader_cls: Type[BaseModelLoader]) -> None:
        """
        Set the default loader to use when no match is found.
        
        Args:
            loader_cls: Default loader class
        """
        self.default_loader = loader_cls
    
    def get_loader_for_model(self, model_name: str, loader_name: Optional[str] = None, 
                            **kwargs) -> BaseModelLoader:
        """
        Get the appropriate loader instance for a model.
        
        Args:
            model_name: Name or path of the model
            loader_name: Optional explicit loader name to use
            **kwargs: Additional parameters to pass to the loader constructor
            
        Returns:
            Loader instance for the model
        """
        # If explicit loader name is provided, use that
        if loader_name and loader_name in self.loaders:
            return self.loaders[loader_name](**kwargs)
        
        # Otherwise try to match using patterns
        for pattern, loader_cls in self.patterns:
            if re.match(pattern, model_name, re.IGNORECASE):
                return loader_cls(**kwargs)
        
        # Fall back to default loader
        if self.default_loader:
            return self.default_loader(**kwargs)
        
        raise ValueError(f"Could not determine appropriate loader for model: {model_name}")
    
    def list_registered_loaders(self) -> Dict[str, Type[BaseModelLoader]]:
        """
        Get all registered loaders.
        
        Returns:
            Dictionary of registered loader names and classes
        """
        return self.loaders.copy()


# Global registry instance
model_registry = ModelRegistry()


def load_model(model_name: str, loader_name: Optional[str] = None, 
               device: str = "cuda", dtype: Optional[torch.dtype] = None, 
               **kwargs) -> Tuple[nn.Module, BaseModelLoader]:
    """
    Convenience function to load a model using the registry.
    
    Args:
        model_name: Name or path of the model
        loader_name: Optional explicit loader name to use
        device: Device to load the model on ('cuda', 'cpu')
        dtype: Data type for model parameters
        **kwargs: Additional parameters to pass to the loader
        
    Returns:
        Tuple of (loaded model, loader instance)
    """
    loader = model_registry.get_loader_for_model(
        model_name, 
        loader_name=loader_name,
        device=device,
        dtype=dtype
    )
    model = loader.load_model(model_name, **kwargs)
    return model, loader


def register_custom_loader(name: str, loader_cls: Type[BaseModelLoader]) -> None:
    """
    Register a custom loader with the global registry.
    
    Args:
        name: Name to register the loader with
        loader_cls: Loader class to register
    """
    model_registry.register_loader(name, loader_cls)


def register_custom_pattern(pattern: str, loader_cls: Type[BaseModelLoader]) -> None:
    """
    Register a custom pattern with the global registry.
    
    Args:
        pattern: Regex pattern to match model names
        loader_cls: Loader class to use for matching models
    """
    model_registry.register_pattern(pattern, loader_cls)