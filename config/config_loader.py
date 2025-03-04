import os
import yaml
from typing import Dict, Optional, Union, Any
import logging
from pathlib import Path

from .config_schema import OptimizerConfig

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> OptimizerConfig:
    """
    Load configuration from a YAML file and return a validated OptimizerConfig object.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses the default config.
    
    Returns:
        OptimizerConfig object with validated configuration
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config doesn't match schema
    """
    if config_path is None:
        # Use default config if none specified
        config_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(config_dir, "default_config.yaml")
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Validate against schema and return config object
    return OptimizerConfig(**config_dict)


def save_config(config: OptimizerConfig, output_path: str) -> None:
    """
    Save a configuration to a YAML file.
    
    Args:
        config: OptimizerConfig object with configuration
        output_path: Path to save the YAML configuration file
    """
    # Convert to dict, filtering out default values
    config_dict = config.dict()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Configuration saved to {output_path}")


def merge_configs(base_config: OptimizerConfig, override_config: Dict[str, Any]) -> OptimizerConfig:
    """
    Merge a base configuration with override values.
    
    Args:
        base_config: Base OptimizerConfig object
        override_config: Dictionary with override values
    
    Returns:
        New OptimizerConfig with merged values
    """
    # Convert to dict, apply overrides, and convert back to config object
    config_dict = base_config.dict()
    
    # Recursively update the config
    def update_nested_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    updated_dict = update_nested_dict(config_dict, override_config)
    return OptimizerConfig(**updated_dict)


def get_optimized_config(model_name: str, hardware_config: Optional[Dict[str, Any]] = None) -> OptimizerConfig:
    """
    Generate an optimized configuration for a specific model and hardware.
    
    Args:
        model_name: Name or path of the model
        hardware_config: Optional hardware configuration
    
    Returns:
        OptimizerConfig with optimized settings
    """
    # Load default config
    config = load_config()
    
    # Set model name
    config.model.model_name_or_path = model_name
    
    # Apply hardware config if provided
    if hardware_config:
        for key, value in hardware_config.items():
            if hasattr(config.hardware, key):
                setattr(config.hardware, key, value)
    
    # Auto-detect GPU count and type if not specified
    if config.hardware.gpu_count == 0:
        # This would use a library like torch.cuda to detect GPUs
        # For now, default to 1 if detection fails
        config.hardware.gpu_count = 1
    
    # Set optimized defaults based on model and hardware
    if config.hardware.gpu_count > 1:
        # Enable tensor parallelism for multi-GPU setups
        config.parallelism.tensor_parallel_size = min(config.hardware.gpu_count, 8)
        
        # Enable optimized kernels for better performance
        config.kernels.use_flash_attention = True
        
        if config.hardware.gpu_count >= 4:
            # Enable sequence parallelism for 4+ GPUs
            config.parallelism.sequence_parallel = True
    
    # Use mixed precision by default for better performance
    config.model.precision = "bf16"
    
    return config