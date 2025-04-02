"""
Project initialization and management utilities.
"""

import os
import sys
import logging
import torch
from typing import Dict, Any

logger = logging.getLogger(__name__)

def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for diagnostics.
    
    Returns:
        Dictionary of system information
    """
    info = {
        "python_version": sys.version,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = torch.cuda.get_device_capability(0)
        info["vram_total"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
        
    return info

def check_gpu_compatibility() -> bool:
    """
    Check if the GPU is compatible with the model requirements.
    
    Returns:
        True if compatible, False otherwise
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Model will run on CPU, which will be very slow.")
        return False
    
    # Check VRAM
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info(f"GPU has {vram_gb:.2f} GB of VRAM")
    
    if vram_gb < 8:
        logger.warning("Less than 8GB of VRAM available. Model may not fit in memory.")
        return False
    
    return True

def print_startup_message() -> None:
    """Print a startup message with system information."""
    info = get_system_info()
    
    print("\n" + "=" * 50)
    print("   Multimodal LLM Service")
    print("=" * 50)
    
    print("\nSystem Information:")
    print(f"- Python: {info['python_version'].split()[0]}")
    
    if info["cuda_available"]:
        print(f"- CUDA: {info['cuda_version']}")
        print(f"- GPU: {info['device_name']}")
        print(f"- VRAM: {info['vram_total']:.2f} GB")
    else:
        print("- CUDA: Not available")
        
    print("\nStarting service...")
    print("=" * 50 + "\n") 