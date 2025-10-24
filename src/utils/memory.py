"""Memory management utilities for adaptive batch sizing and resource checks."""
import psutil
from typing import Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_memory_info() -> Tuple[float, float]:
    """Get current memory usage information.
    
    Returns:
        Tuple of (available_gb, total_gb)
    """
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    return available_gb, total_gb

def check_memory_sufficient(required_gb: float = 2.0) -> bool:
    """Check if sufficient memory is available.
    
    Args:
        required_gb: Minimum required memory in GB
    
    Returns:
        True if sufficient memory available
    """
    available_gb, total_gb = get_memory_info()
    sufficient = available_gb >= required_gb
    
    if not sufficient:
        logger.warning(
            f"Low memory: {available_gb:.1f}GB available of {total_gb:.1f}GB total. "
            f"Minimum recommended: {required_gb:.1f}GB"
        )
    
    return sufficient

def auto_tune_batch_size(default_batch_size: int = 16, model_size_mb: int = 500) -> int:
    """Automatically tune batch size based on available memory.
    
    Args:
        default_batch_size: Default batch size to use if memory is sufficient
        model_size_mb: Estimated model size in MB (default: 500MB for xlm-roberta-base)
    
    Returns:
        Recommended batch size
    """
    available_gb, total_gb = get_memory_info()
    available_mb = available_gb * 1024
    
    # Reserve memory: OS (1GB) + Model + Working space
    reserved_mb = 1024 + model_size_mb
    usable_mb = available_mb - reserved_mb
    
    # Estimate memory per sample (rough approximation)
    # For transformer models: ~10-50MB per sample depending on sequence length
    mb_per_sample = 30  # Conservative estimate
    
    # Calculate max safe batch size
    max_batch_size = max(1, int(usable_mb / mb_per_sample))
    
    # Choose the smaller of default or max
    tuned_batch_size = min(default_batch_size, max_batch_size)
    
    # Apply safety limits
    if tuned_batch_size < 1:
        tuned_batch_size = 1
    elif tuned_batch_size > 32:
        tuned_batch_size = 32  # Cap at reasonable maximum
    
    logger.info(
        f"Memory: {available_gb:.1f}GB available, {total_gb:.1f}GB total. "
        f"Batch size: {tuned_batch_size} (default: {default_batch_size})"
    )
    
    if tuned_batch_size < default_batch_size:
        logger.warning(
            f"Reduced batch size from {default_batch_size} to {tuned_batch_size} due to limited memory"
        )
    
    return tuned_batch_size

def get_memory_recommendations() -> dict:
    """Get memory-based recommendations for app configuration.
    
    Returns:
        Dict with recommended settings
    """
    available_gb, total_gb = get_memory_info()
    
    recommendations = {
        'batch_size': auto_tune_batch_size(),
        'chunk_size': 1000,  # Default
        'can_train': available_gb >= 4.0,
        'warning': None
    }
    
    if available_gb < 2.0:
        recommendations['batch_size'] = 2
        recommendations['chunk_size'] = 500
        recommendations['can_train'] = False
        recommendations['warning'] = (
            f"Very low memory ({available_gb:.1f}GB). "
            "Inference only, training disabled."
        )
    elif available_gb < 4.0:
        recommendations['batch_size'] = 4
        recommendations['warning'] = (
            f"Low memory ({available_gb:.1f}GB). "
            "Training may be slow or unstable."
        )
    elif available_gb < 6.0:
        recommendations['batch_size'] = 8
    else:
        recommendations['batch_size'] = 16
    
    return recommendations
