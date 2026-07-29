"""
Hardware device selection and system environment diagnostics utilities.
"""

import logging
import platform
import sys
import torch


def get_device() -> torch.device:
    """Detect available compute device (CUDA / GPU, MPS, or CPU).

    Returns:
        torch.device: The selected PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Support macOS Apple Silicon MPS if needed in future development
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def log_env_info(logger: logging.Logger) -> None:
    """Diagnose and log the host system environment parameters.

    Logs OS details, Python version, PyTorch version, CUDA version, and GPU info
    to the provided logger instance.

    Args:
        logger: Logger instance to output environment details.
    """
    logger.info("=" * 60)
    logger.info("SYSTEM ENVIRONMENT DIAGNOSTIC")
    logger.info("=" * 60)
    logger.info(f"Operating System: {platform.system()} {platform.release()} ({platform.machine()})")
    logger.info(f"Processor Name:   {platform.processor() or 'Unknown'}")
    logger.info(f"Python Version:   {sys.version.split()[0]} ({sys.executable})")
    logger.info(f"PyTorch Version:  {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available:   {cuda_available}")
    
    if cuda_available:
        logger.info(f"CUDA Version:     {torch.version.cuda}")
        logger.info(f"GPU Count:        {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU Device {i}:     {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA GPU devices are not active or not installed.")
        
    device = get_device()
    logger.info(f"Selected Device:  {device}")
    logger.info("=" * 60)
