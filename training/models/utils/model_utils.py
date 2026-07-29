"""
Model utilities for parameter counting, model size estimation, structure diagnostics, and validations.

Author: AI Research Engineer
Purpose: Engineering helpers for super-resolution model architectures.
Future Work: Incorporate FLOPs estimation hook handlers.
"""

import logging
from typing import Tuple, Any, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """Count the total number of parameters in a PyTorch module.

    Args:
        model (nn.Module): Target PyTorch module.

    Returns:
        int: Total parameter count.
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a PyTorch module.

    Args:
        model (nn.Module): Target PyTorch module.

    Returns:
        int: Trainable parameter count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size(model: nn.Module) -> float:
    """Estimate the memory size of a PyTorch module in megabytes (MB).

    Args:
        model (nn.Module): Target PyTorch module.

    Returns:
        float: Estimated model size in MB.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> None:
    """Log a detailed structural and complexity summary of the model.

    Args:
        model (nn.Module): Target PyTorch module.
        input_size (Tuple[int, ...]): Size of a sample input tensor (e.g. (1, 3, 48, 48)).
    """
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    size_mb = estimate_model_size(model)
    
    logger.info("=" * 60)
    logger.info("MODEL SUMMARY DIAGNOSTICS")
    logger.info("=" * 60)
    logger.info(f"Model Class Name:  {model.__class__.__name__}")
    logger.info(f"Input Shape:       {list(input_size)}")
    logger.info(f"Total Parameters:  {total_params:,}")
    logger.info(f"Trainable Params:  {trainable_params:,}")
    logger.info(f"Non-Trainable:     {total_params - trainable_params:,}")
    logger.info(f"Estimated Size:    {size_mb:.4f} MB")
    logger.info("=" * 60)


def verify_forward_signature(model: nn.Module, input_shape: Tuple[int, ...]) -> bool:
    """Verify that the model executes its forward pass successfully with the given input shape.

    Args:
        model (nn.Module): Target PyTorch module.
        input_shape (Tuple[int, ...]): Spatial tuple representing batch, channels, and dimensions.

    Returns:
        bool: True if forward pass executes without raising exceptions.
    """
    try:
        # Determine the model device
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    try:
        dummy_input = torch.randn(input_shape, device=device)
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
        return True
    except Exception as e:
        logger.error(f"Forward signature validation failed for {model.__class__.__name__}. Error: {e}")
        return False


def verify_device(model: nn.Module, expected_device: torch.device) -> bool:
    """Validate that all parameters inside a module have been successfully transferred to the target device.

    Args:
        model (nn.Module): Target PyTorch module.
        expected_device (torch.device): PyTorch device type expected (e.g. 'cuda', 'cpu').

    Returns:
        bool: True if all parameters match the expected device.
    """
    # Normalize device checking strings
    expected_type = torch.device(expected_device).type
    for name, param in model.named_parameters():
        if param.device.type != expected_type:
            logger.warning(
                f"Parameter {name} on device '{param.device}' does not match expected '{expected_type}'"
            )
            return False
    return True


def verify_dtype(model: nn.Module, expected_dtype: torch.dtype) -> bool:
    """Validate that all parameters inside a module match the expected torch dtype.

    Args:
        model (nn.Module): Target PyTorch module.
        expected_dtype (torch.dtype): Expected data type (e.g. torch.float32, torch.float16).

    Returns:
        bool: True if all parameters match the expected data type.
    """
    for name, param in model.named_parameters():
        if param.dtype != expected_dtype:
            logger.warning(
                f"Parameter {name} of type '{param.dtype}' does not match expected '{expected_dtype}'"
            )
            return False
    return True


def validate_tensor(
    x: torch.Tensor,
    expected_dim: int = 4,
    expected_channels: Optional[int] = None,
    param_name: str = "Input tensor"
) -> None:
    """Validate a tensor's type, dimensions, and channel counts.

    Args:
        x (Any): Object to validate.
        expected_dim (int): Expected number of dimensions. Defaults to 4.
        expected_channels (Optional[int]): Expected number of channels. Defaults to None.
        param_name (str): Name of the tensor for error messages.

    Raises:
        TypeError: If x is not a torch.Tensor.
        ValueError: If x does not have the expected dimensions or channel counts.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{param_name} must be a torch.Tensor, got {type(x)}")
    if len(x.shape) != expected_dim:
        raise ValueError(f"{param_name} must have {expected_dim} dimensions, got shape {x.shape}")
    if expected_channels is not None and x.shape[1] != expected_channels:
        raise ValueError(
            f"{param_name} channel dimension mismatch. Expected {expected_channels} channels, "
            f"got {x.shape[1]} channels (shape: {x.shape})"
        )
