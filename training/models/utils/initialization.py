"""
Weights initialization framework for PyTorch model parameters.

Author: AI Research Engineer
Purpose: Consistent weight initialization strategies to avoid vanishing/exploding gradients.
Future Work: Support custom initialization strategies mapping complex model components.
"""

import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


def initialize_kaiming_normal(model: nn.Module, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "relu") -> None:
    """Initialize convolutional and linear layers using He/Kaiming normal initialization.

    Converts weight parameters and sets biases to 0.

    Args:
        model (nn.Module): Target PyTorch module.
        a (float): Rectifier negative slope parameter. Defaults to 0.0.
        mode (str): Initializer fan mode. Options: 'fan_in', 'fan_out'.
        nonlinearity: Nonlinear activation function name. Defaults to 'relu'.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


def initialize_xavier_normal(model: nn.Module, gain: float = 1.0) -> None:
    """Initialize layers using Xavier/Glorot normal initialization.

    Args:
        model (nn.Module): Target PyTorch module.
        gain (float): Scaling factor. Defaults to 1.0.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


def initialize_normal(model: nn.Module, mean: float = 0.0, std: float = 0.02) -> None:
    """Initialize layer weights with a random normal distribution.

    Args:
        model (nn.Module): Target PyTorch module.
        mean (float): Mean of normal distribution. Defaults to 0.0.
        std (float): Standard deviation of normal distribution. Defaults to 0.02.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, mean=mean, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=mean, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


def initialize_constant(model: nn.Module, val: float = 0.0) -> None:
    """Initialize layer weights and biases to a fixed constant value.

    Args:
        model (nn.Module): Target PyTorch module.
        val (float): Constant value to apply. Defaults to 0.0.
    """
    for m in model.modules():
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.constant_(m.weight, val)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, val)


def custom_initialize(model: nn.Module, method: str, **kwargs) -> None:
    """Dispatcher to initialize module weights based on a string mapping name.

    Args:
        model (nn.Module): Target PyTorch module.
        method (str): Name of initialization strategy ('kaiming', 'xavier', 'normal', 'constant').
        **kwargs: Optional parameter overrides passed to the specific initializer functions.
    """
    cleaned_method = method.strip().lower()
    
    if cleaned_method == "kaiming":
        initialize_kaiming_normal(model, **kwargs)
    elif cleaned_method == "xavier":
        initialize_xavier_normal(model, **kwargs)
    elif cleaned_method == "normal":
        initialize_normal(model, **kwargs)
    elif cleaned_method == "constant":
        initialize_constant(model, **kwargs)
    else:
        logger.warning(
            f"Unknown weight initialization method: '{method}'. Skipping initialization."
        )
