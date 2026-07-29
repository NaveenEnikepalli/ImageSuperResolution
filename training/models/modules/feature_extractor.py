"""Feature Extractor module for the Lightweight Student Super-Resolution model.

Author: Antigravity
Purpose: Implementation of shallow feature extraction using standard 2D convolution and LeakyReLU.
"""

import torch
import torch.nn as nn
from training.models.utils import validate_tensor


class FeatureExtractor(nn.Module):
    """Shallow Feature Extraction module of the Student Super-Resolution network.

    It applies a single 3x3 2D convolution followed by a LeakyReLU activation function.
    No Batch Normalization, residual connections, or attention layers are used.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 48,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        bias: bool = True,
        negative_slope: float = 0.05,
        debug: bool = False,
    ) -> None:
        """Initialize the FeatureExtractor.

        Args:
            in_channels (int): Number of input channels. Defaults to 3.
            out_channels (int): Number of output channels. Defaults to 48.
            kernel_size (int): Kernel size for convolution. Defaults to 3.
            padding (int): Padding for convolution. Defaults to 1.
            stride (int): Stride for convolution. Defaults to 1.
            bias (bool): Enable bias for convolution. Defaults to True.
            negative_slope (float): Negative slope for LeakyReLU. Defaults to 0.05.
            debug (bool): Enable verbose logging of input/output shapes. Defaults to False.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        self.negative_slope = negative_slope
        self.debug = debug

        # Validate configuration parameters
        self._validate_config()

        # Define layers. Rely explicitly on PyTorch default weight initialization.
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
        )
        self.act = nn.LeakyReLU(negative_slope=self.negative_slope)

    def _validate_config(self) -> None:
        """Validate configuration settings.

        Raises:
            ValueError: If configuration values are invalid.
        """
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if self.out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {self.out_channels}")
        if self.kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {self.kernel_size}")
        if self.padding < 0:
            raise ValueError(f"padding must be non-negative, got {self.padding}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.negative_slope < 0.0:
            raise ValueError(f"negative_slope must be non-negative, got {self.negative_slope}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for FeatureExtractor.

        Args:
            x (torch.Tensor): Low-resolution input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Shallow features output of shape (B, out_channels, H, W).
        """
        # x: (B, in_channels, H, W)
        if not torch.jit.is_scripting():
            validate_tensor(x, expected_dim=4, expected_channels=self.in_channels, param_name="Input tensor")

        out = self.conv(x)  # out: (B, out_channels, H, W)
        out = self.act(out)  # out: (B, out_channels, H, W)

        if self.debug:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"[FeatureExtractor Debug Pass]\n"
                f"  Input Shape:    {list(x.shape)}\n"
                f"  Output Shape:   {list(out.shape)}\n"
                f"  Current Device: {x.device}\n"
                f"  Tensor Dtype:   {x.dtype}"
            )

        return out

    def extra_repr(self) -> str:
        """Extra representation string for print(model) diagnostics.

        Returns:
            str: Introspection parameters descriptor.
        """
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, bias={self.bias}, "
            f"negative_slope={self.negative_slope}, debug={self.debug}"
        )
