"""Reconstruction module for final sub-pixel upscaling in Student Model.

Author: Antigravity
Purpose: Implementation of PixelShuffle based restoration and upsampling backend.
"""

import torch
import torch.nn as nn

from training.models.modules.pixelshuffle import PixelShuffle


class Reconstruction(nn.Module):
    """Reconstruction block performing sub-pixel convolution upscaling.

    Architecture:
    1. 3x3 Conv mapping num_features channels to num_features * scale^2.
    2. PixelShuffle layer upscaling the spatial dimensions by scale factor.
    3. Final 3x3 Conv mapping num_features to out_channels (e.g. 3 for RGB).
    """

    def __init__(
        self,
        num_features: int = 48,
        out_channels: int = 3,
        scale: int = 4,
        bias: bool = True,
    ) -> None:
        """Initialize the Reconstruction block.

        Args:
            num_features (int): Input feature map channels. Defaults to 48.
            out_channels (int): Final output channels (e.g., RGB). Defaults to 3.
            scale (int): Integer upscaling factor. Defaults to 4.
            bias (bool): Enable bias for the convolutions. Defaults to True.
        """
        super().__init__()
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")

        self.num_features = num_features
        self.out_channels = out_channels
        self.scale = scale
        self.bias = bias

        # 1. First convolution: Projects features to expanded sub-pixel channel space
        self.conv1 = nn.Conv2d(
            in_channels=self.num_features,
            out_channels=self.num_features * (self.scale ** 2),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self.bias,
        )

        # 2. Upsampler: Rearranges expanded channel dimensions to spatial dimensions
        self.upsampler = PixelShuffle(scale=self.scale)

        # 3. Final convolution: Restores representation channels to target output dimensions (RGB)
        self.conv2 = nn.Conv2d(
            in_channels=self.num_features,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self.bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upscale features to high-resolution space.

        Args:
            x (torch.Tensor): Feature map input tensor of shape (B, num_features, H, W).

        Returns:
            torch.Tensor: Upscaled output image tensor of shape (B, out_channels, H * scale, W * scale).
        """
        assert x.shape[1] == self.num_features, (
            f"Input channels mismatch. Expected {self.num_features}, got {x.shape[1]}"
        )

        # Input x: (B, num_features, H, W)
        # Expand channels via Conv
        expanded = self.conv1(x)  # expanded: (B, num_features * scale^2, H, W)

        # Sub-pixel upscaling
        upscaled = self.upsampler(expanded)  # upscaled: (B, num_features, H * scale, W * scale)

        # Map to final RGB channels
        out = self.conv2(upscaled)  # out: (B, out_channels, H * scale, W * scale)

        return out

    def extra_repr(self) -> str:
        """Extra representation string for print(model) diagnostics.

        Returns:
            str: Introspection parameters descriptor.
        """
        return f"num_features={self.num_features}, out_channels={self.out_channels}, scale={self.scale}, bias={self.bias}"
