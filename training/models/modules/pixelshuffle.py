"""PixelShuffle module wrapper for sub-pixel convolution upscaling.

Author: Antigravity
Purpose: Expose PyTorch's PixelShuffle layer as a standalone module inside the modules package.
"""

import torch
import torch.nn as nn


class PixelShuffle(nn.Module):
    """PixelShuffle layer wrapper.

    Rearranges elements in a tensor of shape (*, C * r^2, H, W) to a tensor of shape
    (*, C, H * r, W * r), where r is the upscale factor.
    """

    def __init__(self, scale: int) -> None:
        """Initialize the PixelShuffle block.

        Args:
            scale (int): Upscaling factor.
        """
        super().__init__()
        if scale <= 0:
            raise ValueError(f"Upscaling scale factor must be positive, got {scale}")
        self.scale = scale
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PixelShuffle upscaling.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C * scale^2, H, W).

        Returns:
            torch.Tensor: Upscaled tensor of shape (B, C, H * scale, W * scale).
        """
        # x: (B, C * scale^2, H, W)
        out = self.pixel_shuffle(x)
        # out: (B, C, H * scale, W * scale)
        return out

    def extra_repr(self) -> str:
        """Provide extra representation details.

        Returns:
            str: Module description containing configuration details.
        """
        return f"scale={self.scale}"
