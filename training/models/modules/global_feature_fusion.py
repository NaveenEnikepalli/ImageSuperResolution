"""Global Feature Fusion module for the Lightweight Student Super-Resolution model.

Author: Antigravity
Purpose: Implementation of block-wise feature concatenation and projection layers.
"""

from typing import List
import torch
import torch.nn as nn


class GlobalFeatureFusion(nn.Module):
    """Global Feature Fusion block.

    Concatenates intermediate RFDB outputs along the channel dimension,
    then applies a 1x1 convolution followed by a 3x3 convolution to project
    the merged channels back to the base feature channels.
    """

    def __init__(
        self,
        num_blocks: int = 3,
        num_features: int = 48,
        bias: bool = True,
    ) -> None:
        """Initialize the GlobalFeatureFusion module.

        Args:
            num_blocks (int): Number of RFDB blocks whose outputs will be fused. Defaults to 3.
            num_features (int): Dimensionality of each RFDB output. Defaults to 48.
            bias (bool): Enable bias for both convolutions. Defaults to True.
        """
        super().__init__()
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")

        self.num_blocks = num_blocks
        self.num_features = num_features
        self.bias = bias

        # 1x1 Convolution: projects concatenated features (num_blocks * num_features) -> num_features
        self.conv1x1 = nn.Conv2d(
            in_channels=self.num_blocks * self.num_features,
            out_channels=self.num_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.bias,
        )

        # 3x3 Convolution: projects num_features -> num_features
        self.conv3x3 = nn.Conv2d(
            in_channels=self.num_features,
            out_channels=self.num_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self.bias,
        )

    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        """Fuse intermediate block representations.

        Args:
            feature_list (List[torch.Tensor]): List of length num_blocks,
                each containing features of shape (B, num_features, H, W).

        Returns:
            torch.Tensor: Fused representation tensor of shape (B, num_features, H, W).
        """
        assert len(feature_list) == self.num_blocks, (
            f"Input list size mismatch. Expected {self.num_blocks}, got {len(feature_list)}"
        )
        for i, feat in enumerate(feature_list):
            assert feat.shape[1] == self.num_features, (
                f"Feature channels mismatch at index {i}. Expected {self.num_features}, got {feat.shape[1]}"
            )

        # Input elements in feature_list: each of shape (B, num_features, H, W)
        # Concatenate along channel dimension (dim=1)
        concat_feats = torch.cat(feature_list, dim=1)  # concat_feats: (B, num_blocks * num_features, H, W)

        # 1x1 Convolution projection
        proj_1x1 = self.conv1x1(concat_feats)  # proj_1x1: (B, num_features, H, W)

        # 3x3 Convolution refinement
        fused = self.conv3x3(proj_1x1)  # fused: (B, num_features, H, W)

        return fused

    def extra_repr(self) -> str:
        """Extra representation string for print(model) diagnostics.

        Returns:
            str: Introspection parameters descriptor.
        """
        return f"num_blocks={self.num_blocks}, num_features={self.num_features}, bias={self.bias}"
