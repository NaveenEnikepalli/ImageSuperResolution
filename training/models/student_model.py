"""Student Model top-level assembly wrapping modular sub-modules.

Author: Antigravity
Purpose: High-level architectural pipeline for the lightweight student model.
"""

import logging
from typing import List, Optional
import torch
import torch.nn as nn

from training.models.modules.feature_extractor import FeatureExtractor
from training.models.modules.rfdb import ResidualFeatureDistillationBlock
from training.models.modules.global_feature_fusion import GlobalFeatureFusion
from training.models.modules.reconstruction import Reconstruction

logger = logging.getLogger(__name__)


class StudentModel(nn.Module):
    """Assembly class aggregating sub-components of the Student Super-Resolution network.

    Coordinates the following modular steps:
    1. Shallow Feature Extraction: (B, 3, H, W) -> (B, num_features, H, W)
    2. Deep Feature Extraction (Sequential RFDB blocks stored in nn.ModuleList):
       Yields intermediate features of shape (B, num_features, H, W)
    3. Global Feature Fusion: Projects block outputs back to (B, num_features, H, W)
    4. Global Skip Connection (Residual shortcut): Fused features + Shallow features
    5. High-Resolution Reconstruction: (B, num_features, H, W) -> (B, 3, H*scale, W*scale)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 48,
        distilled_channels: Optional[int] = None,
        num_blocks: int = 3,
        scale: int = 4,
        negative_slope: float = 0.05,
        debug: bool = False,
    ) -> None:
        """Initialize the StudentModel assembly.

        Args:
            in_channels (int): Input image channels. Defaults to 3.
            out_channels (int): Output image channels. Defaults to 3.
            num_features (int): Number of features/channels in internal layers. Defaults to 48.
            distilled_channels (Optional[int]): Number of distilled channels in RFDB block. Defaults to num_features // 2.
            num_blocks (int): Number of sequential RFDB blocks. Defaults to 3.
            scale (int): Integer upscaling factor. Defaults to 4.
            negative_slope (float): Activation negative slope. Defaults to 0.05.
            debug (bool): Enable diagnostic logs. Defaults to False.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.distilled_channels = distilled_channels if distilled_channels is not None else num_features // 2
        self.num_blocks = num_blocks
        self.scale = scale
        self.negative_slope = negative_slope
        self.debug = debug

        logger.info(
            f"Assembling StudentModel [FROZEN ARCHITECTURE]: scale={self.scale}, "
            f"num_blocks={self.num_blocks}, num_features={self.num_features}, "
            f"distilled_channels={self.distilled_channels}, negative_slope={self.negative_slope}"
        )

        # 1. Shallow Feature Extractor
        self.feature_extractor = FeatureExtractor(
            in_channels=self.in_channels,
            out_channels=self.num_features,
            negative_slope=self.negative_slope,
            debug=self.debug,
        )

        # 2. RFDB Stack (ModuleList of N block modules)
        # Rely explicitly on PyTorch default weight initialization
        self.blocks = nn.ModuleList([
            ResidualFeatureDistillationBlock(
                channels=self.num_features,
                distilled_channels=self.distilled_channels,
                negative_slope=self.negative_slope,
                block_index=i + 1,
                debug=self.debug,
            )
            for i in range(self.num_blocks)
        ])

        # 3. Global Feature Fusion
        self.fusion = GlobalFeatureFusion(
            num_blocks=self.num_blocks,
            num_features=self.num_features,
        )

        # 4. HR Reconstruction Backend (Upsampler & PixelShuffle)
        self.reconstruction = Reconstruction(
            num_features=self.num_features,
            out_channels=self.out_channels,
            scale=self.scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass coordinating features flow across all submodules.

        Args:
            x (torch.Tensor): Input low-resolution image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: High-resolution output image tensor of shape (B, 3, H*scale, W*scale).
        """
        # x: (B, 3, H, W)
        
        # Step 1: Shallow Feature Extraction
        shallow_feats = self.feature_extractor(x)  # shallow_feats: (B, num_features, H, W)

        # Step 2: Sequential Deep Feature Extraction via RFDB Stack
        block_features: List[torch.Tensor] = []
        current_feat = shallow_feats
        for block in self.blocks:
            current_feat = block(current_feat)  # current_feat: (B, num_features, H, W)
            block_features.append(current_feat)

        # Step 3: Global Feature Fusion
        fused_feats = self.fusion(block_features)  # fused_feats: (B, num_features, H, W)

        # Step 4: Global Skip Connection (Residual shortcut)
        final_feats = fused_feats + shallow_feats  # final_feats: (B, num_features, H, W)

        # Step 5: PixelShuffle-based HR Image Reconstruction
        out = self.reconstruction(final_feats)  # out: (B, out_channels, H * scale, W * scale)

        if self.debug:
            logger.info(
                f"[StudentModel Debug Pass]\n"
                f"  Input Shape:          {list(x.shape)}\n"
                f"  Shallow Feats Shape:  {list(shallow_feats.shape)}\n"
                f"  Fused Feats Shape:    {list(fused_feats.shape)}\n"
                f"  Final Feats Shape:    {list(final_feats.shape)}\n"
                f"  Output HR Shape:      {list(out.shape)}"
            )

        return out
