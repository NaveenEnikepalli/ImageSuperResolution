"""Residual Feature Distillation Block (RFDB) for the Student Super-Resolution model.

Author: Antigravity
Purpose: Implementation of the channel-preserving RFDB module with explicit constructor parameters.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn

from training.models.utils import validate_tensor

logger = logging.getLogger(__name__)


class ResidualFeatureDistillationBlock(nn.Module):
    """Residual Feature Distillation Block (RFDB) for Student Super-Resolution network.

    Fulfills the finalized architecture freeze specifications:
    1. Feature Refinement Stage: 3x3 Conv + LeakyReLU. Maps input X -> R of shape (B, C, H, W).
    2. Information Extraction Branch: 1x1 Convolution mapping the ENTIRE refined tensor R
       (C channels) to Distilled Features of shape (B, distilled_channels, H, W).
    3. Feature Preservation Branch: Performs identity slicing mapping C -> C - D by selecting
       the last C - D channels: R[:, distilled_channels:, :, :], with no convs or parameters.
    4. Feature Concatenation: Combines branches along dim=1 to shape (B, C, H, W).
    5. Feature Fusion Stage: 1x1 Conv (C -> C) mapping concatenated branch representations.
    6. Channel-wise Learnable Residual Scaling: Broadcast-scales fusion output using alpha (1, C, 1, 1).
    7. Residual Addition: Output = Input + Alpha * FusionOutput.

    Input shape: (B, C, H, W)
    Output shape: (B, C, H, W)
    """

    def __init__(
        self,
        channels: int = 48,
        distilled_channels: Optional[int] = None,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        bias: bool = True,
        negative_slope: float = 0.05,
        initial_residual_scale: float = 1.0,
        block_index: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        """Initialize the ResidualFeatureDistillationBlock.

        Args:
            channels (int): Number of input and output channels (C). Defaults to 48.
            distilled_channels (Optional[int]): Channels in distillation path (D). Defaults to channels // 2.
            kernel_size (int): Kernel size for refinement convolution. Defaults to 3.
            padding (int): Padding for refinement convolution. Defaults to 1.
            stride (int): Stride for refinement convolution. Defaults to 1.
            bias (bool): Enable bias for all convolutions. Defaults to True.
            negative_slope (float): Negative slope for LeakyReLU. Defaults to 0.05.
            initial_residual_scale (float): Initial residual scale factor. Defaults to 1.0.
            block_index (Optional[int]): Index of block in sequential network for logging.
            debug (bool): Enable verbose logging and shape assertion checks. Defaults to False.
        """
        super().__init__()

        self.num_channels = channels
        self.distilled_channels = distilled_channels if distilled_channels is not None else channels // 2
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        self.negative_slope = negative_slope
        self.initial_residual_scale = initial_residual_scale
        self.block_index = block_index
        self.debug = debug

        # Validate configuration values
        self._validate_config()

        # Stage 1: Feature Refinement Stage (3x3 convolution + LeakyReLU)
        self.refinement_conv = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
        )
        self.act_name = "leaky_relu"
        self.act = nn.LeakyReLU(negative_slope=self.negative_slope)

        # Stage 2: Information Extraction Branch (1x1 convolution mapping C -> D channels)
        self.distill_conv = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.distilled_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.bias,
        )

        # Stage 5: Feature Fusion Stage (1x1 convolution mapping C -> C channels)
        self.fusion_conv = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.bias,
        )

        # Stage 6: Channel-wise Learnable Residual Scaling
        # Rely explicitly on PyTorch default weights initialization
        self.alpha = nn.Parameter(
            torch.full((1, self.num_channels, 1, 1), float(self.initial_residual_scale))
        )

        logger.info(
            f"Initialized ResidualFeatureDistillationBlock"
            f"{f' {self.block_index}' if self.block_index is not None else ''} [FROZEN ARCHITECTURE]: "
            f"num_channels={self.num_channels}, distilled_channels={self.distilled_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"bias={self.bias}, negative_slope={self.negative_slope}, "
            f"initial_residual_scale={self.initial_residual_scale}, debug={self.debug}"
        )

    def _validate_config(self) -> None:
        """Validate configuration settings.

        Raises:
            ValueError: If parameters violate architectural invariants.
        """
        if self.num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {self.num_channels}")
        if self.distilled_channels <= 0:
            raise ValueError(f"distilled_channels must be positive, got {self.distilled_channels}")
        if self.distilled_channels >= self.num_channels:
            raise ValueError(
                f"distilled_channels ({self.distilled_channels}) must be strictly less than "
                f"num_channels ({self.num_channels}) to allow remaining branch channels."
            )
        if self.kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {self.kernel_size}")
        if self.padding < 0:
            raise ValueError(f"padding must be non-negative, got {self.padding}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.negative_slope < 0.0:
            raise ValueError(f"negative_slope must be non-negative, got {self.negative_slope}")
        if self.initial_residual_scale < 0.0:
            raise ValueError(f"initial_residual_scale must be non-negative, got {self.initial_residual_scale}")

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate the input tensor dimensions and shapes.

        Args:
            x (torch.Tensor): Input tensor.
        """
        if not torch.jit.is_scripting():
            validate_tensor(
                x,
                expected_dim=4,
                expected_channels=self.num_channels,
                param_name=f"RFDB{f' {self.block_index}' if self.block_index is not None else ''} input"
            )

    def _feature_refinement(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Feature Refinement Stage (3x3 conv + activation).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Refined local spatial features of shape (B, C, H, W).
        """
        return self.act(self.refinement_conv(x))

    def _information_extraction_branch(self, refined: torch.Tensor) -> torch.Tensor:
        """Apply Information Extraction Branch (1x1 conv mapping entire refined tensor).

        Args:
            refined (torch.Tensor): Entire refined feature map tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Distilled informative features of shape (B, D, H, W).
        """
        return self.distill_conv(refined)

    def _feature_preservation_branch(self, refined: torch.Tensor) -> torch.Tensor:
        """Apply Feature Preservation Branch (Identity slicing).

        Selects the last C - D channels from the refined tensor to preserve complementary information.

        Args:
            refined (torch.Tensor): Entire refined feature map tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Sliced preserved features of shape (B, C - D, H, W).
        """
        return refined[:, self.distilled_channels:, :, :]

    def _feature_fusion(self, concatenated: torch.Tensor) -> torch.Tensor:
        """Fuse branch outputs (1x1 conv mapping combined features).

        Args:
            concatenated (torch.Tensor): Combined features of shape (B, C, H, W).

        Returns:
            torch.Tensor: Fused mixed features of shape (B, C, H, W).
        """
        return self.fusion_conv(concatenated)

    def _apply_residual_scaling(self, fusion_output: torch.Tensor) -> torch.Tensor:
        """Apply learnable channel-wise scaling factor to the fusion output.

        Args:
            fusion_output (torch.Tensor): Fused mixed features.

        Returns:
            torch.Tensor: Scaled features of shape (B, C, H, W).
        """
        return self.alpha * fusion_output

    @torch.jit.unused
    def _log_debug_info(
        self,
        x: torch.Tensor,
        refined: torch.Tensor,
        distilled: torch.Tensor,
        preserved: torch.Tensor,
        concatenated: torch.Tensor,
        fused: torch.Tensor,
        out: torch.Tensor
    ) -> None:
        """Log diagnostic shape, stats, device, and parameter details of all RFDB stages.

        Args:
            x (torch.Tensor): Input tensor.
            refined (torch.Tensor): Refined tensor.
            distilled (torch.Tensor): Distilled tensor.
            preserved (torch.Tensor): Preserved tensor.
            concatenated (torch.Tensor): Concatenated tensor.
            fused (torch.Tensor): Fused tensor.
            out (torch.Tensor): Output tensor.
        """
        block_lbl = f"RFDB Block {self.block_index}" if self.block_index is not None else "RFDB Block"
        
        # Compute mean/std statistics for internal stages for detailed diagnostics
        refined_mean = refined.mean().item()
        refined_std = refined.std().item()
        fused_mean = fused.mean().item()
        fused_std = fused.std().item()
        out_mean = out.mean().item()
        out_std = out.std().item()

        logger.info(
            f"[{block_lbl} Debug Pass]\n"
            f"  Input Shape:        {list(x.shape)}\n"
            f"  Refined Shape:      {list(refined.shape)} (mean={refined_mean:.4f}, std={refined_std:.4f})\n"
            f"  Distilled Shape:    {list(distilled.shape)}\n"
            f"  Preserved Shape:    {list(preserved.shape)}\n"
            f"  Concatenated Shape: {list(concatenated.shape)}\n"
            f"  Fusion Shape:       {list(fused.shape)} (mean={fused_mean:.4f}, std={fused_std:.4f})\n"
            f"  Output Shape:       {list(out.shape)} (mean={out_mean:.4f}, std={out_std:.4f})\n"
            f"  Alpha Mean Value:   {self.alpha.mean().item():.6f}\n"
            f"  Alpha Tensor Shape: {list(self.alpha.shape)}\n"
            f"  Current Device:     {x.device}\n"
            f"  Tensor Dtype:       {x.dtype}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Coordinate forward pass execution across all block submodules.

        Args:
            x (torch.Tensor): Input feature tensor of shape (B, num_channels, H, W).

        Returns:
            torch.Tensor: Residual features output of shape (B, num_channels, H, W).
        """
        # x: (B, C, H, W)
        self._validate_input(x)

        # Stage 1: Feature Refinement Stage (3x3 conv + activation)
        refined = self._feature_refinement(x)  # refined: (B, C, H, W)

        # Stage 2: Information Extraction Branch (1x1 conv mapping entire refined tensor)
        distilled = self._information_extraction_branch(refined)  # distilled: (B, D, H, W)

        # Stage 3: Feature Preservation Branch (Identity slice selecting last C - D channels)
        preserved = self._feature_preservation_branch(refined)  # preserved: (B, C - D, H, W)

        # Stage 4: Feature Concatenation
        concatenated = torch.cat([distilled, preserved], dim=1)  # concatenated: (B, C, H, W)

        # Stage 5: Feature Fusion Stage (1x1 conv)
        fused = self._feature_fusion(concatenated)  # fused: (B, C, H, W)

        # Stage 6: Channel-wise Learnable Residual Scaling
        scaled_fused = self._apply_residual_scaling(fused)  # scaled_fused: (B, C, H, W)

        # Stage 7: Residual Addition
        out = x + scaled_fused  # out: (B, C, H, W)

        if self.debug:
            # Perform intermediate shape assertions in debug mode for mathematical verification
            assert refined.shape == x.shape, f"Refinement shape {refined.shape} mismatch"
            assert distilled.shape == (x.shape[0], self.distilled_channels, x.shape[2], x.shape[3]), f"Distilled shape {distilled.shape} mismatch"
            assert preserved.shape == (x.shape[0], self.num_channels - self.distilled_channels, x.shape[2], x.shape[3]), f"Preserved shape {preserved.shape} mismatch"
            assert concatenated.shape == x.shape, f"Concatenated shape {concatenated.shape} mismatch"
            assert fused.shape == x.shape, f"Fused shape {fused.shape} mismatch"
            assert out.shape == x.shape, f"Output shape {out.shape} mismatch"
            self._log_debug_info(x, refined, distilled, preserved, concatenated, fused, out)

        return out

    def extra_repr(self) -> str:
        """Provide detailed parameter metrics for model printing.

        Returns:
            str: Introspection parameters descriptor.
        """
        block_idx_str = f", block_index={self.block_index}" if self.block_index is not None else ""
        return (
            f"num_channels={self.num_channels}, "
            f"distilled_channels={self.distilled_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"activation={self.act_name}, "
            f"initial_alpha={self.initial_residual_scale:.4f}{block_idx_str}, "
            f"debug={self.debug}"
        )
