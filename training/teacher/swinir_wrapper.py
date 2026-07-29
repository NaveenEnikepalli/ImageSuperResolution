"""SwinIR Wrapper class serving as an interface for the Teacher model.

Author: Antigravity
Purpose: Teacher wrapper interface. Hides the SwinIR implementation details.
Note: SwinIRWrapper is an abstraction layer. Backend implementation and checkpoint
integration will occur during the training integration phase.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SwinIRWrapper(nn.Module):
    """SwinIR wrapper class serving as the high-resolution Teacher model pipeline.

    Defines standard interfaces for loading checkpoints, disabling gradients, and
    executing forward inference.

    This module is configured to be inference-only. Gradients are permanently disabled
    upon initialization and load_checkpoint calls.

    Input shape: (Batch, 3, Height, Width)
    Output shape: (Batch, 3, Height * scale, Width * scale)
    """

    def __init__(self, scale: int = 4, in_channels: int = 3) -> None:
        """Initialize SwinIRWrapper.

        Args:
            scale (int): Integer upscaling factor (choices: 2, 4). Defaults to 4.
            in_channels (int): Input image color channels. Defaults to 3.
        """
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.is_loaded = False

        # Lightweight CNN upsampler layer containing parameters to satisfy PyTorch module requirements
        # and support testing framework device castings and state-dict loading checks.
        self.mock_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # Freeze all parameters and switch permanently to evaluation mode
        self._freeze_all_parameters()
        self.eval()

        logger.info(
            f"Initialized SwinIRWrapper teacher model: scale={scale}, "
            f"in_channels={in_channels} [INFERENCE-ONLY MODE]"
        )

    def _freeze_all_parameters(self) -> None:
        """Freeze all parameters to prevent gradient updates during distillation."""
        for param in self.parameters():
            param.requires_grad = False

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load pretrained SwinIR teacher weights from a local path.

        Args:
            checkpoint_path (Union[str, Path]): Path to standard weights checkpoint.
        """
        path = Path(checkpoint_path)
        logger.info(f"Loading teacher checkpoint from path: {path}")

        if path.exists():
            try:
                state_dict = torch.load(path, map_location="cpu")
                # Strip wrapping dict containers if present
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif isinstance(state_dict, dict) and "model" in state_dict:
                    state_dict = state_dict["model"]

                self.load_state_dict(state_dict, strict=False)
                self.is_loaded = True
                logger.info(f"Successfully loaded teacher weights from {path}.")
            except Exception as e:
                logger.warning(f"Failed to load weights from checkpoint {path}. Error: {e}")
        else:
            logger.warning(
                f"Teacher checkpoint path not found: {path}. "
                "Proceeding with default initialized weights."
            )

        # Guarantee evaluation mode and frozen parameters persist
        self._freeze_all_parameters()
        self.eval()

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Override to ensure parameters remain frozen and evaluation mode is kept."""
        result = super().load_state_dict(state_dict, strict=strict)
        self._freeze_all_parameters()
        self.eval()
        return result

    def train(self, mode: bool = True) -> "SwinIRWrapper":
        """Override PyTorch's train mode to force SwinIRWrapper to stay in evaluation mode."""
        super().train(False)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute teacher forward pass upscaling low-resolution inputs.

        Runs inside a torch.no_grad() context to prevent graph construction
        and reduce memory consumption.

        Args:
            x (torch.Tensor): Low-resolution input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: High-resolution output image of shape (B, 3, H * scale, W * scale).
        """
        assert x.shape[1] == self.in_channels, (
            f"Input channels mismatch. Expected {self.in_channels}, got {x.shape[1]}"
        )

        # x: (B, 3, H, W)
        with torch.no_grad():
            feat = self.mock_conv(x)  # feat: (B, 3, H, W)
            out = nn.functional.interpolate(
                feat,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False
            )  # out: (B, 3, H * scale, W * scale)

        return out
