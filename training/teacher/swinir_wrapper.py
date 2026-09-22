"""SwinIR Wrapper class serving as an interface for the Teacher model.

Author: Antigravity
Purpose: High-level teacher wrapper interface for SwinIR x4 Classical Super-Resolution model.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.teacher.swinir_model import SwinIR

logger = logging.getLogger(__name__)


class SwinIRWrapper(nn.Module):
    """SwinIR wrapper class serving as the high-resolution Teacher model pipeline.

    Defines standard interfaces for loading pretrained checkpoints, disabling gradients,
    window-size padding, and executing inference.

    This module is configured to be inference-only. Gradients are permanently disabled
    upon initialization and load_checkpoint calls.

    Input shape: (Batch, 3, Height, Width)
    Output shape: (Batch, 3, Height * scale, Width * scale)
    """

    def __init__(self, scale: int = 4, in_channels: int = 3) -> None:
        """Initialize SwinIRWrapper with real SwinIR backbone.

        Args:
            scale (int): Integer upscaling factor (choices: 2, 4). Defaults to 4.
            in_channels (int): Input image color channels. Defaults to 3.
        """
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.is_loaded = False

        # Real SwinIR backbone for Classical Image Super-Resolution x4
        self.model = SwinIR(
            upscale=self.scale,
            in_chans=self.in_channels,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=(6, 6, 6, 6, 6, 6),
            embed_dim=180,
            num_heads=(6, 6, 6, 6, 6, 6),
            mlp_ratio=2.0,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )

        # Freeze all parameters and switch permanently to evaluation mode
        self._freeze_all_parameters()
        self.eval()

        logger.info(
            f"Initialized SwinIRWrapper teacher model: scale={scale}, "
            f"in_channels={in_channels} [REAL SWINIR BACKBONE, INFERENCE-ONLY MODE]"
        )

    def _freeze_all_parameters(self) -> None:
        """Freeze all parameters to prevent gradient updates during distillation."""
        for param in self.parameters():
            param.requires_grad = False

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load pretrained SwinIR teacher weights from a local path.

        Performs strict verification of keys and tensor shapes. Fails loudly on missing files
        or incompatible weights.

        Args:
            checkpoint_path (Union[str, Path]): Path to pretrained weights checkpoint.

        Raises:
            FileNotFoundError: If checkpoint path does not exist.
            RuntimeError: If checkpoint state_dict is incompatible with SwinIR architecture.
        """
        if checkpoint_path is None:
            return

        path = Path(checkpoint_path)
        logger.info(f"Loading teacher checkpoint from path: {path}")

        if not path.exists():
            raise FileNotFoundError(f"Teacher checkpoint file not found: {path}")

        try:
            raw_dict = torch.load(path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint file at {path}. Error: {e}")

        # Extract weight container dictionary if wrapped
        if isinstance(raw_dict, dict):
            for key in ["params_ema", "params", "state_dict", "model"]:
                if key in raw_dict and isinstance(raw_dict[key], dict):
                    raw_dict = raw_dict[key]
                    break

        if not isinstance(raw_dict, dict):
            raise RuntimeError(f"Invalid checkpoint format in {path}: expected state_dict dict, got {type(raw_dict)}")

        original_key_count = len(raw_dict)
        model_state = self.model.state_dict()
        model_key_count = len(model_state)
        cleaned_dict = {}

        for k, v in raw_dict.items():
            new_k = k
            if new_k.startswith("module."):
                new_k = new_k[7:]
            if new_k.startswith("netG."):
                new_k = new_k[5:]
            if new_k.startswith("generator."):
                new_k = new_k[10:]
            if new_k.startswith("model.") and new_k[6:] in model_state:
                new_k = new_k[6:]

            # Translate SwinIR ResidualGroup attribute naming:
            # layers.{i}.residual_group.blocks.{j} -> layers.{i}.blocks.{j}
            # layers.{i}.residual_group.conv -> layers.{i}.conv
            if ".residual_group." in new_k:
                new_k = new_k.replace(".residual_group.", ".")

            # Ignore legacy unused patch_embed norm weights from Swin classification backbone
            if new_k in ["patch_embed.norm.weight", "patch_embed.norm.bias"]:
                continue

            cleaned_dict[new_k] = v

        # Populate initialized buffer 'mean' if missing in checkpoint
        if "mean" in model_state and "mean" not in cleaned_dict:
            cleaned_dict["mean"] = model_state["mean"]

        normalized_key_count = len(cleaned_dict)

        # Perform strict key and shape verification
        missing_keys = set(model_state.keys()) - set(cleaned_dict.keys())
        unexpected_keys = set(cleaned_dict.keys()) - set(model_state.keys())
        shape_mismatches = []

        for k in set(model_state.keys()) & set(cleaned_dict.keys()):
            if model_state[k].shape != cleaned_dict[k].shape:
                shape_mismatches.append(f"{k}: expected {model_state[k].shape}, got {cleaned_dict[k].shape}")

        if missing_keys or unexpected_keys or shape_mismatches:
            err_msg = [f"Incompatible SwinIR teacher checkpoint: {path}"]
            if missing_keys:
                err_msg.append(f"  Missing keys ({len(missing_keys)}): {sorted(list(missing_keys))[:5]}")
            if unexpected_keys:
                err_msg.append(f"  Unexpected keys ({len(unexpected_keys)}): {sorted(list(unexpected_keys))[:5]}")
            if shape_mismatches:
                err_msg.append(f"  Shape mismatches ({len(shape_mismatches)}): {shape_mismatches[:5]}")
            raise RuntimeError("\n".join(err_msg))

        self.model.load_state_dict(cleaned_dict, strict=True)
        self.is_loaded = True

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(
            f"Successfully loaded teacher weights from {path} with STRICT VERIFICATION:\n"
            f"  Original Checkpoint Keys:    {original_key_count}\n"
            f"  Normalized Checkpoint Keys:  {normalized_key_count}\n"
            f"  Model Keys:                  {model_key_count}\n"
            f"  Missing Keys:                0\n"
            f"  Unexpected Keys:             0\n"
            f"  Shape Mismatches:            0\n"
            f"  Total Parameters:            {total_params:,}\n"
            f"  Trainable Parameters:        {trainable_params:,}\n"
            f"  Strict Load Status:          SUCCESS"
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
        and reduce memory consumption. Pads non-window-aligned input spatial dimensions
        and crops output to exact (B, 3, 4H, 4W).

        Args:
            x (torch.Tensor): Low-resolution input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: High-resolution output image of shape (B, 3, H * scale, W * scale).
        """
        assert x.shape[1] == self.in_channels, (
            f"Input channels mismatch. Expected {self.in_channels}, got {x.shape[1]}"
        )

        B, C, H, W = x.shape
        window_size = self.model.window_size

        # Calculate window-size padding requirement
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size

        with torch.no_grad():
            if pad_h > 0 or pad_w > 0:
                # Pad spatial dimensions to multiples of window_size
                x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
                out_padded = self.model(x_padded)
                # Crop output back to exact target dimensions
                out = out_padded[:, :, : H * self.scale, : W * self.scale]
            else:
                out = self.model(x)

        return out
