"""Student Model Inference Engine.

Author: Antigravity
Purpose: High-level inference pipeline for Student Super-Resolution models (x2 and x4).
Handles device assignment, checkpoint loading, pre/post-processing, and clamping.
"""

import logging
from pathlib import Path
from typing import Union, Optional
import torch
import numpy as np
from PIL import Image

from training.models.student_model import StudentModel

logger = logging.getLogger(__name__)


class StudentInference:
    """Inference runner for lightweight Student Super-Resolution models."""

    def __init__(
        self,
        scale: int = 4,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize StudentInference engine.

        Args:
            scale (int): Upscaling scale factor (2 or 4). Defaults to 4.
            checkpoint_path (Optional[Union[str, Path]]): Path to trained checkpoint.
            device (Optional[Union[str, torch.device]]): Target compute device.
        """
        if scale not in (2, 4):
            raise ValueError(f"StudentInference only supports scale=2 or scale=4, got scale={scale}")

        self.scale = scale
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Resolve default checkpoint path if not provided
        if checkpoint_path is None:
            checkpoint_path = Path("checkpoints/student") / f"student_x{self.scale}.pth"
        self.checkpoint_path = Path(checkpoint_path)

        # Instantiate StudentModel with scale parameter
        self.model = StudentModel(
            in_channels=3,
            out_channels=3,
            num_features=48,
            distilled_channels=24,
            num_blocks=3,
            scale=self.scale,
            negative_slope=0.05,
        )

        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"StudentInference engine ready [Scale x{self.scale}, Device={self.device}, "
            f"Checkpoint={self.checkpoint_path}]"
        )

    def _load_checkpoint(self) -> None:
        """Load pretrained state dict weights into StudentModel."""
        if not self.checkpoint_path.exists():
            logger.warning(
                f"Student x{self.scale} checkpoint not found at {self.checkpoint_path}. "
                f"Using uninitialized weights (smoke/testing mode)."
            )
            return

        logger.info(f"Loading Student x{self.scale} checkpoint from: {self.checkpoint_path}")
        raw_data = torch.load(self.checkpoint_path, map_location="cpu")

        state_dict = raw_data
        if isinstance(raw_data, dict):
            for k in ["state_dict", "model_state_dict", "model", "student_state_dict"]:
                if k in raw_data and isinstance(raw_data[k], dict):
                    state_dict = raw_data[k]
                    break

        # Remove module prefix if state dict saved from DataParallel
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_k = k[7:] if k.startswith("module.") else k
            cleaned_state_dict[new_k] = v

        self.model.load_state_dict(cleaned_state_dict, strict=False)

    @torch.no_grad()
    def enhance(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Run super-resolution enhancement on an input RGB image.

        Args:
            image (Union[Image.Image, np.ndarray]): Input low-resolution RGB image.

        Returns:
            Image.Image: High-resolution output PIL Image (scale*W x scale*H).
        """
        # Convert PIL to numpy array if required
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            np_img = np.array(pil_img)
        elif isinstance(image, np.ndarray):
            np_img = image
            if np_img.ndim == 2:  # Grayscale
                np_img = np.stack([np_img] * 3, axis=-1)
            elif np_img.shape[2] == 4:  # RGBA
                np_img = np_img[:, :, :3]
        else:
            raise TypeError(f"Expected PIL Image or numpy ndarray, got {type(image)}")

        orig_h, orig_w = np_img.shape[:2]

        # Normalize to range [0.0, 1.0] and convert to tensor (1, 3, H, W)
        tensor_in = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor_in = tensor_in.to(self.device)

        # Forward execution
        sr_tensor = self.model(tensor_in)

        # Clamp output to range [0.0, 1.0]
        sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)

        # Post-process tensor back to uint8 numpy array
        sr_np = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

        target_w = orig_w * self.scale
        target_h = orig_h * self.scale

        out_img = Image.fromarray(sr_np, mode="RGB")

        # Ensure exact output dimensions match target (target_w, target_h)
        if out_img.size != (target_w, target_h):
            logger.warning(
                f"Output dimensions mismatch. Expected ({target_w}, {target_h}), "
                f"got {out_img.size}. Resizing to match exact dimensions."
            )
            out_img = out_img.resize((target_w, target_h), Image.Resampling.BICUBIC)

        return out_img
