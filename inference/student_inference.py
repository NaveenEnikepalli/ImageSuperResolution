"""Student Model Inference Engine.

Author: Antigravity
Purpose: High-level inference pipeline for Student Super-Resolution models (x2 and x4).
Handles device assignment, checkpoint loading, pre/post-processing, and clamping.
"""

import os
import gc
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

        # Resolve checkpoint path: check explicit path, MODEL_DIR env var, then default locations
        root_dir = Path(__file__).resolve().parents[1]
        model_dir_env = os.environ.get("MODEL_DIR")

        if checkpoint_path is not None:
            path_obj = Path(checkpoint_path)
            if not path_obj.is_absolute():
                candidates = []
                if model_dir_env:
                    candidates.append(Path(model_dir_env) / path_obj.name)
                    candidates.append(Path(model_dir_env) / path_obj)
                candidates.append(root_dir / path_obj)
                candidates.append(path_obj)
                path_obj = next((p for p in candidates if p.exists()), candidates[0])
            self.checkpoint_path = path_obj.resolve()
        else:
            candidates = []
            if model_dir_env:
                m_dir = Path(model_dir_env)
                if not m_dir.is_absolute():
                    m_dir = (root_dir / m_dir) if (root_dir / m_dir).exists() else m_dir.resolve()
                candidates.extend([
                    m_dir / f"best_student_x{self.scale}.pth",
                    m_dir / f"student_x{self.scale}.pth",
                    m_dir / f"best_student_x{self.scale}.pt",
                    m_dir / f"student_x{self.scale}.pt",
                ])
            candidates.extend([
                root_dir / "checkpoints" / "student" / f"best_student_x{self.scale}.pth",
                root_dir / "checkpoints" / "student" / f"student_x{self.scale}.pth",
                root_dir / "checkpoints" / f"best_student_x{self.scale}.pth",
                Path("checkpoints/student") / f"best_student_x{self.scale}.pth",
                Path("checkpoints/student") / f"student_x{self.scale}.pth",
            ])
            self.checkpoint_path = next((p.resolve() for p in candidates if p.exists()), candidates[0].resolve())

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
            raise FileNotFoundError(
                f"Required Student x{self.scale} checkpoint not found at: '{self.checkpoint_path}'. "
                f"Please ensure 'best_student_x{self.scale}.pth' exists or configure the MODEL_DIR environment variable."
            )

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

    def _forward_tiled(self, x: torch.Tensor, tile_size: int = 256, tile_pad: int = 16) -> torch.Tensor:
        """Run model in overlapping tiles to strictly constrain peak memory usage on CPU/GPU."""
        batch, channel, height, width = x.shape
        output_height = height * self.scale
        output_width = width * self.scale

        output = torch.zeros((batch, channel, output_height, output_width), dtype=x.dtype, device=x.device)
        stride = tile_size - 2 * tile_pad

        y_starts = list(range(0, height, stride))
        x_starts = list(range(0, width, stride))

        for y_start in y_starts:
            for x_start in x_starts:
                in_y0 = max(y_start - tile_pad, 0)
                in_y1 = min(y_start + stride + tile_pad, height)
                in_x0 = max(x_start - tile_pad, 0)
                in_x1 = min(x_start + stride + tile_pad, width)

                in_tile = x[:, :, in_y0:in_y1, in_x0:in_x1]
                out_tile = self.model(in_tile)

                valid_y0 = y_start
                valid_y1 = min(y_start + stride, height)
                valid_x0 = x_start
                valid_x1 = min(x_start + stride, width)

                tile_crop_y0 = (valid_y0 - in_y0) * self.scale
                tile_crop_y1 = tile_crop_y0 + (valid_y1 - valid_y0) * self.scale
                tile_crop_x0 = (valid_x0 - in_x0) * self.scale
                tile_crop_x1 = tile_crop_x0 + (valid_x1 - valid_x0) * self.scale

                out_y0 = valid_y0 * self.scale
                out_y1 = valid_y1 * self.scale
                out_x0 = valid_x0 * self.scale
                out_x1 = valid_x1 * self.scale

                output[:, :, out_y0:out_y1, out_x0:out_x1] = out_tile[:, :, tile_crop_y0:tile_crop_y1, tile_crop_x0:tile_crop_x1]

        return output

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

        # Forward execution: use tiled inference for larger images (> 384px) to prevent OOM
        if orig_h > 384 or orig_w > 384:
            sr_tensor = self._forward_tiled(tensor_in, tile_size=256, tile_pad=16)
        else:
            sr_tensor = self.model(tensor_in)

        del tensor_in

        # Post-process tensor directly to uint8 on PyTorch to avoid allocating multiple 56+ MiB float32 NumPy arrays
        sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)
        sr_tensor = (sr_tensor.mul(255.0).add_(0.5)).clamp_(0.0, 255.0).to(torch.uint8)
        sr_np = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        del sr_tensor
        gc.collect()

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
