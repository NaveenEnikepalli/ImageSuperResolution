"""SwinIR x8 Classical Super-Resolution Inference Engine.

Author: Antigravity
Purpose: Production inference engine for SwinIR-M DF2K x8 pretrained model.
Handles local PyTorch execution (CPU/CUDA), automatic checkpoint download,
window-size reflection padding, and post-inference cropping.
"""

import logging
import urllib.request
from pathlib import Path
from typing import Union, Optional
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from training.teacher.swinir_model import SwinIR

logger = logging.getLogger(__name__)

# Official SwinIR-M DF2K x8 Checkpoint URL from GitHub Release
SWINIR_X8_URL = (
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pth"
)


class SwinIRInference:
    """Inference runner for official SwinIR-M classical super-resolution x8 model."""

    def __init__(
        self,
        scale: int = 8,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        download_if_missing: bool = True,
    ) -> None:
        """Initialize SwinIRInference engine.

        Args:
            scale (int): Upscaling scale factor (must be 8). Defaults to 8.
            checkpoint_path (Optional[Union[str, Path]]): Local path to SwinIR x8 weights.
            device (Optional[Union[str, torch.device]]): Compute device (CPU or CUDA).
            download_if_missing (bool): Auto-download checkpoint if missing. Defaults to True.
        """
        if scale != 8:
            raise ValueError(f"SwinIRInference currently configured for scale=8, got scale={scale}")

        self.scale = scale
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if checkpoint_path is None:
            root_dir = Path(__file__).resolve().parents[1]
            candidates = [
                root_dir / "checkpoints" / "swinir" / "swinir_x8.pth",
                Path("checkpoints/swinir/swinir_x8.pth"),
            ]
            checkpoint_path = next((p for p in candidates if p.exists()), candidates[0])
        else:
            path_obj = Path(checkpoint_path)
            if not path_obj.exists():
                root_dir = Path(__file__).resolve().parents[1]
                candidate_in_root = root_dir / path_obj
                if candidate_in_root.exists():
                    path_obj = candidate_in_root
            checkpoint_path = path_obj

        self.checkpoint_path = Path(checkpoint_path)
        self.download_if_missing = download_if_missing

        # Official SwinIR-M Classical Super-Resolution x8 architecture parameters
        self.window_size = 8
        self.model = SwinIR(
            upscale=self.scale,
            in_chans=3,
            img_size=64,
            window_size=self.window_size,
            img_range=1.0,
            depths=(6, 6, 6, 6, 6, 6),
            embed_dim=180,
            num_heads=(6, 6, 6, 6, 6, 6),
            mlp_ratio=2.0,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )

        self._ensure_checkpoint_available()
        self._load_checkpoint()

        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"SwinIRInference engine ready [Scale x{self.scale}, Device={self.device}, "
            f"Checkpoint={self.checkpoint_path}]"
        )

    def _ensure_checkpoint_available(self) -> None:
        """Ensure local SwinIR x8 checkpoint exists, downloading automatically if missing."""
        if self.checkpoint_path.exists():
            return

        if not self.download_if_missing:
            logger.warning(
                f"SwinIR x8 checkpoint not found at {self.checkpoint_path} and auto-download disabled."
            )
            return

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"SwinIR x8 checkpoint not found locally. Downloading official pretrained weights from:\n"
            f"  {SWINIR_X8_URL} -> {self.checkpoint_path}"
        )
        try:
            # Download checkpoint with progress reporting
            def _progress(block_num, block_size, total_size):
                if total_size > 0 and block_num % 100 == 0:
                    downloaded = block_num * block_size
                    percent = min(100.0, (downloaded / total_size) * 100.0)
                    logger.info(f"Downloading SwinIR x8 weights: {percent:.1f}% ({downloaded}/{total_size} bytes)")

            urllib.request.urlretrieve(SWINIR_X8_URL, str(self.checkpoint_path), reporthook=_progress)
            logger.info(f"Successfully downloaded SwinIR x8 weights to: {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to download SwinIR x8 weights from {SWINIR_X8_URL}: {e}")
            raise RuntimeError(
                f"Failed to download SwinIR x8 checkpoint: {e}. "
                f"Please manually download {SWINIR_X8_URL} to {self.checkpoint_path}."
            )

    def _load_checkpoint(self) -> None:
        """Load state dictionary into SwinIR model with key normalization."""
        if not self.checkpoint_path.exists():
            logger.warning(
                f"SwinIR x8 checkpoint file missing at {self.checkpoint_path}. "
                f"Running with uninitialized weights."
            )
            return

        logger.info(f"Loading SwinIR x8 state dict from: {self.checkpoint_path}")
        raw_dict = torch.load(self.checkpoint_path, map_location="cpu")

        # Extract weights dict if wrapped
        if isinstance(raw_dict, dict):
            for key in ["params_ema", "params", "state_dict", "model"]:
                if key in raw_dict and isinstance(raw_dict[key], dict):
                    raw_dict = raw_dict[key]
                    break

        model_state = self.model.state_dict()
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

            if ".residual_group." in new_k:
                new_k = new_k.replace(".residual_group.", ".")

            if new_k in model_state and model_state[new_k].shape == v.shape:
                cleaned_dict[new_k] = v

        # Populate mean buffer if missing
        if "mean" in model_state and "mean" not in cleaned_dict:
            cleaned_dict["mean"] = model_state["mean"]

        self.model.load_state_dict(cleaned_dict, strict=False)
        logger.info("Successfully loaded SwinIR x8 weights into model architecture.")

    @torch.no_grad()
    def enhance(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Run SwinIR x8 super-resolution inference on input RGB image.

        Args:
            image (Union[Image.Image, np.ndarray]): Input low-resolution RGB image.

        Returns:
            Image.Image: High-resolution output PIL Image (8W x 8H).
        """
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            np_img = np.array(pil_img)
        elif isinstance(image, np.ndarray):
            np_img = image
            if np_img.ndim == 2:
                np_img = np.stack([np_img] * 3, axis=-1)
            elif np_img.shape[2] == 4:
                np_img = np_img[:, :, :3]
        else:
            raise TypeError(f"Expected PIL Image or numpy ndarray, got {type(image)}")

        orig_h, orig_w = np_img.shape[:2]

        # Convert to tensor (1, 3, H, W) normalized to [0.0, 1.0]
        tensor_in = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor_in = tensor_in.to(self.device)

        # Handle window-size alignment padding
        pad_h = (self.window_size - orig_h % self.window_size) % self.window_size
        pad_w = (self.window_size - orig_w % self.window_size) % self.window_size

        if pad_h > 0 or pad_w > 0:
            logger.info(
                f"SwinIR window-size padding: image size ({orig_w}x{orig_h}), "
                f"padding ({pad_w}, {pad_h}) for window_size={self.window_size}"
            )
            tensor_padded = F.pad(tensor_in, (0, pad_w, 0, pad_h), mode="reflect")
            sr_padded = self.model(tensor_padded)
            # Crop output back to exact target scaled size (8*orig_h, 8*orig_w)
            sr_tensor = sr_padded[:, :, : orig_h * self.scale, : orig_w * self.scale]
        else:
            sr_tensor = self.model(tensor_in)

        # Clamp output to [0.0, 1.0]
        sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)

        # Post-process tensor back to uint8 numpy array
        sr_np = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

        target_w = orig_w * self.scale
        target_h = orig_h * self.scale

        out_img = Image.fromarray(sr_np, mode="RGB")
        if out_img.size != (target_w, target_h):
            out_img = out_img.resize((target_w, target_h), Image.Resampling.BICUBIC)

        return out_img
