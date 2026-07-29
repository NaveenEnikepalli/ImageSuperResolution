"""
Constants configuration module for Lightweight Image Super-Resolution framework.
Defines project-wide constants for validation, default naming, and supported properties.
"""

from typing import Tuple

# Supported dataset names for validation in config
SUPPORTED_DATASETS: Tuple[str, ...] = ("DIV2K", "Set5", "Set14")

# Supported upscaling factors
SUPPORTED_SCALES: Tuple[int, ...] = (2, 4)

# File extensions allowed for input images
SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")

# Default run naming prefix
DEFAULT_EXPERIMENT_NAME: str = "default_distill_run"

# Standard metrics computed during evaluation
DEFAULT_METRICS: Tuple[str, ...] = ("psnr", "ssim")
