"""
Dataset utilities for verification, safe image loading, corrupt file detection, and statistics.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image

from training.utils.constants import SUPPORTED_IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def is_image_file(path: Path) -> bool:
    """Check if the path points to an image file with a supported extension.

    Args:
        path: Path to check.

    Returns:
        bool: True if extension is supported, False otherwise.
    """
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def verify_image(path: Path) -> bool:
    """Verify that an image is not corrupted and can be successfully loaded.

    Args:
        path: Absolute path to target image file.

    Returns:
        bool: True if verified and loadable, False if corrupted.
    """
    if not path.is_file():
        return False
        
    try:
        with Image.open(path) as img:
            # 1. Verify image structure integrity
            img.verify()
        
        # 2. verify() doesn't load data, so we attempt opening and loading pixel data
        with Image.open(path) as img:
            img.load()
            
        return True
    except Exception as e:
        logger.warning(f"Corrupted or invalid image detected at {path}. Error: {e}")
        return False


def load_image_rgb(path: Path) -> Image.Image:
    """Load an image from disk and guarantee it is converted to RGB format.

    Args:
        path: Absolute path to target image file.

    Returns:
        Image.Image: Loaded PIL Image.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found at path: {path}")

    try:
        with Image.open(path) as img:
            img.load()
            if img.mode != "RGB":
                return img.convert("RGB")
            return img.copy()
    except Exception as e:
        logger.error(f"Failed to load image from {path}. Error: {e}")
        raise e


def get_image_files(directory: Path) -> List[Path]:
    """Scan directory recursively for supported, validated image files.

    Args:
        directory: Root directory path.

    Returns:
        List[Path]: Sorted list of validated image files.
    """
    if not directory.exists() or not directory.is_dir():
        logger.error(f"Target directory does not exist: {directory}")
        return []

    # Get sorted image files with supported extensions
    image_paths = sorted([
        p for p in directory.rglob("*")
        if p.is_file() and is_image_file(p)
    ])
    
    return image_paths


def calculate_dataset_stats(image_paths: List[Path], max_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and standard deviation of dataset channels.

    Samples a subset of files to avoid memory overhead on huge datasets.

    Args:
        image_paths: List of absolute image paths.
        max_samples: Maximum files to sample for stats calculation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Channel mean and std arrays of shape (3,).
    """
    if not image_paths:
        return np.zeros(3), np.zeros(3)

    sampled_paths = image_paths
    if len(image_paths) > max_samples:
        indices = np.random.choice(len(image_paths), max_samples, replace=False)
        sampled_paths = [image_paths[i] for i in indices]

    means = []
    stds = []

    for path in sampled_paths:
        try:
            img = load_image_rgb(path)
            # Normalize to [0, 1]
            arr = np.array(img, dtype=np.float32) / 255.0
            # Calculate mean and std along height and width axes (axis 0, 1)
            means.append(np.mean(arr, axis=(0, 1)))
            stds.append(np.std(arr, axis=(0, 1)))
        except Exception:
            continue

    if not means:
        return np.zeros(3), np.zeros(3)

    return np.mean(means, axis=0), np.mean(stds, axis=0)
