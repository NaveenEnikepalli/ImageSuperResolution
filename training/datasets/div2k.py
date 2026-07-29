"""
DIV2K and validation dataset implementation for PyTorch.
Loads high-resolution images, crops/augments them, and generates low-resolution inputs dynamically.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from torch.utils.data import Dataset
import torch

from training.datasets.utils import get_image_files, verify_image, load_image_rgb
from training.datasets.transforms import get_transforms
from training.datasets.degradation import DegradationPipeline

logger = logging.getLogger(__name__)


class DIV2KDataset(Dataset):
    """PyTorch Dataset loading high-resolution images and dynamically generating matching low-resolution patches."""

    def __init__(
        self,
        hr_dir: Path,
        scale: int,
        patch_size: int,
        is_train: bool = True,
        degradation_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the DIV2K dataset.

        Args:
            hr_dir: Absolute path to folder containing High-Resolution target images.
            scale: Upscaling scale factor (choices: 2, 4).
            patch_size: Square dimension of the target crop patch (e.g. 192 for HR, 128 for validation).
            is_train: If True, applies training augmentations and random cropping. Defaults to True.
            degradation_config: Optional dictionary containing custom degradation parameters:
                - blur_kernel: int
                - blur_sigma: float
                - noise_std: float
                - jpeg_quality: int
        """
        self.hr_dir = Path(hr_dir).resolve()
        self.scale = scale
        self.patch_size = patch_size
        self.is_train = is_train

        # Verify dataset folder existence
        if not self.hr_dir.exists():
            raise FileNotFoundError(f"High-resolution image directory does not exist at: {self.hr_dir}")

        # Scan for supported image files
        logger.info(f"Scanning directory for image files: {self.hr_dir}")
        all_files = get_image_files(self.hr_dir)
        if not all_files:
            raise ValueError(f"No supported image files found in {self.hr_dir}")

        # Verify load integrity and filter out corrupted images
        logger.info("Verifying image files integrity...")
        self.image_paths = []
        for p in all_files:
            if verify_image(p):
                self.image_paths.append(p)
            else:
                logger.warning(f"Skipping corrupted or unreadable image file: {p}")

        if not self.image_paths:
            raise ValueError(f"No valid, uncorrupted image files found in {self.hr_dir}")

        logger.info(f"Successfully loaded and verified {len(self.image_paths)} images.")

        # Build transform pipelines
        self.transforms = get_transforms(self.patch_size, is_train=self.is_train)

        # Build degradation pipeline
        degr_params = degradation_config or {}
        self.degradation = DegradationPipeline(
            scale=self.scale,
            blur_kernel=degr_params.get("blur_kernel", 21),
            blur_sigma=degr_params.get("blur_sigma", 0.0),
            noise_std=degr_params.get("noise_std", 0.0),
            jpeg_quality=degr_params.get("jpeg_quality", 100)
        )

    def __len__(self) -> int:
        """Return total number of validated images in the dataset.

        Returns:
            int: Image count.
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load HR image, crop/augment it to a patch, and generate the corresponding LR tensor.

        Args:
            index: Dataset index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple (LR_tensor, HR_tensor),
                both normalized to range [0.0, 1.0].
        """
        path = self.image_paths[index]
        
        # Load image safely
        hr_img = load_image_rgb(path)

        # Apply spatial crop and geometric augmentations on the HR image
        hr_patch = self.transforms(hr_img)

        # Apply degradation pipeline to crop and yield matching normalized (LR, HR) tensors
        lr_tensor, hr_tensor = self.degradation(hr_patch)

        return lr_tensor, hr_tensor
