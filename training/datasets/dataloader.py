"""
Dataloader builder utilities for training and validation datasets.
"""

import logging
from pathlib import Path
from typing import Dict
from torch.utils.data import DataLoader

from training.utils.config import ExperimentConfig
from training.datasets.div2k import DIV2KDataset

logger = logging.getLogger(__name__)


def create_train_loader(config: ExperimentConfig) -> DataLoader:
    """Create PyTorch DataLoader for training DIV2K dataset.

    Args:
        config: Centralized experiment configurations object.

    Returns:
        DataLoader: Instantiated training DataLoader.
    """
    logger.info("Initializing training DIV2K dataset...")
    
    # Resolve HR dir path from dataset config
    hr_dir = Path(config.dataset.train_hr_path)
    
    # Build dataset instance
    dataset = DIV2KDataset(
        hr_dir=hr_dir,
        scale=config.model.scale,
        patch_size=config.dataset.patch_size,
        is_train=True,
        degradation_config={
            "blur_kernel": config.degradation.blur_kernel,
            "blur_sigma": config.degradation.blur_sigma,
            "noise_std": config.degradation.noise_std,
            "jpeg_quality": config.degradation.jpeg_quality
        }
    )

    # Determine multi-worker optimization settings
    num_workers = config.dataset.num_workers
    persistent_workers = True if num_workers > 0 else False

    # Instantiate DataLoader
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        drop_last=True
    )
    
    logger.info(
        f"Training DataLoader initialized with batch_size={config.dataset.batch_size}, "
        f"num_workers={num_workers}, persistent_workers={persistent_workers}"
    )
    return loader


def create_validation_loader(config: ExperimentConfig) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders for each validation dataset specified in the config.

    Args:
        config: Centralized experiment configurations object.

    Returns:
        Dict[str, DataLoader]: Mapping of dataset name (e.g., 'Set5') to its DataLoader.
    """
    loaders = {}
    
    # Loop over all configured validation datasets
    for name, path_str in config.dataset.val_datasets.items():
        logger.info(f"Initializing validation dataset '{name}' from: {path_str}...")
        
        val_dir = Path(path_str)
        if not val_dir.exists():
            logger.warning(f"Validation directory for '{name}' not found at: {val_dir}. Skipping.")
            continue

        # In validation, evaluate on full scale patches (usually standard image size or a large patch).
        # We can use a center crop equal to dataset patch_size, or larger.
        # Set5/Set14 are small, so patch_size = config.dataset.patch_size is appropriate.
        dataset = DIV2KDataset(
            hr_dir=val_dir,
            scale=config.model.scale,
            patch_size=config.dataset.patch_size,
            is_train=False,
            # Validation downsampling is pure bicubic downsampling (no noise/blur/compression)
            degradation_config={
                "blur_kernel": 21,
                "blur_sigma": 0.0,
                "noise_std": 0.0,
                "jpeg_quality": 100
            }
        )

        # Batch size 1 is standard for validation to handle arbitrary image aspect ratios if needed
        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Single worker is safer and faster for validation sets
            pin_memory=True
        )
        
        loaders[name] = loader
        logger.info(f"Validation DataLoader for '{name}' initialized with {len(dataset)} images.")

    return loaders
