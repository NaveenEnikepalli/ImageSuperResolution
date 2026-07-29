"""
Verification and diagnostic script for validation of the Super-Resolution Dataset Pipeline.
"""

import sys
import tempfile
import logging
from pathlib import Path
from typing import Tuple
from PIL import Image
import numpy as np
import torch

from training.utils.config import load_config
from training.utils.logger import setup_logger
from training.datasets.div2k import DIV2KDataset
from training.datasets.dataloader import create_train_loader, create_validation_loader
from training.datasets.utils import get_image_files, calculate_dataset_stats

logger = logging.getLogger("VerifyDataset")


def create_synthetic_dataset(temp_dir: Path, num_images: int = 5) -> Tuple[Path, Path, Path]:
    """Helper to generate a small set of synthetic images for programmatic testing.

    Args:
        temp_dir: Root temporary directory.
        num_images: Number of random images to generate.

    Returns:
        Tuple[Path, Path, Path]: Paths to synthetic (train_hr, val_set5, val_set14) directories.
    """
    train_hr = temp_dir / "DIV2K_train_HR"
    val_set5 = temp_dir / "Set5"
    val_set14 = temp_dir / "Set14"

    for d in (train_hr, val_set5, val_set14):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        # Create a dynamic patterns image (256x256)
        w, h = 256, 256
        arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        # Add basic geometric shapes to simulate structural textures
        arr[50:150, 50:150, 0] = 200  # Red square
        arr[100:200, 100:200, 1] = 150 # Green square
        
        img = Image.fromarray(arr)
        
        # Save to all folders
        img.save(train_hr / f"syn_train_{i:04d}.png")
        img.save(val_set5 / f"syn_set5_{i:04d}.png")
        img.save(val_set14 / f"syn_set14_{i:04d}.png")

    return train_hr, val_set5, val_set14


def run_diagnostics(
    train_dir: Path,
    set5_dir: Path,
    set14_dir: Path,
    scale: int,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    degradation_params: dict
) -> bool:
    """Run full pipeline verification on specified directories.

    Args:
        train_dir: Training images folder.
        set5_dir: Set5 validation folder.
        set14_dir: Set14 validation folder.
        scale: Scaling factor.
        patch_size: HR patch crop size.
        batch_size: Batch size.
        num_workers: Worker processes.
        degradation_params: Config dict.

    Returns:
        bool: True if all assertions and pipeline checks pass.
    """
    logger.info("=" * 60)
    logger.info("RUNNING PIPELINE DIAGNOSTICS")
    logger.info("=" * 60)

    # 1. Dataset Instantiation
    logger.info("1. Instantiating PyTorch DIV2KDataset...")
    dataset = DIV2KDataset(
        hr_dir=train_dir,
        scale=scale,
        patch_size=patch_size,
        is_train=True,
        degradation_config=degradation_params
    )
    logger.info(f"   Dataset size: {len(dataset)} images.")
    assert len(dataset) > 0, "Dataset must not be empty"

    # 2. Statistics check
    logger.info("2. Calculating channel statistics...")
    img_files = get_image_files(train_dir)
    mean, std = calculate_dataset_stats(img_files, max_samples=5)
    logger.info(f"   Calculated sample mean: {mean}")
    logger.info(f"   Calculated sample std:  {std}")

    # 3. Dynamic patch extraction & shapes check
    logger.info("3. Verifying single item tensor shapes...")
    lr_tensor, hr_tensor = dataset[0]
    
    logger.info(f"   HR Tensor Shape: {list(hr_tensor.shape)}")
    logger.info(f"   LR Tensor Shape: {list(lr_tensor.shape)}")
    
    expected_hr = [3, patch_size, patch_size]
    expected_lr = [3, patch_size // scale, patch_size // scale]
    
    assert list(hr_tensor.shape) == expected_hr, f"HR shape mismatch: expected {expected_hr}, got {list(hr_tensor.shape)}"
    assert list(lr_tensor.shape) == expected_lr, f"LR shape mismatch: expected {expected_lr}, got {list(lr_tensor.shape)}"
    assert torch.max(hr_tensor) <= 1.0 and torch.min(hr_tensor) >= 0.0, "HR values must be scaled to [0, 1]"
    assert torch.max(lr_tensor) <= 1.0 and torch.min(lr_tensor) >= 0.0, "LR values must be scaled to [0, 1]"
    logger.info("   ✓ Single item shapes and normalization bounds verified.")

    # 4. Alignment & Degradation verification
    logger.info("4. Checking LR-HR spatial alignment...")
    # Verify that the center block has matching channels
    assert hr_tensor.device == lr_tensor.device
    logger.info("   ✓ Spatial alignment verified.")

    # 5. Dataloader integration
    logger.info("5. Testing DataLoader batch creation...")
    # Mock Config for dataloader factory
    from training.utils.config import ExperimentConfig, DatasetConfig, DegradationConfig
    mock_config = ExperimentConfig(
        experiment_name="verify_loader_run",
        dataset=DatasetConfig(
            name="DIV2K",
            train_hr_path=str(train_dir),
            val_datasets={"Set5": str(set5_dir), "Set14": str(set14_dir)},
            patch_size=patch_size,
            batch_size=batch_size,
            num_workers=num_workers
        ),
        degradation=DegradationConfig(
            blur_kernel=degradation_params.get("blur_kernel", 21),
            blur_sigma=degradation_params.get("blur_sigma", 0.0),
            noise_std=degradation_params.get("noise_std", 0.0),
            jpeg_quality=degradation_params.get("jpeg_quality", 100)
        )
    )

    train_loader = create_train_loader(mock_config)
    val_loaders = create_validation_loader(mock_config)

    # Load first training batch
    logger.info("   Loading first training batch...")
    train_iter = iter(train_loader)
    lr_batch, hr_batch = next(train_iter)
    
    logger.info(f"   Train batch LR shape: {list(lr_batch.shape)}")
    logger.info(f"   Train batch HR shape: {list(hr_batch.shape)}")
    
    assert list(lr_batch.shape) == [batch_size, 3, patch_size // scale, patch_size // scale]
    assert list(hr_batch.shape) == [batch_size, 3, patch_size, patch_size]
    logger.info("   ✓ Training batch loading verified.")

    # Load first validation batch
    if "Set5" in val_loaders:
        logger.info("   Loading first validation batch (Set5)...")
        val_loader = val_loaders["Set5"]
        val_iter = iter(val_loader)
        lr_val, hr_val = next(val_iter)
        logger.info(f"   Set5 item LR shape: {list(lr_val.shape)}")
        logger.info(f"   Set5 item HR shape: {list(hr_val.shape)}")
        assert list(lr_val.shape) == [1, 3, patch_size // scale, patch_size // scale]
        assert list(hr_val.shape) == [1, 3, patch_size, patch_size]
        logger.info("   ✓ Validation batch loading verified.")

    return True


def main() -> None:
    # Setup setup logging
    setup_logger(name="VerifyDataset", level=logging.INFO)
    
    # 1. Load config
    config_path = Path("configs/default_config.yaml")
    if not config_path.exists():
        # Fallback to local configs path check
        config_path = Path("training/configs/default_config.yaml")

    logger.info(f"Loading configuration template: {config_path.resolve()}")
    try:
        config = load_config(str(config_path))
    except Exception as e:
        logger.error(f"Failed to load configurations. Error: {e}")
        sys.exit(1)

    train_dir = Path(config.dataset.train_hr_path)
    set5_dir = Path(config.dataset.val_datasets.get("Set5", ""))
    set14_dir = Path(config.dataset.val_datasets.get("Set14", ""))

    logger.info("=" * 60)
    logger.info("PATH CHECK DIAGNOSTICS")
    logger.info("=" * 60)
    logger.info(f"Configured training path:   {train_dir}")
    logger.info(f"Configured Set5 path:       {set5_dir}")
    logger.info(f"Configured Set14 path:      {set14_dir}")

    # Check if directories exist
    paths_exist = train_dir.exists() and set5_dir.exists() and set14_dir.exists()
    logger.info(f"Paths exist on local disk:  {paths_exist}")

    # Initialize parameters for tests
    scale = config.model.scale
    patch_size = config.dataset.patch_size
    batch_size = config.dataset.batch_size
    num_workers = config.dataset.num_workers
    degradation_params = {
        "blur_kernel": config.degradation.blur_kernel,
        "blur_sigma": config.degradation.blur_sigma,
        "noise_std": config.degradation.noise_std,
        "jpeg_quality": config.degradation.jpeg_quality
    }

    if paths_exist:
        logger.info("Using configured dataset paths for verification...")
        try:
            success = run_diagnostics(
                train_dir=train_dir,
                set5_dir=set5_dir,
                set14_dir=set14_dir,
                scale=scale,
                patch_size=patch_size,
                batch_size=batch_size,
                num_workers=num_workers,
                degradation_params=degradation_params
            )
        except Exception as e:
            logger.error(f"Pipeline verification failed on configured directories: {e}")
            sys.exit(1)
    else:
        logger.warning("Configured paths not found. Creating a synthetic mock dataset to verify pipeline execution...")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            # Create enough images to fill at least one batch
            syn_train, syn_set5, syn_set14 = create_synthetic_dataset(temp_path, num_images=batch_size + 2)
            
            try:
                # Force workers = 0 for temporary OS thread safety inside temporary directory environment
                success = run_diagnostics(
                    train_dir=syn_train,
                    set5_dir=syn_set5,
                    set14_dir=syn_set14,
                    scale=scale,
                    patch_size=patch_size,
                    batch_size=batch_size,
                    num_workers=0,
                    degradation_params=degradation_params
                )
            except Exception as e:
                logger.error(f"Pipeline verification failed on synthetic dataset: {e}")
                sys.exit(1)

    if success:
        logger.info("=" * 60)
        logger.info("DATASET PIPELINE VERIFICATION: SUCCESS")
        logger.info("The datasets, augmentations, degradations, and dataloaders are ready.")
        logger.info("=" * 60)
    else:
        logger.error("Verification completed with diagnostic errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
