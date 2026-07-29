"""
Unit tests validating Phase 2 Dataset Pipeline components.
"""

import tempfile
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from training.utils.config import ExperimentConfig, DatasetConfig, DegradationConfig
from training.datasets.utils import is_image_file, verify_image, load_image_rgb, get_image_files
from training.datasets.transforms import RandomCrop, CenterCrop, RandomHorizontalFlip, get_transforms
from training.datasets.degradation import DegradationPipeline
from training.datasets.div2k import DIV2KDataset
from training.datasets.dataloader import create_train_loader, create_validation_loader


def _create_dummy_image(path: Path, size: Tuple[int, int] = (256, 256)) -> None:
    """Create a dummy image file for test validation."""
    arr = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_dataset_utils() -> None:
    """Verify image loading, file checks, and load validations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        img_path = temp_path / "test_img.png"
        corrupt_path = temp_path / "corrupt_img.png"
        txt_path = temp_path / "test.txt"

        # Write valid image
        _create_dummy_image(img_path)
        # Write corrupted image (truncated metadata)
        with open(corrupt_path, "w", encoding="utf-8") as f:
            f.write("corrupted image content")
        # Write plain text file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("text content")

        # 1. Test image suffix checking
        assert is_image_file(img_path)
        assert not is_image_file(txt_path)

        # 2. Test directory scanning
        files = get_image_files(temp_path)
        assert img_path in files
        assert corrupt_path in files
        assert txt_path not in files

        # 3. Test verification
        assert verify_image(img_path)
        assert not verify_image(corrupt_path)

        # 4. Test safe loading
        pil_img = load_image_rgb(img_path)
        assert pil_img.mode == "RGB"
        assert pil_img.size == (256, 256)


def test_transforms() -> None:
    """Verify crop and augmentations dimensions and properties."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "transform_img.png"
        _create_dummy_image(img_path, size=(200, 150))
        img = load_image_rgb(img_path)

        # 1. Random Crop
        crop_50 = RandomCrop(50)
        cropped = crop_50(img)
        assert cropped.size == (50, 50)

        # 2. Center Crop
        center_100 = CenterCrop(100)
        centered = center_100(img)
        assert centered.size == (100, 100)

        # 3. Flip
        flip = RandomHorizontalFlip(p=1.0)
        flipped = flip(img)
        assert flipped.size == img.size


def test_degradation_pipeline() -> None:
    """Verify that degradation scales patch shapes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "degr_img.png"
        _create_dummy_image(img_path, size=(128, 128))
        img = load_image_rgb(img_path)

        # Scale factor 4 downsampler
        pipeline = DegradationPipeline(
            scale=4,
            blur_kernel=5,
            blur_sigma=1.2,
            noise_std=0.01,
            jpeg_quality=75
        )

        lr_tensor, hr_tensor = pipeline(img)

        # Check tensor normalization
        assert torch.max(lr_tensor) <= 1.0
        assert torch.min(lr_tensor) >= 0.0

        # Check shape output logic
        assert list(hr_tensor.shape) == [3, 128, 128]
        assert list(lr_tensor.shape) == [3, 32, 32]


def test_div2k_dataset_instantiation() -> None:
    """Verify dataset loading, skips corruption, and outputs tensors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        
        # Write 3 valid and 1 corrupted images
        for i in range(3):
            _create_dummy_image(temp_path / f"img_{i}.png", size=(256, 256))
        with open(temp_path / "corrupt.png", "w", encoding="utf-8") as f:
            f.write("corrupt")

        dataset = DIV2KDataset(
            hr_dir=temp_path,
            scale=2,
            patch_size=128,
            is_train=True
        )

        # Should verify and filter out corrupt.png, leaving exactly 3 images
        assert len(dataset) == 3

        lr_tensor, hr_tensor = dataset[0]
        assert list(hr_tensor.shape) == [3, 128, 128]
        assert list(lr_tensor.shape) == [3, 64, 64]


def test_dataloader_creation() -> None:
    """Verify train and validation dataloaders return batch shape formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        train_dir = temp_path / "train"
        val_dir = temp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        # Create enough images to fill training batch size 4
        for i in range(6):
            _create_dummy_image(train_dir / f"t_{i}.png", size=(256, 256))
            _create_dummy_image(val_dir / f"v_{i}.png", size=(256, 256))

        config = ExperimentConfig(
            experiment_name="test_dataloader",
            dataset=DatasetConfig(
                name="DIV2K",
                train_hr_path=str(train_dir),
                val_datasets={"val_subset": str(val_dir)},
                patch_size=128,
                batch_size=4,
                num_workers=0
            )
        )

        # 1. Train loader
        train_loader = create_train_loader(config)
        assert isinstance(train_loader, DataLoader)
        train_iter = iter(train_loader)
        lr_batch, hr_batch = next(train_iter)
        assert list(lr_batch.shape) == [4, 3, 32, 32]
        assert list(hr_batch.shape) == [4, 3, 128, 128]

        # 2. Validation loaders
        val_loaders = create_validation_loader(config)
        assert "val_subset" in val_loaders
        val_loader = val_loaders["val_subset"]
        assert isinstance(val_loader, DataLoader)
        val_iter = iter(val_loader)
        lr_val, hr_val = next(val_iter)
        assert list(lr_val.shape) == [1, 3, 32, 32]
        assert list(hr_val.shape) == [1, 3, 128, 128]
