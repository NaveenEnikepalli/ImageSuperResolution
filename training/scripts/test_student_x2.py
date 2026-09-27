"""Verification script for Student x2 Model architecture and training configuration.

Author: Antigravity
Purpose: Validate StudentModel instantiated with scale=2, verify forward pass shape,
parameter count, and configuration loader.
"""

import sys
import logging
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.models.student_model import StudentModel
from training.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestStudentX2")


def verify_student_x2() -> bool:
    """Verify Student x2 architecture and configuration."""
    logger.info("=== Verifying Student x2 Architecture ===")

    # 1. Load config
    config_path = Path("training/configs/student_x2_training.yaml")
    if not config_path.exists():
        logger.error(f"Config file missing: {config_path}")
        return False

    config = load_config(config_path)
    assert config.model.scale == 2, f"Expected scale=2, got {config.model.scale}"

    # 2. Instantiate model with scale=2
    model = StudentModel(
        in_channels=3,
        out_channels=3,
        num_features=48,
        distilled_channels=24,
        num_blocks=3,
        scale=2,
        negative_slope=0.05
    )

    model.eval()

    # 3. Test forward pass with dummy tensor (1, 3, 96, 96)
    x = torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        out = model(x)

    expected_shape = (1, 3, 192, 192)
    assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Student x2 Model initialized successfully!")
    logger.info(f"Input shape: {list(x.shape)} -> Output shape: {list(out.shape)}")
    logger.info(f"Total parameter count: {params:,}")
    return True


if __name__ == "__main__":
    success = verify_student_x2()
    if success:
        print("Student x2 verification PASSED!")
        sys.exit(0)
    else:
        print("Student x2 verification FAILED!")
        sys.exit(1)
